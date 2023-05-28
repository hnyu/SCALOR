import argparse
import os
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from movi_data_set import MoviDataset
from log_utils import log_summary
from utils import save_ckpt, load_ckpt, print_scalor
from common import *
import parse

from tensorboardX import SummaryWriter

from scalor import SCALOR


def compute_ari(seg,
                gt_seg,
                num_groups,
                ignore_background=True):
    """Converted from the JAX version:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py#L111
    """
    # Following SAVI, it may be that num_groups <= max(segmentation). We prune
    # out extra objects here. For example, movi_e has an instance id of 64 in an
    # outlier video.
    # https://github.com/google-research/slot-attention-video/blob/main/savi/lib/preprocessing.py#L414
    gt_seg = torch.where(gt_seg >= num_groups, torch.zeros_like(gt_seg), gt_seg)
    seg = torch.where(seg >= num_groups, torch.zeros_like(seg), seg)

    seg = torch.nn.functional.one_hot(seg, num_groups).to(torch.float32)
    gt_seg = torch.nn.functional.one_hot(gt_seg, num_groups).to(torch.float32)

    if ignore_background:
        # remove background (id=0).
        gt_seg = gt_seg[..., 1:]

    N = torch.einsum('bthwc,bthwk->bck', gt_seg, seg)  # [B,c,k]
    A = N.sum(-1)  # row-sum  [B,c]
    B = N.sum(-2)  # col-sum  [B,k]
    num_points = A.sum(1)  # [B]

    rindex = (N * (N - 1)).sum((1, 2))  # [B]
    aindex = (A * (A - 1)).sum(1)  # [B]
    bindex = (B * (B - 1)).sum(1)  # [B]

    expected_rindex = aindex * bindex / torch.clamp(
        num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def evaluation(model, args, device, eval_loader, writer, global_step):
    model.eval()
    with torch.no_grad():
        aris, fg_aris = [], []
        total = 0
        for sample, gt_seg, _ in eval_loader:
            total += sample.shape[0]
            imgs = sample.to(device)
            gt_seg = gt_seg.to(device).long()
            seg = model(imgs)[1]
            gt_seg = gt_seg.squeeze(2)
            seg = seg.squeeze(2)
            fg_ari = compute_ari(seg, gt_seg, num_groups=50)
            ari = compute_ari(seg, gt_seg, num_groups=50, ignore_background=False)
            fg_aris.append(fg_ari)
            aris.append(ari)
        mean_fg_ari = sum(fg_aris) / len(fg_aris)
        mean_ari = sum(aris) / len(aris)
        writer.add_scalar('eval/fg_ari', mean_fg_ari.mean(), global_step=global_step)
        writer.add_scalar('eval/ari', mean_ari.mean(), global_step=global_step)
        print("Total evaluated: ", total)
    model.train()


def main(args):

    args.color_t = torch.rand(700, 3)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    device = torch.device(
       "cuda" if not args.nocuda and torch.cuda.is_available() else "cpu")

    train_data = MoviDataset(args=args, train=True)
    eval_data = MoviDataset(args=args, train=False)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    eval_loader = DataLoader(
        eval_data, batch_size=10, shuffle=False, num_workers=args.workers, drop_last=False)

    num_train = len(train_data)

    model = SCALOR(args)
    model.to(device)
    model.train()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = \
            load_ckpt(model, optimizer, args.last_ckpt, device)

    writer = SummaryWriter(args.summary_dir)

    args.global_step = global_step

    log_tau_gamma = np.log(args.tau_end) / args.tau_ep

    for epoch in range(int(args.start_epoch), args.epochs):
        local_count = 0
        last_count = 0
        end_time = time.time()


        for batch_idx, (sample, seg, counting_gt) in enumerate(train_loader):

            tau = np.exp(global_step * log_tau_gamma)
            tau = max(tau, args.tau_end)
            args.tau = tau

            global_step += 1

            log_phase = global_step % args.print_freq == 0 or global_step == 1
            args.global_step = global_step
            args.log_phase = log_phase

            imgs = sample.to(device)

            y_seq, seg_seq, log_like, kl_z_what, kl_z_where, kl_z_depth, \
            kl_z_pres, kl_z_bg, log_imp, counting, \
            log_disc_list, log_prop_list, scalor_log_list = model(imgs)


            log_like = log_like.mean(dim=0)
            kl_z_what = kl_z_what.mean(dim=0)
            kl_z_where = kl_z_where.mean(dim=0)
            kl_z_depth = kl_z_depth.mean(dim=0)
            kl_z_pres = kl_z_pres.mean(dim=0)
            kl_z_bg = kl_z_bg.mean(0)

            total_loss = - (log_like - kl_z_what - kl_z_where - kl_z_depth - kl_z_pres - kl_z_bg)

            optimizer.zero_grad()
            total_loss.backward()

            clip_grad_norm_(model.parameters(), args.cp)
            optimizer.step()

            local_count += imgs.data.shape[0]

            if log_phase:

                time_inter = time.time() - end_time
                end_time = time.time()

                count_inter = local_count - last_count

                print_scalor(global_step, epoch, local_count, count_inter,
                               num_train, total_loss, log_like, kl_z_what, kl_z_where,
                               kl_z_pres, kl_z_depth, time_inter)

                writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
                writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
                writer.add_scalar('train/What_KL', kl_z_what.item(), global_step=global_step)
                writer.add_scalar('train/Where_KL', kl_z_where.item(), global_step=global_step)
                writer.add_scalar('train/Pres_KL', kl_z_pres.item(), global_step=global_step)
                writer.add_scalar('train/Depth_KL', kl_z_depth.item(), global_step=global_step)
                writer.add_scalar('train/Bg_KL', kl_z_bg.item(), global_step=global_step)
                # writer.add_scalar('train/Bg_alpha_KL', kl_z_bg_mask.item(), global_step=global_step)
                writer.add_scalar('train/tau', tau, global_step=global_step)

                log_summary(args, writer, imgs, y_seq, global_step, log_disc_list,
                            log_prop_list, scalor_log_list, prefix='train')

                last_count = local_count

            if global_step % args.eval_freq == 0:
                evaluation(model, args, device, eval_loader, writer, global_step)

            if global_step % args.save_epoch_freq == 0 or global_step == 1:
                save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                          local_count, args.batch_size, num_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCALOR')
    args = parse.parse(parser)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)

