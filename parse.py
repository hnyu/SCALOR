import torch
from common import *

def parse(parser):
    parser.add_argument('--data-dir', default='./data', metavar='DIR',
                        help='Path to the data folder')
    parser.add_argument('--nocuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run (default: 4000)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 20)')
    parser.add_argument('--lr', '--learning-rate', default=4e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--cp', '--clip-gradient', default=1.0, type=float,
                        metavar='CP', help='rate of gradient clipping')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print batch frequency (default: 100)')
    parser.add_argument('--generate-freq', '-gp', default=500, type=int,
                        metavar='N', help='generation frequency (on training set) (default: 500)')
    parser.add_argument('--eval-freq', '-ef', default=2000, type=int,
                        metavar='N', help='eval frequency (on val set) (default: 500)')
    parser.add_argument('--save-epoch-freq', '-s', default=5000, type=int,
                        metavar='N', help='save epoch frequency (default: 5000)')
    parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                        help='path to save checkpoints')
    parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                        help='path to save summary')
    parser.add_argument('--tau-end', default=0.5, type=float, metavar='T',
                        help='initial temperature for gumbel')
    parser.add_argument('--tau-ep', default=2e4, type=float, metavar='E',
                        help='exponential decay factor for tau')
    parser.add_argument('--seed', default=666, type=int,
                        help='Fixed random seed.')
    parser.add_argument('--sigma', default=0.1, type=float, metavar='S',
                        help='Sigma for log likelihood.')
    parser.add_argument('--phase-parallel', default=True, type=bool,
                        help='Multi-GPUs')
    parser.add_argument('--phase-simplify-summary', default=True, type=bool,
                        help='Only show image in summary log.')
    parser.add_argument('--max-num-obj', default=10, type=int,
                        help='max number of obj instances in a video')

    args = parser.parse_args()

    # common.cfg overrides
    parser.add_argument('--size-anc', type=float)
    parser.add_argument('--var-s', type=float)
    parser.add_argument('--z-pres-anneal-end-value', type=float)
    parser.add_argument('--explained-ratio-threshold', type=float)

    args = parser.parse_args()

    # override defaults from common.py
    for k, v in cfg.items():
        if k not in args or vars(args)[k] is None:
            vars(args)[k] = v

    args.color_t = torch.rand(args.color_num, 3)

    if "movi_d" in args.data_dir or "movi_e" in args.data_dir:
        args.max_num_obj = 23

    print("Max obj instances: ", args.max_num_obj)

    return args
