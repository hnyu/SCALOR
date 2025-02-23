z_what_dim = 64
z_where_scale_dim = 2  # sx sy
z_where_shift_dim = 2  # tx ty
z_pres_dim = 1
glimpse_size = 32
img_h = 128
img_w = img_h
img_encode_dim = 64
z_depth_dim = 1
bg_what_dim = 2

temporal_rnn_hid_dim = 128
temporal_rnn_out_dim = temporal_rnn_hid_dim
propagate_encode_dim = 32
z_where_transit_bias_net_hid_dim = 128
z_depth_transit_net_hid_dim = 128

z_pres_hid_dim = 64
z_what_from_temporal_hid_dim = 64
z_what_enc_dim = 128

prior_rnn_hid_dim = 64
prior_rnn_out_dim = prior_rnn_hid_dim

DEBUG = False

object_act_size = 21
seq_len = 6
phase_obj_num_contrain = True
phase_rejection = True

temporal_img_enc_hid_dim = 64
temporal_img_enc_dim = 128
z_where_bias_dim = 4
temporal_rnn_inp_dim = 128
prior_rnn_inp_dim = 128
bg_prior_rnn_hid_dim = 32
where_update_scale = .2

pres_logit_factor = 8.8

conv_lstm_hid_dim = 64

cfg = {
    'num_img_summary': 3,
    'num_cell_h': 8,
    'num_cell_w': 8,
    'phase_conv_lstm': True,
    'phase_no_background': False,
    'phase_eval': True,
    'phase_boundary_loss': False,
    'phase_generate': False,
    'phase_nll': False,
    'phase_gen_disc': True,
    'phase_bg_alpha_curriculum': False,
    'bg_alpha_curriculum_steps': 200,
    'bg_alpha_curriculum_value': 0.9,
    'using_bg_sbd': False,
    'gen_disc_pres_probs': 0.1,
    'observe_frames': 5,
    'size_anc': 0.2,   # object scale mean
    'var_s': 0.1,      # object scale var
    'ratio_anc': 1.,   # h-w ratio mean
    'var_anc': 0.5,    # h-w ratio var
    'train_station_cropping_origin': 240,
    'color_num': 500,
    'explained_ratio_threshold': 0.2, # proposal rejection threshold
    'tau_imp': 0.25,
    'z_pres_anneal_end_value': 1e-3,
    'phase_do_remove_detach': True,
    'remove_detach_step': 30000,
    'max_num_obj': 45 # Remove this constrain in discovery.py by setting phase_obj_num_contrain to False if you have enough GPU memory.
}
