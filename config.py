# data_preprocessing
train_data_dir = r'D:\usr\dataset\mvtec_anomaly_detection\tile\train\good'
augment_num = 10000
im_resize = (256, 256)
patch_size = (128, 128) # network input size
grayscale = False # read image on 'grayscale' or 'bgr' mode
p_rotate = 1
rotate_angle_vari = 45 # rotate image between [-angle, +angle]
p_rotate_crop = 1 # crop inner rotated image
p_crop = 1 if patch_size != im_resize else 0
p_horizonal_flip = 0.3
p_vertical_flip = 0.3
aug_dir = r'D:\usr\anomaly detection\AE_results\train_patches\tile'

# network
input_size = patch_size
input_channel = 1 if grayscale else 3
not_simplified = True
flc = 32 # number of the first hidden layer channels
z_dim = 512 # Dimension of the latent space vector

# training
chechpoint_dir = r'D:\usr\anomaly detection\AE_results\chechpoints\tile_l2' 
epochs = 10000
early_stop_n = 200
save_model_frequency = 5
batch_size = 128
loss = 'l2_loss' # ['ssim_loss', 'ssim_l2_loss', 'l2_loss']
weight = 10 # weight of the l2_loss item if loss=='ssim_l2_loss'
lr = 2e-4
decay = 1e-5
valid_data_ratio = 0.2
do_aug = False # whether to do data augmentation before training
save_snapshot = True # whether to save reconstruction result of valid positive samples when training finished

# testing
test_dir = r'D:\usr\dataset\mvtec_anomaly_detection\tile\test'
sub_folder = ['good', 'crack','rough','glue_strip','gray_stroke','oil']
save_dir = r'D:\usr\anomaly detection\AE_results\reconst\tile_l2'
weight_file = None # if set None, the latest weight file will be automatically selected 
stride = 32 # step length of the sliding window
threshold = None # if set None, threshold will be automatically estimated based on valid positive samples
depress_edge_ratio = 0.2 # reduce residual value in edge area 
depress_edge_pixel = 10 # boundary of edge area
percent = 95 # for estimating threshold based on valid positive samples
use_ssim = False # otherwise using l1 per pixel distance
ssim_win_size = 11 # ssim kernel size
