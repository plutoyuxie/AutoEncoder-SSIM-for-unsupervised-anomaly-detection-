import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='leather')
        self.parser.add_argument('--train_data_dir', type=str, default=None)
        self.parser.add_argument('--test_dir', type=str, default=None)
        self.parser.add_argument('--sub_folder', type=list, nargs='*', default=None)
        self.parser.add_argument('--aug_dir', type=str, default=None)
        self.parser.add_argument('--chechpoint_dir', type=str, default=None)
        self.parser.add_argument('--save_dir', type=str, default=None)

        self.parser.add_argument('--augment_num', type=int, default=5000)
        self.parser.add_argument('--im_resize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--patch_size', type=int, default=128, help='then crop to this size')
        self.parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
        self.parser.add_argument('--p_rotate', type=float, default=0.3, help='probability to do image rotation')
        self.parser.add_argument('--rotate_angle_vari', type=float, default=45.0, help='rotate image between [-angle, +angle]')
        self.parser.add_argument('--p_rotate_crop', type=float, default=1.0, help='probability to crop inner rotated image')
        self.parser.add_argument('--p_horizonal_flip', type=float, default=0.3, help='probability to do horizonal flip')
        self.parser.add_argument('--p_vertical_flip', type=float, default=0.3, help='probability to do vertical flip')

        self.parser.add_argument('--simplified', action='store_true', help='whether to use a simplified network')
        self.parser.add_argument('--z_dim', type=int, default=512, help='dimension of the latent space vector')
        self.parser.add_argument('--flc', type=int, default=32, help='number of the first hidden layer channels')

        self.parser.add_argument('--epochs', type=int, default=100, help='maximum training epochs')
        self.parser.add_argument('--early_stop_n', type=int, default=50, help='stop training after n epochs non-improving')
        self.parser.add_argument('--save_model_frequency', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--loss', type=str, default='ssim_loss', help='loss type in [ssim_loss, ssim_l2_loss, l2_loss]')
        self.parser.add_argument('--weight', type=int, default=10, help='weight of the l2_loss item if using ssim_l2_loss')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of Adam')
        self.parser.add_argument('--decay', type=float, default=1e-5, help='decay of Adam')	
        self.parser.add_argument('--valid_data_ratio', type=float, default=0.2, help='ratio of valid data')
        self.parser.add_argument('--do_aug', action='store_true', help='whether to do data augmentation before training')
        self.parser.add_argument('--save_snapshot', action='store_true', help='whether to save reconstruction result of valid positive samples when training finished')

        self.parser.add_argument('--weight_file', type=str, default=None, help='if set None, the latest weight file will be automatically selected')
        self.parser.add_argument('--stride', type=int, default=32, help='step length of the sliding window')
        self.parser.add_argument('--ssim_threshold', type=float, default=None, help='ssim threshold for testing')
        self.parser.add_argument('--l1_threshold', type=float, default=None, help='l1 threshold for testing')
        self.parser.add_argument('--depress_edge_ratio', type=float, default=0.2, help='suppress residual response in edge area when using ssim loss')
        self.parser.add_argument('--depress_edge_pixel', type=int, default=10, help='boundary of edge area')
        self.parser.add_argument('--percent', type=float, default=98.0, help='for estimating threshold based on valid positive samples')
        self.parser.add_argument('--ssim_win_size', type=int, default=11, help='ssim kernel size')      

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        if not self.opt.train_data_dir:
            self.opt.train_data_dir = 'D:/user/dataset/mvtec_anomaly_detection/'+self.opt.name+'/train/good'
        if not self.opt.test_dir:
            self.opt.test_dir = 'D:/user/dataset/mvtec_anomaly_detection/'+self.opt.name+'/test'
        if not self.opt.sub_folder:
            self.opt.sub_folder = os.listdir(self.opt.test_dir)
        if not self.opt.aug_dir:
            self.opt.aug_dir = 'D:/user/anomaly detection/AE_results/'+self.opt.name+'/train_patches'
        if not self.opt.chechpoint_dir:
            self.opt.chechpoint_dir = 'D:/user/anomaly detection/AE_results/'+self.opt.name+'/chechpoints/'+self.opt.loss
        if not self.opt.save_dir:
            self.opt.save_dir = 'D:/user/anomaly detection/AE_results/'+self.opt.name+'/reconst/ssim_l1_metric_'+self.opt.loss

        if not os.path.exists(self.opt.chechpoint_dir):
            os.makedirs(self.opt.chechpoint_dir)
        if not os.path.exists(self.opt.aug_dir):
            os.makedirs(self.opt.aug_dir)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        self.opt.input_channel = 1 if self.opt.grayscale else 3
        self.opt.p_crop = 1 if self.opt.patch_size != self.opt.im_resize else 0

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        file_name = os.path.join(self.opt.chechpoint_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
