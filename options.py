import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--name', type=str, default='leather')
        self.parser.add_argument('--train_data_dir', type=str, default=None)
        self.parser.add_argument('--test_dir', type=str, default=None)
        self.parser.add_argument('--sub_folder', type=list, nargs='*', default=None)
        self.parser.add_argument('--do_aug', action='store_true', help='whether to do data augmentation before training')
        self.parser.add_argument('--aug_dir', type=str, default=None)
        self.parser.add_argument('--chechpoint_dir', type=str, default=None)
        self.parser.add_argument('--save_dir', type=str, default=None)

        self.parser.add_argument('--augment_num', type=int, default=10000)
        self.parser.add_argument('--im_resize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--patch_size', type=int, default=128, help='then crop to this size')
        self.parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
        self.parser.add_argument('--p_rotate', type=float, default=0.3, help='probability to do image rotation')
        self.parser.add_argument('--rotate_angle_vari', type=float, default=45.0, help='rotate image between [-angle, +angle]')
        self.parser.add_argument('--p_rotate_crop', type=float, default=1.0, help='probability to crop inner rotated image')
        self.parser.add_argument('--p_horizonal_flip', type=float, default=0.3, help='probability to do horizonal flip')
        self.parser.add_argument('--p_vertical_flip', type=float, default=0.3, help='probability to do vertical flip')

        self.parser.add_argument('--z_dim', type=int, default=100, help='dimension of the latent space vector')
        self.parser.add_argument('--flc', type=int, default=32, help='number of the first hidden layer channels')

        self.parser.add_argument('--epochs', type=int, default=200, help='maximum training epochs')
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--loss', type=str, default='ssim_loss', help='loss type in [ssim_loss, ssim_l1_loss, l2_loss]')
        self.parser.add_argument('--weight', type=int, default=1, help='weight of the l1_loss item if using ssim_l1_loss')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of Adam')
        self.parser.add_argument('--decay', type=float, default=1e-5, help='decay of Adam')	


        self.parser.add_argument('--weight_file', type=str, default=None, help='if set None, the latest weight file will be automatically selected')
        self.parser.add_argument('--stride', type=int, default=32, help='step length of the sliding window')
        self.parser.add_argument('--ssim_threshold', type=float, default=None, help='ssim threshold for testing')
        self.parser.add_argument('--l1_threshold', type=float, default=None, help='l1 threshold for testing')
        self.parser.add_argument('--percent', type=float, default=98.0, help='for estimating threshold based on valid positive samples')
        self.parser.add_argument('--bg_mask', type=str, default=None, help='background mask, B means black, W means white')

    def parse(self):
        DATASET_PATH = 'D:/yujiawei/dataset/mvtec_anomaly_detection'
        self.opt = self.parser.parse_args()

        if not self.opt.train_data_dir:
            self.opt.train_data_dir = DATASET_PATH+'/'+self.opt.name+'/train/good'
        if not self.opt.test_dir:
            self.opt.test_dir = DATASET_PATH+'/'+self.opt.name+'/test'
        if not self.opt.sub_folder:
            self.opt.sub_folder = os.listdir(self.opt.test_dir)
        if not self.opt.aug_dir:
            self.opt.aug_dir = './train_patches/'+self.opt.name
        if not self.opt.chechpoint_dir:
            self.opt.chechpoint_dir = './results/'+self.opt.name+'/chechpoints/'+self.opt.loss
        if not self.opt.save_dir:
            self.opt.save_dir = './results/'+self.opt.name+'/reconst/ssim_l1_metric_'+self.opt.loss

        if not os.path.exists(self.opt.chechpoint_dir):
            os.makedirs(self.opt.chechpoint_dir)
        if not os.path.exists(self.opt.aug_dir):
            os.makedirs(self.opt.aug_dir)
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)

        self.opt.input_channel = 1 if self.opt.grayscale else 3
        self.opt.p_crop = 1 if self.opt.patch_size != self.opt.im_resize else 0
        self.opt.mask_size = self.opt.patch_size if self.opt.im_resize - self.opt.patch_size < self.opt.stride else self.opt.im_resize

        return self.opt
