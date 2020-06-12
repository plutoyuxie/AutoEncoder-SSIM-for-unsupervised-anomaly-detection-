import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from glob import glob
import cv2
import os

from utils import read_img, get_patch, patch2img, set_img_color
from network import AutoEncoder
import config as cfg

# network
autoencoder = AutoEncoder(cfg)

if cfg.weight_file:
    autoencoder.load_weights(cfg.chechpoint_dir + '/' + cfg.weight_file)
else:
    file_list = os.listdir(cfg.chechpoint_dir)
    latest_epoch = max([int(i.split('-')[0]) for i in file_list if 'hdf5' in i])
    print('load latest weight file: ', latest_epoch)
    autoencoder.load_weights(glob(cfg.chechpoint_dir + '/' + str(latest_epoch) + '*.hdf5')[0])
autoencoder.summary()

def get_residual_map(img_path, cfg):
    test_img = read_img(img_path, cfg.grayscale)
    
    if test_img.shape[:2] != cfg.im_resize:
        test_img = cv2.resize(test_img, cfg.im_resize)

    test_img_ = test_img / 255.

    if test_img.shape[:2] == cfg.patch_size:
        test_img_ = np.expand_dims(test_img_, 0)
        decoded_img = autoencoder.predict(test_img_)
    else:
        patches = get_patch(test_img_, cfg.patch_size[0], cfg.patch_size[1], cfg.stride)
        patches = autoencoder.predict(patches)
        decoded_img = patch2img(patches, cfg.im_resize, cfg.patch_size, cfg.stride)
    
    rec_img = np.reshape((decoded_img * 255.).astype('uint8'), test_img.shape)

    if cfg.grayscale:
        if cfg.use_ssim:
            residual_map = 1 - ssim(test_img, rec_img, win_size=cfg.ssim_win_size, full=True)[1]
        else: # l1 per pixel distance
            residual_map = np.abs(test_img / 255. - rec_img / 255.)
    else:
        if cfg.use_ssim:
            residual_map = 1 - ssim(np.mean(test_img, axis=2), np.mean(rec_img, axis=2),
                                    win_size=cfg.ssim_win_size, full=True)[1]
        else: # l1 per pixel distance
            residual_map = np.mean(np.abs(test_img / 255. - rec_img / 255.), axis=2)
    
    return test_img, rec_img, residual_map


def get_threshold(cfg):
    if cfg.threshold:
        threshold = cfg.threshold
    else:
        # estimate threshold
        valid_good_list = glob(cfg.train_data_dir + '/*png')
        num_valid_data = int(np.ceil(len(valid_good_list) * cfg.valid_data_ratio))
        total_rec = []
        for img_path in valid_good_list[-num_valid_data:]:
            _, _, residual_map = get_residual_map(img_path, cfg)
            total_rec.append(residual_map)
        total_rec = np.array(total_rec)
        threshold = float(np.percentile(total_rec, [cfg.percent]))
    print('threshold: ', threshold)

    return threshold


def get_results(file_list, cfg):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    for img_path in file_list:
        img_name = img_path.split('\\')[-1][:-4]
        c = '' if not cfg.sub_folder else k
        test_img, rec_img, residual_map = get_residual_map(img_path, cfg)
        
        tmp_weight = np.ones(cfg.im_resize) * cfg.depress_edge_ratio
        tmp_weight[cfg.depress_edge_pixel:cfg.im_resize[0]-cfg.depress_edge_pixel, 
                cfg.depress_edge_pixel:cfg.im_resize[1]-cfg.depress_edge_pixel] = 1
        residual_map *= tmp_weight

        mask = np.zeros(cfg.im_resize)
        mask[residual_map > threshold] = 1

        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255

        vis_img = set_img_color(test_img, mask, weight_foreground=0.3, grayscale=cfg.grayscale)
        
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_residual.png', mask)
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_origin.png', test_img)
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_rec.png', rec_img)
        cv2.imwrite(cfg.save_dir+'/'+c+'_'+img_name+'_visual.png', vis_img)


if __name__ == '__main__':
    threshold = get_threshold(cfg)
    if cfg.sub_folder:
        for k in cfg.sub_folder:
            test_list = glob(cfg.test_dir+'/'+k+'/*png')
            get_results(test_list, cfg)
    else:
        test_list = glob(cfg.test_dir+'/*png')
        get_results(test_list, cfg)
