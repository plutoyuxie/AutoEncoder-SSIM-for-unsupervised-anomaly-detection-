from tensorflow.keras.utils import Sequence
from network import AutoEncoder

import numpy as np
from skimage.io import imread, imsave
from glob import glob

# setting
BATCH_SIZE = 128
D_DIM = 512
TEST_DIR = r'D:\user\AE_defect\test_patches\good\image\*.png'
CHECHPOINT_DIR = r'D:\user\AE_defect\chechpoints\tranp_ssim_d_512_epoch_1000\430-0.18871.hdf5'
SAVE_DIR = r'D:\user\AE_defect\reconst\good'

# network
autoencoder = AutoEncoder(D_DIM)
autoencoder.load_weights(CHECHPOINT_DIR)

# data
test_list = glob(TEST_DIR)
test_imgs = np.array([imread(filename) for filename in test_list])
test_imgs = test_imgs.astype('float32') / 255.

for i in range(int(np.ceil(len(test_list)/BATCH_SIZE))):
    test_patch = test_imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    decoded_imgs = autoencoder.predict(test_patch)
    for j in range(len(test_patch)):
        real_, rec_ = (test_patch[j]*255.).astype('uint8'), (decoded_imgs[j]*255.).astype('uint8')
        imsave(SAVE_DIR+'/'+str(i*BATCH_SIZE+j)+'_real.png', real_)
        imsave(SAVE_DIR+'/'+str(i*BATCH_SIZE+j)+'_rec.png', rec_)




