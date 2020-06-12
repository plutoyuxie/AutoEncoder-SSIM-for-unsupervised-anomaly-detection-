from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import os

from network import AutoEncoder
import config as cfg
from utils import generate_image_list, augment_images, read_img


class data_flow(Sequence):
    def __init__(self, filenames, batch_size, grayscale):
        self.filenames = filenames
        self.batch_size = batch_size
        self.grayscale = grayscale

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([read_img(filename, self.grayscale) for filename in batch_x])
        
        batch_x = batch_x / 255.
        return batch_x, batch_x

# data
if cfg.aug_dir and cfg.do_aug:
    if not os.path.exists(cfg.aug_dir):
        os.makedirs(cfg.aug_dir)

    img_list = generate_image_list(cfg)
    augment_images(img_list, cfg)

dataset_dir = cfg.aug_dir if cfg.aug_dir else cfg.train_data_dir
file_list = glob(dataset_dir + '/*')
num_valid_data = int(np.ceil(len(file_list) * cfg.valid_data_ratio))
data_train = data_flow(file_list[:-num_valid_data], cfg.batch_size, cfg.grayscale)
data_valid = data_flow(file_list[-num_valid_data:], cfg.batch_size, cfg.grayscale)

# loss
if cfg.loss == 'ssim_loss':
    
    @tf.function
    def ssim_loss(gt, y_pred, max_val=1.0):
        return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))
    
    loss = ssim_loss
elif cfg.loss == 'ssim_l2_loss':
    
    @tf.function
    def ssim_l2_loss(gt, y_pred, max_val=1.0):
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))
        L2 = tf.reduce_mean(tf.square(gt - y_pred))
        return ssim_loss + L2 * cfg.weight
    
    loss = ssim_l2_loss
else:
    loss = 'mse'

# network
autoencoder = AutoEncoder(cfg)
optimizer = Adam(lr=cfg.lr, decay=cfg.decay)
autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['mae'] if loss == 'mse' else ['mse'])
autoencoder.summary()

earlystopping = EarlyStopping(patience=cfg.early_stop_n)

checkpoint = ModelCheckpoint(os.path.join(cfg.chechpoint_dir, '{epoch:02d}-{val_loss:.5f}.hdf5'), save_best_only=True,
                            period=cfg.save_model_frequency, mode='auto', verbose=1, save_weights_only=True)

if not os.path.exists(cfg.chechpoint_dir):
    os.makedirs(cfg.chechpoint_dir)

autoencoder.fit(data_train, epochs=cfg.epochs, validation_data=data_valid, callbacks=[checkpoint, earlystopping])

# show reconstructed images
if cfg.save_snapshot:
    decoded_imgs = autoencoder.predict(data_valid)
    n = len(decoded_imgs)
    save_snapshot_dir = cfg.chechpoint_dir +'/snapshot/'
    if not os.path.exists(save_snapshot_dir):
        os.makedirs(save_snapshot_dir)
    for i in range(n):
        cv2.imwrite(save_snapshot_dir+str(i)+'_rec_valid.png', (decoded_imgs[i]*255).astype('uint8'))

