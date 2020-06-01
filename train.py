import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from network import AutoEncoder

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from glob import glob
import os

# setting
EPOCHS = 1000
SAVE_MODEL_FREQUENCY = 10
Early_STOP_N = 20
BATCH_SIZE = 128
D_DIM = 512
DATASET_DIR = r'D:\user\AE_defect\train_patches\carpet\train\good\*.png'
CHECHPOINT_DIR = r'D:\user\AE_defect\chechpoints\tranp_ssim_d_512_epoch_1000'
LOAD_DATA_ONLY_ONCE = False
USE_SSIM_LOSS = True
SHOW_RESULT = True

# network
autoencoder = AutoEncoder(D_DIM)

# loss
if USE_SSIM_LOSS:
	@tf.function
	def ssim_loss(gt, y_pred, max_val=1.0):
		return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))

	autoencoder.compile(optimizer=Adam(lr=2e-4, decay=1e-5), loss=ssim_loss, metrics=['mse'])  # or 'mse'\'mae'
else:
	autoencoder.compile(optimizer=Adam(lr=2e-4, decay=1e-5), loss='mse', metrics=['mae'])
autoencoder.summary()

earlystopping = EarlyStopping(patience=Early_STOP_N)
checkpoint = ModelCheckpoint(os.path.join(CHECHPOINT_DIR,'{epoch:02d}-{val_loss:.5f}.hdf5'), 
                                    period=SAVE_MODEL_FREQUENCY, mode='auto', verbose=1, save_weights_only=True)

# data
file_list = glob(DATASET_DIR)

if LOAD_DATA_ONLY_ONCE:
	''' loading the whole dataset into memory, which can save training time if the dataset is small'''
	all_imgs = np.array([imread(filename) for filename in file_list])
	all_imgs = all_imgs.astype('float32') / 255.
	
	autoencoder.fit(all_imgs, all_imgs, validation_split=1800/9800, batch_size=BATCH_SIZE, shuffle=True,
					epochs=EPOCHS, callbacks=[checkpoint, earlystopping])
else:
	class data_flow(Sequence):
		def __init__(self, filenames, batch_size):
			self.filenames = filenames
			self.batch_size = batch_size

		def __len__(self):
			return int(np.ceil(len(self.filenames) / float(self.batch_size)))

		def __getitem__(self, idx):
			batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
			batch_x = np.array([imread(filename) for filename in batch_x])
			batch_x = batch_x.astype('float32') / 255.
			return batch_x, batch_x

	data_train = data_flow(file_list[:-1800], BATCH_SIZE)
	data_valid = data_flow(file_list[-1800:], BATCH_SIZE)

	autoencoder.fit(data_train, epochs=EPOCHS, validation_data=data_valid, callbacks=[checkpoint, earlystopping])

# show reconstructed images
if SHOW_RESULT:
    if LOAD_DATA_ONLY_ONCE:
	    x_test = all_imgs[-5:]
    else:
        test_imgs = file_list[-5:]
        x_test = np.array([imread(filename) for filename in test_imgs])
        x_test = x_test.astype('float32') / 255.

    decoded_imgs = autoencoder.predict(x_test)
    n = len(x_test)
    plt.figure(figsize=(8, 3))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(128, 128,3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i].reshape(128, 128,3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
