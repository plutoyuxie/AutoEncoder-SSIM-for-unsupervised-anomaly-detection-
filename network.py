from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model


def AutoEncoder(cfg):
    input_img = Input(shape=(cfg.patch_size, cfg.patch_size, cfg.input_channel))

    h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
    h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    if cfg.patch_size==256:
        h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc*4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    encoded = Conv2D(cfg.z_dim, (8, 8), strides=1, activation='linear', padding='valid')(h)

    h = Conv2DTranspose(cfg.flc, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
    h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc*4, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(cfg.flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    if cfg.patch_size==256:
        h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)

    decoded = Conv2DTranspose(cfg.input_channel, (4, 4), strides=2, activation='sigmoid', padding='same')(h)

    return Model(input_img, decoded)