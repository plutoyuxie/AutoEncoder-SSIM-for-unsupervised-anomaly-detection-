from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model


def AutoEncoder(cfg):
    h_inner, w_inner = cfg.patch_size//2**4, cfg.patch_size//2**4

    input_img = Input(shape=(cfg.patch_size, cfg.patch_size, cfg.input_channel))

    if not cfg.simplified:
        h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
        h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        encoded = Conv2D(cfg.z_dim, (h_inner, w_inner), strides=1, activation='linear', padding='valid')(h)

        h = Conv2DTranspose(cfg.flc, (h_inner, w_inner), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
        h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*4, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2DTranspose(cfg.flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)

    else:
        h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
        h = Conv2D(cfg.flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2D(cfg.flc*8, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        encoded = Conv2D(cfg.z_dim, (h_inner, w_inner), strides=1, activation='linear', padding='valid')(h)

        h = Conv2DTranspose(cfg.flc*8, (h_inner, w_inner), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
        h = Conv2DTranspose(cfg.flc*4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2DTranspose(cfg.flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
        h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)

    decoded = Conv2DTranspose(cfg.input_channel, (4, 4), strides=2, activation='linear', padding='same')(h)

    return Model(input_img, decoded)