from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model

# network
def AutoEncoder(d_dim):
    input_img = Input(shape=(128, 128, 3))
    hidden_layers = Conv2D(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
    hidden_layers = Conv2D(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(64, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(128, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    encoded = Conv2D(d_dim, (8, 8), strides=1, activation='linear', padding='valid')(hidden_layers)

    hidden_layers = Conv2DTranspose(32, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
    hidden_layers = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(128, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2DTranspose(64, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(64, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2DTranspose(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2D(32, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    hidden_layers = Conv2DTranspose(32, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(hidden_layers)
    decoded = Conv2DTranspose(3, (4, 4), strides=2, activation='linear', padding='same')(hidden_layers)
    return Model(input_img, decoded)