import autoencoders.images_from_text as ift
reload(ift)
import os
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw
from keras.preprocessing.image import img_to_array, array_to_img









import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

img_width = 200
img_height = 200
font_size = 4
batch_size = 20
original_dim = img_width * img_height
latent_dim = 2
intermediate_dim = 256
nb_epoch = 2

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)



test = ift.train_set_from_dir('data/hadoop_images/test')
x_train = test[:20000, :].astype('float32') / 255.
x_test = test[-4600:, :].astype('float32') / 255.
x_train.shape, x_test.shape

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=10,
        batch_size=batch_size,
        validation_data=(x_test, x_test))








encoder = Model(x, z_mean)


decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
new_img_generator = Model(decoder_input, _x_decoded_mean)

ara = new_img_generator.predict(np.array([[5,-15]]))
array_to_img(ara.reshape((1, img_width, img_height))).resize((250, 250))


from IPython.display import display
for a in range(-200, 200, 100):
    for b in range(-200, 200, 100):
        ara = new_img_generator.predict(np.array([[a,b]]))
        ima = array_to_img(ara.reshape((1, img_width, img_height))).resize((250, 250))
        print a, b
        display(ima)