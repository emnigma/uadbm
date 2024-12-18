import numpy as np
import tensorflow as tf
# from tensorflow.compat.v1.layers import Dense
# from tensorflow.nn import leaky_relu
from tensorflow.python.keras.layers import Flatten, Conv2D, Dropout, Dense

from tensorflow.python.keras.activations import leaky_relu

from models.customlayers import build_unified_decoder, build_unified_encoder


def constrained_adversarial_autoencoder(z, x, dropout_rate, dropout, config):
    outputs = {}

    with tf.compat.v1.variable_scope('Encoder'):
        encoder = build_unified_encoder(x.get_shape().as_list(), config.intermediateResolutions)

        temp_out = x
        for layer in encoder:
            temp_out = layer(temp_out)

        temp_temp_out = temp_out
        intermediate_conv = Conv2D(temp_temp_out.get_shape().as_list()[3] // 8, 1, padding='same')
        dropout_layer = Dropout(dropout_rate)
        temp_out = intermediate_conv(temp_out)

        reshape = temp_out.get_shape().as_list()[1:]
        z_layer = Dense(config.zDim)
        dec_dense = Dense(np.prod(reshape))

        outputs['z_'] = z_ = dropout_layer(z_layer(Flatten()(temp_out)), dropout)

    with tf.compat.v1.variable_scope('Decoder'):
        decoder = build_unified_decoder(config.outputWidth, config.intermediateResolutions, config.numChannels)
        intermediate_conv_reverse = Conv2D(temp_temp_out.get_shape().as_list()[3], 1, padding='same')

        reshaped = tf.reshape(dropout_layer(dec_dense(z_)), [-1, *reshape])
        temp_out = intermediate_conv_reverse(reshaped)

        # Decode: z -> x_hat
        for layer in decoder:
            temp_out = layer(temp_out)

        outputs['x_hat'] = temp_out

    with tf.compat.v1.variable_scope('Encoder'):
        # mapping reconstruction to latent space for constrained part
        for layer in encoder:
            temp_out = layer(temp_out)
        outputs['z_rec'] = dropout_layer(z_layer(Flatten()(intermediate_conv(temp_out))))

    # Discriminator
    with tf.compat.v1.variable_scope('Discriminator'):
        discriminator = [
            Dense(100, activation=leaky_relu),
            Dense(50, activation=leaky_relu),
            Dense(1)
        ]

        # fake
        temp_out = z_
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_'] = temp_out

        # real
        temp_out = z
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d'] = temp_out

        # adding noise
        epsilon = tf.random_uniform([config.batchsize, 1], minval=0., maxval=1.)
        outputs['z_hat'] = z_hat = z + epsilon * (z - z_)

        temp_out = z_hat
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_hat'] = temp_out

    return outputs
