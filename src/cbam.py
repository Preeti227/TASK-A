from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation,
    Reshape, Multiply, Lambda, Concatenate, Conv2D
)
import tensorflow as tf

def cbam_block(x, ratio=8):
    channel = x.shape[-1]

    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)

    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal')
    shared_dense_two = Dense(channel, kernel_initializer='he_normal')

    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_out = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    x = Multiply()([x, channel_attention])

    avg_pool_spatial = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool_spatial = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid',
                               kernel_initializer='he_normal', use_bias=False)(concat)
    x = Multiply()([x, spatial_attention])
    return x
