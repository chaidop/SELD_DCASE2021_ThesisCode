import einops
import keras
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange
#keras.backend.set_image_data_format("channels_first")
from .attention import Attention

keras.backend.set_image_data_format('channels_first')
def Swish(inputs):
    keras.backend.set_image_data_format('channels_first')
    return inputs * tf.sigmoid(inputs)


def GLU(inputs, dim):
    keras.backend.set_image_data_format('channels_first')
    out, gate = tf.split(inputs, 2, axis=dim)
    return out * tf.sigmoid(gate)


def DepthwiseLayer(inputs, chan_in, chan_out, kernel_size, padding):
    keras.backend.set_image_data_format('channels_first')
    conv = keras.layers.Conv1D(chan_out, 1, groups=chan_in)
    #inputs = tf.reshape(inputs, [-1])
    padded = tf.zeros(
        [chan_in * chan_in] - tf.shape(inputs), dtype=inputs.dtype
    )
    #inputs = tf.concat([inputs, padded], 0)
    #inputs = tf.reshape(inputs, [-1, self.chan_in, self.chan_in])

    return conv(inputs)


def Scale(inputs, scale):
        keras.backend.set_image_data_format('channels_first')
        return inputs * scale


def PreNorm(inputs, dim, fn):
    keras.backend.set_image_data_format('channels_first')
    norm = keras.layers.LayerNormalization(axis=-1)

    inputs = norm(inputs)
    return fn(inputs)


def FeedForward(inputs, dim, mult=4, dropout=0.0 ):
    keras.backend.set_image_data_format('channels_first')
    inputs = Swish(inputs)
    keras.backend.set_image_data_format('channels_first')
    net = keras.Sequential(
        [
            keras.layers.Dense(dim * mult, activation=None),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(dim, input_dim=dim * mult),
            keras.layers.Dropout(dropout),
        ]
    )
    return net(inputs)


def BatchNorm(inputs, causal):
        keras.backend.set_image_data_format('channels_first')
        if not causal:
            keras.backend.set_image_data_format('channels_first')
            return keras.layers.BatchNormalization(axis=-1)(inputs)
        return tf.identity(inputs)


def ConformerConvModule(
        dim,
        inputs,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.0,
        
    ):
        keras.backend.set_image_data_format('channels_first')
        inner_dim = dim * expansion_factor
        if not causal:
            padding = (kernel_size // 2, kernel_size // 2 - (kernel_size + 1) % 2)
        else:
            padding = (kernel_size - 1, 0)
        keras.backend.set_image_data_format('channels_first')
        inputs =  keras.layers.LayerNormalization(axis=-1)(inputs)
        inputs = keras.layers.Conv1D(filters=inner_dim * 2, kernel_size=1)(inputs)
        inputs = GLU(inputs, dim=-1)
        inputs  =DepthwiseLayer(
                    inputs, inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
                )
        keras.backend.set_image_data_format('channels_first')
        inputs = BatchNorm(inputs, causal=causal)
        inputs = Swish( inputs)
        keras.backend.set_image_data_format('channels_first')
        inputs = keras.layers.Conv1D(filters=dim, kernel_size=1)(inputs)
        inputs = keras.layers.Dropout(dropout)(inputs)

        return inputs


def ConformerBlock(
        dim,
        inputs, 
        mask=None,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
    ):
        keras.backend.set_image_data_format('channels_first')
        norm = keras.layers.LayerNormalization(axis=-1)
        inputs = norm(inputs)
        keras.backend.set_image_data_format('channels_first')
        ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, inputs=inputs)
        ff1 = Scale(0.5, ff1)
        inputs = ff1 + inputs

        inputs = norm(inputs)
        #11/5/2022 added the loop
        attn = inputs
        #for i in range(24):
        keras.backend.set_image_data_format('channels_first')
        attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, inputs=attn
        )
        
        inputs = attn + inputs

        conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
            inputs=inputs
        )
        inputs = conv + inputs
        keras.backend.set_image_data_format('channels_first')
        norm = keras.layers.LayerNormalization(axis=-1)
        inputs = norm(inputs)
        keras.backend.set_image_data_format('channels_first')
        ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, inputs=inputs)
        ff2 = Scale(0.5, ff2)
        inputs = ff2 + inputs
        keras.backend.set_image_data_format('channels_first')
        post_norm = keras.layers.LayerNormalization(axis=-1)(inputs)
        inputs = post_norm

        return inputs