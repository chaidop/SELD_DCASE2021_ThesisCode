import einops
import keras
from matplotlib import axis
from pyparsing import rest_of_line
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange
import math
import numpy as np

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(d_model, position: int = 10000):
  print("axis ", np.newaxis)
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]
  #(1, position, d_model)
  return tf.cast(pos_encoding, dtype=tf.float32)
def Attention(dim, inputs, heads=8, dim_head=64, dropout=0.0, max_pos_emb=256, context=None, mask=None, context_mask=None):
    inner_dim = dim_head * heads
    sheads = heads
    sscale = dim_head ** -0.5
    keras.backend.set_image_data_format('channels_first')
    sto_q = keras.layers.Dense(inner_dim, use_bias=False)
    sto_kv = keras.layers.Dense(inner_dim * 2, use_bias=False)
    sto_out = keras.layers.Dense(dim)
    keras.backend.set_image_data_format('channels_first')
    smax_pos_emb = max_pos_emb
    srel_pos_emb = keras.layers.Embedding(2 * max_pos_emb + 1, dim_head)
    keras.backend.set_image_data_format('channels_first')
    sdropout = keras.layers.Dropout(dropout)

    n = inputs.shape[-2]
    heads = sheads
    max_pos_emb = smax_pos_emb
    if context is None:
        has_context = False
        context = inputs
    else:
        has_context = True

    kv = tf.split(sto_kv(context), num_or_size_splits=2, axis=-1)
    q, k, v = (sto_q(inputs), *kv)

    #q, k, v = map(
    #    lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), (q, k, v)
    #)
    keras.backend.set_image_data_format('channels_first')
    q = tf.reshape(q, [-1,heads, n, dim_head])
    k = tf.reshape(k, [-1,heads, n, dim_head])
    v = tf.reshape(v, [-1,heads, n, dim_head])
    #dots = tf.einsum("b h i d, b h j d -> b h i j", q, k) * sscale
    dots = tf.matmul(q, k, transpose_b=True) * sscale

    seq = tf.range(n)
    dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
    dist = (
        tf.clip_by_value(
            dist, clip_value_min=-max_pos_emb, clip_value_max=max_pos_emb
        )
        + max_pos_emb
    )
    print(srel_pos_emb, dist)
    rel_pos_emb = srel_pos_emb(dist)
    pos_encoding = rel_pos_emb[np.newaxis, ...]

    rel_pos_emb = tf.cast(pos_encoding, dtype=tf.float32)
    #pos_attn = tf.einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * sscale



    #this is matrix mult by d (axis = 4)
    #my mhsa: pos_emb: b h n d
    #this:    pos_emb: b n r d

    print(q.shape, rel_pos_emb.shape)
    #b h n d => b n h d
    q = tf.transpose(q, perm=[0, 2, 1, 3])
    print(q.shape, rel_pos_emb.shape)
    #b n h r
    pos_attn = tf.matmul(q, rel_pos_emb, transpose_b=True)
    #b h n r
    pos_attn = tf.transpose(pos_attn, perm=[0,2, 1 , 3])
    print(pos_attn)
    #mmp = tf.matmul(q, rel_pos_emb)
    dots = dots + pos_attn

    if mask is not None or context_mask is not None:
        if mask is not None:
            mask = tf.ones(*inputs.shape[:2])
        if not has_context:
            if context_mask is None:
                context_mask = mask
        else:
            if context_mask is None:
                context_mask = tf.ones(*context.shape[:2])
        mask_value = -tf.experimental.numpy.finfo(dots.dtype).max
        mask = rearrange(mask, "b i -> b () i ()") * rearrange(
            context_mask, "b j -> b () () j"
        )
        dots = tf.where(mask, mask_value, dots)

    attn = tf.nn.softmax(dots, axis=-1)

    #out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)
    ##CUSTOM
    # b h d j
    #tf.transpose(v, perm=[0,1,3,2])
    out = tf.matmul(attn,v)
    out = tf.reshape(out, [-1, n, inner_dim])#rearrange(out, "b h n d -> b n (h d)")
    out = sto_out(out)
    return sdropout(out)