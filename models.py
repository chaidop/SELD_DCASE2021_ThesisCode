import os
import configparser
from keras import backend as K
import keras
import tensorflow as tf
import tensorflow.keras
from keras.layers import Layer
from keras.models import load_model, Model
from keras.layers import Permute, Reshape, Lambda, Bidirectional, Conv2DTranspose, dot
from keras.layers import Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Activation, BatchNormalization, TimeDistributed, Dropout, LayerNormalization
from keras.layers import GRU, Dense, Input, Activation, Conv2D, MaxPooling2D
from keras.layers import Dot, add, multiply, concatenate, subtract, GlobalMaxPooling1D
from keras.layers import UpSampling2D, GlobalMaxPooling2D

import numpy as np

##https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb?hl=ro#scrollTo=ncyS-Ms3i2x_

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self,*, num_heads, dim_head = 64):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = dim_head * num_heads

    assert d_model % self.num_heads == 0

    self.depth = dim_head

    #get projected vectors
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    (B, N, S, H) , see official tensorflows github for MultiHeadAttention
    https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/multi_head_attention.py#L123-L516
    """

    ##d_model is output dimensions
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    #
    scaled_attention = Dropout(0.2)(scaled_attention)

    scaled_attention =  K.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = Reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

# Normalization ->  Pos_encoding + MHSA -> Dropout
## See file:///C:/Users/pouli/Documents/Mathimata/Eidiko_Thema/INFO/online_material/conformer_speech_recognition.pdf
#(Conformer: Convolution-augmented Transformer for Speech Recognition)
class MultiHeadAttentionModule(tf.keras.layers.Layer):
  def __init__(self,*, dim_head, num_heads):
    super(MultiHeadAttentionModule, self).__init__()
    self.num_heads = num_heads
    self.d_model = dim_head * num_heads

  def call(self, v, k, q, mask):
    inputs = q
    inputs = keras.layers.LayerNormalization(inputs)
    pos_embedding = positional_encoding(seq_length , self.d_model)
    mhsa = MultiHeadAttention(self.d_model, num_heads)
    output, attention_weights = mhsa(inputs, inputs, inputs, mask, pos_embedding)
    Dropout(0.2)

    return output

class attentionLayer(Layer):
    def __init__(self, ** kwargs):
        """"
        Class-wise attention pooling layer
                Args:
                Attributes:
            kernel: tensor
            bias: tensor	
        """
        super(attentionLayer, self).__init__( ** kwargs)

    def build(self, input_shape):

        kernel_shape = [1] * len(input_shape)
        bias_shape = tuple([1] * (len(input_shape)-1))
        
        kernel_shape[-1] = input_shape[-1]
        kernel_shape = tuple(kernel_shape)
        
        ## init wieghts with 0
        self.kernel = self.add_weight(
                shape = kernel_shape, 
                initializer = keras.initializers.Zeros(), 
                name = '%s_kernel'%self.name)
        ## init bias with 0
        self.bias = self.add_weight(
                shape = bias_shape, 
                initializer = keras.initializers.Zeros(), 
                name = '%s_bias'%self.name)

        super(attentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        weights = K.sum(inputs * self.kernel, axis = -1) + self.bias
        return weights

    def compute_output_shape(self, input_shape):
        out_shape = []
        for i in range(len(input_shape)-1):
            out_shape += [input_shape[i]]
        return tuple(out_shape)

def scaled_dot_product_attention(input_vector):
        Q, V, K = input_vector
        ## Q*K^T
        QK_mult = Q*K.transpose()
        dk = K.shape[1]
        ## the scaled self attention equation (page 4, "Attetnion is all you need")
        attention_weights = tf.nn.softmax(QK_mult/tf.math.sqrt(dk))
        attention = attention_weights*V

        return attention, attention_weights

    def compute_output_shape(self, input_shape):
        out_shape = []
        for i in range(len(input_shape)-1):
            out_shape += [input_shape[i]]
        return tuple(out_shape)

class Conformer(Layer):
    def __init__(
            self,
            spec_cnn):
        super(Conformer, self).__init__()
        self.ff = FeedForward(spec_cnn)//2

    def call(self, spec_cnn):
        self.ff = FeedForward(spec_cnn)//2
        spec_cnn = spec_cnn + self.ff
        ##### MHSA layer ##############
        layer = MultiHeadAttentionModule(num_heads=8, key_dim=12)
        output_tensor, weights = layer(spec_cnn, spec_cnn, spec_cnn)
        spec_cnn = spec_cnn + output_tensor
        #######################
        spec_cnn = ConvolutionModule(spec_cnn, 64)
        spec_cnn = spec_cnn + FeedForward(spec_cnn)//2
        spec_cnn = LayerNormalization(512)(spec_cnn)
        ###############
        Dense(256, activation = 'relu')(spec_cnn)
        Dense(128, activation = 'relu')(spec_cnn)
        Dense(36, activation = 'tanh')(spec_cnn)

        return spec_cnn

def FeedForward(spec_cnn, encoder_dim: int = 512,
            expansion_factor: int = 4):
    
    
    spec_cnn = LayerNormalization(encoder_dim)(spec_cnn)
    spec_cnn = Dense(encoder_dim, encoder_dim*expansion_factor, activation = None)(spec_cnn) ## Linear layer
    temp = spec_cnn
    print("FFN SHAPE ", temp.shape)
    ##swish activation function
    spec_cnn = tf.keras.activations.sigmoid(spec_cnn) * spec_cnn
    spec_cnn = Dropout(0.02)(spec_cnn)    
    spec_cnn = Dense(encoder_dim*expansion_factor, encoder_dim, activation = None)(spec_cnn)

    return temp+spec_cnn

def ConvolutionModule(spec_cnn, nb_cnn2d_filt):
    temp = spec_cnn
    #pointwise convolution
    spec_cnn = Conv1D(filters=nb_cnn2d_filt, kernel_size=(1,1), padding='same')(spec_cnn)
    spec_cnn = Activation('glu')(spec_cnn)
    #1D depthwise conv

    spec_cnn = LayerNormalization(512)(spec_cnn)
    ##swish activation function
    spec_cnn = tf.keras.activations.sigmoid(spec_cnn) * spec_cnn
    #pointwise convolution
    spec_cnn = Conv1D(filters=nb_cnn2d_filt, kernel_size=(1,1), padding='same')(spec_cnn)
    spec_cnn = Dropout(dropout_rate)(spec_cnn)

    return temp+spec_cnn

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.nxewaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)