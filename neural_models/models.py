from matplotlib.pyplot import axis
from sklearn.preprocessing import scale
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

import keras
from keras.layers import Layer, Add
from keras.models import load_model, Model
from keras.layers import Permute, Reshape, Lambda, Bidirectional, Conv2DTranspose, dot
from keras.layers import Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Activation, BatchNormalization, TimeDistributed, Dropout
from keras.layers import GRU, Dense, Input, Activation, Conv2D, MaxPooling2D, Conv1D
from keras.layers import Dot, add, multiply, concatenate, subtract, GlobalMaxPooling1D
from keras.layers import UpSampling2D, GlobalMaxPooling2D


#from keras_layer_normalization import LayerNormalization
from keras import backend as K

import numpy as np
from . import layer_normalization

##https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb?hl=ro#scrollTo=ncyS-Ms3i2x_

class MultiHeadAttention(keras.layers.Layer):
  def __init__(self,*, num_heads, dim_head = 64):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = dim_head

    assert self.d_model % self.num_heads == 0

    self.depth = dim_head//self.num_heads

    #get projected vectors
    self.wq = Dense(self.d_model)
    self.wk = Dense(self.d_model)
    self.wv = Dense(self.d_model)

    self.dense = Dense(self.d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    (B, N, S, H) , see official tensorflows github for MultiHeadAttention
    https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/multi_head_attention.py#L123-L516
    """

    ##d_model is output dimensions
    x = tf.reshape(x, (batch_size, x.shape[-2], self.num_heads, self.depth))##original was -1 instead of 60 (2nd place)
    print(x)
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs, v, k, q , mask, pos_embedding):
    batch_size = q.shape[1]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    pos_emb = self.dense(pos_embedding)#(batch_size, 10000, d_model)
    pos_emb = self.split_heads(pos_emb,pos_emb.shape[0])#(batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        [q, k, v], pos_emb)
    #
    scaled_attention = Dropout(0.2)(scaled_attention)

    
    #scaled_attention =  K.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = K.permute_dimensions(scaled_attention, (0,2,1,3))
    concat_attention = Reshape((-1 , self.d_model))(scaled_attention )  # (batch_size, seq_len_q, d_model)
    #scaled_attention = Permute((2,3,1))(scaled_attention)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    #print(K.is_keras_tensor(output))
    #K.print_tensor(output)
    return output, attention_weights


def split_heads(x, batch_size, num_heads, depth):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    (B, N, S, H) , see official tensorflows github for MultiHeadAttention
    https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/multi_head_attention.py#L123-L516
    """

    ##d_model is output dimensions
    #x = Reshape((-1, x.shape[-2], num_heads, depth))(x )##original was -1 instead of 60 (2nd place)
    #print(x)
    #trans = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2, 4)))
    #x = trans(x)
    x = Reshape((batch_size, x.shape[-2], num_heads, depth))(x)##original was -1 instead of 60 (2nd place)
    print(x)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2, 4)))(x)
    return x

def MultiHeadAttention_fun( q, v, k, pos_embedding, num_heads = 8, dim_head = 64):
    d_model = num_heads*dim_head
    depth = dim_head//num_heads

    #get projected vectors
    wq = Dense(d_model)
    wk = Dense(d_model)
    wv = Dense(d_model)
    dense = Dense(d_model)

    batch_size = q.shape[1]

    q = wq(q)  # (batch_size, seq_len, d_model)
    k = wk(k)  # (batch_size, seq_len, d_model)
    v = wv(v)  # (batch_size, seq_len, d_model)

    q = split_heads(q, batch_size, num_heads, dim_head)  # (batch_size, num_heads, seq_len_q, depth)
    k = split_heads(k, batch_size, num_heads, dim_head)  # (batch_size, num_heads, seq_len_k, depth)
    v = split_heads(v, batch_size, num_heads, dim_head)  # (batch_size, num_heads, seq_len_v, depth)

    pos_emb = dense(pos_embedding)#(batch_size, 10000, d_model)
    print("POS EMB SHAPE ", pos_emb.shape)
    pos_emb = split_heads(pos_emb, pos_emb.shape[0], num_heads, dim_head)#(batch_size, num_heads, seq_len_v, depth)
    print("POS SPLIT EMB SHAPE ", pos_emb.shape)#(batch_size, num_heads, seq_len_v, depth)
    #pos_emb = Reshape((pos_emb.shape[-3],pos_emb.shape[-2],pos_emb.shape[-1]))(pos_emb)
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        [q, k, v], pos_emb)
    #
    scaled_attention = Dropout(0.2)(scaled_attention)
    #scaled_attention =  K.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    permuter = Lambda(lambda x: K.permute_dimensions(x, (0,1,3,2,4)))
    scaled_attention = permuter(scaled_attention)
    concat_attention = Reshape((scaled_attention.shape[-4],scaled_attention.shape[-3] , scaled_attention.shape[-2]*scaled_attention.shape[-1]))(scaled_attention )  # (batch_size, seq_len_q, d_model)
    print(concat_attention)
    #scaled_attention = Permute((2,3,1))(scaled_attention)
    print(scaled_attention)
    output = dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def MultiHeadAttentionModule_fun(num_heads, dim_head, inputs ):
    num_heads = num_heads
    d_model = dim_head * num_heads
    temp = inputs
    inputs = layer_normalization.LayerNormalization()(inputs)
    print(inputs)
    pos_embedding = positional_encoding(d_model= d_model, position=256)#### position = q.shape[2]
    #pos_embedding (1, seq_posemb, d_model)
    output, attention_weights = MultiHeadAttention_fun(q = inputs , v = inputs, k = inputs , pos_embedding=pos_embedding, num_heads=num_heads, dim_head=dim_head)
    output = Dropout(0.2)(output)

    return output, output

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
# Normalization ->  Pos_encoding + MHSA -> Dropout
## See file:///C:/Users/pouli/Documents/Mathimata/Eidiko_Thema/INFO/online_material/conformer_speech_recognition.pdf
#(Conformer: Convolution-augmented Transformer for Speech Recognition)
class MultiHeadAttentionModule(keras.layers.Layer):
  def __init__(self,*, dim_head, num_heads):
    super(MultiHeadAttentionModule, self).__init__()
    self.num_heads = num_heads
    self.d_model = dim_head * num_heads

  def call(self, inputs, v, k, q, mask):
    print("HEREEE ", q)

    temp = inputs
    inputs = layer_normalization.LayerNormalization()(inputs)
    print(inputs)
    pos_embedding = positional_encoding(d_model= self.d_model)#### position = q.shape[2]
    #pos_embedding (1, seq_posemb, d_model)
    #mhsa = MultiHeadAttention(num_heads= self.num_heads, dim_head= self.d_model )
    
    #output, attention_weights = mhsa(inputs, q= inputs, v= inputs,k=inputs, mask=mask, pos_embedding=pos_embedding)
    output, attention_weights = MultiHeadAttention_fun(q = inputs , v = inputs, k = inputs , pos_embedding=pos_embedding)
    output = Dropout(0.2)(output)

    return output, output

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

def scaled_dot_product_attention(input_vector, pos_emb):
    Q, K, V = input_vector
    ## Q*K^T
    #K = tf.transpose(K, perm=[0,1,3,2])
    QK_mult = tf.matmul(Q,K, transpose_b=True) #(...,..., seq_len, seq_len)
    print(QK_mult) 
    dk = tf.cast(K.shape[-1], tf.float32)   #depth
    print(1/tf.math.sqrt(dk))
    #postitional_encoding*Q
    pose = tf.matmul(Q,pos_emb, transpose_b=True) #(...,..., seq_len, seq_len_posemb)
    #pose = tf.transpose(pose , perm=[0, 1, 3, 2])

    product = (QK_mult + pose[:,:,:,:,:QK_mult.shape[-1]] )/tf.math.sqrt(dk)
    print(product)
    ## the scaled self attention equation (page 4, "Attention is all you need")
    attention_weights = tf.nn.softmax(product, axis=-1)#(...,..., seq_len, seq_len)
    attention = tf.matmul(attention_weights, V)#(...,..., seq_len, depth)

    return attention, attention_weights

    def compute_output_shape(self, input_shape):
        out_shape = []
        for i in range(len(input_shape)-1):
            out_shape += [input_shape[i]]
    return tuple(out_shape)
def scaled_dot_product_attentionORIG(input_vector, pos_emb):
    Q, K, V = input_vector
    ## Q*K^T
    #K = tf.transpose(K, perm=[0,1,3,2])
    QK_mult = tf.matmul(Q,K, transpose_b=True) #(...,..., seq_len, seq_len)
    print(QK_mult) 
    dk = tf.cast(K.shape[-1], tf.float32)   #depth
    print(1/tf.math.sqrt(dk))
    #postitional_encoding*Q
    pose = tf.matmul(Q,pos_emb, transpose_b=True) #(...,..., seq_len, seq_len_posemb)
    #pose = tf.transpose(pose , perm=[0, 1, 3, 2])
    pose_cut = pose[:,:,:,:,:QK_mult.shape[-1]]
    product = (QK_mult + pose_cut )/tf.math.sqrt(dk)
    print(product)
    ## the scaled self attention equation (page 4, "Attention is all you need")
    attention_weights = tf.nn.softmax(product, axis=-1)#(...,..., seq_len, seq_len)
    attention = tf.matmul(attention_weights, V)#(...,..., seq_len, depth)
    return attention, attention_weights

class Conformer(Layer):
    def __init__(
            self, **kwargs):
        super(Conformer, self).__init__(**kwargs)

    def call(self, spec_cnn, dconv_kernel_size):
        num_heads=8
        dim_head=64
        print(spec_cnn)
        res_spec = spec_cnn
        print("FFN")
        spec_cnn = FeedForward(spec_cnn, encoder_dim=spec_cnn.shape[-1])//2
        spec_cnn = res_spec + spec_cnn
        print(spec_cnn)
        ##### MHSA layer ##############
        print("MHSA MHSA")
        #mhsa1 = MultiHeadAttentionModule(num_heads=8, dim_head=64)#####dim_head was 512
        #output_tensor, weights = mhsa1(spec_cnn, q=spec_cnn, v=spec_cnn, k=spec_cnn, mask= 1)
        output_tensor, weights = MultiHeadAttentionModule_fun(inputs = spec_cnn, num_heads=num_heads, dim_head=dim_head )
        output_tensor = Dense(spec_cnn.shape[-1])(output_tensor)
        spec_cnn = spec_cnn + output_tensor
        print(spec_cnn)
        #######################
        spec_cnn = ConvolutionModule(spec_cnn, spec_cnn.shape[1], dconv_kernel_size)#on the 2nd inpu it was 512(d_model)
        print(spec_cnn)
        #######################
        temp = FeedForward(spec_cnn, encoder_dim=spec_cnn.shape[-1])//2
        spec_cnn = Add()([spec_cnn , temp]) 
        spec_cnn = layer_normalization.LayerNormalization()(spec_cnn)
        print(spec_cnn)
        return spec_cnn

def Conformer_fun(spec_cnn, dconv_kernel_size):
    dim_head = 32
    num_heads = 4
    print(spec_cnn)
    res_spec = spec_cnn
    print("FFN")
    spec_cnn = FeedForward(spec_cnn, encoder_dim=spec_cnn.shape[-1])//2
    spec_cnn = Add()([spec_cnn , res_spec])
    print(spec_cnn)
    ##### MHSA layer ##############
    print("MHSA MHSA")
    #mhsa1 = MultiHeadAttentionModule(num_heads=8, dim_head=64)#####dim_head was 512
    #output_tensor, weights = mhsa1(spec_cnn, q=spec_cnn, v=spec_cnn, k=spec_cnn, mask= 1)
    output_tensor, weights = MultiHeadAttentionModule_fun(inputs = spec_cnn, num_heads=num_heads, dim_head=dim_head )
    output_tensor = Dense(spec_cnn.shape[-1])(output_tensor)
    spec_cnn = spec_cnn + output_tensor
    print(spec_cnn)
    #######################
    spec_cnn = ConvolutionModule(spec_cnn, spec_cnn.shape[1], dconv_kernel_size)#on the 2nd inpu it was 512(d_model)
    print(spec_cnn)
    #######################
    temp = FeedForward(spec_cnn, encoder_dim=spec_cnn.shape[-1])//2
    spec_cnn = Add()([spec_cnn , temp]) 
    spec_cnn = layer_normalization.LayerNormalization()(spec_cnn)
    print(spec_cnn)
    return spec_cnn
        
def FeedForward(spec_cnn, encoder_dim: int = 512,
            expansion_factor: int = 4):
    
    temp = spec_cnn
    #print(K.is_keras_tensor(temp))
    #K.print_tensor(spec_cnn)
    print("HERE")
    
    spec_cnn = layer_normalization.LayerNormalization()(spec_cnn)
    print(spec_cnn)
    spec_cnn = Dense(encoder_dim*expansion_factor, activation = None)(spec_cnn) ## Linear layer
    print(spec_cnn)
    
    print("FFN SHAPE ", temp)
    ##swish activation function
    spec_cnn = tf.keras.activations.sigmoid(spec_cnn) * spec_cnn
    spec_cnn = Dropout(0.02)(spec_cnn)    
    spec_cnn = Dense(encoder_dim, activation = None)(spec_cnn)
    print("FFN SHAPE ", spec_cnn)
    added = Add()([temp, spec_cnn])
    return added

def ConvolutionModule(spec_cnn, nb_cnn2d_filt, dconv_kernel_size: int = 31):
    temp = spec_cnn
    nb_cnn2d_filt = 2*nb_cnn2d_filt
    #(None, 256, 60, 2) = (B, C, T, F), B: batch_size, C: channels, T: time length, F: features or frequency length
    
    ####### Tensorflow added (None, 60, 2, 256)->(None,60,512)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    spec_cnn = Reshape((spec_cnn.shape[-3], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
    ###
    #pointwise convolution (B, C, T, F)->(B, 2*C, T, F)
    conv = Conv1D(filters=2* nb_cnn2d_filt, kernel_size=1, padding='same')(spec_cnn)
    # GLU Part (https://github.com/IRIS-AUDIO/SELD/blob/669ead73ce1e0db7bafef96d9f4037f9cf2cd0b7/modules.py)
    conv_1, conv_2 = tf.split(conv, 2, axis=-1)
    conv_2 = tf.keras.activations.sigmoid(conv_2)
    conv = conv_1 * conv_2
    spec_cnn = conv
    #2D depthwise conv
    kernel_size = dconv_kernel_size
    spec_cnn = Conv1D(filters=nb_cnn2d_filt ,kernel_size = kernel_size, padding='same')(spec_cnn)#,groups=nb_cnn2d_filt tensorflow 2.4.0
    spec_cnn = layer_normalization.LayerNormalization()(spec_cnn)
    ##swish activation function
    spec_cnn = tf.keras.activations.sigmoid(spec_cnn) * spec_cnn
    #pointwise convolution
    spec_cnn = Conv1D(filters=nb_cnn2d_filt, kernel_size=1, padding='same')(spec_cnn)
    spec_cnn = Dropout(0.02)(spec_cnn)
    #spec_cnn = Reshape((temp.shape[-3], spec_cnn.shape[-2],temp.shape[-1]))(spec_cnn)
    print(spec_cnn)

    ####### Tensorflow added (None, 60, 2, 256)->(None,60,512)
    spec_cnn = Reshape((temp.shape[-2],temp.shape[-3],temp.shape[-1]))(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    ###
    add = Add()([temp, spec_cnn])
    return add

##from https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb?hl=ro#scrollTo=jpEox7gJ8FCI
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

  class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv2d(keras.layers.Layer):
    #padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    def __init__(self, chan_in, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = tf.keras.layers.Conv2D(chan_in, kernel_size, groups = chan_in)

    def forward(self, x):
        x = tf.pad(x, self.padding)
        return self.conv(x)
