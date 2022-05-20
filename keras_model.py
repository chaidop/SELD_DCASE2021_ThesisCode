#
# The SELDnet architecture
#f
########## CHANGED import keras. ... to import tensorflow.keras....
#import tensorflow
#from tensorflow import keras

#import tensorflow.keras
from keras.layers import Lambda,Bidirectional, Conv2D, Conv1D, MaxPooling2D, Input, Concatenate, Add, AveragePooling2D, Flatten, ZeroPadding2D, Multiply, GlobalAveragePooling2D, ELU ##CUSTONM CODE (to Add kai AveragePooling)
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
from optimizers import *
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np
from neural_models.models import Conformer, Conformer_fun
from neural_models.dense_conv import DenseNet
import os

from keras import backend as K
from neural_models.conformer_tf import *
#K.tensorflow_backend._get_available_gpus()

from numba import jit, cuda


import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')

for device in physical_devices:
    print(device)
    tf.config.experimental.set_memory_growth(device, True)


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
'''
import tensorboard
tensorboard.__version__
'''
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
''' 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''
'''
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)
'''
### https://github.com/Machine-Listeners-Valencia/seld-dcase2021/blob/master/keras_model.py
def squeeze_excite_block(input, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply([input, se])
    return x


def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = Multiply([input_tensor, se])
    return x
def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = Add()([cse, sse])
    return x
####
def res_identity(x, filters):
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block
    f1, f2 = filters
    print("\n--------IDENTITY---------\n")
    #first block
    x = Conv2D(f1, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    print(x)
    x = BatchNormalization()(x)
    print(x)
    x = Activation('relu')(x)
    print(x)
    #second block # bottleneck (but size kept same with padding)
    #x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    print(x)
    x = BatchNormalization()(x)
    print(x)
    # x = Activation(activations.relu)(x)

    # add the input
    x = Add()([x, x_skip])
    print(x)
    x = Activation('relu')(x)
    print(x)

    return x

def res_identity18(x, filters):
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added
    x_skip = x
    # copy tensor to variable called x_skip
    temp1, temp2 = x.shape[-2], x.shape[-1]
    print(temp1, temp2)
    ##reshape for 1d conv
    #(64, 60, 16)->(60, 16, 64), cause conv1d works on last dim
    x = Permute((2,3,1))(x)
    print(x)
    #x = Reshape((x.shape[-3],x.shape[-2]*x.shape[-1]))(x)

    print("xskip ",x_skip)
    f1, f2 = filters
    # Layer 1
    x = Conv1D(f1, 3, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(x)
    # Layer 2
    x = Conv1D(f2, 3, padding = 'same')(x)
    x = BatchNormalization()(x)
    print(x)
    
    #x = Reshape((temp1, temp2, x.shape[-1]))(x)
    x = Permute((3,1,2))(x)
    # Add Residue
    x = Add()([x, x_skip])
    print(x)
    x = Activation('relu')(x)
    print(x)

    return x

def res_conv(x, s, filters):
    '''
    here the input size changes'''
    x_skip = x
    f1, f2 = filters
    print('\n------------ CONV BLOCK ------------\n')
    # first block
    x = Conv2D(f1, kernel_size=(3,3), strides=(s, s), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    print(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    print(x)
    x = Activation('relu')(x)
    print(x)

    # second block
    #x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)

    #third block
    x = Conv2D(f1, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    print(x)
    x = BatchNormalization()(x)
    print(x)

    # shortcut, Processing Residue with conv(1,1)
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    print(x)
    x_skip = BatchNormalization()(x_skip)
    print(x)

    # add
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x

def res_conv18(x, s, filters):
    '''
    here the input size changes'''
    # copy tensor to variable called x_skip
    x_skip = x
    f1, f2 = filters
    # Layer 1
    x = Conv2D(f1, kernel_size=(3,3), padding = 'same', strides = (1,1))(x)
    print("what",x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(f2, kernel_size=(3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(f2, kernel_size=(1,1), strides = (1,1))(x_skip)
    # Add Residue
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x

#12/5/2022 resnet parameters from 'A Model Ensemble Approach for Sound Event Localization and Detection'
def resnet_2020(t_pool_size,f_pool_size, spec_cnn, nb_cnn2d_filt):
    # 1st stage
    # here we perform maxpooling, see the figure above
    print("hello\n")
    print(spec_cnn)
    # frm here on only conv block and identity block, no pooling
    print("\n############ STAGE 1 ##############\n")
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt, nb_cnn2d_filt))
    print(spec_cnn)

    spec_cnn = Dropout(0.2)(spec_cnn)
    # 3rd stage
    print("\n############ STAGE 2 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=1, filters=(nb_cnn2d_filt*2, nb_cnn2d_filt*2))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*2,nb_cnn2d_filt*2))
    print(spec_cnn)

    #spec_cnn = MaxPooling2D(pool_size=(t_pool_size[1],f_pool_size[1]))(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    # 4th stage
    print("\n############ STAGE 3 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=2, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))

    #spec_cnn = MaxPooling2D(pool_size=(t_pool_size[2], f_pool_size[2]))(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    # 5th stage
    print("\n############ STAGE 4 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=2, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    
    #spec_cnn = Conv2D(nb_cnn2d_filt, kernel_size=(1,1), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(spec_cnn)
    #spec_cnn = BatchNormalization(axis=1)(spec_cnn)
    #spec_cnn = Activation('relu')(spec_cnn)
    print(spec_cnn)

    return spec_cnn
#implement the Resnet34 architecture
def resnet34(t_pool_size,f_pool_size, spec_cnn, nb_cnn2d_filt):
    # 1st stage
    # here we perform maxpooling, see the figure above
    print("hello\n")
    print(spec_cnn)
    # frm here on only conv block and identity block, no pooling
    print("\n############ STAGE 1 ##############\n")
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt, nb_cnn2d_filt))
    print(spec_cnn)
    # 3rd stage
    print("\n############ STAGE 2 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=1, filters=(nb_cnn2d_filt*2, nb_cnn2d_filt*2))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*2,nb_cnn2d_filt*2))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*2, nb_cnn2d_filt*2))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*2, nb_cnn2d_filt*2))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*2, nb_cnn2d_filt*2))
    print(spec_cnn)

    #spec_cnn = MaxPooling2D(pool_size=(t_pool_size[1], f_pool_size[1]))(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    # 4th stage
    print("\n############ STAGE 3 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=1, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt))
    print(spec_cnn)

    #spec_cnn = MaxPooling2D(pool_size=(t_pool_size[2], f_pool_size[2]))(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    # 5th stage
    print("\n############ STAGE 4 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=1, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt))
    print(spec_cnn)

    print(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    
    #spec_cnn = Conv2D(nb_cnn2d_filt, kernel_size=(3,3), strides=(1, 2), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(spec_cnn)
    print(spec_cnn)

    return spec_cnn

#implement the Resnet34 architecture
def resnet18(input_im):
    x = input_im
    #2nd stage
    # from here on only conv block and identity block, no pooling
    x = res_conv18(x, s=1, filters=(64, 64))
    x = res_identity18(x, filters=(64, 64))
    x = res_identity18(x, filters=(64, 64))
    print("1",x)
    # 3rd stage
    x = res_conv18(x, s=2, filters=(128, 128))
    x = res_identity18(x, filters=(128, 128))
    x = res_identity18(x, filters=(128, 128))
    print("2",x)
    # 4th stage
    x = res_conv18(x, s=2, filters=(256, 64))
    x = res_identity18(x, filters=(256, 64))
    x = res_identity18(x, filters=(256, 64))
    print("3",x)
    # 4th stage
    x = res_conv18(x, s=2, filters=(512, 64))
    x = res_identity18(x, filters=(512, 64))
    x = res_identity18(x, filters=(512, 64))
    # ends with average pooling and dense connection
    #x = AveragePooling2D((2, 2), padding='same')(x)
    print(x)
    print("128")
    print(x)
   
    x = Dropout(0.2)(x)
    #x = Dense(512, activation = 'relu')(x)
    print(x)
    #x = Dense(12, activation = 'softmax')(x) ## 12 = nb_classes(the total number of sound events)

    return x  
  
#implement the Resnet50 architecture
def resnet50(input_im):

  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=2, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = AveragePooling2D((2, 2), padding='same')(x)
  print("128")
  x = Flatten()(x)

  return x
# function optimized to run on gpu
#@cuda.jit  
def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, is_accdoa,
              model_approach, depth, decoder, dconv_kernel_size, nb_conf, simple_parallel): ####### CUSTOM CODE

    #tf.config.experimental.list_physical_devices('GPU')
    #tf.debugging.set_log_device_placement(True)

    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    # model definition
    spec_start = Input(shape=( data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    #print(spec_c_cnn)
    ###### end #####
    #spec_cnn = ZeroPadding2D(padding=(2, 2))(spec_cnn)
    if model_approach == 0 or model_approach == 4:
        for i, convCnt in enumerate(f_pool_size):
            spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
            spec_cnn = BatchNormalization()(spec_cnn)
            spec_cnn = Activation('relu')(spec_cnn)
            spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
            spec_cnn = Dropout(dropout_rate)(spec_cnn)
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        print('model 0')

    ##### RESNET50 IMPLEMENTATION ##########
    if model_approach == 1 or model_approach == 2:
        keras.backend.set_image_data_format('channels_first')
        # 1st stage
        # here we perform maxpooling, see the figure above
        x = spec_cnn
        #x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, kernel_size=(7, 7), padding='same')(x)
        print(x)
        x = BatchNormalization()(x)
        print(x)
        x = Activation('relu')(x)
        print('relu',x)
        x = MaxPooling2D(pool_size=(t_pool_size[0], f_pool_size[0]))(x)
        print(x)
        spec_cnn = x
        print("hello\n")
        if model_approach == 1:
            spec_cnn = resnet18(spec_cnn)
        elif model_approach == 2:
            spec_cnn = resnet34(t_pool_size,f_pool_size,spec_cnn, nb_cnn2d_filt)
        #added maxpool for reshaping output
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        #print(spec_c_cnn)
        ##need to pool to bring sequence to (60,64,2) dimension(seq-len, mel-bands, idk(??))
        spec_cnn = AveragePooling2D(pool_size=(1,4), padding='same')(spec_cnn)
        spec_cnn = Reshape((spec_cnn.shape[1], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
        #12/5/2022
        print(spec_cnn)
        spec_cnn = Dropout(0.2)(spec_cnn)
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        #print(spec_c_cnn)
        #17/5/2022 shouldnt use softmax
        #spec_cnn = Dense(12, activation = 'sigmoid')(spec_cnn) ## 12 = nb_classes(the total number of sound events)

        print("pool ",spec_cnn)
        
    if model_approach == 3:
        #print("INITIAL SHAPE ", spec_cnn)
        ###(10, 300, 64) (CHANNELS, timesteps(sequence length per sample), mel-spectogramms per audio file)
        ### subsampling (DCASE2021_Zhang_67_t3.pdf)
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        #print("(64, 300, 64) ",spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(5,4))(spec_cnn)
        #print("(64, 60, 16) ", spec_cnn)
        ###(64, 60, 16)
        spec_cnn = Conv2D(128, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        ###(128, 60, 4)
        spec_cnn = Conv2D(256, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,2))(spec_cnn)
        ###(256, 60, 2)
        ### Conformer
        #print("Before conformer ", spec_cnn)
        #model = Conformer()
        print("model printed")
        ##Zhang and Ko use 2 and 3 conformers respectively
        
        #print(spec_c_cnn)
        for i in range(depth):
            spec_cnn = Conformer_fun( spec_cnn, dconv_kernel_size=31)
        
        print("Conformer out ", spec_cnn.shape, spec_cnn[:,1,1])
        #(60,512) with models_2d.py
        #(256, 60, 2) with models.py
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        spec_cnn = Reshape((spec_cnn.shape[-3], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
        #(60, 512)
        ###### DENSE LAYERS #########
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(128, activation = 'relu')(spec_cnn)
        #Dense(36, activation = 'tanh')(spec_cnn)
        #print(spec_cnn)

    elif model_approach == 5:
        ##Ko's implementation kinda
        ###STAGE 1 
        temp = spec_cnn1 = spec_cnn2 = spec_cnn
        
        spec_cnn1 = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn1)
        spec_cnn1 = BatchNormalization()(spec_cnn1)
        spec_cnn2 = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn2)
        spec_cnn2 = BatchNormalization()(spec_cnn2)
        spec_cnn1 = Add()([spec_cnn1, spec_cnn2])
        spec_cnn1 = Activation('relu')(spec_cnn1)
        spec_cnn2 = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(temp)
        a = Concatenate(axis=-1)([spec_cnn1,spec_cnn2])

        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(a)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Add()([temp, spec_cnn])
        spec_cnn = Activation('relu')(spec_cnn)
        b = Concatenate(axis=-1)([temp,spec_cnn])

        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
        spec_cnn = Permute((2, 1, 3))(spec_cnn)

    elif model_approach == 6:
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        #print("(64, 300, 64) ",spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        #print("(64, 60, 16) ", spec_cnn)
        ###(64, 60, 16)
        spec_cnn = Conv2D(256, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(5,2))(spec_cnn)

        #(256, 60, 2)
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        #(60, 512)
        spec_cnn = Reshape((spec_cnn.shape[-3], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
        #(60, 128)
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)#i need to lower the dimension cause i got memory error 256 for best model
        #(128, 60)
        #spec_cnn = Permute((2, 1))(spec_cnn)
        for i in range(depth):
            spec_cnn = ConformerBlock(dim = spec_cnn.shape[-1], inputs= spec_cnn)
        print("output conformer ", spec_cnn.shape)
        
        #spec_cnn = Permute((2, 1))(spec_cnn)
        #(60, 128)
        ###### DENSE LAYERS #########
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(128, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(36, activation = 'tanh')(spec_cnn)
    elif model_approach == 10:
        #(10,300,64)
        print("approach 6! ",spec_cnn)
        '''
        #keras.backend.set_image_data_format('channels_first')
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        #print("(64, 300, 64) ",spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        #print("(64, 60, 16) ", spec_cnn)
        ###(64, 60, 16)
        spec_cnn = Conv2D(256, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(5,2))(spec_cnn)
        '''
        #zhang
        '''conv_size = [nb_cnn2d_filt, nb_cnn2d_filt*2, nb_cnn2d_filt*4]
        for i in range(3):
            spec_cnn = Conv2D(conv_size[i], kernel_size=(3, 3), padding='same')(spec_cnn)
            print(spec_cnn)
            spec_cnn = BatchNormalization(axis=1)(spec_cnn)
            spec_cnn = Activation('relu')(spec_cnn)
            spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i],f_pool_size[i]))(spec_cnn)'''
        
        #resne34-conformer
        keras.backend.set_image_data_format('channels_first')
        # 1st stage
        # here we perform maxpooling, see the figure above
        x = spec_cnn
        #x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, kernel_size=(7, 7), padding='same')(x)
        print(x)
        x = BatchNormalization()(x)
        print(x)
        x = Activation('relu')(x)
        print('relu',x)
        x = MaxPooling2D(pool_size=(t_pool_size[0], f_pool_size[0]))(x)
        print(x)
        spec_cnn = x
        print("hello\n")
        spec_cnn = resnet34(t_pool_size,f_pool_size,spec_cnn, nb_cnn2d_filt)
        #added maxpool for reshaping output
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        #print(spec_c_cnn)
        ##need to pool to bring sequence to (60,64,2) dimension(seq-len, mel-bands, idk(??))
        spec_cnn = AveragePooling2D(pool_size=(1,4), padding='same')(spec_cnn)
        print(spec_cnn)
        spec_cnn = Dropout(0.2)(spec_cnn)
        #(256, 60, 2)
        #spec_cnn = Permute((2, 1, 3))(spec_cnn)
        print('permute', spec_cnn)
        spec_cnn = Reshape((spec_cnn.shape[-3], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
        #(60, 512)
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)#i need to lower the dimension cause i got memory error 256 for best model
        print(spec_cnn)
        #(128, 60)
        #spec_cnn = Permute((2, 1))(spec_cnn)
        for i in range(depth):
            spec_cnn = ConformerBlock(dim = spec_cnn.shape[-1], inputs= spec_cnn)
        print("output conformer ", spec_cnn.shape)
        
        #spec_cnn = Permute((2, 1))(spec_cnn)
        #(60, 128)
        ###### DENSE LAYERS #########
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(128, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(36, activation = 'tanh')(spec_cnn)
        """
        #old version
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        #print("(64, 300, 64) ",spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        #print("(64, 60, 16) ", spec_cnn)
        ###(64, 60, 16)
        spec_cnn = Conv2D(256, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(5,2))(spec_cnn)
        """
        """
        ### subsampling (DCASE2021_Zhang_67_t3.pdf), last used 26/4/2022
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        #print("(64, 300, 64) ",spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(5,4))(spec_cnn)
        #print("(64, 60, 16) ", spec_cnn)
        ###(64, 60, 16)
        spec_cnn = Conv2D(128, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,4))(spec_cnn)
        ###(128, 60, 4)
        spec_cnn = Conv2D(256, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1,2))(spec_cnn)
        ###(256, 60, 2)

        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        spec_cnn = Reshape((spec_cnn.shape[-3], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
        #(150, 512)
        #25/4/22
        spec_cnn = Dense(128)(spec_cnn)
        spec_cnn = Permute((2, 1))(spec_cnn)
        #(256, 150)
        #for i in range(depth):
        spec_cnn = ConformerBlock(dim = spec_cnn.shape[-1], inputs= spec_cnn)
        print("output conformer ", spec_cnn.shape)
        
        #(60, 256)
        spec_cnn = Permute((2, 1))(spec_cnn)
        ###### DENSE LAYERS #########
        spec_cnn = Dense(512, activation = 'relu')(spec_cnn)#, kernel_regularizer='l1'
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)#, kernel_regularizer='l1'
        spec_cnn = Dense(128, activation = 'relu')(spec_cnn)#, kernel_regularizer='l2'
        spec_cnn = Dense(36, activation = 'tanh')(spec_cnn)#, kernel_regularizer='l2'
        """
    elif model_approach == 7:
        spec_cnn = Conv2D(64, kernel_size=(7, 7), padding='same')(spec_cnn)
        print(spec_cnn)
        spec_cnn = BatchNormalization(axis=1)(spec_cnn)
        print(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[0],f_pool_size[0]))(spec_cnn)
        spec_cnn = resnet_2020(t_pool_size=t_pool_size, f_pool_size=f_pool_size, spec_cnn=spec_cnn, nb_cnn2d_filt=nb_cnn2d_filt)
        print('resnet out ', spec_cnn)
        #added maxpool for reshaping output
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        #print(spec_c_cnn)
        ##need to pool to bring sequence to (60,64,2) dimension(seq-len, mel-bands, idk(??))
        spec_cnn = AveragePooling2D(pool_size=(1,4), padding='same')(spec_cnn)
        spec_cnn = Reshape((spec_cnn.shape[1], spec_cnn.shape[-1]*spec_cnn.shape[-2]))(spec_cnn)
        #12/5/2022
        print(spec_cnn)
        spec_cnn = Dropout(0.2)(spec_cnn)
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        
        #(128, 60)
        for i in range(depth):
            spec_cnn = ConformerBlock(dim = spec_cnn.shape[-1], inputs= spec_cnn)
        print("output conformer ", spec_cnn.shape)
        
        #spec_cnn = Permute((2, 1))(spec_cnn)
        #(60, 128)
        ###### DENSE LAYERS #########
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(128, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(36, activation = 'tanh')(spec_cnn)
    elif model_approach == 8:
        print(spec_cnn)
        spec_cnn = DenseNet(inputs = spec_cnn, growth_rate=nb_cnn2d_filt//2, layers=2, nb_filters=nb_cnn2d_filt)
        spec_cnn = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(spec_cnn)
    #Najaro-Alcajar squeeze excitation
    elif model_approach == 9:
        spec_aux = spec_cnn
        spec_cnn = Conv2D(nb_cnn2d_filt, 3, padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = ELU()(spec_cnn)
        spec_cnn = Conv2D(nb_cnn2d_filt, 3, padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)

        spec_aux = Conv2D(nb_cnn2d_filt, 1, padding='same')(spec_aux)
        spec_aux = BatchNormalization()(spec_aux)

        spec_cnn = Add()([spec_cnn, spec_aux])
        spec_cnn = ELU()(spec_cnn)

        spec_cnn = channel_spatial_squeeze_excite(spec_cnn, ratio=ratio)

        spec_cnn = Add()([spec_cnn, spec_aux])

        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

        spec_cnn = Permute((2, 1, 3))(spec_cnn)
    """
        #####
        #spec_cnn = ZeroPadding2D(padding=(6, 6))(spec_cnn)
        print("========== ZEROPADD: ========")
        print(spec_cnn)
        # 1st stage
        # here we perform maxpooling, see the figure above

        spec_cnn = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(spec_cnn)
        print("========== CONV2D: ========")
        print(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        print("========== BATCH: ========")
        print(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        print("========== RELU: ========")
        print(spec_cnn)
        spec_cnn = MaxPooling2D((3, 3), strides=(2, 2))(spec_cnn)
        print("========== MAX POOL: ========")
        print(spec_cnn)
        # 2nd stage
        # frm here on only conv block and identity block, no pooling

        spec_cnn = res_conv(spec_cnn, s=1, filters=(64, 256))
        spec_cnn = res_identity(spec_cnn, filters=(64, 256))
        spec_cnn = res_identity(spec_cnn, filters=(64, 256))

        # 3rd stage

        spec_cnn = res_conv(spec_cnn, s=2, filters=(128, 512))
        spec_cnn = res_identity(spec_cnn, filters=(128, 512))
        spec_cnn = res_identity(spec_cnn, filters=(128, 512))
        spec_cnn = res_identity(spec_cnn, filters=(128, 512))

        # 4th stage

        spec_cnn = res_conv(spec_cnn, s=2, filters=(256, 1024))
        spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
        spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
        spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
        spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
        spec_cnn = res_identity(spec_cnn, filters=(256, 1024))

        # 5th stage

        spec_cnn = res_conv(spec_cnn, s=2, filters=(512, 2048))
        spec_cnn = res_identity(spec_cnn, filters=(512, 2048))
        spec_cnn = res_identity(spec_cnn, filters=(512, 2048))

        # ends with average pooling and dense connection

        spec_cnn = AveragePooling2D((2, 2), padding='same')(spec_cnn)

        spec_cnn = Flatten()(spec_cnn)
    
    ####
    
    model = keras.applications.ResNet50(
        include_top=True, weights=None, input_tensor=spec_cnn,
        input_shape=None, pooling=max, classes=12,
    )
    """
    
########### END RESNET 50 ######################
    # RNN
    #print(spec_cnn)
    print("data_out[-2]:")
    print(data_out[-2])
    print(spec_cnn.shape)

    # RNN    
    spec_rnn = Reshape((data_out[-2] if is_accdoa else data_out[0][-2], -1))(spec_cnn)
	
    for nb_rnn_filt in rnn_size:
        if decoder == 0:
            keras.backend.set_image_data_format('channels_last')
            spec_rnn = Bidirectional(
                GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                    return_sequences=True),
                merge_mode='mul'
            )(spec_rnn)
        elif decoder == 1:
            keras.backend.set_image_data_format('channels_last')
            spec_rnn = Bidirectional(
                LSTM(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                    return_sequences=True),
                merge_mode='mul'
            )(spec_rnn)
    print(spec_cnn)
    # FC - DOA
    doa = spec_rnn
    #Kos sed cofnormer
    #spec_rnn = Conformer_fun( spec_rnn, dconv_kernel_size=7, num_heads=2 , dim_head=6)
    if simple_parallel == True:
        x_output = TimeDistributed(Dense(data_out[-1], kernel_initializer=keras.initializers.GlorotUniform(seed=None)))(doa)
        x_output = Activation('relu')(x_output)
        x_output = Dropout(dropout_rate)(x_output)
        x_output = TimeDistributed(Dense(12, kernel_initializer=keras.initializers.GlorotUniform(seed=None)))(x_output)
        x_output = Activation('tanh', name='doa_out_x')(x_output)
        x_output = Dropout(dropout_rate)(x_output)

        y_output = TimeDistributed(Dense(data_out[-1], kernel_initializer=keras.initializers.GlorotUniform(seed=None)))(doa)
        y_output = Activation('relu')(y_output)
        y_output = Dropout(dropout_rate)(y_output)
        y_output = TimeDistributed(Dense(12, kernel_initializer=keras.initializers.GlorotUniform(seed=None)))(y_output)
        y_output = Activation('tanh', name='doa_out_y')(y_output)
        y_output = Dropout(dropout_rate)(y_output)
        z_output = TimeDistributed(Dense(data_out[-1], kernel_initializer=keras.initializers.GlorotUniform(seed=None)))(doa)
        z_output = Activation('relu')(z_output)
        z_output = Dropout(dropout_rate)(z_output)
        z_output = TimeDistributed(Dense(12, kernel_initializer=keras.initializers.GlorotUniform(seed=None)))(z_output)
        z_output = Activation('tanh', name='doa_out_z')(z_output)
        z_output = Dropout(dropout_rate)(z_output)

        doa = keras.layers.concatenate([x_output, y_output, z_output], axis=-1)  # (batch_size, time_steps, 3 * n_classes)
    else:
        for nb_fnn_filt in fnn_size:
            doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
            doa = Dropout(dropout_rate)(doa)
        print(spec_cnn)
        doa = TimeDistributed(Dense(data_out[-1] if is_accdoa else data_out[1][-1]))(doa)
        doa = Activation('tanh', name='doa_out')(doa)
        print(spec_cnn)
    #model = None
    # CUSTOM added BinaryCrossentropy loss as seen in Xytos diplwmatikh,
    if is_accdoa:
        model = Model(inputs=spec_start, outputs=doa)
        
        model.compile(optimizer=Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['accuracy'])
    else:
        # FC - SED
        sed = spec_rnn
        for nb_fnn_filt in fnn_size:
            sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
            sed = Dropout(dropout_rate)(sed)
        sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
        sed = Activation('sigmoid', name='sed_out')(sed)

        if doa_objective is 'mse':
            model = Model(inputs=spec_start, outputs=doa)
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
        elif doa_objective is 'masked_mse':
            doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
            model = Model(inputs=spec_start, outputs=doa)
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
        else:
            print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
            exit()
    model.summary()
    return model


def masked_mse(y_gt, model_out):
    nb_classes = 12 #TODO fix this hardcoded value of number of classes
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :nb_classes] >= 0.5
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, nb_classes:] - model_out[:, :, nb_classes:]) * sed_out))/keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective, 
                    model_approach):### CUSTOM
    if doa_objective is 'mse':
        ##CUSTOM need to check if it is using custom layer conformer, to add parameter
        #if model_approach == 3:
        #    return load_model(model_file, custom_objects={'Conformer': Conformer})
        return load_model(model_file, custom_objects={'AdaBelief': AdaBelief})
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()