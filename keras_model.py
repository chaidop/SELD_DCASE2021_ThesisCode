#
# The SELDnet architecture
#
########## CHANGED import keras. ... to import tensorflow.keras....
from tensorflow import keras
import pdb
from keras.layers import Lambda,Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, Add, AveragePooling2D, Flatten, ZeroPadding2D ##CUSTONM CODE (to Add kai AveragePooling)
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend
keras.backend.set_image_data_format('channels_first')
 
from IPython import embed
import numpy as np
import os
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

import torch
from models import Conformer, Conformer_fun

'''
config = ConfigProto()
config.gpu_options.allow_growth = 0.5
session = InteractiveSession(config=config)
'''

import keras.backend.tensorflow_backend as K
#K.set_session(session)

#gpu cant run any model, so use cpu with: 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import tensorflow.keras
import keras
#tf.keras.backend.set_image_data_format('channels_first')
from numba import jit, cuda

print("EEEEEEEEEEEEEEEEEEEEEEEEEEE")
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''

"""
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)
"""
def res_identity(x, filters):
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block
    f1, f2 = filters
    print("\n--------IDENTITY---------\n")
    #first block
    x = Conv2D(f1, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
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
    # copy tensor to variable called x_skip
    x_skip = x
    f1, f2 = filters
    # Layer 1
    x = Conv2D(f1, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(f1, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)

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
    x = Conv2D(f1, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
    print(x)
    x = BatchNormalization()(x)
    print(x)

    # shortcut, Processing Residue with conv(1,1)
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x_skip)
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
    x = Conv2D(f1, kernel_size=(3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(f1, kernel_size=(1,1), strides = (1,1))(x_skip)
    # Add Residue
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x

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
    spec_cnn = res_conv18(spec_cnn, s=2, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*4, nb_cnn2d_filt*4))
    print(spec_cnn)

    #spec_cnn = MaxPooling2D(pool_size=(t_pool_size[2], f_pool_size[2]))(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    # 5th stage
    print("\n############ STAGE 4 ##############\n")
    spec_cnn = res_conv18(spec_cnn, s=2, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt*8))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt*8))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt*8))
    print(spec_cnn)
    spec_cnn = res_identity18(spec_cnn, filters=(nb_cnn2d_filt*8, nb_cnn2d_filt*8))
    print(spec_cnn)

    print(spec_cnn)
    spec_cnn = Dropout(0.2)(spec_cnn)
    spec_cnn = Dense(512, activation = 'relu')(spec_cnn)
    print(spec_cnn)
    spec_cnn = Dense(12, activation = 'softmax')(spec_cnn) ## 12 = nb_classes(the total number of sound events)

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
    x = res_conv18(x, s=2, filters=(128, 512))
    x = res_identity18(x, filters=(128, 512))
    x = res_identity18(x, filters=(128, 512))
    print("2",x)
    # 4th stage
    x = res_conv18(x, s=2, filters=(256, 1024))
    x = res_identity18(x, filters=(256, 1024))
    x = res_identity18(x, filters=(256, 1024))
    print("3",x)
    # ends with average pooling and dense connection
    #x = AveragePooling2D((2, 2), padding='same')(x)
    print(x)
    print("128")
    print(x)
   
    x = Dropout(0.2)(x)
    x = Dense(512, activation = 'relu')(x)
    print(x)
    x = Dense(12, activation = 'softmax')(x) ## 12 = nb_classes(the total number of sound events)

    return x  
# function optimized to run on gpu
#@cuda.jit  
def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, is_accdoa,
              model_approach, 
              dconv_kernel_size,
              nb_conf): ####### CUSTOM CODE

    #tf.config.experimental.list_physical_devices('GPU')
    #tf.debugging.set_log_device_placement(True)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    print("data in ", data_in)
    # CNN
    spec_cnn = spec_start
    #print(spec_cnn)
    ###### end #####
    #spec_cnn = ZeroPadding2D(padding=(3, 3))(spec_cnn)
    if model_approach == 0:
        for i, convCnt in enumerate(f_pool_size):
            spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
            spec_cnn = BatchNormalization()(spec_cnn)
            spec_cnn = Activation('relu')(spec_cnn)
            spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
            spec_cnn = Dropout(dropout_rate)(spec_cnn)
        spec_cnn = Permute((2, 1, 3))(spec_cnn)

    if model_approach == 1 or model_approach == 2:
        # 1st stage
        # here we perform maxpooling, see the figure above
        x = spec_cnn
        #x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, kernel_size=(7, 7), padding='same')(x)
        print(x)
        x = BatchNormalization()(x)
        print(x)
        x = Activation('relu')(x)
        print(x)
        x = MaxPooling2D(pool_size=(t_pool_size[0], f_pool_size[0]))(x)
        print(x)
        spec_cnn = x
        print("hello\n")
        if model_approach == 1:
            spec_cnn = resnet18(spec_cnn)
        elif model_approach == 2:
            spec_cnn = resnet34(t_pool_size,f_pool_size,spec_cnn, nb_cnn2d_filt)
        #added maxpool for reshaping output
        #spec_cnn = Permute((2, 1, 3))(spec_cnn)
        print(spec_cnn)
        spec_cnn = Permute((2, 1, 3))(spec_cnn)
        print(spec_cnn)
        ##need to pool to bring sequence to (60,64,2) dimension(seq-len, mel-bands, idk(??))
        spec_cnn = AveragePooling2D(pool_size=(4,6), padding='same')(spec_cnn)
        print("pool ",spec_cnn)
        
    ##### Resnet34 IMPLEMENTATION ##########
    ### 23/03/2022:: put code for this in above 'if' option
    ##this is the original that i run for the results in eidiko thema
    if model_approach == 200:
        spec_cnn = resnet34(t_pool_size,f_pool_size,spec_cnn, nb_cnn2d_filt)
        # ends with average pooling and dense connection
        #spec_cnn = BatchNormalization()(spec_cnn)
        #spec_cnn = Activation('relu')(spec_cnn)
        
        #spec_cnn = AveragePooling2D(pool_size=(2,1), padding='same')(spec_cnn)
        #print(spec_cnn)

        #spec_cnn = Dropout(dropout_rate)(spec_cnn)
        #spec_cnn = Dense(512, activation = 'relu')(spec_cnn)
        #print(spec_cnn)
        #spec_cnn = Dense(12, activation = 'softmax')(spec_cnn) ## 12 = nb_classes(the total number of sound events)
        #print(spec_cnn)
        print("184")
        print(spec_cnn)
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
    
    model = keras.applications.Resnet34(
        include_top=True, weights=None, input_tensor=spec_cnn,
        input_shape=None, pooling=max, classes=12,
    )
    """
    
    ########### END RESNET 34 ######################
    if model_approach == 3:
        #print("INITIAL SHAPE ", spec_cnn)
        ###(10, 300, 64) (CHANNELS, timesteps(sequence length per sample), mel-spectogramms per audio file)
        ### subsampling (DCASE2021_Zhang_67_t3.pdf)
        spec_cnn = keras.layers.Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
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
        model = Conformer(spec_cnn)
        print("model printed")
        ##Zhang and Ko use 2 and 3 conformers respectively
        #for i in range(depth):
        spec_cnn = model( spec_cnn, dconv_kernel_size=dconv_kernel_size)
        
        #spec_cnn = Conformer_fun( spec_cnn, dconv_kernel_size=dconv_kernel_size) #(None, 256, 60, 2)
        print("Conformer out ", spec_cnn.shape)
        ###### RESHAPING (60,512) ########
        permuter=Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3))) #(None, 60, 256, 2)
        spec_cnn = permuter(spec_cnn)  
        spec_cnn = Reshape((spec_cnn.shape[-3], spec_cnn.shape[-2]*spec_cnn.shape[-1]))(spec_cnn)#(None, 60, 512)
        print("Lambda out ", spec_cnn.shape) 
        ###### DENSE LAYERS #########
        spec_cnn = Dense(256, activation = 'relu')(spec_cnn)
        spec_cnn = Dense(128, activation = 'relu')(spec_cnn)
        #Dense(36, activation = 'tanh')(spec_cnn)
        #print(spec_cnn)

    # RNN
    #print(spec_cnn)
    print("data_out[-2]:")
    print(data_out[-2])
    spec_rnn = Reshape((data_out[-2] if is_accdoa else data_out[0][-2], -1))(spec_cnn)
    print(spec_rnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
        #LSTM(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                #return_sequences=True)(spec_rnn)
    print(spec_cnn)
    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)
    print(spec_cnn)
    doa = TimeDistributed(Dense(data_out[-1] if is_accdoa else data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)
    print(spec_cnn)
    model = None
    if is_accdoa:
        model = Model(inputs=spec_start, outputs=doa)
        model.compile(optimizer=Adam(), loss='mse')
    else:
        # FC - SED
        sed = spec_rnn
        for nb_fnn_filt in fnn_size:
            sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
            sed = Dropout(dropout_rate)(sed)
        sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
        sed = Activation('sigmoid', name='sed_out')(sed)

        if doa_objective is 'mse':
            model = Model(inputs=spec_start, outputs=[sed, doa])
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
        elif doa_objective is 'masked_mse':
            doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
            model = Model(inputs=spec_start, outputs=[sed, doa_concat])
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


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()
