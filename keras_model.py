#
# The SELDnet architecture
#

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, ZeroPadding2D, AveragePooling2D, Flatten, Add
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers import add, multiply, GlobalAveragePooling2D, ELU
import keras.backend as K
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.activations import relu
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow.keras
import keras
import tensorflow as tf

keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np

def res_identity(x, filters):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block
  f1, f2 = filters

  #first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input
  x = Add()([x, x_skip])
  x = Activation(relu)(x)

  return x

def res_conv(x, s, filters):
  '''
  here the input size changes'''
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tensorflow.keras.regularizers.l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add
  x = Add()([x, x_skip])
  x = Activation(relu)(x)

  return x
#implement the Resnet50 architecture
def resnet50(input_im):

  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
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

  # define the model

  model = Model(inputs=input_im, outputs=x, name='Resnet50')

  return model

def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, is_accdoa):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    ######## skip connection #####
    #xskip = spec_cnn
    #x = Add()([x, x_skip])
    #x = Activation(relu)(x)
    ###### end #####
    #spec_cnn = ZeroPadding2D(padding=(3, 3))(spec_cnn)
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    ##### RESNET50 IMPLEMENTATION ##########
    # frm here on only conv block and identity block, no pooling
    """"
    spec_cnn = res_conv(spec_cnn, s=1, filters=(64, 256))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(64, 256))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(64, 256))
    print(spec_cnn)
    # 3rd stage

    spec_cnn = res_conv(spec_cnn, s=2, filters=(128, 512))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(128, 512))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(128, 512))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(128, 512))
    print(spec_cnn)

    # 4th stage

    spec_cnn = res_conv(spec_cnn, s=2, filters=(256, 1024))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(256, 1024))
    print(spec_cnn)

    # 5th stage

    spec_cnn = res_conv(spec_cnn, s=2, filters=(60, 15))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(60, 15))
    print(spec_cnn)
    spec_cnn = res_identity(spec_cnn, filters=(60, 15))
    print(spec_cnn)

    # ends with average pooling and dense connection

    spec_cnn = AveragePooling2D((2, 2), padding='same')(spec_cnn)
    print(spec_cnn)
    print("184")
    #spec_cnn = Flatten()(spec_cnn)
    print(spec_cnn)
    """

    """
    spec_cnn = ZeroPadding2D(padding=(6, 6))(spec_cnn)
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
    spec_cnn = Activation(tf.keras.activations.relu)(spec_cnn)
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
    """
    """
    tensorflow.keras.applications.resnet50.ResNet50(
        include_top=True, weights=None, input_tensor=spec_cnn,
        input_shape=None, pooling=max, classes=30,
    )
    """
########### END RESNET 50 ######################
    # RNN
    print(spec_cnn)
    print("data_out[-2]:")
    print(data_out[-2])
    spec_rnn = Reshape((data_out[-2] if is_accdoa else data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
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