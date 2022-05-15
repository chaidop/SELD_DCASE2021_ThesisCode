#An implementation of DenseNet from the paper "Densely Connected Convolutional Networks"
#code inspiration ref: https://github.com/bhaskar-gaur/DenseNet-Keras/blob/master/
import keras
from keras.layers import Conv2D, BatchNormalization, Concatenate, Activation, AveragePooling2D, Dense, Flatten, Permute, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K
from keras.initializers import random_normal
import numpy as np
keras.backend.set_image_data_format('channels_last')#the default for this implementation

global bn_axis

bn_axis = 3

def composite_fun(inputs, growth_rate=12):
    ##bottleneck (1x1 conv)
    output = BatchNormalization(axis=-1)(inputs)
    output = Activation('relu')(output)
    output = Conv2D(4 * growth_rate, 1, use_bias=False)(output)
    ##3x3 conv
    output = BatchNormalization(axis=-1)(output)
    output = Activation('relu')(output)
    output = Conv2D(growth_rate, 3, padding='same')(output)

    return output

def dense_block(inputs, growth_rate, layers):
    temp = input
    for _ in range(layers):
        output = composite_fun(inputs, growth_rate)
        concat = Concatenate(axis=-1)([temp,output])
        temp = concat
    return temp

#block after dense blocks, which downsamples the dense block output
#its needed since the concatenation increases the dimensions
def transition_block(inputs, nb_filters, compression = 0.5):
    output = BatchNormalization(axis=-1)(inputs)#bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    output = Activation('relu')(output)
    #1x1 conv for downsampling
    output = Conv2D(int(keras.backend.int_shape(output)[bn_axis]*compression), 1, use_bias=False ,padding='same',kernel_regularizer=l2(0.001), kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(nb_filters*compression))))))(output)
    output = AveragePooling2D(pool_size=(2,2), strides=(2,2))(output)
    
    return output

def DenseNet(inputs, growth_rate, layers, nb_filters, nb_classes = 12, depth = 3):
    #permute to make channels_last
    output = Permute((3,1,2))(output)

    input = keras.layers.Input(shape=output.shape)

    output = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(input)
    output = Conv2D(nb_filters, (7,7), use_bias=False , padding='same')(output)
    output = BatchNormalization(axis=-1)(output)
    output = Activation('relu', name='conv1/relu')(output)

    output = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(output)
    output = MaxPooling2D(3, strides=2, name='pool1')(output)
    
    #output = MaxPooling2D(pool_size=(5,4))(output)
    
    #3 dense blocks as in the paper
    for _ in range(depth):
        concat_out = dense_block(inputs=output, growth_rate=growth_rate, layers=layers)
        output = transition_block(concat_out, nb_filters)

    output = dense_block(inputs=output, growth_rate=growth_rate, layers=layers)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = AveragePooling2D(pool_size=(2,2))(output)
    #output = Flatten()(output)
    output = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.001))(output)

    #make it channels_first again
    output = Permute((2,3,1))(output)

    return output