#An implementation of DenseNet from the paper "Densely Connected Convolutional Networks"
#code inspiration ref: https://github.com/bhaskar-gaur/DenseNet-Keras/blob/master/
import keras
from keras.layers import Conv2D, BatchNormalization, Concatenate, Activation, AveragePooling2D, Dense, Flatten, Permute, MaxPooling2D
from keras.regularizers import l2
import keras.backend as K
from keras.initializers import random_normal
import numpy as np
#keras.backend.set_image_data_format('channels_last')#the default for this implementation

global bn_axis

bn_axis = 1

def composite_fun(inputs, growth_rate=12):
    ##bottleneck (1x1 conv)
    output = BatchNormalization()(inputs)
    output = Activation('relu')(output)
    output = Conv2D(4 * growth_rate, 1, use_bias=False)(output)
    ##3x3 conv
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Conv2D(growth_rate, 3, padding='same')(output)

    return output

def dense_block(inputs, growth_rate, layers):
    temp = inputs
    for _ in range(layers):
        output = composite_fun(inputs, growth_rate)
        concat = Concatenate(axis=bn_axis)([temp,output])#axis=-1 originally for channels last
        temp = concat
    return temp

#block after dense blocks, which downsamples the dense block output
#its needed since the concatenation increases the dimensions
def transition_block(layer, inputs, nb_filters, compression = 0.5):
    print(compression, nb_filters)
    print(compression*nb_filters)
    output = BatchNormalization()(inputs)#bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    output = Activation('relu')(output)
    #1x1 conv for downsampling
    print(np.sqrt(2.0/(9*int(nb_filters*compression))))
    output = Conv2D(int(keras.backend.int_shape(output)[bn_axis]*compression), 1, use_bias=False ,padding='same',kernel_regularizer=l2(0.001), kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(nb_filters*compression))))))(output)
    
    #CUSTOM downsample only 3 times, if there are 4 and above layers dont
    if layer < 3:
        output = AveragePooling2D(pool_size=(1,2), padding='same')(output)#original was (2,2,) with stride (2,2)
    
    return output

def DenseNet(inputs, growth_rate, layers, nb_filters, nb_classes = 12, depth = 3):
    output = inputs
    #permute if it is channels_last
    #output = Permute((3,2,1))(inputs)

    print(inputs)
    #output = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(output)
    output = Conv2D(nb_filters, (7,7), use_bias=False , padding='same')(output)
    output = BatchNormalization()(output)
    output = Activation('relu', name='conv1/relu')(output)

    soutput = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(output)
    #output = MaxPooling2D(2, strides=2, name='pool1')(output)#original had 3 and strides = 2
    
    output = MaxPooling2D(pool_size=(5,2))(output)
    
    #3 dense blocks as in the paper
    for i in range(depth):
        concat_out = dense_block(inputs=output, growth_rate=growth_rate, layers=layers)
        output = transition_block(i, concat_out, nb_filters)

    output = dense_block(inputs=output, growth_rate=growth_rate, layers=layers)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = AveragePooling2D(pool_size=(1,2), padding='same')(output)
    #output = Flatten()(output)
    #output = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(0.001))(output)

    #make it channels_first again
    #output = Permute((2,3,1))(output)

    return output