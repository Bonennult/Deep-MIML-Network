#Adapted from https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294
#ImageNet Labels from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import copy

import keras.layers as KL
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys
sys.setrecursionlimit(3000)

bn_axis = 3

class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
                    

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    #1x1
    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = KL.Activation('relu', name=conv_name_base + '2a_relu')(x)

    #3x3 with padding
    x = KL.ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = KL.Activation('relu', name=conv_name_base + '2b_relu')(x)

    #1x1
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    #Size did not change, direcly add input
    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    #1x1 and downsample /2
    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = KL.Activation('relu', name=conv_name_base + '2a_relu')(x)

    #3x3 with padding
    x = KL.ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = KL.Activation('relu', name=conv_name_base + '2b_relu')(x)

    #1x1
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    #size changed, downsample input by the same amount before adding
    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet101_model(weights_path=None):
    '''Instantiate the ResNet101 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5
    img_input = KL.Input(shape=(224, 224, 3), name='data')                       #224x224x3

    #conv1
    x = KL.ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)            #230x230x3
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)   #(230-7+1)/2 --> 112x112x64
    x = KL.BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)      
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = KL.Activation('relu', name='conv1_relu')(x)
    x = KL.MaxPooling2D((3, 3), padding="same", strides=(2, 2), name='pool1')(x)  #(112-3+1)/2 --> 56x56x64

    #conv2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))       #MaxPooling already downsampled => 56x56X256 
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')                   #56x56x256

    #conv3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')                     #stride=2 => 28x28x512
    for i in range(1,4):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))      #28x28x512

    #conv4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')                    #stride=2 => 14x14x1024      
    for i in range(1,23):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))     #14x14x1024

    #conv5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')                    #stride=2 => 7x7x2018
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')                #7x7x2048

    #pool, flatten, softmax 
    x_fc = KL.AveragePooling2D((7, 7), name='avg_pool')(x)                        #1x1x2048
    x_fc = KL.Flatten()(x_fc)                                                     #1x2048 (vector) 
    x_fc = KL.Dense(1000, activation='softmax', name='fc1000')(x_fc)              #1x1000

    model = Model(img_input, x_fc)

    # load weights
    model.load_weights(weights_path, by_name=True)

    return model

def resnet_predict(model, image_file):

    #import matplotlib.pyplot as plt

    #read image in BGR format
    #im = cv2.imread('images/tiger.jpg')
    im = cv2.imread(image_file)

    #im = np.flip(im, 2)  
    '''plt.figure(1)
    plt.imshow(im)
    plt.figure(2)
    plt.imshow(np.flip(im, 2))  #BGR to RGB
    '''

    #resize to 224x224x3
    im = cv2.resize(im, (224, 224)).astype(np.float32)

    # Remove train image mean
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    #print("image size:", im.shape)

    # Insert a new dimension for the batch_size
    im = np.expand_dims(im, axis=0)
    #print("batch_shape", im.shape)

    softmax = model.predict(im)
    prediction = np.argmax(softmax)

    import json
    with open('labels.json','r') as f:
        ImageNet_Labels = json.load(f)
    
    #print("softmax shape:", softmax.shape)

    #print("Prediction ID:", prediction)
    #print("Class:", ImageNet_Labels[str(prediction)])
    #print("Score:", softmax[0,np.argmax(softmax)])
    #print()

    #print(out[0,282], ImageNet_Labels['282'])
    return softmax[0]