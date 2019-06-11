import keras
from keras import models
from keras.models import Model
from keras.layers import Activation, Conv2D, \
    MaxPooling2D, Input, Concatenate, Conv2DTranspose, add, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data_utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
import time

class DeepLabV2(object):
    def __init__(self, img_shape, num_of_class, actf = 'relu',
        learning_rate = 0.001):

        '''
        Arguments :
        img_shape - shape of input image
        actf - activation function for network training
        learning_rate - learning rate for training
        '''
        self.num_of_class = num_of_class
        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape

        self.model = self.build_model()

    # ASPP(Atrous Spatial Pyramid Pooling)
    def aspp(self, x, atrous_rates = [6, 12, 18, 24], feature_maps = 1024, filter_size = (3, 3)):
        layers = []
        for i, rate in enumerate(atrous_rates):
            layer = Conv2D(feature_maps, filter_size, dilation_rate = (rate, rate), activation = self.actf
                , padding = 'same', kernel_initializer = 'he_normal', name = 'ASPP' + str(i))(x)
            layer =  Conv2D(feature_maps, (1,1), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'ASPP_conv' + str(i)+ '1')(layer)
            layer =  Conv2D(self.num_of_class, (1,1), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'ASPP_conv' + str(i) + '2')(layer)
            layers.append(layer)
        return add(layers)

    # convolution block
    def conv_block(self, x, layers, feature_maps, name = None, filter_size = (3, 3), max_pool = True,
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        for i in range(layers):
            x = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                            padding = 'same', kernel_initializer = 'he_normal', name = name + '_conv' +str(i+1))(x)
        if max_pool:
            x = MaxPooling2D(pooling_filter_size, strides = pooling_strides, name = name + '_pool')(x)

        return x

    # astrous convolution block
    def atrous_conv_block(self, x, layers, feature_maps, name = None, filter_size = (3, 3), atrous_rate = (2, 2),
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        for i in range(layers):
            x = Conv2D(feature_maps , filter_size , activation = self.actf, dilation_rate = atrous_rate,
                            padding = 'same', kernel_initializer = 'he_normal', name = name + 'atrous_conv' +str(i+1))(x)

        return x

    # build network based on VGG16
    # 1x1 convolution for increasing accuracy
    def build_model(self):
        inputs = Input(self.img_shape)

        # conv block1, output stride = 2
        block_1 = self.conv_block(inputs, 2, 64, 'block1')
        # conv block2, output stride = 4
        block_2 = self.conv_block(block_1, 2, 128, 'block2')
        # conv block3, output stride = 8
        block_3 = self.conv_block(block_2, 3, 256, 'block3')
        # conv block4, output stride = 8
        block_4 = self.conv_block(block_3, 3, 512, max_pool = False, name = 'block4')
        # conv block4, output stride = 8
        block_5 = self.atrous_conv_block(block_4, 3, 512, name = 'block5')

        # Atrous Spatial Pyramid Pooling, 8umsampling
        aspp_output = self.aspp(block_5)
        aspp_output = Conv2DTranspose(self.num_of_class, (16,16), strides = (8, 8), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'aspp_upsampling')(aspp_output)
        aspp_output = Conv2D(self.num_of_class, (1,1), padding = 'same')(aspp_output)

        aspp_output = Activation('softmax')(aspp_output)

        model = Model(inputs = inputs, outputs = aspp_output)
        model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = self.learning_rate), metrics=['accuracy'])
        return model

    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, shuffle = True):

        self.history = self.model.fit(X_train, Y_train, validation_split = val_split,
                                          epochs = epoch, batch_size = batch_size, shuffle =  shuffle)
        return self.history

    # train with data augmentation
    def train_generator(self, x_train, y_train, x_test, y_test, name_model, epoch = 10, batch_size = 32, val_split = 0.2, min_lr = 1e-06):

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=[0.7, 1.3]
        )

        val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_gen = train_datagen.flow(
            x_train,
            y_train,
            batch_size = batch_size,
            shuffle=True
        )

        val_gen = val_datagen.flow(
            x_test,
            y_test,
            batch_size = batch_size,
            shuffle=False
        )

        save_dir = './save_model/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)

        cb_checkpoint = ModelCheckpoint(save_dir + name_model + '.h5', monitor = 'val_acc', save_best_only = True, verbose = 1)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 5, verbose = 1, min_lr = min_lr)

        self.history = self.model.fit_generator(train_gen,
                                                validation_data=val_gen,
                                                epochs=epoch,
                                                callbacks=[cb_checkpoint, reduce_lr])
        return self.history
    # predict test data
    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)

        return pred_classes

    # show architecture
    def show_model(self):
        return print(self.model.summary())

    def saved_model_use(self, save_dir = None):
        if save_dir == None:
            return print('No path')

        self.model.load_weights(save_dir)

        return print("Loaded model from '{}'".format(save_dir))
