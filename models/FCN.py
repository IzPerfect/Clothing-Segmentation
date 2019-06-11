import keras
from keras import models
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, \
    MaxPooling2D, UpSampling2D,Input, Concatenate, Conv2DTranspose, add
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data_utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
import time


class FCN8s(object):
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

    # build network based on VGG16 which is pre-trained on ImageNet dataset
    # 1x1 convolution for increasing accuracy
    # upsample(Conv2DTranspose : find best upsampling)
    # FCN-8s-skip-layers : ((pool5*2upsample + pool4)*2upsample + pool3)*8upsample
    def build_model(self):
        vgg16_original = VGG16(weights = 'imagenet', include_top = False, input_shape = self.img_shape)
        inputs = vgg16_original.input

        layer_dict = dict([(layer.name, layer) for layer in vgg16_original.layers])
        for vgg_layer in vgg16_original.layers[:]:
            vgg_layer.trainable = False

        block_3 = layer_dict['block3_pool'].output
        pool3 = Conv2D(self.num_of_class, (1,1), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'skip_pool3')(block_3)
        # conv block4, output stride = 16
        block_4 = layer_dict['block4_pool'].output
        pool4 = Conv2D(self.num_of_class, (1,1), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'skip_pool4')(block_4)
        # conv block4, output stride = 32
        block_5 = layer_dict['block5_pool'].output

        # fully connected  => fully convolutional
        fc1 = Conv2D(4096, (7,7), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'conv6')(block_5)
        fc2 = Conv2D(4096, (1,1), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal',name = 'conv7')(fc1)

        fc2 = Conv2D(self.num_of_class, (1,1), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal',name = 'conv8')(fc2)
        pool5 = Conv2DTranspose(self.num_of_class, (4,4), strides = (2, 2) ,activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'skip_pool5')(fc2)

        # (pool5*2upsample + pool4) * 2upsample
        skip_layer_sum1 = Conv2DTranspose(self.num_of_class, (4,4), strides = (2, 2) , activation = self.actf, padding = 'same', kernel_initializer = 'he_normal', name = 'skip_sum1')(add([pool4, pool5]))

        # ((skip_layer_sum1 + pool3)*8upsample
        skip_layer_sum2 = Conv2DTranspose(self.num_of_class, (16,16), strides = (8, 8), padding = 'same', name = 'skip_sum2')(add([pool3, skip_layer_sum1]))
        skip_layer_sum2 = Conv2D(self.num_of_class, (1,1), padding = 'same')(skip_layer_sum2)
        skip_layer_sum2 = Activation('softmax')(skip_layer_sum2)

        model = Model(inputs = inputs, outputs = skip_layer_sum2)
        model.compile(loss = 'categorical_crossentropy', optimizer= Adam(lr = self.learning_rate), metrics=['accuracy'])
        return model

    # train model
    def train(self, x_train, y_train, epoch = 10, batch_size = 32, val_split = 0.2, shuffle = True):

        self.history = self.model.fit(x_train, y_train, validation_split = val_split,
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
