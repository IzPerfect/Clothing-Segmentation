import numpy as np
import scipy.io as spio
import os
import cv2
import matplotlib.pyplot as plt
import glob
from keras.utils import to_categorical

import numpy as np
from PIL import Image


def mat_to_img(label_names):
    for i, img_name in enumerate(label_names):
        mask_ex = spio.loadmat(label_path + label_list[i])
        img = Image.fromarray(mask_ex['groundtruth'])
        img.save(save_dir+ str(i+1) + '.png')# png right number

def change_class(x):
    num_of_data = x.shape[0]
    x = x.reshape([num_of_data,-1])
    result = np.zeros(x.shape)
    classes = [0, 41, 19]

    for k in range(num_of_data):
        for i, v in enumerate(x[k]):
            if v not in classes:
                result[k][i] = 3 # clothes
            else:
                if v == 0:
                    result[k][i] = 0 # background
                elif v == 41:
                    result[k][i] = 1 # skin
                else:
                    result[k][i] = 2 # hair

    return result

pwd = os.getcwd() # current path

image_path = pwd + '/photos/'
label_path = pwd + '/annotations/pixel-level/'

valid_exts = [".jpg", ".mat"] # file extensions

image_list = [i for i in sorted(os.listdir(image_path)) if os.path.splitext(i)[1].lower() in valid_exts]
label_list = [i for i in sorted(os.listdir(label_path)) if os.path.splitext(i)[1].lower() in valid_exts]

save_dir = "./label_images/" # path where you want to save

if not os.path.exists(save_dir): # if there is no exist, make the path
    os.makedirs(save_dir)


mat_to_img(label_list)

IMG_H, IMG_W = 384, 256

x_train = np.zeros((len(image_list), IMG_H, IMG_W, 3), dtype=np.uint8)
y_train = np.zeros((len(label_list), IMG_H, IMG_W, 1), dtype=np.uint8)

save_dir = "./images/" # path where you want to save

if not os.path.exists(save_dir): # if there is no exist, make the path
    os.makedirs(save_dir)

for i, image in enumerate(image_list):
    img = cv2.imread(image_path + image, 1)
    img = cv2.resize(img, dsize=(IMG_W, IMG_H))
    cv2.imwrite(save_dir + str(i+1) + '.png', img)
    x_train[i] = img


save_dir = './label_images/resize/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, image in enumerate(label_list):
    img = cv2.imread('./label_images/' + str(i+1) + '.png' , cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(IMG_W, IMG_H), interpolation= cv2.INTER_NEAREST)
    cv2.imwrite(save_dir + str(i+1) + '.png', img)
    y_train[i] = np.expand_dims(img, axis = 2)

save_dir = "./dataset/" # path where you want to save

if not os.path.exists(save_dir): # if there is no exist, make the path
    os.makedirs(save_dir)

np.save(save_dir+'x_train.npy', x_train[:900])
np.save(save_dir+'y_train.npy', y_train[:900])
np.save(save_dir+'x_test.npy', x_train[900:1000])
np.save(save_dir+'y_test.npy', y_train[900:1000])

x_train = np.load('dataset/x_train.npy').astype(np.float32)
x_test = np.load('dataset/x_test.npy').astype(np.float32)
y_train = np.load('dataset/y_train.npy').astype(np.float32)
y_test = np.load('dataset/y_test.npy').astype(np.float32)



y_label_train = change_class(y_train)
y_label_train = y_label_train.reshape([-1, IMG_H, IMG_W,1])
y_label_test = change_class(y_test)
y_label_test = y_label_test.reshape([-1, IMG_H, IMG_W,1])

y_label_train_onehot = to_categorical(y_label_train, num_classes = 4)
y_label_test_onehot = to_categorical( y_label_test, num_classes = 4)

save_dir = "./dataset/" # path where you want to save
np.save(save_dir+'y_train_onehot.npy', y_label_train_onehot)
np.save(save_dir+'y_test_onehot.npy', y_label_test_onehot)
