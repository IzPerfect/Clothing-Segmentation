import numpy as np
import matplotlib.pyplot as plt
import keras
import math
import os

# plot history for accuracy
def plot_acc(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)

# plot history for loss
def plot_loss(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)

# show results using each model
def show_result(model, x, y, title = 'Result', save_file = False, save_file_name = 'show_result', IMG_HEIGHT = 384, IMG_WIDTH = 256):
    if len(x) != len(y):
        raise Exception('Check the number of images and labels')
    else:
        pass

    x = x.reshape([1, IMG_HEIGHT, IMG_WIDTH, 3])
    pred = model.predict(x)
    pred = np.argmax(pred, axis = 3)

    y = y.reshape([1, IMG_HEIGHT, IMG_WIDTH, 4])
    gt = np.argmax(y, axis = 3)

    fig = plt.figure(figsize = (12, 5))
    plt.suptitle(title, fontsize=25)
    plt.subplot(1,4,1)
    plt.imshow(np.squeeze(x))
    plt.title('Image')
    plt.subplot(1,4,2)
    plt.imshow(np.squeeze(gt))
    plt.title('Ground truth')
    plt.subplot(1,4,3)
    plt.imshow(np.squeeze(pred))
    plt.title('Prediction')
    plt.subplot(1,4,4)
    plt.imshow(np.squeeze(x))
    masked_imclass = np.ma.masked_where(np.squeeze(pred) == 0, np.squeeze(pred))
    plt.imshow( masked_imclass, alpha = 0.4)
    plt.title('Overlay')
    plt.show()

    # save results
    if save_file == True:
        save_dir = './result/show_result/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)
        fig.savefig(save_dir + save_file_name)
    return

# save image segmented by skin, hair, clothes
def iamge_segmentation(model, x, title = 'Each Segmentation',save_file = False, save_file_name = 'iamge_segmentation', IMG_HEIGHT = 384, IMG_WIDTH = 256):
    x = x.reshape([1, IMG_HEIGHT, IMG_WIDTH, 3])

    pred = model.predict(x)
    pred = np.argmax(pred, axis = 3)
    empty = np.zeros((pred.shape))

    fig = plt.figure(figsize = (12, 5))
    plt.subplot(1,4,1)
    plt.imshow(np.squeeze(x))
    plt.title('Image')
    plt.subplot(1,4,2)
    plt.imshow(np.squeeze(x))
    masked_imclass = np.ma.masked_where(np.squeeze(pred) == 1, np.squeeze(empty))
    plt.imshow( masked_imclass, alpha = 1)
    plt.title('Skin')
    plt.subplot(1,4,3)
    plt.imshow(np.squeeze(x))
    masked_imclass = np.ma.masked_where(np.squeeze(pred) == 2, np.squeeze(empty))
    plt.imshow( masked_imclass, alpha = 1)
    plt.title('Hair')
    plt.subplot(1,4,4)
    plt.imshow(np.squeeze(x))
    masked_imclass = np.ma.masked_where(np.squeeze(pred) == 3, np.squeeze(empty))
    plt.imshow( masked_imclass, alpha = 1)
    plt.title('Clothes')
    plt.show()
    if save_file == True:
        save_dir = './result/iamge_segmentation/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)
        fig.savefig(save_dir + save_file_name)

    return
