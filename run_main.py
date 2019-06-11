import numpy as np
from data_utils import *
from models.FCN import *
from models.DeepLabV2 import *
from models.UNet import *
import argparse

# get arguments
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest = 'model',default = 'unet', help ='select model(fcn, deeplabv2, unet)(default : unet)')
    parser.add_argument('--epoch', dest = 'epoch', type = int, default = 1, help ='training epoch (default : 1)')
    parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 8, help ='training batch_size (default : 8)')
    parser.add_argument('--learning_rate', dest = 'learning_rate', type = float, default = 0.001, help ='training learning rate (default : 0.001)')

    return parser.parse_args()

# main function
def main(args):
    # data load
    IMG_HEIGHT = 384
    IMG_WIDTH = 256
    num_of_class = 4

    x_train = np.load('dataset/x_train.npy').astype(np.float32)
    x_test = np.load('dataset/x_test.npy').astype(np.float32)
    y_train = np.load('dataset/y_train_onehot.npy').astype(np.float32)
    y_test = np.load('dataset/y_test_onehot.npy').astype(np.float32)

    if args.model == 'unet':
        model = UNet(img_shape = x_train[0].shape, num_of_class = num_of_class, learning_rate = args.learning_rate)
    elif args.model == 'fcn':
        model = FCN8s(img_shape = x_train[0].shape, num_of_class = num_of_class, learning_rate = args.learning_rate)
    elif args.model == 'deeplabv2':
        model = DeepLabV2(img_shape = x_train[0].shape, num_of_class = num_of_class, learning_rate = args.learning_rate)
    # model train
    history = model.train_generator(x_train, y_train,
                                x_test, y_test,
                                args.model,
                                epoch = args.epoch,
                                batch_size = args.batch_size)


    # model history plot - loss and accuracy
    plot_acc(history)
    plot_loss(history)
    plt.show()

if __name__ == '__main__':

    args = arg_parser()
    print("Args : ", args)
    main(args)
