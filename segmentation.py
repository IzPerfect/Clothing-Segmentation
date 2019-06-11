import numpy as np
from data_utils import *
import cv2
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model

# get arguments
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', dest = 'model_path',default = './save_model/UNet_model.h5', help ='load model path')
    parser.add_argument('--image_path', dest = 'image_path', help ='image path')

    return parser.parse_args()

# main function
def main(args):
    # data load
    IMG_HEIGHT = 384
    IMG_WIDTH = 256

    img = cv2.imread(args.image_path, 1)
    img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img)

    # load model
    model = load_model(args.model_path)
    test_img = model.predict(img.reshape([1, 384, 256, 3])/255.)
    test_img = np.array(test_img)
    argm = np.argmax(test_img, axis=3)
    fig = plt.figure()
    plt.imshow(np.squeeze(argm))

    # save and show the result
    save_dir = './MyFile/'
    if not os.path.exists(save_dir): # if there is no exist, make the path
        os.makedirs(save_dir)
    fig.savefig(save_dir + 'segmented.png')


if __name__ == '__main__':

    args = arg_parser()
    print("Args : ", args)
    main(args)
