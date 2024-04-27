import os
import random

from numpy import asarray
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import generate_augmented

HORSE = [1, 0, 0, 0]
PENGUIN = [0, 1, 0, 0]
TURTLE = [0, 0, 1, 0]
OTHER = [0, 0, 0, 1]

IMAGE_RESOLUTION = (256, 256)


def load_class_set(directory: str, _class):
    dataset = []
    for file in os.listdir(directory):
        image = img_to_array(load_img(os.path.join(directory, file)))
        dataset += [[image, _class]]
    return dataset

def load_set(directory: str):
    dataset = []
    dataset += load_class_set(directory + '/horses', HORSE)
    dataset += load_class_set(directory + '/penguins', PENGUIN)
    dataset += load_class_set(directory + '/turtles', TURTLE)
    dataset += load_class_set(directory + '/other', OTHER)
    return dataset


def load_all_dataset(directory, augmented=False, samples_per_class=1000):
    train_set = load_set(directory + '/train')
    valid_set = load_set(directory + '/valid')
    test_set = load_set(directory + '/test')

    return train_set, valid_set, test_set


def load_split(split: str, normalize = True):
    '''
     arg: 'Split1', 'Split2', 'Split3'
    :param normalize:
    :param split: Split1, Split2, Split3
    :return:
    '''
    # create sets
    train_set, valid_set, test_set = load_all_dataset('Splits/' + split)

    x_train_set = [val[0].ravel() for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0].ravel() for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0].ravel() for val in test_set]
    y_test_set = [val[1] for val in test_set]

    #convert to numpy!
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    if normalize:
        # preprocess data
        def preprocess(array):
            return array.astype("float32") / 255

        x_train_set = preprocess(x_train_set)
        x_valid_set = preprocess(x_valid_set)
        x_test_set = preprocess(x_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set
