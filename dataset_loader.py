import os

import cv2
from numpy import asarray
import numpy as np
import random

HORSE = [1, 0, 0, 0]
PENGUIN = [0, 1, 0, 0]
TURTLE = [0, 0, 1, 0]
OTHER = [0, 0, 0, 1]


def load_horses():
    dataset = []

    dir = 'dataset/horses'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, HORSE]
    return dataset


def load_penguins():
    dataset = []

    dir = 'dataset/penguins'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, PENGUIN]
    return dataset


def load_turles():
    dataset = []

    dir = 'dataset/turtles'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, TURTLE]
    return dataset


def load_others():
    dataset = []

    dir = 'dataset/other'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, OTHER]
    return dataset


def load_all_dataset():
    '''

    :return: str
    '''

    # return load_horses() + load_penguins() + load_turles() + load_others()
    return load_penguins()


def split1():
    dataset = load_all_dataset()

    # create sets
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    valid_set = dataset[0::10]
    test_set = dataset[1::10]

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0] for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0] for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0] for val in test_set]
    y_test_set = [val[1] for val in test_set]

    # but why?
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set

def split2():
    dataset = load_all_dataset()

    # create sets
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    valid_set = dataset[0::10]
    test_set = dataset[1::10]

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0] for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0] for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0] for val in test_set]
    y_test_set = [val[1] for val in test_set]

    # but why?
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set