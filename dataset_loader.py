from math import sin, cos, radians
import os
import random

import cv2
from numpy import asarray
import numpy as np

import generate_augmented

HORSE = [1, 0, 0, 0]
PENGUIN = [0, 1, 0, 0]
TURTLE = [0, 0, 1, 0]
OTHER = [0, 0, 0, 1]

IMAGE_RESOLUTION = (256,256)
def load_dataset(directory, classs, augmented=False, samples_per_class=1000) -> ([[[], []]],[[[], []]]):
    dataset = []
    augmented = []
    samples = 0

    if augmented:
        first_iter = True
        while True:
            for filename in os.listdir(directory):
                if first_iter:
                    # normalize
                    img = generate_augmented.augment(directory + '/' + filename, False, 0, IMAGE_RESOLUTION)
                else:
                    # augment
                    img = generate_augmented.augment(directory + '/' + filename,False,45,IMAGE_RESOLUTION)

                numpydata = asarray(img)
                # MODIFY DATA
                if first_iter:
                    dataset += [[numpydata, classs]]
                else:
                    augmented += [[numpydata, classs]]
                samples += 1
                if samples == samples_per_class:
                    return dataset,augmented
            first_iter = False
    else:
        for filename in os.listdir(directory):
            # img = cv2.imread(directory + '/' + filename)
            # previously was function above, but numpy.array was angry about it (cause not coherent size)
            img = generate_augmented.augment(directory + '/' + filename, False, 0, IMAGE_RESOLUTION)
            numpydata = asarray(img)

            dataset += [[numpydata, classs]]
            samples += 1
            if samples == samples_per_class:
                return dataset, []
        return dataset,[]


def split_dataset_into_sets(dataset,augmented):

    original_amount = len(dataset)
    augmented_amount = len(augmented)
    random.shuffle(dataset)

    valid_set_size = int((augmented_amount + original_amount)/10)

    # create sets
    valid_set = dataset[0:valid_set_size]
    test_set = dataset[valid_set_size:2*valid_set_size]
    train_set = augmented + dataset[2*valid_set_size:]
    '''
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    valid_set = dataset[0::10]
    test_set = dataset[1::10]
    '''
    return train_set, valid_set, test_set


def concat_sets(train_set, train_temp, valid_set, valid_temp, test_set, test_temp):
    train_set += train_temp
    valid_set += valid_temp
    test_set += test_temp


def load_all_dataset(directory, augmented=False, samples_per_class=1000):

    original, augmented = load_dataset(directory + '/horses', HORSE, augmented, samples_per_class)
    train_set, valid_set, test_set = split_dataset_into_sets(original,augmented)

    original, augmented = load_dataset(directory + '/penguins', PENGUIN, augmented, samples_per_class)
    train_temp, valid_temp, test_temp = split_dataset_into_sets(original,augmented)
    concat_sets(train_set, train_temp, valid_set, valid_temp, test_set, test_temp)

    original, augmented = load_dataset(directory + '/turtles', TURTLE, augmented, samples_per_class)
    train_temp, valid_temp, test_temp =split_dataset_into_sets(original,augmented)
    concat_sets(train_set, train_temp, valid_set, valid_temp, test_set, test_temp)

    original, augmented = load_dataset(directory + '/other', OTHER, augmented, samples_per_class)
    train_temp, valid_temp, test_temp = split_dataset_into_sets(original,augmented)
    concat_sets(train_set, train_temp, valid_set, valid_temp, test_set, test_temp)

    return train_set, valid_set, test_set


def split1():
    # create sets
    train_set, valid_set, test_set = load_all_dataset('dataset_src')

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0].ravel() for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0].ravel() for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0].ravel() for val in test_set]
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
    # create sets
    train_set, valid_set, test_set = load_all_dataset('dataset_src', augmented=True, samples_per_class=10)

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0].ravel() for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0].ravel() for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0].ravel() for val in test_set]
    y_test_set = [val[1] for val in test_set]

    # but why?
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set


def split3():
    # create sets
    train_set, valid_set, test_set = load_all_dataset('dataset_src', augmented=True, samples_per_class=10)

    train_set += valid_set

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0].ravel() for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0].ravel() for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0].ravel() for val in test_set]
    y_test_set = [val[1] for val in test_set]

    # but why?
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set
