import gc
import math

import keras.backend
import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_learning_curve(train, valid, test):
    length = len(train)
    x = range(length)

    plt.plot(x, train, label="train")
    plt.plot(x, valid, label="valid")
    plt.plot(x, test, label="test")

    plt.legend(loc='upper center')

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import tensorflow

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()


    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))

def plot_confusion(model, x_set, y_set):
    BATCH_SIZE = 400
    y_prediction = []

    for i in range(math.ceil(len(x_set) / BATCH_SIZE)):
        y_prediction_temp = model.predict_on_batch(x_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        y_prediction_temp = np.argmax(y_prediction_temp, axis=1)
        y_prediction += list(y_prediction_temp)
        #reset_keras()
    y_test = np.argmax(y_set, axis=1)
    # Create confusion matrix and normalizes it over predicted (columns)
    confusion = confusion_matrix(y_test, y_prediction, normalize='true')
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['horse','penguin','turtle','other']).plot()
    plt.show()


def sensitivity_specifity(model, x_set, y_set):
    y_prediction = model.predict(x_set)
    y_prediction = np.argmax(y_prediction, axis=1)
    y_test = np.argmax(y_set, axis=1)
    confusion = confusion_matrix(y_test, y_prediction)
    for animal_index in range(4):
        true_positive = confusion[animal_index][animal_index]
        false_positive = sum(row[animal_index] if index != animal_index else 0 for index,row in enumerate( confusion))
        true_negative = sum(
            x if row_index != animal_index and index != animal_index else 0
            for row_index, row in enumerate(confusion)
            for index, x in enumerate(row)
        )
        false_negative = sum(x if index != animal_index else 0 for index,x in enumerate( confusion[animal_index]))

        sensitivity = true_positive/(true_positive+false_negative)
        specifity = true_negative/(true_negative+false_positive)

        print(f"Class {animal_index}: sensitivity: {sensitivity:.3f}, specifity: {specifity:.3f}")
