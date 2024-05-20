import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.models
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import dataset_loader
import tensorflow as tf
import plotter
from model import create_model
import sys
import os

PATH = os.path.basename("models")
PATH = os.path.join(PATH, sys.argv[1] + '.keras')

LOSS = 0
BATCH_SIZE = 200

SPLITS = {
    'MODEL1': 'Split1',
    'MODEL2': 'Split2',
    'MODEL3': 'Split3',
    'MODEL1_ok': 'Split1',
    'MODEL1_ok_copy': 'Split1',
    'MODEL2_good': 'Split2',
    'MODEL2_good_copy': 'Split2',
    'MODEL3_good': 'Split3',
    'MODEL3_good_copy': 'Split3',
}

# tf.config.list_physical_devices('GPU')
# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 10, 2), random_state=1, max_iter=1000)
print("loading data")
x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set = dataset_loader.load_split(
    SPLITS[sys.argv[1]])
print("data loaded! Lets test!")

# clf.fit(x_train_set, y_train_set)

# print(clf.predict(x_train_set))
# plotter.plot_learning_curve(clf)

if not tf.test.is_gpu_available():
    print("OH NO")

model = tensorflow.keras.models.load_model(PATH)
# plot graph
model.summary()
plotter.plot_confusion(model, x_train_set, y_train_set)
plotter.plot_confusion(model, x_valid_set, y_valid_set)
plotter.plot_confusion(model, x_test_set, y_test_set)

plotter.sensitivity_specifity(model, x_test_set, y_test_set)

