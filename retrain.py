import gc
import random
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
BATCH_SIZE = 400

# tf.config.list_physical_devices('GPU')
# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 10, 2), random_state=1, max_iter=1000)
print("loading data")
x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set = dataset_loader.load_split('Split2')
print("data loaded! Lets fit!")

# clf.fit(x_train_set, y_train_set)

# print(clf.predict(x_train_set))
# plotter.plot_learning_curve(clf)

if not tf.test.is_gpu_available():
    print("OH NO")

model = tensorflow.keras.models.load_model(PATH)

# training loop
MAX_ITERATIONS = 1000
MAX_STALE_ITERATIONS = 20

stale_iterations = 0
iteration = 0
best_valid = 0

# calc loss before training
for i in range(int(len(x_train_set) / BATCH_SIZE)):
    best_valid += model.test_on_batch(x_train_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                                      y=y_train_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])[LOSS]

start = time.time()
train_loss = []
valid_loss = []
test_loss = []

with tf.device('/device:GPU:0'):
    while stale_iterations < MAX_STALE_ITERATIONS:
        # shuffle
        indices = np.arange(x_train_set.shape[0])
        np.random.shuffle(indices)
        x_train_set = x_train_set[indices]
        y_train_set = y_train_set[indices]

        iteration += 1
        stale_iterations += 1
        for i in range(int(len(x_train_set) / BATCH_SIZE)):
            # model.fit(x_train_set, y_train_set,validation_data=(x_valid_set,y_valid_set),batch_size=50,epochs=10)#, callbacks=MyCustomCallback())
            model.train_on_batch(x=x_train_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                                 y=y_train_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])

        # calc loss for training
        loss = 0
        for i in range(int(len(x_train_set) / BATCH_SIZE)):
            loss += model.test_on_batch(x_train_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                                        y=y_train_set[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])[LOSS]
        train_loss += [loss / len(x_train_set)]

        valid = model.test_on_batch(x_valid_set, y=y_valid_set)[LOSS] / len(x_valid_set)
        valid_loss += [valid]
        test_loss += [0]  #[model.test_on_batch(x_test_set, y=y_test_set)[LOSS]/len(x_test_set)]

        if best_valid > valid:
            best_valid = valid
            stale_iterations = 0
            model.save(PATH)

        print("iteration:", iteration)

end = time.time()
print("learning time:", end - start)

model = tensorflow.keras.models.load_model(PATH)
# plot graph
plotter.plot_learning_curve(train_loss, valid_loss, test_loss)
# plot confusion matrix

# Predict
plotter.plot_confusion(model, x_valid_set, y_valid_set)

print(list(zip(model.predict(x_valid_set), y_valid_set)))
