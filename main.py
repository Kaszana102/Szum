import gc

import tensorflow.keras.models
import tensorflow.keras as keras

import dataset_loader
import tensorflow as tf
import plotter
from model import create_model
import sys
import os

PATH = os.path.basename("models")
PATH = os.path.join(PATH, sys.argv[1]+'.keras')

LOSS = 0
#tf.config.list_physical_devices('GPU')
#clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 10, 2), random_state=1, max_iter=1000)
print("loading data")
x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set = dataset_loader.load_split('Split2')
print("data loaded! Lets fit!")

# clf.fit(x_train_set, y_train_set)

# print(clf.predict(x_train_set))
# plotter.plot_learning_curve(clf)

if not tf.test.is_gpu_available():
    print("OH NO")

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


model = create_model()

# traning loop
MAX_ITERATIONS = 1000
MAX_STALE_ITERATIONS = 10

stale_iterations = 0
iteration = 0
best_valid = 1000

train_loss = []
valid_loss = []
test_loss = []
with tf.device('/device:GPU:0'):
    while (stale_iterations < MAX_STALE_ITERATIONS and iteration < MAX_ITERATIONS) or iteration < 100:
        iteration += 1
        stale_iterations += 1

        model.fit(x_train_set, y_train_set, validation_data=(x_valid_set, y_valid_set),batch_size=2)#, callbacks=MyCustomCallback())
        train_loss += [model.evaluate(x_train_set, y_train_set)[LOSS]]
        valid = model.evaluate(x_valid_set, y_valid_set)
        valid_loss += [valid[LOSS]]
        test_loss += [model.evaluate(x_test_set, y_test_set)[LOSS]]

        if best_valid > valid[LOSS]:
            best_valid = valid[LOSS]
            stale_iterations = 0
            model.save(PATH)
model = tensorflow.keras.models.load_model(PATH)
plotter.plot_learning_curve(train_loss, valid_loss, test_loss)
#print(list(zip(model.predict(x_test_set), y_test_set)))
