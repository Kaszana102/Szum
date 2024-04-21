import random

import dataset_loader
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import plotter
from model import create_model

LOSS = 0

#clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 10, 2), random_state=1, max_iter=1000)
x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set = dataset_loader.split2()

print("data loaded! Lets fit!")
# clf.fit(x_train_set, y_train_set)

# print(clf.predict(x_train_set))
# plotter.plot_learning_curve(clf)

if not tf.test.is_gpu_available():
    print("OH NO")


# preprocess data
def preprocess(array):
    return array.astype("float32") / 255


x_train_set = preprocess(x_train_set)
x_valid_set = preprocess(x_valid_set)
x_test_set = preprocess(x_test_set)

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
while (stale_iterations < MAX_STALE_ITERATIONS and iteration<MAX_ITERATIONS) or iteration < 100 :
    iteration+=1
    stale_iterations+=1

    model.fit(x_train_set, y_train_set, validation_data=(x_valid_set, y_valid_set))
    train_loss += [model.evaluate(x_train_set, y_train_set)[LOSS]]
    valid = model.evaluate(x_valid_set, y_valid_set)
    valid_loss += [valid[LOSS]]
    test_loss += [model.evaluate(x_test_set, y_test_set)[LOSS]]

    if best_valid < valid[LOSS]:
        best_valid = valid[LOSS]
        stale_iterations = 0

plotter.plot_learning_curve(train_loss, valid_loss, test_loss)
print(list(zip(model.predict(x_train_set),y_train_set)))
