import random

import dataset_loader
from sklearn.neural_network import MLPClassifier
import plotter

clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 10, 2), random_state=1, max_iter=1000)
x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set = dataset_loader.split2()

print("data loaded! Lets fit!")
clf.fit(x_train_set, y_train_set)

print(clf.predict(x_train_set))
plotter.plot_learning_curve(clf)
