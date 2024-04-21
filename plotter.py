import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
def plot_learning_curve(network : MLPClassifier):
    plt.plot(range(network.n_iter_) , network.loss_curve_)
    plt.show()