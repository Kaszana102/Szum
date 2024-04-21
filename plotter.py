import matplotlib.pyplot as plt


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
