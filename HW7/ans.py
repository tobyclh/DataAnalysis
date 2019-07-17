import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)


def generate_data(sample_size=90, n_class=3):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y


def optimize(x, y, n_class, h, l):
    k = np.exp(-(x - x[:, None]) ** 2 / (2 * h ** 2))
    theta = np.zeros((len(x), n_class))
    for i in range(n_class):
        ki = k[:, y == i]
        theta[y == i, i] = np.linalg.solve(
            ki.T.dot(ki) + l * np.identity(int(np.sum(np.where(y == i, 1, 0)))),
            ki.T.dot(np.where(y == i, 1., 0.)))
    return theta


def predict(X, x, theta, h):
    K = np.exp(-(X - x[:, None]) ** 2 / (2 * h ** 2))
    prediction = K.T.dot(theta)
    unnormalized_prob = np.maximum(prediction, 0.)
    return unnormalized_prob / unnormalized_prob.sum(axis=1, keepdims=True)


def visualize(x, y, theta, h):
    X = np.linspace(-5., 5., 100)
    prob = predict(X, x, theta, h)

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)

    plt.plot(X, prob[:, 0], c='blue')
    plt.plot(X, prob[:, 1], c='red')
    plt.plot(X, prob[:, 2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')

    plt.savefig('lecture7-h1.png')


x, y = generate_data(sample_size=90, n_class=3)
theta = optimize(x, y, n_class=3, h=1., l=.1)
visualize(x, y, theta, h=1.)