import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
np.random.seed(0)  # set the random seed for reproducibility
GAMMA = 1
def data_generate(n=50):
    x = np.random.randn(n, 3)
    x[:25, 0] -= 15
    x[25:, 0] -= 5
    x[:2, 0] += 10
    x[:, -1] = 1
    y = np.ones(n)
    y[:25] = -1
    return x, y

x, y = data_generate()
SIGMA = np.eye(3)
MEAN = np.mean(x, axis=0)
for i in range(10000):
    sample = random.randint(0, 49)
    _x, _y = x[sample], y[sample]
    Beta = _x.T.dot(SIGMA).dot(_x) + GAMMA
    new_mean = MEAN - (MEAN.T.dot(_x.T) - _y) * SIGMA.dot(_x)/Beta
    new_sigma = SIGMA - SIGMA.dot(_x.dot(_x.T)).dot(SIGMA)/Beta
    if sum(abs(MEAN - new_mean)) < 1e-5:
        break
    MEAN = new_mean
    SIGMA = new_sigma
    print(MEAN, SIGMA)