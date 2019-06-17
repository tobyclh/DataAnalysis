import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility

def data_generate(n=50):
    x = np.random.randn(n, 1)
    x[:25, 0] -= 15
    x[25:, 0] -= 5
    x[:2, 0] += 10
    x[:, -1] = 1
    y = np.ones(n)
    x[:25, 0] = -1
    return x, y

x, y = data_generate()
theta = np.random.randn(3)
SIGMA = np.cov(x)
MEAN = np.mean(x, axis=0)
for i in range(10000):
    new_mean = MEAN - MEAN*