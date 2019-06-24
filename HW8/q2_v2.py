import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
np.random.seed(0)  # set the random seed for reproducibility
GAMMA = 5
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
SIGMA = np.cov(x.T)
MEAN = np.mean(x, axis=0)
for i in range(10000):
    beta = 0
    mean = 0
    sigma = 0
    for j in range(50):
        beta += x[j].T.dot(SIGMA).dot(x[j])
    beta /= 50
    for j in range(50):
        mean += MEAN - (MEAN.dot(x[j]) - y[j]) * SIGMA.dot(x[j])/beta
        sigma += SIGMA - SIGMA.dot(x[j].dot(x[j].T)).dot(SIGMA)/beta
    mean /= 50
    sigma /= 50
        
    if sum(abs(MEAN - mean)) < 1e-5:
        break
    MEAN = mean
    SIGMA = sigma
    print(MEAN, SIGMA)