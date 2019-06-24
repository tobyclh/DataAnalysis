import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
np.random.seed(0)  # set the random seed for reproducibility
GAMMA = 10
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
a = np.mean(np.clip(1-MEAN.dot(x.T)*y, 0, None) * y) # C
b = np.zeros(50)
c = np.zeros([50, 3])
d = np.zeros([50, 3, 3])
for i in range(50):
    b[i] = x[i].dot(SIGMA).dot(x[i])
    c[i] = SIGMA.dot(x[i])
    d[i] = SIGMA.dot(x[i].dot(x[i].T)).dot(SIGMA)
b = b.mean() + GAMMA
c = c.mean(0)
u_head = MEAN + a/b*c
sigma_head = SIGMA - d.mean() / b
print(u_head)
print(sigma_head)