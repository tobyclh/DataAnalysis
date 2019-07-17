import numpy as np
import matplotlib
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)


def data_generation(n=1000):
    a = 3. * np.pi * np.random.rand(n)
    x = np.stack(
        [a * np.cos(a), 30. * np.random.random(n), a * np.sin(a)], axis=1)
    return a, x

def knn(xs, k=4):
    """reall slow implmentation of KNN"""
    knn = []
    for i, x in enumerate(xs):
        dis = ((x - xs)**2).sum(1)
        indices = np.argpartition(dis, k+1)[:k+1]
        indices = indices.tolist()
        indices.remove(i) #remove self
        # assert len(indices) == k
        knn.append(indices)
    return knn

def knn2W(knn):
    n_data = len(knn)
    W = np.zeros([n_data, n_data])
    for i, indices in enumerate(knn):
        for idx in indices:
            W[i, idx] = 1
            W[idx, i] = 1
    return W


def LapEig(x, d=2):
    indices = knn(x)
    W = knn2W(indices)
    D = np.diag(W.sum(1))
    L = D - W
    eigval, eigvec = linalg.eig(L, D)
    eigen_pairs = [[np.abs(eigval[i]),eigvec[:, i]] for i in range(len(eigval))]
    eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0], reverse=False)
    w = np.stack([eig[1].real for eig in eigen_pairs], axis=-1)
    z = w[:, 1:d+1]
    return z
    


def visualize(x, z, a):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 1], z[:, 0], c=a, marker='o')
    fig.savefig('lecture10-h2.png')


n = 1000
a, x = data_generation(n)
z = LapEig(x)
visualize(x, z, a)