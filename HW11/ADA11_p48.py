import numpy as np
import matplotlib
from scipy.linalg import eig

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(46)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
        # x[sample_size // 2:int(3*sample_size // 4), 0] -= 4
        # x[sample_size // 2:int(3*sample_size // 4), 0] -= 4

    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y):
    """Fisher Discriminant Analysis.
    Implement this function

    Returns
    -------
    T : (1, 2) ndarray
        The embedding matrix.
    """
    x -= x.mean(0)
    c = y.max()
    S_b = np.zeros([2, 2])
    S_w = np.zeros([2, 2])
    mu_ys = []
    nys = []
    for i in range(1, c+1):
        n_y = (y==i).sum()
        nys.append(n_y)
        mu_y = x[y==i].mean(0)
        mu_ys.append(mu_y)
        a = x[y==i]-mu_y
        S_w += a.T.dot(a)
    mu_ys = np.array(mu_ys)
    S_b = nys*mu_ys.dot(mu_ys.T)
    a = np.linalg.solve(S_w, S_b)
    eigval, eigvec = np.linalg.eig(a)
    eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
    eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
    w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real))
    # print(w.T)
    return w[0:1]
    # print(v)
    # return v


def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 9,
             np.array([-T[:, 1], T[:, 1]]) * 9, 'k-')
    plt.legend()
    plt.savefig('lecture11-h1.png')


sample_size = 200
# x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
T = fda(x, y)
visualize(x, y, T)
