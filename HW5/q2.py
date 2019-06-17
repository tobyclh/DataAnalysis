import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from argparse import ArgumentParser
np.random.seed(1)

parser = ArgumentParser()
parser.add_argument('--bandwidth', default=10, type=float)
parser.add_argument('--recal', action='store_true')
opt = parser.parse_args()

def kernel(x, c):
    return np.exp(-(np.linalg.norm(x - c) ** 2) / (2 * opt.bandwidth ** 2))

def build_design_mat(x1, x2):
    return pairwise_distances(x1, x2, metric=kernel, n_jobs=-1)

def optimize_param(design_mat, y):
    return np.linalg.solve(
        design_mat.T.dot(design_mat),
        design_mat.T.dot(y))

def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data).T.dot(theta)

def build_confusion_matrix(train_data, test_data, thetas):
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    if Path('predicts.npy').exists() and not opt.recal:
        predicts = np.load('predicts.npy')
    else:
        predicts = []
        for theta in tqdm(thetas):
            prediction = predict(train_data, test_data, theta)
            predicts.append(prediction)
        predicts = np.stack(predicts)
        np.save('predicts.npy', predicts)
    output = np.argmax(predicts, axis=0)
    print(f'Output : {output}')
    for i in range(10):
        prediction = output[i*200:(i+1)*200].astype(np.int)
        for j in range(10):
            confusion_matrix[j, i] = np.sum(
                np.where(prediction == j, 1, 0))
    return confusion_matrix

def load_data(mode='train'):
    datum = []
    labels = []
    for i in range(10):
        data = np.loadtxt(f'data/digit_{mode}{i}.csv', delimiter=',')
        n_data, size_sq = data.shape
        _size = int(size_sq**0.5)
        label = np.ones(n_data) * i
        datum.append(data)
        labels.append(label)
    if mode == 'train':
        method = np.concatenate
    else:
        method = np.concatenate
    datum, labels = method(datum), method(labels)
    return datum, labels

x, y = load_data('train')

if Path('design_mat.npy').exists() and not opt.recal:
    design_mat = np.load('design_mat.npy')
else:
    design_mat = build_design_mat(x, x)
    np.save('design_mat.npy', design_mat)

if Path('thetas.npy').exists() and not opt.recal:
    thetas = np.load('thetas.npy')
else:
    thetas = []
    for i in tqdm(range(10)):
        _y = -np.ones_like(y)
        _y[i*500:(i+1)*500] = 1
        theta = optimize_param(design_mat, _y)
        thetas.append(theta)
    np.save('thetas.npy', thetas)

t_x, t_y = load_data('test')
confusion_matrix = build_confusion_matrix(x, t_x, thetas)
print('confusion matrix:')
print(confusion_matrix)
