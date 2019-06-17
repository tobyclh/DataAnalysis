import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

cwd = os.getcwd()
DATAPATH = 'data'
H = 0.5


def load_data(datapath):
    X_train = np.array([]).reshape(-1, 256)
    X_test = np.array([]).reshape(-1, 256)
    y_train = np.array([])
    y_test = np.array([])
    for category in range(10):
        train_df = pd.read_csv(os.path.join(
            datapath, "digit_train{}.csv".format(category)), header=None)
        test_df = pd.read_csv(os.path.join(
            datapath, "digit_test{}.csv".format(category)), header=None)
        X_train = np.concatenate((X_train, train_df.values), axis=0)
        X_test = np.concatenate((X_test, test_df.values), axis=0)
        y_train = np.concatenate(
            (y_train, np.array([category for _ in range(train_df.shape[0])])), axis=0)
        y_test = np.concatenate(
            (y_test, np.array([category for _ in range(test_df.shape[0])])), axis=0)
    return X_train, X_test, y_train, y_test


def kernel(X, X_train):
    """ generate guassian kernel from dataset """
    def gaussian(x: float, y: float, h: float = H) -> float:
        return np.exp(-(np.linalg.norm(x-y))**2 / (2*h**2))
    X_kernel = np.array([[gaussian(x_i, x_j) for x_j in X_train]
                         for x_i in X]).reshape(len(X), len(X_train))
    return X_kernel


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # metrics
    print('accuracy: ', accuracy_score(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


# Load data into DataFrame
X_train, X_test, y_train, y_test = load_data(DATAPATH)

# Apply gaussian kernel
X_test = kernel(X_test, X_train)
X_train = kernel(X_train, X_train)
#print(X_train.shape, X_test.shape)

# train
W = {}
X_TX = np.matmul(X_train.T, X_train)
# print(X_TX.shape)
for category in range(10):
    y_train_cate = np.array([1 if y == category else -1 for y in y_train])
    W[category] = np.linalg.solve(X_TX, np.matmul(X_train.T, y_train_cate))
# print(W[0].shape)

# test
y_pred = np.zeros((X_test.shape[0], 10)).T
for category in range(10):
    y_pred[category] = np.matmul(X_test, W[category])
y_pred = y_pred.T
y_pred = [np.argmax(y) for y in y_pred]

# Evaluate
plot_confusion_matrix(y_test, y_pred)
