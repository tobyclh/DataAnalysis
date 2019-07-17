import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = -1
    y[n_positive:] = 1
    return x, y


def A(x_as, x_bs):
    n_a, n_b = x_as.shape[0], x_bs.shape[0]
    a = 0
    for x_a in x_as:
        for x_b in x_bs:
            a += np.linalg.norm(x_a - x_b, ord=2)
    return a/(n_a*n_b)

def b(x_train_c, x_test):
    n_c, n_t = x_train_c.shape[0], x_test.shape[0]
    b = 0
    for x_a in x_train_c:
        for x_b in x_test:
            b += np.linalg.norm(x_a - x_b, ord=2)
    return b/(n_c*n_t)

def get_pie(train_x, train_y, test_x):
    xp = train_x[train_y==1]
    xn = train_x[train_y==-1]
    Apn = A(xp, xn)
    Ann = A(xn, xn)
    App = A(xp, xp)
    bp = b(xp, test_x)
    bn = b(xn, test_x)
    pie = (Apn - Ann - bp + bn)/(2*Apn - App - Ann)
    return np.clip(pie, 0, 1)

def cwls(train_x, train_y, test_x):
    """[summary]
    
    Parameters
    ----------
    train_x : 100x2
        [description]
    train_y : 100
        [description]
    test_x : [type]
        [description]
    """
    pie = get_pie(train_x, train_y, test_x)
    PIE = train_y.copy().astype(float)
    PIE[train_y==1] = pie
    PIE[train_y==-1] = (1-pie)
    PIE = np.diag(PIE) #100x100
    train_x = np.concatenate((train_x, np.ones([100, 1])), axis=1)
    a = train_x.T.dot(PIE).dot(train_x)
    b = train_x.T.dot(PIE).dot(train_y)
    theta = np.linalg.solve(a, b)
    print(f'theta {theta}')
    return theta



    

def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1],
                    marker='$O$', c='blue')
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
                    marker='$X$', c='red')
        plt.savefig('lecture9-h3-{}-weighted.png'.format(name))


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
