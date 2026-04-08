import numpy as np
from mnist_loader import data_load

def vectorized_result(j):
    e = np.zeros((10, 1), dtype=np.float32)
    e[int(j)] = 1.0
    return e

def wrap_x(x):
    # x: (784, ) -> (784,1)
    return np.reshape(x, (784, 1)).astype(np.float32, copy=False)

def make_train_batch(X, y, batch_idx):
    # output: list of (x(784,1), y_onehot(10,1))
    return [(wrap_x(X[i]), vectorized_result(y[i])) for i in batch_idx]

def iter_batch_indices(n, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]

def load_raw():
    # return (tr_d, va_d, te_d) sama seperti mnist_loader.data_load()
    return data_load()