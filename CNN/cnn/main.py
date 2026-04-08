import numpy as np
from preprocess_data import load_raw, make_train_batch

(tr_d, va_d, te_d) = load_raw()
X_train, y_train = tr_d

batch_idx = np.arange(10)
mini_batch = make_train_batch(X_train, y_train, batch_idx)

print(len(mini_batch))
print(mini_batch[0][0].shape, mini_batch[0][1].shape)  # (784,1) (10,1)
