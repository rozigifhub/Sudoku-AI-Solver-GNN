import numpy as np
from pathlib import Path
# from preprocess_data import load_raw, make_train_batch
from neural_network import Network
# from preprocess_data import load_cell_for_inference
from preprocess_data import load_cells_from_folder
import sys

# (tr_d, va_d, te_d) = load_raw()
# X_train, y_train = tr_d
# X_test, y_test = te_d

# batch_idx = np.arange(10)
# mini_batch = make_train_batch(X_train, y_train, batch_idx)

# print(len(mini_batch))
# print(mini_batch[0][0].shape, mini_batch[0][1].shape)  # (784,1) (10,1)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_relu_softmax(784,32,16,10)_mbc64_eta0_01_NewDataset.npz"

if MODEL_PATH.exists():
    net = Network.load(MODEL_PATH)
    print("loaded saved model")
else:
    sys.exit("Model tidak ditemukan")
# net = Network([784, 32, 16, 10])
# net.SGD(
#     X_train,
#     y_train,
#     epochs=5,
#     mini_batch_size=64,
#     eta=0.01,
#     batch_maker=make_train_batch,
#     X_test=X_test,
#     y_test=y_test
#     )
# net.save("models/mnist_relu_softmax(784,32,16,10)_mbc64_eta0_01_NewDataset.npz")
# print("Model baru sudah ke save")


# out = net.feedforward(mini_batch[0][0])
# print(out.shape, out.sum())
# net.SGD(X_train, y_train, epochs=1, mini_batch_size=10, eta=0.1, batch_maker=make_train_batch)


cells = load_cells_from_folder("out_cells")
digits = [str(net.predict_digit(x)) for _, x in cells]
print("".join(digits))