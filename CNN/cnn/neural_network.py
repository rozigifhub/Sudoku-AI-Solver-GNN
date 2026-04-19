#Libraries
from pathlib import Path
import numpy as np

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases: list[np.ndarray] = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights: list[np.ndarray] = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for  i, (b, w) in  enumerate(zip(self.biases, self.weights)):
            z = w @ a + b
            if i == len(self.weights) - 1:
                a = softmax(z)
            else:
                a = ReLU(z)
        return a

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = w @ activation + b
            zs.append(z)

            if i == len(self.weights) - 1:
                activation = softmax(z)
            else:
                activation = ReLU(z)

            activations.append(activation)

        # Softmax + cross-entropy derivative simplifies to prediction - target.
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            delta = (self.weights[-layer + 1].T @ delta) * ReLU_prime(z)
            nabla_b[-layer] = delta
            nabla_w[-layer] = delta @ activations[-layer - 1].T

        return nabla_b, nabla_w

    def SGD(self, X_train, y_train, epochs, mini_batch_size, eta, batch_maker, seed=0, X_test=None, y_test=None):
        rng = np.random.default_rng(seed)
        n = X_train.shape[0]

        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, mini_batch_size):
                batch_idx = perm[start:start + mini_batch_size]
                mini_batch = batch_maker(X_train, y_train, batch_idx)  # list of (x,y)
                self.update_mini_batch(mini_batch, eta)

            
            if X_test is not None and y_test is not None:
                test = self.accuracy(X_test, y_test)
                print(f"epoch {epoch+1}/{epochs} done, akurasi {test:.2f}%")
            else:
                print(f"epoch {epoch+1}/{epochs} done")


    def accuracy(self, X, y):
        correct = 0

        for x_raw, label in zip(X, y):
            x = np.reshape(x_raw, (784, 1)).astype(np.float32, copy=False)
            prediction = self.feedforward(x)
            predicted_label = np.argmax(prediction)

            if predicted_label == int(label):
                correct += 1

        return correct / len(y) * 100

    def is_blank_input(self, x, pixel_threshold=0.5, min_active_pixels=8):
        x = np.reshape(x, (784, 1)).astype(np.float32, copy=False)
        active_pixels = np.count_nonzero(x > pixel_threshold)
        return active_pixels < min_active_pixels

    def predict_digit(self, x, pixel_threshold=0.5, min_active_pixels=8):
        x = np.reshape(x, (784, 1)).astype(np.float32, copy=False)

        if self.is_blank_input(x, pixel_threshold=pixel_threshold, min_active_pixels=min_active_pixels):
            return 0

        prediction = self.feedforward(x)
        return int(np.argmax(prediction))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, np.ndarray] = {"sizes": np.array(self.sizes, dtype=np.int64)}
        for i, w in enumerate(self.weights):
            data[f"w{i}"] = w
        for i, b in enumerate(self.biases):
            data[f"b{i}"] = b

        np.savez(path, **data)  # pyright: ignore[reportArgumentType]

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            sizes = data["sizes"].astype(int).tolist()
            net = cls(sizes)
            net.weights = [np.asarray(data[f"w{i}"], dtype=np.float64) for i in range(len(sizes) - 1)]
            net.biases = [np.asarray(data[f"b{i}"], dtype=np.float64) for i in range(len(sizes) - 1)]
        return net
# ReLU activation di hidden layer
def ReLU(z):
    return np.maximum(0.01 * z, z)

# Softmax activation di output
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

# Membuat output ReLU jadi 1 dan 0
def ReLU_prime(z):
    return np.where(z > 0, 1, 0.01).astype(np.float32)






