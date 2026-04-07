import numpy as np
from pathlib import Path

def data_load():
    repo_root = Path(__file__).resolve().parent.parent
    npz_path = repo_root / "Digit_Data_CNN" / "mnist.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Generate it once via: "
            r"python Digit_Data_CNN\convert_mnist_pkl_to_npz.py"
        )

    with np.load(npz_path, allow_pickle=False) as d:
        training_data = (d["x_train"], d["y_train"])
        validation_data = (d["x_valid"], d["y_valid"])
        test_data = (d["x_test"], d["y_test"])
    return training_data, validation_data, test_data

def data_load_wrapper():
    tr_d, va_d, te_d = data_load()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


