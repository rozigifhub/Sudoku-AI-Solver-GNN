import numpy as np
from pathlib import Path
from PIL import Image

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


def preprocess_cell_image(cell_img: Image.Image, out_size=28, gray_threshold=0.35, pad=4, min_active_pixels=30):
    gray_img = cell_img.convert("L")
    gray = np.asarray(gray_img, dtype=np.float32) / 255.0

    # Sudoku cells are dark digit on bright background.
    ink = (gray < gray_threshold).astype(np.float32)

    if ink.sum() < min_active_pixels:
        return np.zeros((out_size * out_size, 1), dtype=np.float32)

    ys, xs = np.where(ink > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = ink[y0:y1, x0:x1]

    h, w = cropped.shape
    side = max(h, w) + 2 * pad
    canvas = np.zeros((side, side), dtype=np.float32)

    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top:top + h, left:left + w] = cropped

    resized = Image.fromarray((canvas * 255).astype(np.uint8), mode="L").resize(
        (out_size, out_size),
        Image.Resampling.LANCZOS,
    )
    x = np.asarray(resized, dtype=np.float32) / 255.0
    return np.reshape(x, (out_size * out_size, 1))


def load_cell_for_inference(path, out_size=28, gray_threshold=0.35, pad=4, min_active_pixels=30):
    cell_path = Path(path)
    with Image.open(cell_path) as cell_img:
        return preprocess_cell_image(cell_img, out_size=out_size, gray_threshold=gray_threshold, pad=pad, min_active_pixels=min_active_pixels)


def load_cells_from_folder(folder, pattern="cell_r*_c*.png", out_size=28, gray_threshold=0.35, pad=4, min_active_pixels=30):
    folder_path = Path(folder)
    cell_paths = sorted(folder_path.glob(pattern))
    return [
        (
            cell_path,
            load_cell_for_inference(cell_path, out_size=out_size, gray_threshold=gray_threshold, pad=pad, min_active_pixels=min_active_pixels),
        )
        for cell_path in cell_paths
    ]

