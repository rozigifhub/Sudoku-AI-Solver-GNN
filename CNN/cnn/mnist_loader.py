from __future__ import annotations

import argparse
from pathlib import Path
import gzip
import pickle

import numpy as np
from PIL import Image

def data_load(path: str | Path | None = None):
    """
    Fungsi yang dipakai `main.py`.

    Return:
      (training_data, validation_data, test_data)
    dengan format:
      training_data = (X_train, y_train)
      validation_data = (X_valid, y_valid)
      test_data = (X_test, y_test)

    Default:
      - kalau ada `<repo_root>/Digit_Data_CNN/mnist.npz` -> pakai itu (recommended)
      - kalau belum ada -> fallback ke `<repo_root>/Digit_Data_CNN/mnist.pkl.gz`
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if path is None:
        # Prefer future-proof .npz if it exists (no pickle warning, safer than pickle)
        npz_path = repo_root / "Digit_Data_CNN" / "mnist_sudoku.npz"
        if npz_path.exists():
            return load_mnist_npz(npz_path)
        # Fallback to legacy pickle (one-time conversion recommended)
        pkl_path = repo_root / "Digit_Data_CNN" / "mnist_sudoku_digits.pkl.gz"
        return load_mnist_pkl_gz(pkl_path)

    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_mnist_npz(path)
    return load_mnist_pkl_gz(path)


def load_mnist_npz(path: str | Path):
    """
    Load MNIST dari `mnist.npz` (output converter).

    Return format sama seperti load_mnist_pkl_gz():
      (train_set, valid_set, test_set) where set = (X, y)
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        x_train = d["x_train"]
        y_train = d["y_train"]
        x_valid = d["x_valid"]
        y_valid = d["y_valid"]
        x_test = d["x_test"]
        y_test = d["y_test"]
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_mnist_pkl_gz(path: str | Path):
    """
    Load MNIST dari file `mnist.pkl.gz` (pickle + gzip).

    Return: (train_set, valid_set, test_set)
    Masing-masing set: (X, y)
      - X: np.ndarray shape (N, 784) float32, nilai 0..1
      - y: np.ndarray shape (N,) int
    """
    path = Path(path)
    with gzip.open(path, "rb") as f:
        # encoding latin1 umum untuk pickle MNIST versi lama (Python 2)
        return pickle.load(f, encoding="latin1")


def save_sample_pngs(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path,
    n: int = 20,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(int(n), int(X.shape[0]))
    for i in range(n):
        img = (X[i].reshape(28, 28) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(out_dir / f"mnist_{i:03d}_label{int(y[i])}.png")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent

    ap = argparse.ArgumentParser(description="Load mnist_sudoku_digits.pkl.gz and export a few sample PNGs.")
    ap.add_argument(
        "--path",
        type=Path,
        default=repo_root / "Digit_Data_CNN" / "mnist_sudoku_digits.pkl.gz",
        help="Path to mnist.pkl.gz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=repo_root / "CNN" / "mnist_samples_sudokuGenerator",
        help="Output folder for sample PNGs.",
    )
    ap.add_argument("--n", type=int, default=20, help="How many sample images to export.")
    args = ap.parse_args()

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_mnist_pkl_gz(args.path)

    print("train:", X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)
    print("valid:", X_valid.shape, y_valid.shape, X_valid.dtype, y_valid.dtype)
    print("test :", X_test.shape, y_test.shape, X_test.dtype, y_test.dtype)
    print("X range:", float(X_train.min()), float(X_train.max()))

    save_sample_pngs(X_train, y_train, out_dir=args.out, n=args.n)
    print(f"saved samples to: {args.out}")
