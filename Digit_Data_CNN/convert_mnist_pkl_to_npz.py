from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
import warnings

import numpy as np


def load_mnist_pkl_gz(path: Path):
    # Pickle MNIST lama kadang memicu warning dari NumPy versi baru.
    try:
        from numpy.exceptions import VisibleDeprecationWarning  # type: ignore

        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
    except Exception:
        pass
    with gzip.open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def save_mnist_npz(
    out_path: Path,
    train_set,
    valid_set,
    test_set,
    compressed: bool = True,
) -> None:
    (x_train, y_train) = train_set
    (x_valid, y_valid) = valid_set
    (x_test, y_test) = test_set

    save = np.savez_compressed if compressed else np.savez
    save(
        out_path,
        x_train=np.asarray(x_train),
        y_train=np.asarray(y_train),
        x_valid=np.asarray(x_valid),
        y_valid=np.asarray(y_valid),
        x_test=np.asarray(x_test),
        y_test=np.asarray(y_test),
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_inp = repo_root / "Digit_Data_CNN" / "mnist.pkl.gz"
    default_out = repo_root / "Digit_Data_CNN" / "mnist.npz"

    p = argparse.ArgumentParser(description="Convert mnist.pkl.gz (pickle) to mnist.npz (NumPy).")
    p.add_argument("--in", dest="inp", type=Path, default=default_inp, help="Input mnist.pkl.gz path.")
    p.add_argument("--out", type=Path, default=default_out, help="Output mnist.npz path.")
    p.add_argument("--no-compress", action="store_true", help="Use np.savez (uncompressed).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp: Path = args.inp
    out: Path = args.out

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    train_set, valid_set, test_set = load_mnist_pkl_gz(inp)
    save_mnist_npz(out, train_set, valid_set, test_set, compressed=not args.no_compress)

    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
