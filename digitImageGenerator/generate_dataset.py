from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_font = base_dir / "fonts" / "Roboto-Regular.ttf"
    default_out = base_dir / "mnist_sudoku_digits.pkl.gz"

    parser = argparse.ArgumentParser(
        description="Generate Sudoku digit dataset in MNIST-like .pkl.gz format."
    )
    parser.add_argument("--font", type=Path, default=default_font, help="Path to Roboto .ttf font file.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output .pkl.gz path.")
    parser.add_argument("--train-per-digit", type=int, default=5000, help="Train samples per digit.")
    parser.add_argument("--valid-per-digit", type=int, default=500, help="Validation samples per digit.")
    parser.add_argument("--test-per-digit", type=int, default=1200, help="Test samples per digit.")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument("--digit-size", type=int, default=28, help="Output image size.")
    return parser.parse_args()


def resolve_font(font_path: Path) -> Path:
    if font_path.exists():
        return font_path

    raise FileNotFoundError(
        f"Roboto font not found at {font_path}. "
        "Put Roboto-Regular.ttf in digitImageGenerator/fonts or pass --font <path>."
    )


def render_digit(
    digit: int,
    font_path: Path,
    rng: np.random.Generator,
    out_size: int = 28,
    canvas_size: int = 96,
) -> np.ndarray:
    bg = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(bg)

    font_size = int(rng.integers(48, 72))
    font = ImageFont.truetype(str(font_path), size=font_size)
    text = str(digit)

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    jitter_x = int(rng.integers(-6, 7))
    jitter_y = int(rng.integers(-6, 7))
    x = (canvas_size - text_w) // 2 - bbox[0] + jitter_x
    y = (canvas_size - text_h) // 2 - bbox[1] + jitter_y

    draw.text((x, y), text, fill=255, font=font)

    if rng.random() < 0.35:
        blur_radius = float(rng.uniform(0.2, 0.7))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.asarray(bg, dtype=np.float32) / 255.0
    arr = np.where(arr > 0.45, 1.0, 0.0).astype(np.float32)

    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return np.zeros((out_size * out_size,), dtype=np.float32)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    pad = int(rng.integers(3, 6))
    side = max(h, w) + 2 * pad
    square = np.zeros((side, side), dtype=np.float32)
    top = (side - h) // 2
    left = (side - w) // 2
    square[top : top + h, left : left + w] = cropped

    resized = Image.fromarray((square * 255).astype(np.uint8), mode="L").resize(
        (out_size, out_size),
        Image.Resampling.LANCZOS,
    )
    final_arr = np.asarray(resized, dtype=np.float32) / 255.0
    final_arr = np.where(final_arr > 0.1, final_arr, 0.0).astype(np.float32)
    return final_arr.reshape(-1)


def build_split(
    digits: list[int],
    per_digit: int,
    font_path: Path,
    rng: np.random.Generator,
    out_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in digits:
        for _ in range(per_digit):
            images.append(render_digit(digit, font_path=font_path, rng=rng, out_size=out_size))
            labels.append(digit)

    X = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main() -> int:
    args = parse_args()
    font_path = resolve_font(args.font)
    rng = np.random.default_rng(args.seed)
    digits = list(range(1, 10))

    X_train, y_train = build_split(
        digits,
        per_digit=args.train_per_digit,
        font_path=font_path,
        rng=rng,
        out_size=args.digit_size,
    )
    X_valid, y_valid = build_split(
        digits,
        per_digit=args.valid_per_digit,
        font_path=font_path,
        rng=rng,
        out_size=args.digit_size,
    )
    X_test, y_test = build_split(
        digits,
        per_digit=args.test_per_digit,
        font_path=font_path,
        rng=rng,
        out_size=args.digit_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wb") as f:
        pickle.dump(
            (
                (X_train, y_train),
                (X_valid, y_valid),
                (X_test, y_test),
            ),
            f, # pyright: ignore[reportArgumentType]
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"saved: {args.out}")
    print(f"train: {X_train.shape}, labels: {y_train.shape}")
    print(f"valid: {X_valid.shape}, labels: {y_valid.shape}")
    print(f"test : {X_test.shape}, labels: {y_test.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
