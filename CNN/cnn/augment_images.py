from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def augment_image(
    image_path: Path,
    out_path: Path,
    rotation: float = 0.0,
    brightness: float = 1.0,
    blur: float = 0.0,
) -> None:
    img = Image.open(image_path).convert("RGB")

    if rotation != 0:
        img = img.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")

    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# def process_folder(
#     input_dir: Path,
#     output_dir: Path,
#     rotation: float,
#     brightness: float,
#     blur: float,
# ) -> None:
#     for image_path in input_dir.glob("*"):
#         if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
#             continue

#         out_path = output_dir / image_path.name
#         augment_image(
#             image_path=image_path,
#             out_path=out_path,
#             rotation=rotation,
#             brightness=brightness,
#             blur=blur,
#         )

def process_folder(
    input_dir: Path,
    output_dir: Path,
    rotation: float,
    brightness: float,
    blur: float,
) -> None:
    for image_path in input_dir.glob("*"):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue

        if not should_process(image_path.name):
            continue

        out_path = output_dir / image_path.name
        augment_image(
            image_path=image_path,
            out_path=out_path,
            rotation=rotation,
            brightness=brightness,
            blur=blur,
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate, change brightness, and blur images.")
    parser.add_argument("--input", type=Path, required=True, help="Input image or folder")
    parser.add_argument("--output", type=Path, required=True, help="Output image or folder")
    parser.add_argument("--rotation", type=float, default=0.0, help="Rotation in degrees")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness factor, e.g. 0.8 or 1.2")
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius")
    return parser.parse_args()

def should_process(name: str) -> bool:
    stem = Path(name).stem.lower()  # contoh: board26
    if not stem.startswith("board"):
        return False

    num_text = stem.replace("board", "", 1)
    if not num_text.isdigit():
        return False

    num = int(num_text)
    return 26 <= num <= 50

def main() -> int:
    args = parse_args()

    if args.input.is_file():
        augment_image(
            image_path=args.input,
            out_path=args.output,
            rotation=args.rotation,
            brightness=args.brightness,
            blur=args.blur,
        )
    elif args.input.is_dir():
        process_folder(
            input_dir=args.input,
            output_dir=args.output,
            rotation=args.rotation,
            brightness=args.brightness,
            blur=args.blur,
        )
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
