from PIL import Image
from pathlib import Path

# LOAD GAMBAR
base = Path(__file__).resolve().parent.parent  # folder beating_michiko
img_path = base / "10kGambarSudoku" / "gambar1.PNG"

img = Image.open(img_path).convert("RGB");
print("size: ", img.size)

# MENGHITUNG UKURAN IMG, INNER IMAGE DAN CELL
w, h = img.size
if w != h:
    raise SystemExit (f"gambar harus square")

pad = 1
innerImage = w - 2 * pad
cell = innerImage // 9
inset = 6
if cell * 9 != innerImage:
    raise SystemExit(f"innerImage={innerImage} tidak habis dibagi 9 (pad salah atau gambar tidak pas)")

out_dir = base / "out_cells"
out_dir.mkdir(parents=True, exist_ok=True)

# MENGHITUNG POSISI CROPPING DAN SAVED CROP
count = 0 
for row in range(9):
    for column in range(9):
        x0 = pad + column * cell + inset
        y0 = pad + row * cell + inset
        x1 = pad + (column + 1) * cell - inset
        y1 = pad + (row + 1) * cell - inset

        cell_img = img.crop((x0, y0, x1, y1))
        cell_img.save(out_dir / f"cell_r{row}_c{column}.png")
        count += 1

print("saved: ", count, "cells to", out_dir)