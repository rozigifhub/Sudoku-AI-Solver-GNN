# 31/03/2026:
- Pillow crop gambar dari kiri atas (0,0) ke kanan bawah (61, 61) tidak pakai minus dan sebaliknya. 
# 01/04/2026:
- I made image loader using pillow, then make automatic program to calculate cell in every size of image.
- I made a loop to calculate where to crop the image in every cell.
# 02/04/2026:
- added mnist dataset and mnist loader in directory Digit_Data_CNN(mnist) and CNN(loader)
# 07/04/2026:
- Tujuan: “Future-proof loader MNIST (hindari pickle warning)”

- Perubahan kode:
Added: Digit_Data_CNN\convert_mnist_pkl_to_npz.py
Updated: CNN/cnn/mnist_loader.py (prefer .npz, fallback .pkl.gz)
Updated: src2/data_loader.py (load .npz)

- Command yang dipakai:
python Digit_Data_CNN\convert_mnist_pkl_to_npz.py
python CNN\cnn\main.py

- Hasil:
Shapes: train (50000,784), valid (10000,784), test (10000,784)
Warning pickle: hanya muncul saat convert (sekali), loader normal pakai .npz

