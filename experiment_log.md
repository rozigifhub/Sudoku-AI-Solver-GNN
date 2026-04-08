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

# Experiment log (2026-04-08, Asia/Jakarta)

- Fokus: rapihin pipeline MNIST untuk training NN dari nol (NumPy).
Progress: buat data_load_wrapper() untuk ubah X jadi shape (784,1) dan label train jadi one-hot (10,1).

- Catatan teknis: train pakai one-hot, valid/test tetap label int untuk evaluasi.

- Risiko/issue: zip() itu iterator sekali pakai → bisa “habis” setelah 1 kali loop epoch.

Next: ubah zip(...) jadi list(zip(...)) agar bisa dipakai berulang; mulai implement Network.feedforward().

- Besok (2026-04-09) yang dikerjakan:

- Preprocess: ganti training_data/validation_data/test_data jadi list(zip(...)), lalu print len(...) dan shape sample pertama.

- Network: tambah fungsi sigmoid + feedforward(a) dan test output shape (10,1) untuk 1 sample input.

- Setelah itu baru lanjut: SGD() skeleton (epoch loop + mini-batch split) tanpa backprop dulu (cuma nyiapin struktur).