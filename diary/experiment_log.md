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

# Experiment log (2026-04-08, Asia/Jakarta)
- hal penting diingat:
batch_idx = np.arange(10)
itu bikin mini batch ku berisi 10 imgae + 10 label

mini_batch[0] = sample pertama bentuk tuple(x0, y0)
mini_batch[0][0] = x0 (gambar)
mini_batch[0][1] = y0 (label)

mini_batch[1] = sample kedua (x1, y1)
mini_batch[1][0] = x1 (gambar/input sample kedua)
mini_batch[1][1] = y1 (label sample kedua)
dst..
hasil hari ini: 
![](diary/test-image/image.png)

Goal hari ini: “Switch ke opsi 2 (wrap per mini-batch), bukan wrap 50k di awal.” (selesai)

Bug yang ketemu + fix:
load_raw() sempat return function (lupa ()), fix: return data_load()
make_train_batch sempat return label int, fix: pakai vectorized_result(y[i]) biar shape (10,1)

Sanity check yang lolos: len(mini_batch)=10, x.shape=(784,1), y.shape=(10,1)
hasil:
10
(784, 1) (10, 1)

TODO besok:
Implement sigmoid(z) + feedforward(a) di neural_network.py

Test: input x dari mini_batch[0][0] → output shape (10,1)
Bikin iter_batch_indices(n, batch_size) untuk shuffle+split mini-batch index