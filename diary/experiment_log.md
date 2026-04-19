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

Target besok:
Implement ReLU(z) + softmax(z) (untuk output) + feedforward(a) di neural_network.py

Test: input x dari mini_batch[0][0] → output shape (10,1)
Bikin iter_batch_indices(n, batch_size) untuk shuffle+split mini-batch index

# Experiment log (2026-04-08, Asia/Jakarta)
Target kemarin:
- feedforward (Done)
- ReLU + softmax (Done)
Target hari ini:
- SGD(training_data, epochs, mini_batch_size, eta, test_data=None)
Di dalamnya: shuffle index/data, pecah jadi mini-batch, panggil placeholder update_mini_batch(...)

- update_mini_batch untuk sekarang cukup pass atau print shape (nanti baru diisi backprop)

# Experiment log (2026-04-16, Asia/Jakarta)
Epoch 10:
baseline epoch 10, network[784,30,10], eta= 0.01, mbc=32 : 93.58%
epoch = 10, eta = 0.05, mbc = 128, network[784,30,10] = 93.79%
epoch = 10, eta = 0.01, mbc = 64, network[784,30,10] = 93.75%

epoch 20  
network[784,32, 16,10], eta = 0.01, mbc = 64 : 95.06%

Penambahan: 
- ReLU prime(z)
- SGD
- update_mini_batch
- backprop

# Experiment log (19/04/2026)
Ink thresshold = 0.05 dan min_pixel = 8
hasil: tidak bagus
Fokus hari ini: nyambungin model MNIST ke real case Sudoku dari folder out_cells dan rapihin preprocessing inferensi.

Perubahan yang dilakukan:

ImgProcessing/crop1image.py

hasil crop cell sekarang langsung di-resize ke 28x28
source image diarahkan ke 10kGambarSudoku/image.png
bug perhitungan grid dibenerin (innerImage)
pad dibuat otomatis sesuai ukuran gambar
CNN/cnn/preprocess_data.py
ditambah helper inferensi untuk load cell dari out_cells

preprocessing diubah jadi:
grayscale
threshold biner
blank filtering
crop bounding box digit
center ke canvas
resize ke 28x28
CNN/cnn/neural_network.py
dipakai method predict_digit() untuk inferensi cell
Temuan penting:

gambar real case aktif ternyata 549x549, bukan 551x551
asumsi cell lama jadi tidak valid, jadi out_cells harus di-crop ulang
domain mismatch masih besar:
model dilatih di MNIST
real case pakai digit Sudoku yang rapi
akibatnya hasil inferensi masih sering meleset (5 terbaca 6, dst.)
Hasil hari ini:

out_cells sudah berhasil digenerate ulang menjadi 81 cell ukuran 28x28
blank cell dan digit cell sekarang sudah lebih rapi masuk ke preprocessing
pipeline inferensi real case sudah jalan, tapi akurasi masih belum cukup bagus untuk dipakai final
Kesimpulan:

bottleneck utama sekarang bukan lagi loading atau resize
masalah terbesar adalah mismatch antara data training MNIST dan digit Sudoku real case
Rencana berikutnya:

buat dataset Sudoku sendiri dengan class 0..9 (0 untuk blank)
pertimbangkan pakai CNN setelah dataset Sudoku siap
jangan lanjut tuning kecil-kecilan model MNIST terlalu lama karena limit utamanya sudah jelas

# Experiment log (2026/04/19)

Fokus hari ini: membangun dataset synthetic digit Sudoku dan menguji model pada real case board Sudoku.

Tujuan:

membuat dataset digit Sudoku yang lebih sesuai domain dibanding MNIST
memastikan model bisa dipakai membaca digit di setiap cell Sudoku nyata
Perubahan yang dilakukan:

membuat folder digitImageGenerator/
menambahkan script digitImageGenerator/generate_dataset.py
mendesain dataset synthetic format mirip MNIST (.pkl.gz)
dataset berisi digit 1..9 dengan font Roboto
blank tidak dimasukkan ke dataset, karena blank ditangani di preprocessing / inference
pipeline preprocessing dan crop cell tetap dipakai untuk pengujian real case
Spesifikasi dataset synthetic:

format output:
((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
kelas: 1..9
gambar disiapkan sebagai input classifier
target jumlah data besar untuk train dan test

pendekatan visual: render digit rapi, crop, center, resize, flatten
Hasil eksperimen:

- model tidak hanya berjalan di dataset synthetic
- model juga berhasil diuji pada real case
model bisa membaca digit pada tiap cell Sudoku dari board yang di-crop
- ini menunjukkan pipeline preprocessing + model sudah cukup cocok untuk kasus aktual yang sedang diuji
Temuan penting:

- dataset synthetic sangat membantu karena bentuk digit lebih mirip Sudoku rapi dibanding MNIST tulisan tangan
- preprocessing tetap menjadi bagian penting agar hasil real case akurat
keberhasilan real case menunjukkan bahwa pendekatan yang dipakai sekarang sudah usable, bukan hanya bagus di data buatan
Kesimpulan:

- hari ini berhasil membuat fondasi dataset synthetic Sudoku
- sekaligus berhasil membuktikan bahwa model bisa membaca digit pada cell Sudoku nyata
- ini menjadi milestone penting karena model tidak hanya bagus di training environment, tetapi juga berhasil dipakai pada data real case

next step: 
- uji ulang ke lebih banyak board Sudoku nyata untuk cek konsistensi generalisasi