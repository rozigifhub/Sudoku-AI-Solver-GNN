from mnist_loader import data_load
import numpy as np

# Load data mentah
training_data, validation_data, test_data = data_load()

# Pisahkan gambar dan label
X_train, y_train = training_data

print("=== CONTOH 1 ===")
print("Gambar (784 nilai pixel):")
print(X_train[0])           # print gambar pertama
print("Label:", y_train[0]) # print label pertama
print("\n")

print("=== CONTOH 2 ===")
print("Gambar (784 nilai pixel):")
print(X_train[1])           # print gambar kedua
print("Label:", y_train[1]) # print label kedua
print("\n")

# Print shapes khusus untuk data_load() (RAW)
print("=== SHAPE RAW DARI data_load() ===")
print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)

print("X_val shape   :", validation_data[0].shape)
print("y_val shape   :", validation_data[1].shape)

print("X_test shape  :", test_data[0].shape)
print("y_test shape  :", test_data[1].shape)
