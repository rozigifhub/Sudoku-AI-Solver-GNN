from neural_network import Network
from data_loader import data_load_wrapper

training_data, validation_data, test_data = data_load_wrapper()

net = Network([784, 64, 32, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
