for executing the program:
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_neuralNetwork()
import neuralNet
net = neuralNet.Network([784, 30, 10])
net.stochasticGradientDesecent(training_data, 30, 10, 3.0, test_data=test_data)
