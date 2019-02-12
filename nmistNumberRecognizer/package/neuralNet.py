import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        #sizes is the list which specifies how many neurons on each layes
        #every connection has a weight
        #every neuron has a bias
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        #print((sizes[:-1], sizes[1:]))
        #print(self.weights)
        #print(self.biases)
    def feed_forward(self, input):
        for bias, weight in zip(self.biases, self.weights):
            input = sigmoid(np.dot(weight, input) + bias)
        return input
    def stochasticGradientDesecent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        test_data = list(test_data)
        training_data = list(training_data)
        if(test_data):
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            #creating mini batchs of training data, by shuffling the data first.
            for mini_batch in mini_batchs:
                self.updateMiniBatch(mini_batch, eta)
            if(test_data):
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete" .format(i))

    def updateMiniBatch(self, mini_batch, eta):
        nabla_biases = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weights = [np.zeros(weight.shape) for weight in self.weights]
        #specifies the size of every part in self.biases and self.weights
        for x,y in mini_batch:
            delta_nabla_biases, delta_nabla_weights = self.backpropagation(x,y)
            nabla_biases = [nb+dnb for nb, dnb in zip(nabla_biases, delta_nabla_biases)]
            nabla_weights = [nw+dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
        self.biases = [bias-(eta/len(mini_batch))*nabla_bias for bias, nabla_bias in zip(self.biases, nabla_biases)]
        self.weights = [weight - (eta / len(mini_batch)) * nabla_weight for weight, nabla_weight in zip(self.weights, nabla_weights)]
    def backpropagation(self, x, y):
        nabla_biases = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weight = [np.zeros(weight.shape) for weight in self.weights]
        #feed forward
        activation = x
        activations = [x]
        z_vectors = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(z_vectors[-1])
        nabla_biases[-1] = delta
        nabla_weight[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = z_vectors[-i]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_biases[-i] = delta
            nabla_weight[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_biases, nabla_weight
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)),y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)
    def costDerivative(self, outputActivations, y):
        return (outputActivations - y)
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))

net = Network([2,3,1])
