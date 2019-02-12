import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('C:/Users/ejhon/Desktop/nmistNumberRecognizer/package/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_neuralNetwork():
    # training_data, validation_data, test_data = load_data()
    # training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    # training_results = [vectorized_results(y) for y in training_data[1]]
    # training_data = zip(training_inputs, training_results)
    # validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    # validation_data = zip(validation_inputs, validation_data[1])
    # test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    # test_data = zip(test_inputs, test_data[1])
    # return (training_data, validation_data, test_data)
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1));
    e[j] = 1.0
    return e

load_data_neuralNetwork()