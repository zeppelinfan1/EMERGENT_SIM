"""
NEURAL NETWORK BRAIN
"""
import cv2
# IMPORTS
import numpy as np
import os, pickle, copy

# OBJECTS
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):

        # Remember input values
        self.inputs = inputs
        # Calculate output
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularizers
        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dl1
        if self.weight_regularizer_l2 > 0:
            self.dweights += self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += self.bias_regularizer_l2 * self.biases
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):

        return self.weights, self.biases

    def set_parameters(self, weights, biases):

        self.weights = weights
        self.biases = biases

class Layer_Dropout:

    def __init__(self, rate):

        self.rate = 1 - rate

    def forward(self, inputs, training):

        self.inputs = inputs
        if not training:
            self.output = inputs.copy()

            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

class Layer_Input:

    def forward(self, inputs, training):

        self.output = inputs

class Activation_ReLU:

    def forward(self, inputs, training):

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):

        return outputs

class Activation_Softmax:

    def forward(self, inputs, training):

        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):

        return np.argmax(outputs, axis=1)

class Activation_Sigmoid:

    def forward(self, inputs, training):

        pass

    def backward(self, dvalues):

        pass

    def predictions(self, outputs):

        pass


class Activation_Linear:

    def forward(self, inputs, training):

        pass

    def backward(self, dvalues):

        pass

    def predictions(self, outputs):

        pass

class Optimizer_SGD:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):

        pass

    def pre_update_params(self):

        pass

    def update_params(self, layer):

        pass

    def post_update_params(self):

        pass

class Optimizer_Adagrad:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):

        pass

    def pre_update_params(self):

        pass

    def update_params(self, layer):

        pass

    def post_update_params(self):

        pass


class Optimizer_RMSprop:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):

        pass

    def pre_update_params(self):

        pass

    def update_params(self, layer):

        pass

    def post_update_params(self):

        pass


class Optimizer_Adam:

    def __init__(self, learning_rate=1., decay=0., momentum=0.):

        pass

    def pre_update_params(self):

        pass

    def update_params(self, layer):

        pass

    def post_update_params(self):

        pass

class Loss:

    def regularization_loss(self):

        pass

    def remember_trainable_layers(self, trainable_layers):

        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):

        pass

    def calculate_accumulated(self, *, include_regularization=False):

        pass

    def new_pass(self):

        pass

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        pass

    def backward(self, dvalues, y_true):

        pass

class Activation_Softmax_Loss_CategoricalCrossEntropy():

    def backward(self, dvalues, y_true):

        pass

class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        pass

    def backward(self, dvalues, y_true):

        pass

class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):

        pass

    def backward(self, dvalues, y_true):

        pass


class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):

        pass

    def backward(self, dvalues, y_true):

        pass

class Accuracy:

    def calculate(self, predictions, y):

        pass

    def calculate_accumulated(self):

        pass

    def new_pass(self):

        pass

class Accuracy_CategoricalAccuracy:

    def __init__(self, *, binary=False):

        pass

    def init(self, y):

        pass

    def compare(self, predictions, y):

        pass

class Accuracy_Regression(Accuracy):

    def __init__(self):

        pass

    def init(self, y, reinit=False):

        pass

    def compare(self, predictions, y):

        pass

class Model:

    def __init__(self):

        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):

        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):

        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):

        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        pass

    def evaluate(self, X_val, y_val, *, batch_size=None):

        pass

    def predict(self, X, *, batch_size=None):

        pass

    def forward(self, X, training):

        pass

    def backward(self, output, y):

        pass

    def get_parameters(self):

        pass

    def set_parameters(self):

        pass

    def save_parameters(self, path):

        pass

    def load_parameters(self, path):

        pass

    def save(self, path):

        pass

    @staticmethod
    def load(path):

        pass

    def load_mnist_dataset(self, path):

        pass

    def create_data_mnist(self, path):

        pass


def create_data(samples, classes):

    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):

        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r*np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y

"""SPIRAL EXAMPLE
"""
if __name__ == "__main__":

    X, y = create_data(samples=1000, classes=4)
    X_test, y_test = create_data(samples=100, classes=4)

    model = Model()
    model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.1))
    model.add(Layer_Dense(2, 512))
    model.add(Activation_Softmax())