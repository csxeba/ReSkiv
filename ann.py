import numpy as np


class Input:

    def __init__(self, indim):
        self.trainable = False
        self.outdim = indim

    def connect(self, layer):
        pass

    @staticmethod
    def feedforward(X):
        return X

    def backpropagate(self, error):
        pass


class Dense:

    def __init__(self, neurons):
        self.trainable = True
        self.neurons = neurons

        self.W = None
        self.b = np.zeros((neurons,))

        self.gW = None
        self.gb = np.zeros_like(self.b)

        self.outdim = neurons
        self.inputs = None

    def connect(self, layer):
        indim = layer.outdim
        self.W = np.random.randn(indim, self.neurons) / indim
        self.gW = np.zeros_like(self.W)

    def feedforward(self, X):
        self.inputs = X
        return X.dot(self.W) + self.b

    def backpropagate(self, error):
        self.gW = self.inputs.T.dot(error)
        self.gb = error.sum(axis=0)
        return error.dot(self.W.T)


class ReLU:

    def __init__(self):
        self.trainable = False
        self.outdim = None
        self.mask = None

    def connect(self, layer):
        self.outdim = layer.outdim

    def feedforward(self, X):
        self.mask = X < 0.
        X[self.mask] = 0.
        return X

    def backpropagate(self, error):
        error[self.mask] = 0.
        return error


class Network:

    def __init__(self, inshape, layers=()):
        self.layers = []
        self.memory = []  # RMSprop memory
        self.layers.append(Input(inshape))
        for layer in layers:
            layer.connect(self.layers[-1])
            self.layers.append(layer)
            if layer.trainable:
                self.memory.append((np.zeros_like(layer.W), np.zeros_like(layer.b)))

    @staticmethod
    def softmax(X):
        eX = np.exp(X)
        return eX / eX.sum(axis=1, keepdims=True)

    def predict(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)
        return self.softmax(X)

    def backpropagate(self, error):
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)

    def parameter_update(self, eta=0.1, decay=0.9):
        for i, layer in enumerate(lyr for lyr in self.layers if lyr.trainable):
            W, gW = layer.W, layer.gW
            b, gb = layer.b, layer.gb
            mW, mb = self.memory[i]
            mW = decay * mW + (1. - decay) * gW**2
            mb = decay * mb + (1. - decay) * gb**2
            self.memory[i] = (mW, mb)
            layer.W -= ((eta * gW) / np.sqrt(mW + 1e-8))
            layer.b -= ((eta * gb) / np.sqrt(mb + 1e-8))

    def learn_batch(self, X, Y, rewards=None, eta=0.1, decay=0.9):
        predictions = self.predict(X)
        network_delta = predictions - Y
        if rewards is not None:
            network_delta *= rewards
        self.backpropagate(network_delta)
        self.parameter_update(eta, decay)
