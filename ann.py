import numpy as np


def cross_entropy(A, Y):
    return -np.sum(Y * np.log(A))


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

    def __init__(self, neurons, lmbd=0.01):
        self.trainable = True
        self.neurons = neurons
        self.lmbd = lmbd

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
        l2 = self.lmbd * error.shape[0]
        self.gW = self.gW * l2 + self.inputs.T.dot(error)
        self.gb = self.gb * l2 + error.sum(axis=0)
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


class Tanh:

    def __init__(self):
        self.trainable = False
        self.output = None
        self.outdim = None

    def connect(self, layer):
        self.outdim = layer.outdim

    def feedforward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backpropagate(self, error):
        return error * (1. - self.output**2)


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
    def load(path):
        import gzip
        import pickle

        with gzip.open(path) as handle:
            model = pickle.load(handle)
        return model

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
        cost = cross_entropy(predictions, Y)
        network_delta = predictions - Y
        if rewards is not None:
            network_delta *= rewards
        self.backpropagate(network_delta)
        self.parameter_update(eta, decay)
        return cost

    def evaluate(self, X, Y):
        pred = self.predict(X)
        cost = cross_entropy(pred, Y) / Y.shape[0]
        pred_cls = pred.argmax(axis=1)
        Y_cls = Y.argmax(axis=1)
        eq = pred_cls == Y_cls
        # noinspection PyTypeChecker
        acc = np.mean(eq)
        return cost, acc

    def save(self, path=None):
        import os
        import gzip
        import pickle

        if path is None:
            path = os.getcwd()

        with gzip.open("net.pgz", "wb") as handle:
            pickle.dump(self, handle)
        # print("Saved ANN to", path)
        return path
