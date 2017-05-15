import numpy as np


def cross_entropy(A, Y):
    return -np.sum(Y * np.log(A))


def cross_entropy2(A: np.ndarray, Y: np.ndarray):
    return -np.sum(Y * np.log(A) + (1. - Y) * np.log(1. - A))


class SGD:

    def __init__(self, eta=0.01):
        self.eta = eta

    def __call__(self, W, gW):
        return W - gW * self.eta


class RMSProp:

    def __init__(self, eta=0.001, decay=0.9, epsilon=1e-8):
        self.eta = eta
        self.decay = decay
        self.epsilon = epsilon
        self.mW = 0.

    def __call__(self, W, gW):
        self.mW = self.decay * self.mW + (1. - self.decay) * gW ** 2.
        W -= ((self.eta * gW) / np.sqrt(self.mW + self.epsilon))
        return W


class _Input:

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

    def get_weights(self, unfold=True):
        if unfold:
            return np.concatenate((self.W.ravel(), self.b))
        return [self.W, self.b]

    def get_gradients(self, unfold=True):
        if unfold:
            return np.concatenate((self.gW.ravel(), self.gb))
        return [self.gW, self.gb]

    def set_weights(self, W, fold=True):
        if fold:
            wsz = self.W.size
            self.W = W[:wsz].reshape(self.W.shape)
            self.b = W[wsz:]
        else:
            self.W, self.b = W

    @property
    def nparams(self):
        return self.W.size + self.b.size


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

    def __init__(self, inshape, layers=(), **optimizer_params):
        self.layers = []
        self.memory = []  # RMSprop memory
        self.layers.append(_Input(inshape))
        for layer in layers:
            layer.connect(self.layers[-1])
            self.layers.append(layer)
            if layer.trainable:
                self.memory.append((np.zeros_like(layer.W), np.zeros_like(layer.b)))
        self.optimizer = RMSProp(**optimizer_params)

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

    @staticmethod
    def sigmoid(X):
        return 1. / (1. + np.exp(-X))

    @classmethod
    def default(cls, inshape, outshape, lmbd_global=0.):
        return cls(inshape, layers=(
            Dense(200, lmbd=lmbd_global), Tanh(),
            Dense(60, lmbd=lmbd_global), Tanh(),
            Dense(outshape, lmbd=lmbd_global)
        ))

    def predict(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)
        return self.softmax(X)

    def backpropagate(self, error):
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)

    def epoch(self, X, Y, discount_rwds=None, bsize=50):
        assert len(X) == len(Y)
        if bsize is None:
            bsize = len(X)
        allx = len(X)
        if discount_rwds is None:
            batch_stream = ((s, X[s:s + bsize], Y[s:s + bsize]) for s in range(0, len(X), bsize))
        else:
            assert len(X) == len(discount_rwds)
            batch_stream = (
                (s, X[s:s+bsize], Y[s:s+bsize], discount_rwds[s:s+bsize])
                for s in range(0, len(X), bsize)
            )
        costs = []
        for batch in batch_stream:
            cost = self.learn_batch(*batch[1:])
            costs.append(cost / bsize)
            print("\rANN Fitting... {:>.2%} Cost: {:>.4f}"
                  .format(batch[0] / allx, np.mean(costs)), end="")
        print("\rANN Fitting... {:>.2%} Cost: {:>.4f}"
              .format(1., np.mean(costs)))

    def learn_batch(self, X, Y, discount_rwds=None):
        predictions = self.predict(X)
        cost = cross_entropy(predictions, Y)
        network_delta = predictions - Y
        if discount_rwds is not None:
            network_delta *= discount_rwds
        self.backpropagate(network_delta)
        self.set_weights(self.optimizer(self.get_weights(), self.get_gradients()))
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

    def get_weights(self, unfold=True):
        W = [lyr.get_weights(unfold) for lyr in self.layers if lyr.trainable]
        return np.concatenate(W) if unfold else W

    def get_gradients(self, unfold=True):
        gW = [lyr.get_gradients(unfold) for lyr in self.layers if lyr.trainable]
        return np.concatenate(gW) if unfold else gW

    def set_weights(self, W, fold=True):
        if fold:
            start = 0
            for layer in self.layers:
                if not layer.trainable:
                    continue
                end = start + layer.nparams
                layer.set_weights(W[start:end])
                start = end
        else:
            for w, layer in zip(W, self.layers):
                if not layer.trainable:
                    continue
                layer.set_weights(w)
