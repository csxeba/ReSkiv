from .optimization import *


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


class Trainable:

    trainable = True

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


class Dense(Trainable):

    def __init__(self, neurons, lmbd=0.01):
        self.neurons = (neurons if isinstance(neurons, int)
                        else np.prod(neurons))
        self.lmbd = lmbd

        self.W = None
        self.b = np.zeros((neurons,))

        self.gW = None
        self.gb = np.zeros_like(self.b)

        self.outdim = neurons
        self.inputs = None

    def connect(self, layer):
        indim = layer.outdim
        if isinstance(indim, tuple):
            indim = np.prod(indim)
        self.W = np.random.randn(indim, self.neurons) / indim
        self.gW = np.zeros_like(self.W)

    def feedforward(self, X):
        self.inputs = X
        return X.dot(self.W) + self.b

    def backpropagate(self, error):
        l2 = self.lmbd / error.shape[0]
        self.gW = self.gW * l2 + self.inputs.T.dot(error)
        self.gb = self.gb * l2 + error.sum(axis=0)
        return error.dot(self.W.T)


class LSTM(Trainable):

    def __init__(self, neurons, bias_init_factor=3.):
        self.time = 0
        self.Z = 0
        self.G = neurons * 3
        self.Zs = []
        self.gates = []
        self.cache = []

        self.s = Sigmoid()
        self.r = ReLU()

        self.neurons = neurons
        self.outdim = neurons
        self.bias_init_factor = bias_init_factor

        self.inputs = None

        self.W = None
        self.b = None
        self.gW = None
        self.gb = None
        self.C = np.zeros((self.neurons,))

    def connect(self, layer):
        inshape = layer.outdim
        self.Z = inshape[-1] + self.neurons
        self.W = np.random.randn(self.Z, self.neurons * 4) / self.Z
        self.b = np.zeros((self.neurons * 4,)) + self.bias_init_factor
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def feedforward(self, X):
        time, dim = X.shape
        self.time = time
        self.inputs = X
        self.Zs, self.gates, self.cache = [], [], []
        output = np.zeros((self.neurons,))

        for t in range(time):
            Z = np.concatenate((self.inputs[t], output))

            preact = Z @ self.W + self.b  # type: np.ndarray
            preact[:self.G] = self.s.feedforward(preact[:self.G])
            preact[self.G:] = self.r.feedforward(preact[self.G:])

            f, i, o, cand = np.split(preact, 4, axis=-1)

            self.C = self.C * f + i * cand
            output = self.C * o

            self.Zs.append(Z)
            self.gates.append(preact)
            self.cache.append([output, self.C.copy(), preact])

        return np.stack([cache[0] for cache in self.cache], axis=0)

    def backpropagate(self, error):
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        delta = error

        actprime = self.r.backpropagate
        sigprime = self.s.backpropagate

        dC = np.zeros_like(delta[-1])
        dX = np.zeros_like(self.inputs)
        dZ = np.zeros_like(self.Zs[0])

        for t in range(-1, -(self.time+1), -1):
            output, state, preact = self.cache[t]
            f, i, o, cand = np.split(self.gates[t], 4, axis=-1)

            # Add recurrent delta to output delta
            delta[t] += dZ[-self.neurons:]

            # Backprop into state
            dC += delta[t] * o * state

            state_yesterday = 0. if t == -self.time else self.cache[t-1][1]
            # Calculate the gate derivatives
            df = state_yesterday * dC
            di = cand * dC
            do = state * delta[t]
            dcand = i * dC * actprime(cand)  # Backprop nonlinearity
            dgates = np.concatenate((df, di, do, dcand), axis=-1)
            dgates[:self.G] *= sigprime(self.gates[t][:self.G])  # Backprop nonlinearity

            dC *= f

            self.gb += dgates.sum(axis=0)
            self.gW += self.Zs[t].T @ dgates

            dZ = dgates @ self.W.T

            dX[t] = dZ[:-self.neurons]

        return dX


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


class Sigmoid:

    def __init__(self):
        self.trainable = False
        self.output = None
        self.outdim = None

    def connect(self, layer):
        self.outdim = layer.outdim

    def feedforward(self, X):
        self.output = 1. / (1. + np.exp(-X))
        return self.output

    def backpropagate(self, error):
        return error * self.output * (1. - self.output)


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

    def __init__(self, inshape, layers=(), optimizer="rmsprop"):
        self.layers = []
        self.memory = []  # RMSprop memory
        self.layers.append(_Input(inshape))
        for layer in layers:
            layer.connect(self.layers[-1])
            self.layers.append(layer)
            if layer.trainable:
                self.memory.append((np.zeros_like(layer.W), np.zeros_like(layer.b)))
        self.cost = cross_entropy
        if isinstance(optimizer, str):
            self.optimizer = {"sgd": SGD(),
                              "adam": Adam(),
                              "rmsprop": RMSProp()}
        else:
            self.optimizer = optimizer

    @staticmethod
    def load(path):
        import gzip
        import pickle

        with gzip.open(path) as handle:
            model = pickle.load(handle)
        return model

    @staticmethod
    def softmax(X):
        eX = np.exp(X - X.max(axis=0))
        return eX / eX.sum(axis=1, keepdims=True)

    @classmethod
    def default(cls, inshape, outshape, lmbd_global=0.):
        return cls(inshape, layers=(
            Dense(200, lmbd=lmbd_global), Tanh(),
            Dense(60, lmbd=lmbd_global), Tanh(),
            Dense(outshape, lmbd=lmbd_global)
        ))

    def prediction(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)
        return self.softmax(X)

    def backpropagation(self, error):
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)
        return error

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
        predictions = self.prediction(X)
        cost = cross_entropy(predictions, Y)
        network_delta = predictions - Y
        if discount_rwds is not None:
            network_delta *= discount_rwds
        self.backpropagation(network_delta)
        self.set_weights(self.optimizer.optimize(self.get_weights(), self.get_gradients()))
        return cost

    def evaluate(self, X, Y):
        pred = self.prediction(X)
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

        with gzip.open("online.agent", "wb") as handle:
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
