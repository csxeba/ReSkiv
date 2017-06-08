import numpy as np


class Experience:

    def __init__(self, limit=10000):
        self.limit = limit
        self.X = None
        self.Y = None

    @property
    def N(self):
        return len(self.X)

    def initialize(self, X, Y):
        self.X = X
        self.Y = Y

    def incorporate(self, X, Y):
        m = len(X)
        N = self.N
        if N == self.limit:
            narg = np.arange(N)
            np.random.shuffle(narg)
            marg = narg[:m]
            self.X[marg] = X
            self.Y[marg] = Y
            return
        self.X = np.concatenate((self.X, X))
        self.Y = np.concatenate((self.Y, Y))
        if N > self.limit:
            narg = np.arange(N)
            np.random.shuffle(narg)
            self.X = self.X[narg]
            self.Y = self.Y[narg]

    def accumulate(self, X, Y):
        if self.X is None:
            self.initialize(X, Y)
        else:
            self.incorporate(X, Y)

        if self.N > self.limit:
            arg = np.arange(self.N)
            np.random.shuffle(arg)

    def get_batch(self, m):
        narg = np.arange(self.N)
        np.random.shuffle(narg)
        marg = narg[:m]
        return self.X[marg], self.Y[marg]
