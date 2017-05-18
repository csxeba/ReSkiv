import numpy as np


def cross_entropy(A, Y):
    return -np.sum(Y * np.log(A))


def cross_entropy2(A: np.ndarray, Y: np.ndarray):
    return -np.sum(Y * np.log(A) + (1. - Y) * np.log(1. - A))


class SGD:

    def __init__(self, eta=0.01):
        self.eta = eta

    def optimize(self, W, gW):
        return W - gW * self.eta


class RMSProp:

    def __init__(self, eta=0.1, decay=0.9, epsilon=1e-8):
        self.eta = eta
        self.decay = decay
        self.epsilon = epsilon
        self.mW = 0.

    def optimize(self, W, gW):
        self.mW = self.decay * self.mW + (1. - self.decay) * gW ** 2.
        W -= ((self.eta * gW) / np.sqrt(self.mW + self.epsilon))
        return W


class Adam:

    def __init__(self, eta=0.1, decay_memory=0.9, decay_velocity=0.999, epsilon=1e-8):
        self.eta = eta
        self.decay_memory = decay_memory
        self.decay_velocity = decay_velocity
        self.epsilon = epsilon

        self.velocity = 0.
        self.memory = 0.

    # noinspection PyTypeChecker
    def optimize(self, W, gW):
        if self.velocity is None or self.memory is None:
            self.velocity = np.zeros_like(W)
            self.memory = np.zeros_like(W)
        self.velocity = self.decay_velocity * self.velocity + (1. - self.decay_velocity) * gW
        self.memory = self.decay_memory * self.memory + (1. - self.decay_memory) * gW**2
        update = (self.eta * self.velocity) / np.sqrt(self.memory + self.epsilon)
        return W - update
