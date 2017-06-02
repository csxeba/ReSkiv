from os.path import exists

import numpy as np

from learning.ann import Network, Dense, Tanh


def parse_supervised_data(path=None):
    if path is None:
        path = "./supervised.data"
    if not exists(path):
        raise RuntimeError("No data to train on!")
    chain = open(path).read()
    lines = chain.split("\n")
    X, Y = [], []
    for line in lines:
        Xchain, Ychain = line.split(";")
        X.append(list(map(float, Xchain.split(","))))
        Y.append(list(map(float, Ychain.split(","))))
    return np.array(X), np.array(Y)


def get_network(path=None):
    if path is not None:
        return Network.load(path)
    net = Network((6,), [
        Dense(300, lmbd=0.), Tanh(),
        Dense(60, lmbd=0.), Tanh(),
        Dense(
    ])


X, Y = parse_supervised_data(".")

