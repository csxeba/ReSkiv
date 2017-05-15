from keras.models import Sequential
from keras.layers import Dense, Conv2D


class Network(Sequential):

    @classmethod
    def default(cls, inshape):
