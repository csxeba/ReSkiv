import matplotlib
from matplotlib import pyplot as plt

from ReSkiv.environment import Game


matplotlib.use("Qt5Agg")


def prepro(I):
    I = I[::4, ::4, 2]  # downsample by factor of 4
    return I

game = Game("spazz", 30, (450, 400), escape_allowed=False)
frame = game.reset()
rounds = 1
while 1:
    print("\rRound:", rounds)
    plt.imshow(prepro(frame))
    plt.show()

    info = game.step()
    if info is None:
        break

    frame, reward, done = info
    rounds += 1
