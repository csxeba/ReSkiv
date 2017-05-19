import numpy as np
from scipy.ndimage import distance_transform_edt as dte


def calc_meand(screensize):
    """
    Calculates the mean euclidean distance between
    all points in a rectangle
    """
    Ls = screensize
    Lw, Lh = screensize
    Lw2, Lh2 = screensize**2
    Lw3, Lh3 = screensize**3
    d = np.linalg.norm(Ls)
    a1 = (5/2)*(Lw2/Lh)*np.log((Lh + d) / Lw)
    a2 = d*(3 - (Lw2/Lh2) - (Lh2/Lw2))
    return (1/15) * ((Lw3/Lh2)+(Lh3/Lw2)+a2+a1)


def downsample(I, ds=4):
    """Downsamples and scales an image taken from the environment"""
    return I[::ds, ::ds, 2] / 255.


def prepro_convolutional(I, ds=None):
    I = I[::ds, ::ds, 2] if ds else I[:, :, 2]
    I = I.astype(float) / 255.
    return I[None, None, :, :]


def prepro_recurrent(X):
    """
    Recurrent networks receive 3-dimensional data.
    First dim is the time axis,
    Second is the batch,
    Third is the actual number of parameters
    """
    return X[:, None, ...]  # time = batches, batches = 1


def steep_hills(game):
    sx, sy = game.size
    peaks = np.ones(game.size)
    peaks[(0, sx - 1, 0, sx - 1), (0, 0, sy - 1, sy - 1)] = 0.
    valleys = np.ones_like(peaks)
    for e in game.enemies:
        peaks[tuple(e.coords)] = 0.
    valleys[tuple(game.square.coords)] = 0.

    hills = dte(valleys) - dte(peaks)
    hills = 255. * ((hills - hills.min()) / (hills.max() - hills.min()))
    hills = np.repeat(hills[..., None], 3, axis=-1)
    hills[..., 2] = 0.
    hills[..., 1] = 255. - hills[..., 0]
    return hills


def prepro_hills(game, ds=2):
    ds = 1 if ds < 1 else ds

    sx, sy = game.size // ds
    peaks = np.ones(game.size // ds)
    peaks[(0, sx - 1, 0, sx - 1), (0, 0, sy - 1, sy - 1)] = 0.
    valleys = np.ones_like(peaks)
    for e in game.enemies:
        peaks[tuple(e.coords // ds)] = 0.
    valleys[tuple(game.square.coords // ds)] = 0.

    peaks = dte(peaks) + 1e-5
    peaks = (peaks / game.maxdist) ** -1.5
    valleys = dte(valleys) + 1e-5
    valleys = (valleys / game.maxdist) ** -1.5
    return peaks - valleys


def calculate_gradient(game, ds, deg=1):

    if deg < 1:
        deg = 1

    hills = prepro_hills(game, ds=ds)

    pc = tuple(game.player.coords // ds)
    px, py = pc

    grads = [hills[px:px+deg, py:py+deg]]
    i = 0
    while i < deg:
        grads.append(np.gradient(grads[i]))
        i += 1

    grads = [np.array([g[0].mean(), g[1].mean()]) for g in grads[1:]]

    return grads[0] if deg == 1 else grads



def discount_rewards(rwd, gamma=0.99):
    """
    Compute the discounted reward backwards in time
    """
    discounted_r = np.zeros_like(rwd)
    running_add = 0
    for t in range(len(rwd)-1, -1, -1):
        running_add = running_add * gamma + rwd[t]
        discounted_r[t] = running_add

    return discounted_r
