import numpy as np


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


def downsample_image(I, ds=4):
    """Downsamples and scales an image taken from the environment"""
    I = I[::ds, ::ds, 2].astype(float) / 255.
    return I


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
