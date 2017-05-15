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


def prepro(I, ds=4):
    I = I[::ds, ::ds, 2].astype(float)
    I -= I.mean()
    I /= I.std()
    return I[:, :, None]


def discount_rewards(rwd, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rwd)
    running_add = 0
    for t in range(len(rwd)-1, -1, -1):
        running_add = running_add * gamma + rwd[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /= (discounted_r.std() + 1e-8 * 2.)
    return discounted_r[:, None]
