"""
TODO : allow order types of noise ? for now only random normal
"""


import numpy as np


MUS = (0, 0, 0, 0, 0, 0, 0)


def genseq_t(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian t-noise (flicker) with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, T)
    arr = np.repeat(np.repeat(val[:, np.newaxis], V, axis=1)[:, :, np.newaxis], H, axis=2)
    return arr


def genseq_v(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian v-noise with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, V)
    arr = np.repeat(np.repeat(val[np.newaxis, :], T, axis=0)[:, :, np.newaxis], H, axis=2)
    return arr


def genseq_h(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian h-noise with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, H)
    arr = np.repeat(np.repeat(val[np.newaxis, :], T, axis=0)[:, np.newaxis, :], V, axis=1)
    return arr


def genseq_tv(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian tv-noise with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, T*V)
    val = np.reshape(val, (T, V))
    arr = np.repeat(val[:, :, np.newaxis], H, axis=2)
    return arr


def genseq_th(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian th-noise with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, T*H)
    val = np.reshape(val, (T, H))
    arr = np.repeat(val[:, np.newaxis, :], V, axis=1)
    return arr


def genseq_vh(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian vh-noise (fixed pattern noise) with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, V*H)
    val = np.reshape(val, (V, H))
    arr = np.repeat(val[np.newaxis, :, :], T, axis=0)
    return arr


def genseq_tvh(T, V, H, mu, sigma):
    """Generate a 3D T-V-H sequence of gaussian tvh-noise ("photon" noise) with mean mu and std sigma."""
    val = np.random.normal(mu, sigma, T*V*H)
    arr = np.reshape(val, (T,V,H))
    return arr


def genseq_all_seq(T, V, H, sigmas, mus=MUS):
    """Generate all 7 3D T-V-H sequences of noises, and return them as well as the total sequence :
     - t sequence
     - v sequence
     - h sequence
     - tv sequence
     - th sequence
     - vh sequence
     - tvh sequence
     - sum of all of these
     The corresponding std and means must be passed in a tuple. Default means are 0 for all noise.
     """
    sigma_t, sigma_v, sigma_h, sigma_tv, sigma_th, sigma_vh, sigma_tvh = sigmas
    mu_t, mu_v, mu_h, mu_tv, mu_th, mu_vh, mu_tvh = mus
    arr_t   = genseq_t(T, V, H, mu_t, sigma_t)
    arr_v   = genseq_v(T, V, H, mu_v, sigma_v)
    arr_h   = genseq_h(T, V, H, mu_h, sigma_h)
    arr_tv  = genseq_tv(T, V, H, mu_tv, sigma_tv)
    arr_th  = genseq_th(T, V, H, mu_th, sigma_th)
    arr_vh  = genseq_vh(T, V, H, mu_vh, sigma_vh)
    arr_tvh = genseq_tvh(T, V, H, mu_tvh, sigma_tvh)
    arr_tot = np.sum([arr_t, arr_v, arr_h, arr_tv, arr_th, arr_vh, arr_tvh], axis=0)
    return arr_t, arr_v, arr_h, arr_tv, arr_th, arr_vh, arr_tvh, arr_tot


def genseq_3dnoise_seq(T, V, H, sigmas, mus=MUS):
    """Generate a 3D T-V-H sequence with all noises with standard-deviation sigmas and means mus.
     The corresponding std and means must be passed in a tuple. Default means are 0 for all noise.
     """
    return genseq_all_seq(T, V, H, sigmas, mus)[-1]
