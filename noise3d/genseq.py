import numpy as np

MUS = (0, 0, 0, 0, 0, 0, 0)


def genseq_t(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, T)
    arr = np.repeat(np.repeat(val[:, np.newaxis], V, axis=1)[:, :, np.newaxis], H, axis=2)
    return arr

def genseq_v(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, V)
    arr = np.repeat(np.repeat(val[np.newaxis, :], T, axis=0)[:, :, np.newaxis], H, axis=2)
    return arr

def genseq_h(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, H)
    arr = np.repeat(np.repeat(val[np.newaxis, :], T, axis=0)[:, np.newaxis, :], V, axis=1)
    return arr
    
def genseq_tv(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, T*V)
    val = np.reshape(val, (T, V))
    arr = np.repeat(val[:, :, np.newaxis], H, axis=2)
    return arr

def genseq_th(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, T*H)
    val = np.reshape(val, (T, H))
    arr = np.repeat(val[:, np.newaxis, :], V, axis=1)
    return arr

def genseq_vh(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, V*H)
    val = np.reshape(val, (V, H))
    arr = np.repeat(val[np.newaxis, :, :], T, axis=0)
    return arr

def genseq_tvh(T, V, H, mu, sigma):
    val = np.random.normal(mu, sigma, T*V*H)
    arr = np.reshape(val, (T,V,H))
    return arr


def genseq_all_seq(T, V, H, sigmas, mus=MUS):
    
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
    return genseq_all_seq(T, V, H, sigmas, mus)[-1]