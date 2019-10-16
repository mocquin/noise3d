import numpy as np

from .opr import dt, dv, dh, idt, idv, idh
from .opr import n_s, n_t, n_v, n_h, n_tv, n_th, n_vh, n_tvh
from .opr import n_dt, n_dv, n_dh, n_dtdv, n_dtdh, n_dvdh

def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)



DTYPE = np.float64
DDOF = 1

## classic approach
def var_s(seq): return np.var(n_s(seq), dtype=DTYPE,  ddof=DDOF)
def var_nt(seq): return np.var(n_t(seq), dtype=DTYPE,  ddof=DDOF)
def var_nv(seq): return np.var(n_v(seq), dtype=DTYPE,  ddof=DDOF)
def var_nh(seq): return np.var(n_h(seq), dtype=DTYPE,  ddof=DDOF)
def var_ntv(seq): return np.var(n_tv(seq), dtype=DTYPE,  ddof=DDOF)
def var_nth(seq): return np.var(n_th(seq), dtype=DTYPE,  ddof=DDOF)
def var_nvh(seq): return np.var(n_vh(seq), dtype=DTYPE,  ddof=DDOF)
def var_ntvh(seq): return np.var(n_tvh(seq), dtype=DTYPE,  ddof=DDOF)

def get_all_3D_noise_var(seq):
    return (var_nt(seq), var_nv(seq), var_nh(seq), var_ntv(seq), var_nth(seq), var_nvh(seq), var_ntvh(seq)), np.var(seq, dtype=DTYPE, ddof=DDOF)


## matrix approach
def var_dt(seq): return np.var(n_dt(seq), dtype=DTYPE,  ddof=DDOF)
def var_dv(seq): return np.var(n_dv(seq), dtype=DTYPE,  ddof=DDOF)
def var_dh(seq): return np.var(n_dh(seq), dtype=DTYPE,  ddof=DDOF)
def var_dtdv(seq): return np.var(n_dtdv(seq), dtype=DTYPE,  ddof=DDOF)
def var_dtdh(seq): return np.var(n_dtdh(seq), dtype=DTYPE,  ddof=DDOF)
def var_dvdh(seq): return np.var(n_dvdh(seq), dtype=DTYPE,  ddof=DDOF)
def var_tot(seq): return np.var(seq, dtype=DTYPE,  ddof=DDOF)

def get_all_3d_mean_var(seq):
    return var_dvdh(seq), var_dtdh(seq), var_dtdv(seq), var_dh(seq), var_dv(seq), var_dt(seq), var_tot(seq)

## Fast compute
def get_all_3d_noise_var_fast(seq, ddof=DDOF):
    # 7 images de base
    seq_dt = dt(seq)
    seq_dv = dv(seq)
    seq_dh = dh(seq)
    seq_dtdv = dv(seq_dt)
    seq_dtdh = dh(seq_dt)
    seq_dvdh = dv(seq_dh)
    
    var_nt = np.var(seq_dvdh, dtype=DTYPE, ddof=ddof)
    var_nv = np.var(seq_dtdh, dtype=DTYPE, ddof=ddof)
    var_nh = np.var(seq_dtdv, dtype=DTYPE, ddof=ddof)
    
    var_nth = np.var(seq_dv - seq_dtdv - seq_dvdh, dtype=DTYPE, ddof=ddof)
    var_ntv = np.var(seq_dh - seq_dvdh - seq_dtdh, dtype=DTYPE, ddof=ddof)
    var_nvh = np.var(seq_dt - seq_dtdv - seq_dtdh, dtype=DTYPE, ddof=ddof)
    
    tot = np.var(seq, dtype=DTYPE, ddof=ddof)
    var_ntvh = tot - (var_nt + var_nv + var_nh + var_ntv + var_nth + var_nvh)
    
    return (var_nt, var_nv, var_nh, var_ntv, var_nth, var_nvh, var_ntvh), tot
    



M_classic = np.array([
[ 1,  0,  0,  0,  0,  0,  0],
[ 0,  1,  0,  0,  0,  0,  0],
[ 0,  0,  1,  0,  0,  0,  0],
[ 1,  1,  0,  1,  0,  0,  0],
[ 1,  0,  1,  0,  1,  0,  0],
[ 0,  1,  1,  0,  0,  1,  0],
[ 1,  1,  1,  1,  1,  1,  1],   
])


def compute_M_UBO(T, V, H):
    mat = np.array([
    [ 1,  0,  0,             1/V,             1/H,              0,                                  1/(V*H)],
    [ 0,  1,  0,             1/T,               0,             1/H,                                 1/(T*H)],
    [ 0,  0,  1,               0,             1/T,             1/V,                                 1/(T*V)],
    [ 1,  1,  0, (V+T+T*V)/(T*V),             1/H,             1/H,                   (T + V + T*V)/(T*V*H)],
    [ 1,  0,  1,             1/V, (H+T+T*H)/(T*H),             1/V,                       (H+T+H*T)/(T*V*H)],
    [ 0,  1,  1,             1/T,             1/T, (H+V+H*V)/(V*H),                       (H+V+H*V)/(T*V*H)],
    [ 1,  1,  1, (V+T+T*V)/(T*V), (H+T+T*H)/(T*H), (H+V+H*V)/(V*H), (H+T+V+T*V + H*T + V*H + T*V*H)/(T*V*H)]
    ])

    return mat


def compute_M_corrected(T, V, H):
    mat = np.array([
    [                   1,                   0,                   0,                  1/V,                 1/H,                    0,  1/(V*H)],
    [                   0,                   1,                   0,                  1/T,                   0,                  1/H,  1/(T*H)],
    [                   0,                   0,                   1,                    0,                 1/T,                  1/V,  1/(T*V)],
    [     V*(T-1)/(T*V-1),     T*(V-1)/(T*V-1),                   0,                    1, V*(T-1)/((T*V-1)*H),  T*(V-1)/((T*V-1)*H),      1/H],
    [     H*(T-1)/(T*H-1),                   0,     T*(H-1)/(T*H-1),  H*(T-1)/((T*H-1)*V),                   1,  T*(H-1)/((T*H-1)*V),      1/V],
    [                   0,     H*(V-1)/(V*H-1),     V*(H-1)/(V*H-1),  H*(V-1)/((V*H-1)*T), V*(H-1)/((V*H-1)*T),                    1,      1/T],
    [ V*H*(T-1)/(T*V*H-1), T*H*(V-1)/(T*V*H-1), T*V*(H-1)/(T*V*H-1),  H*(T*V-1)/(T*V*H-1), V*(T*H-1)/(T*V*H-1),  T*(V*H-1)/(T*V*H-1),        1],
    ])
    return mat


# General function
def _get_all_3d_variance_from_matrix(seq, M):
    vec_var_D = np.array(get_all_3d_mean_var(seq))
    
    T, V, H = seq.shape
    M_inv = np.linalg.inv(M)
    
    vec_var_sigma = np.matmul(M_inv, vec_var_D)
    return tuple(vec_var_sigma), sum(vec_var_sigma)
    

# With classic matrix
def get_all_3d_classic_var_matrix(seq):
    return _get_all_3d_variance_from_matrix(seq, M_classic)

# With UBO
def get_all_3d_UBO_var_matrix(seq):
    T, V, H = seq.shape
    M_UBO = compute_M_UBO(T, V, H)
    return _get_all_3d_variance_from_matrix(seq, M_UBO)

# With corrected matrix
def get_all_3d_corrected_var_matrix(seq):
    T, V, H = seq.shape
    M_corrected = compute_M_corrected(T, V, H)
    return _get_all_3d_variance_from_matrix(seq, M_corrected)


# NETD comparison
def var_netd(seq, axis=0, ddof=1):
    """2D Spatial mean of 1D temporel variance"""
    # prendre ddof=1 pour estimateur non biaisé
    return np.mean(np.var(seq, axis=axis, ddof=ddof))

def std_netd(seq, axis=0, ddof=1):
    return np.mean(np.std(seq, axis=axis, ddof=ddof))

def var_fpn(seq, axis=0, ddof=1):
    return np.var(np.mean(seq, axis=axis), ddof=ddof)


def compare_netd_tvh(seq, v=True):
    """Compares the classic NETD and the tvh noise."""
    netd = var_netd(seq)
    tvh = var_ntvh(seq) 
    if v:
        print("Classic NETD : {}".format(netd))
        print("TVH noise : {}".format(tvh))
    return netd, tvh