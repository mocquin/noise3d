import numpy as np

from .opr import dt, dv, dh, idt, idv, idh
from .opr import n_s, n_t, n_v, n_h, n_tv, n_th, n_vh, n_tvh
from .opr import n_dt, n_dv, n_dh, n_dtdv, n_dtdh, n_dvdh
from .opr import NAMES


DTYPE = np.float64
DDOF = 1


## classic approach
def var_s(seq, ddof=DDOF):
    """Return the variance of the mean sequence."""
    return np.var(n_s(seq), dtype=DTYPE,  ddof=ddof)


def var_nt(seq, ddof=DDOF):
    """Return the t noise sequence variance."""
    return np.var(n_t(seq), dtype=DTYPE,  ddof=ddof)


def var_nv(seq, ddof=DDOF):
    """Return the v noise sequence variance."""
    return np.var(n_v(seq), dtype=DTYPE,  ddof=ddof)


def var_nh(seq, ddof=DDOF):
    """Return the h noise sequence variance."""
    return np.var(n_h(seq), dtype=DTYPE,  ddof=ddof)


def var_ntv(seq, ddof=DDOF):
    """Return the tv noise sequence variance."""
    return np.var(n_tv(seq), dtype=DTYPE,  ddof=ddof)


def var_nth(seq, ddof=DDOF):
    """Return the th noise sequence variance."""
    return np.var(n_th(seq), dtype=DTYPE,  ddof=ddof)


def var_nvh(seq, ddof=DDOF):
    """Return the vh noise sequence variance."""
    return np.var(n_vh(seq), dtype=DTYPE,  ddof=ddof)


def var_ntvh(seq, ddof=DDOF):
    """Return the tvh noise sequence variance."""
    return np.var(n_tvh(seq), dtype=DTYPE,  ddof=ddof)


def get_all_3d_noise_var_classic(seq, names=False):
    """Return all the 7 noise variance and total variance.
    
    Equivalent to extract all 7 noise sequences and take 
    their variances.
    Returns 8 noise variances (7 3D noise component and 
    total variance) in a tuple.
    If names==True, also returns a tuple of strings 
    containing the variances names.
    """
    res = var_nt(seq), var_nv(seq), var_nh(seq), var_ntv(seq), var_nth(seq), var_nvh(seq), var_ntvh(seq), np.var(seq, dtype=DTYPE, ddof=DDOF)
    return res + (NAMES+('tot', ), ) if names else res


## matrix approach
def var_dt(seq, ddof=DDOF):
    """Return dt noise variance.
    
    Used for the matrix approach."""
    return np.var(n_dt(seq), dtype=DTYPE,  ddof=ddof)


def var_dv(seq, ddof=DDOF):
    """Return dv noise variance.
    
    Used for the matrix approach."""
    return np.var(n_dv(seq), dtype=DTYPE,  ddof=ddof)


def var_dh(seq, ddof=DDOF):
    """Return dh noise variance.
    
    Used for the matrix approach."""
    return np.var(n_dh(seq), dtype=DTYPE,  ddof=ddof)


def var_dtdv(seq, ddof=DDOF):
    """Return dtdv noise variance.
    
    Used for the matrix approach."""
    return np.var(n_dtdv(seq), dtype=DTYPE,  ddof=ddof)


def var_dtdh(seq, ddof=DDOF):
    """Return dtdh noise variance.
    
    Used for the matrix approach."""
    return np.var(n_dtdh(seq), dtype=DTYPE,  ddof=ddof)


def var_dvdh(seq, ddof=DDOF):
    """Return dvdh noise variance.
    
    Used for the matrix approach."""
    return np.var(n_dvdh(seq), dtype=DTYPE,  ddof=ddof)


def var_tot(seq, ddof=DDOF):
    """Return the sequence variance"""
    return np.var(seq, dtype=DTYPE,  ddof=ddof)


def get_all_3d_mean_var(seq, names=False):
    """Return all 7 mean noise variance of the matrix approach.
    
    The 6 mean and the total variance."""
    res = var_dvdh(seq), var_dtdh(seq), var_dtdv(seq), var_dh(seq), var_dv(seq), var_dt(seq), var_tot(seq)
    return res + (MEAN_NAMES + ('tot', ), ) if names else res


## Fast compute
def get_all_3d_noise_var_fast(seq, ddof=DDOF, names=False):
    """Faster and equivalent way of getting the 3D noise variances."""
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
    res = var_nt, var_nv, var_nh, var_ntv, var_nth, var_nvh, var_ntvh, tot
    return res + (NAMES + ("tot", ), ) if names else res
    

# MATRIX APPROACH
# classic mixin matrix
M_classic = np.array([
[ 1,  0,  0,  0,  0,  0,  0],
[ 0,  1,  0,  0,  0,  0,  0],
[ 0,  0,  1,  0,  0,  0,  0],
[ 1,  1,  0,  1,  0,  0,  0],
[ 1,  0,  1,  0,  1,  0,  0],
[ 0,  1,  1,  0,  0,  1,  0],
[ 1,  1,  1,  1,  1,  1,  1],   
])



def compute_M_corrected(T, V, H, VH=None):
    """
    Compute unbiased mixin matrix
    """
    if VH is None:
        VH = V*H
    # if not mask is used, VH should be equal to V*H
    mat = np.array([
    [                1,                   0,                  0,                 1/V,                 1/H,                   0,  1/(VH)],
    [                0,                   1,                  0,                 1/T,                   0,                 1/H, 1/(T*H)],
    [                0,                   0,                  1,                   0,                 1/T,                 1/V, 1/(T*V)],
    [  V*(T-1)/(T*V-1),     T*(V-1)/(T*V-1),                  0,                   1, V*(T-1)/((T*V-1)*H), T*(V-1)/((T*V-1)*H),     1/H],
    [  H*(T-1)/(T*H-1),                   0,    T*(H-1)/(T*H-1), H*(T-1)/((T*H-1)*V),                   1, T*(H-1)/((T*H-1)*V),     1/V],
    [                0,      H*(V-1)/(VH-1),     V*(H-1)/(VH-1),  H*(V-1)/((VH-1)*T),  V*(H-1)/((VH-1)*T),                   1,     1/T],
    [VH*(T-1)/(T*VH-1),  T*H*(V-1)/(T*VH-1), T*V*(H-1)/(T*VH-1),  H*(T*V-1)/(T*VH-1),  V*(T*H-1)/(T*VH-1),   T*(VH-1)/(T*VH-1),       1],
    ], dtype=float)
    return mat


def _get_all_3d_variance_from_matrix(seq, M):
    """
    Generic function to compute noise variances from mean variances.
    This is a helper function for :
     - get_all_3d_classic_var_matrix
     - get_all_3d_corrected_var_matrix
    """
    vec_var_D = np.array(get_all_3d_mean_var(seq))
    M_inv = np.linalg.inv(M)
    vec_var_sigma = tuple(np.matmul(M_inv, vec_var_D))
    return vec_var_sigma + (sum(vec_var_sigma),)
    

def get_all_3d_classic_var_matrix(seq, names=False):
    """
    Compute classic noise variances using the matrix mixin approach.
    """
    res =  _get_all_3d_variance_from_matrix(seq, M_classic)
    return res + (NAMES + ('tot', ), ) if names else res


def get_all_3d_corrected_var_matrix(seq, names=False):
    """
    Compute unbiased noise variances using the matrix mixin approach.
    """
    Tv, Vv, Hv, VHv = get_valid_counts(seq)
    M_corrected = compute_M_corrected(Tv, Vv, Hv, VHv)
    res = _get_all_3d_variance_from_matrix(seq, M_corrected)
    return res + (NAMES + ('tot', ), ) if names else res


def get_noise_vars(seq, method="fast", ddof=DDOF, names=False):
    if method == "fast":
        return get_all_3d_noise_var_fast(seq, ddof=DDOF, names=names)
    elif method == "classic":
        return get_all_3d_noise_var_classic(seq, ddof=DDOF, names=names)
    elif method == "matrix_classic":
        return get_all_3d_classic_var_matrix(seq, names=names)
    elif method == "matrix_corrected":
        return get_all_3d_corrected_var_matrix(seq, names=names)
    else:
        raise NotImplementedError("Method must be among classic, fast, matrix_classic and matrix_corrected")


def get_valid_counts(seq):
    # shapes
    Ts, Vs, Hs = seq.shape 
    # valid counts
    #Tv, Vv, Hv = (np.ma.count(seq, axis=0), np.ma.count(seq, axis=1), np.ma.count(seq, axis=2))
    #print(Tv.shape)
    #print(Vv.shape)
    #print(Hv.shape)
    # valid counts in 2D plane
    # if not mask is used, VH should be equal to V*H
    VHv = np.ma.count(seq[0]) # count the valid values on first image
    return Ts, Vs, Hs, VHv
    

# Classical noise estimators
def var_netd(seq, axis=0, ddof=1):
    """
    Compute the NETD variance the "usual" way.
    """
    return np.mean(np.var(seq, axis=axis, ddof=ddof))


def var_fpn(seq, axis=0, ddof=1):
    """
    Compute the fpn variance the 'usual' way.
    """
    return np.var(np.mean(seq, axis=axis), ddof=ddof)


# For debugging/testing
#def get_all_3d_UBO_var_matrix(seq):
#    """
#    Compute UBO noise variances using the matrix mixin approach.
#    """
#    Tv, Vv, Hv, VHv = get_valid_counts(seq)    
#    M_UBO = compute_M_UBO(Tv, Vv, Hv, VHv)
#    return _get_all_3d_variance_from_matrix(seq, M_UBO)
#
#
#def compute_M_UBO(T, V, H, VH=None):
#    """
#    Compute mixin matrix from paper ####
#    """
#    if VH is None:
#        VH = V*H
#    # if not mask is used, VH should be equal to V*H
#    mat = np.array([
#    [ 1,  0,  0,             1/V,             1/H,              0,                                  1/(VH)],
#    [ 0,  1,  0,             1/T,               0,             1/H,                                 1/(T*H)],
#    [ 0,  0,  1,               0,             1/T,             1/V,                                 1/(T*V)],
#    [ 1,  1,  0, (V+T+T*V)/(T*V),             1/H,             1/H,                   (T + V + T*V)/(T*VH)],
#    [ 1,  0,  1,             1/V, (H+T+T*H)/(T*H),             1/V,                       (H+T+H*T)/(T*VH)],
#    [ 0,  1,  1,             1/T,             1/T, (H+V+H*V)/(VH),                       (H+V+H*V)/(T*VH)],
#    [ 1,  1,  1, (V+T+T*V)/(T*V), (H+T+T*H)/(T*H), (H+V+H*V)/(VH), (H+T+V+ T*V + H*T + VH + T*VH)/(T*VH)]
#    ])
#
#    return mat
