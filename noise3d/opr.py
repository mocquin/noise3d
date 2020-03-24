""""""

import numpy as np

from .genseq import NAMES


MEAN_NAMES = ('dvdh','dtdh','dtdv','dh','dv','dt')


# Base operators
def dt(seq):
    """Means the 3D input sequence along the T axis.
    Remember the convention of this package : T on 0 axis, V on 1 axis, H on 2 axis.
    Returns a sequence with same dimension as input."""
    return np.mean(seq, axis=0, dtype=np.float64, keepdims=True)


def dv(seq):
    """Means the 3D input sequence along the V axis.
    Remember the convention of this package : T on 0 axis, V on 1 axis, H on 2 axis.
    Returns a sequence with same dimension as input."""
    return np.mean(seq, axis=1, dtype=np.float64, keepdims=True)


def dh(seq):
    """Means the 3D input sequence along the H axis.
    Remember the convention of this package : T on 0 axis, V on 1 axis, H on 2 axis.
    Returns a sequence with same dimension as input."""
    return np.mean(seq, axis=2, dtype=np.float64, keepdims=True)


def idt(seq):
    """Filters the T noise of the 3D input sequence.
    Returns a sequence with same shape as input, thanks to numpy's broadcast."""
    return seq - dt(seq)


def idv(seq):
    """Filters the V noise of the 3D input sequence.
    Returns a sequence with same shape as input, thanks to numpy's broadcast."""
    return seq - dv(seq)


def idh(seq):
    """Filters the H noise of the 3D input sequence.
    Returns a sequence with same shape as input, thanks to numpy's broadcast."""
    return seq - dh(seq)


# Noise sequence
## linear approach
def n_s(seq):
    """Extract the mean of the 3D input sequence, by applying all 3 base mean-operators.
    Equivalent to np.mean(seq)."""
    return dt(dv(dh(seq)))


def n_t(seq):
    """Extract the temporal t noise sequence of the 3D 
    input sequence, with same shape."""
    return dv(dh(idt(seq)))


def n_v(seq):
    """Extract the vertical v noise sequence of the 3D 
    input sequence, with same shape."""
    return dt(dh(idv(seq)))


def n_h(seq):
    """Extract the horizontal h noise sequence of the 3D 
    input sequence, with same shape."""
    return dv(dt(idh(seq)))


def n_tv(seq):
    """Extract the temporel-vertical tv noise sequence of the 3D 
    input sequence, with same shape."""
    return dh(idv(idt(seq)))


def n_th(seq):
    """Extract the temporal-horizontal th noise sequence of the 3D 
    input sequence, with same shape."""
    return dv(idh(idt(seq)))


def n_vh(seq):
    """Extract the vertical-horizontal vh noise sequence of the 3D 
    input sequence, with same shape."""
    return dt(idv(idh(seq)))


def n_tvh(seq):
    """Extract the vertical temporel-vertical-horizontal noise 
    sequence of the 3D input sequence, with same shape."""
    return idh(idv(idt(seq)))


def get_all_3d_noise_seq(seq, names=False):
    """Extract all 7 noise sequences of the 3D input sequence, with same shape.
    Returns the 7 sequences plus input seq in a tuple."""
    seqs = n_t(seq), n_v(seq), n_h(seq), n_tv(seq), n_th(seq), n_vh(seq), n_tvh(seq), seq
    return seqs + (NAMES, ) if names else seqs


## matrix approach
def n_dt(seq):
    """Applies a temporal mean to the 3D input sequence.
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    return dt(seq)


def n_dv(seq):
    """Applies a vertical mean to the 3D input sequence.
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    return dv(seq)


def n_dh(seq):
    """Applies a horizontal mean to the 3D input sequence.
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    return dh(seq)


def n_dtdv(seq):
    """Applies a temporal-vertical mean to the 3D input sequence.
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    return dt(dv(seq))


def n_dtdh(seq):
    """Applies a temporal-horizontal mean to the 3D input sequence.
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    return dt(dh(seq))


def n_dvdh(seq):
    """Applies a vertical-horizontal mean to the 3D input sequence.
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    return dv(dh(seq))


def get_all_3D_mean_seq(seq, names=False):
    """Returns 6 filtered sequences in the following order :
        dvdh, dtdh, dtdv, dh, dv, dt
    This operator is used in a matrix-approach to 
    compute all 7 noise sequences."""
    seqs = n_dvdh(seq), n_dtdh(seq), n_dtdv(seq), n_dh(seq), n_dv(seq), n_dt(seq)
    return seqs + (MEAN_NAMES, ) if names else seqs


#### Basic extraction of noise sequences
def get_all_3d_noise_seq_fast(seq3d, names=False):
    """Extract all 7 noise sequences of a 3D sequence.
    
    This basic approach consists in filtering noise in some
    directions and extracting it from the base sequence.
    
    Returns all noise sequences in a tuple, as well as the 
    original sequence, in the following order : 
        (t, v, h, tv, th, vh, tvh, input)
        
    This function is called "fast" beacause it limits the 
    number of dt, dv, dh operators calls.
    In most cases, it should lead to quite accurate results.
    """
    # 7 base mean sequences
    seq_dt = dt(seq3d)
    seq_dv = dv(seq3d)
    seq_dh = dh(seq3d)
    seq_dtdv = dv(seq_dt)
    seq_dtdh = dh(seq_dt)
    seq_dvdh = dv(seq_dh)
    # overall mean
    seq_dtdvdh = dh(seq_dtdv)
    # 7 noise sequences
    seq_s = seq_dtdvdh
    seq_t = seq_dvdh - seq_dtdvdh
    seq_v = seq_dtdh - seq_dtdvdh
    seq_h = seq_dtdv - seq_dtdvdh
    seq_tv = seq_dh - seq_dtdh - seq_dvdh + seq_dtdvdh 
    seq_th = seq_dv - seq_dtdv - seq_dvdh + seq_dtdvdh
    seq_vh = seq_dt - seq_dtdv - seq_dtdh + seq_dtdvdh
    seq_tvh = seq3d - (seq_t + seq_v + seq_h + seq_tv + seq_th + seq_vh)
    res = seq_t, seq_v, seq_h, seq_tv, seq_th, seq_vh, seq_tvh, seq3d
    return res.append(NAMES) if names else res


#def var_UBO_t(seq):
#    T, V, H = seq.shape
#    return 1/(T-1)* np.sum(dv(dh(idt(seq)))**2)
#def var_UBO_v(seq):
#    T, V, H = seq.shape
#    return 1/(V-1)* np.sum(dt(dh(idv(seq)))**2)
#def var_UBO_h(seq):
#    T, V, H = seq.shape
#    return 1/(H-1)* np.sum(dt(dv(idh(seq)))**2)
#def var_UBO_tv(seq):
#    T, V, H = seq.shape
#    return 1/(T-1)*1/(V-1) * np.sum(dh(idt(idv(seq)))**2)
#def var_UBO_th(seq):
#    T, V, H = seq.shape
#    return 1/(T-1)*1/(H-1) * np.sum(dv(idt(idh(seq)))**2)
#def var_UBO_vh(seq):
#    T, V, H = seq.shape
#    return 1/(H-1)*1/(V-1) * np.sum(dt(idv(idh(seq)))**2)
#def var_UBO_tvh(seq):
#    T, V, H = seq.shape
#    return 1/(T-1)*1/(V-1)*1/(H-1) * np.sum(idt(idv(idh(seq)))**2)
#
#
#def get_3D_UBO_var(seq):
#    return var_UBO_t(seq), var_UBO_v(seq), var_UBO_h(seq), var_UBO_tv(seq), var_UBO_th(seq), var_UBO_vh(seq), var_UBO_tvh(seq)
#
#
#def get_3D_UBO_corrected_var(seq):
#    T, V, H = seq.shape
#    
#    var_UBO_tvh_val = var_UBO_tvh(seq)
#    
#    var_UBO_vh_val = var_UBO_vh(seq) - 1/T * var_UBO_tvh_val
#    var_UBO_th_val = var_UBO_th(seq) - 1/T * var_UBO_tvh_val
#    var_UBO_tv_val = var_UBO_tv(seq) - 1/T * var_UBO_tvh_val
#    
#    var_UBO_t_val = var_UBO_t(seq) - 1/(V*H)*var_UBO_tvh_val - 1/V * var_UBO_th_val - 1/H * var_UBO_tv_val
#    var_UBO_v_val = var_UBO_v(seq) - 1/(V*T)*var_UBO_tvh_val - 1/V * var_UBO_vh_val - 1/T * var_UBO_tv_val
#    var_UBO_h_val = var_UBO_h(seq) - 1/(H*T)*var_UBO_tvh_val - 1/T * var_UBO_th_val - 1/H * var_UBO_vh_val
#    
#    vec_var = var_UBO_t_val, var_UBO_v_val, var_UBO_h_val, var_UBO_tv_val, var_UBO_th_val, var_UBO_vh_val, var_UBO_tvh_val
#    
#    var_tot = np.sum(vec_var)
#    return vec_var, var_tot