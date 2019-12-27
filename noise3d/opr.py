import numpy as np

# opérateurs
def dt(seq): return np.mean(seq, axis=0, dtype=np.float64, keepdims=True)
def dv(seq): return np.mean(seq, axis=1, dtype=np.float64, keepdims=True)
def dh(seq): return np.mean(seq, axis=2, dtype=np.float64, keepdims=True)

def idt(seq): return seq - dt(seq)
def idh(seq): return seq - dh(seq)
def idv(seq): return seq - dv(seq)

# Séquences de bruits
## linear approach
def n_s(seq): return dt(dv(dh(seq)))
def n_t(seq): return dv(dh(idt(seq)))
def n_v(seq): return dt(dh(idv(seq)))
def n_h(seq): return dv(dt(idh(seq)))
def n_tv(seq): return dh(idv(idt(seq)))
def n_th(seq): return dv(idh(idt(seq)))
def n_vh(seq): return dt(idv(idh(seq)))
def n_tvh(seq): return idh(idv(idt(seq)))

def get_all_3d_noise_seq(seq):
    return n_t(seq), n_v(seq), n_h(seq), n_tv(seq), n_th(seq), n_vh(seq), n_tvh(seq), seq

## matrix approach
def n_dt(seq): return dt(seq)
def n_dv(seq): return dv(seq)
def n_dh(seq): return dh(seq)
def n_dtdv(seq): return dt(dv(seq))
def n_dtdh(seq): return dt(dh(seq))
def n_dvdh(seq): return dv(dh(seq))

def get_all_3D_mean_seq(seq):
    return n_dvdh(seq), n_dtdh(seq), n_dtdv(seq), n_dh(seq), n_dv(seq), n_dt(seq)



#### Pour calculer les séquences de bruits
def get_all_3d_noise_seq_fast(seq):
    # 7 images de base
    seq_dt = dt(seq)
    seq_dv = dv(seq)
    seq_dh = dh(seq)
    seq_dtdv = dv(seq_dt)
    seq_dtdh = dh(seq_dt)
    seq_dvdh = dv(seq_dh)
    # moyenne
    seq_dtdvdh = dh(seq_dtdv)
    
    # Pour calculs coefs bruit
    seq_s = seq_dtdvdh
    seq_t = seq_dvdh - seq_dtdvdh
    seq_v = seq_dtdh - seq_dtdvdh
    seq_h = seq_dtdv - seq_dtdvdh
    seq_tv = seq_dh - seq_t - seq_dtdvdh 
    seq_th = seq_dv - seq_t - seq_dtdvdh
    seq_vh = seq_dt - seq_v - seq_dtdvdh
    seq_tvh = imgSeq - (seq_t + seq_v + seq_h + seq_tv + seq_th + seq_vh)
    
    vec_seq = seq_s, seq_t, seq_v, seq_h, seq_tv, seq_th, seq_vh, seq_tvh
    
    return vec_seq


   



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