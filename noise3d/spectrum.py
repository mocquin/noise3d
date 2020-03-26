import numpy as np

from .opr import dt, dv, dh, idt, idv, idh, NAMES
from .opr import n_s, n_t, n_v, n_h, n_tv, n_th, n_vh, n_tvh
from .opr import n_dt, n_dv, n_dh, n_dtdv, n_dtdh, n_dvdh


def _compute_spectrum_correction_matrix(T, V, H):
    """
    Compute spectrum mixin correction matrix
    """
    mat = 1/(T*V*H) * np.array(
        [
            [V*H*(T-1), 0, 0, H*(T-1), V*(T-1), 0, T-1],
            [0, T*H*(V-1), 0, H*(V-1), 0, T*(V-1), V-1],
            [0, 0, T*V*(H-1), 0, V*(H-1), T*(H-1), H-1],
            [0, 0, 0, H*(T-1)*(V-1), 0, 0, (T-1)*(V-1)],
            [0, 0, 0, 0, V*(T-1)*(H-1), 0, (T-1)*(H-1)],
            [0, 0, 0, 0, 0, T*(V-1)*(H-1), (V-1)*(H-1)],
            [0, 0, 0, 0, 0, 0, (T-1)*(V-1)*(H-1)],
        ]
    )
    return mat


def compute_meas_psd(seq):
    """
    Compute measured psd.
    Used for matrix approach.
    """
    T, V, H = seq.shape
    seq = seq - np.mean(seq)
    data_fft = np.fft.fftn(seq, axes=(0,1,2))
    data_fft_mod2 = np.real(data_fft * np.conjugate(data_fft))

    psd_tm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)
    psd_vm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)
    psd_hm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)
    psd_tvm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)
    psd_thm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)
    psd_vhm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)
    psd_tvhm = np.zeros(data_fft_mod2.shape)#, dtype=np.complex)

    psd_tm[:, 0, 0] = data_fft_mod2[:, 0, 0]
    psd_vm[0, :, 0] = data_fft_mod2[0, :, 0]
    psd_hm[0, 0, :] = data_fft_mod2[0, 0, :]
    psd_tvm[:, :, 0] = data_fft_mod2[:, :, 0] 
    psd_tvm = psd_tvm - psd_tm - psd_vm
    psd_thm[:, 0, :] = data_fft_mod2[:, 0, :]
    psd_thm = psd_thm - psd_tm - psd_hm
    psd_vhm[0, :, :] = data_fft_mod2[0, :, :]
    psd_vhm = psd_vhm - psd_vm - psd_hm
    psd_tvhm = data_fft_mod2 - (psd_tm + psd_vm + psd_hm + psd_tvm + psd_thm + psd_vhm)

    return psd_tm, psd_vm, psd_hm, psd_tvm, psd_thm, psd_vhm, psd_tvhm


def compute_psd(seq, names=False):
    """
    Compute noises psd.
    Relies on compute_meas_psd.
    """
    T, V, H = seq.shape
    correction_mat = _compute_spectrum_correction_matrix(T, V, H)
    inv_mat = np.linalg.inv(correction_mat)
    psd_m = np.asarray(compute_meas_psd(seq))
    res = tuple(np.einsum("ij,jklm->iklm", inv_mat, psd_m))
    return res + (NAMES, ) if names else res


def compute_var(seq, names=False):
    """
    Compute noise variances using spectrums.
    Using the matrix mixin approach.
    Relis on compute_psd.
    """
    T, V, H = seq.shape
    vec_psd = compute_psd(seq)
    vec_var = np.sum(vec_psd, axis=(1,2,3))/(T*V*H)**2
    t, v, h, tv, th, vh, tvh = vec_var
    res = t, v, h, tv, th, vh, tvh, np.sum(vec_var) # **2 : 1 DFT conventon, 1 to go into power units
    return res + (NAMES + ("tot",), ) if names else res


def var_psd_astrid(seq, numeric_invert=False, names=False):
    """Stand-alone method to compute bias corrected variances.
    
    Gives same values as corrected matrix method.
    
    numeric_invert allows to numericaly invert the mixing matrix. Faster if false.
    
    """
    #seq = seq - np.mean(seq)
    # Linear system matrix between sums (Es) and variances
    T, V, H = seq.shape
    
    # FT and norm of sequence
    psd_seq = np.fft.fftn(seq, axes=(0,1,2))
    norm_psd = np.multiply(psd_seq, np.conjugate(psd_seq))
    if not np.all(np.isreal(norm_psd)):
        raise ValueError("norm should be real")
    norm_psd = norm_psd.astype(np.float128, copy=False)
    
    # Generate indexes of cube sequence
    vt = np.linspace(0, T-1, T)
    vv = np.linspace(0, V-1, V)
    vh = np.linspace(0, H-1, H)
    mt, mv, mh  = np.meshgrid(vt, vv, vh, indexing="ij")
    
    # Sum over specific 3D region
    E3 = np.sum(norm_psd[np.logical_and(np.logical_and(mt!=0, mv!=0), mh!=0)])
    E4 = np.sum(norm_psd[np.logical_and(np.logical_and(mt==0, mv!=0), mh!=0)])
    E5 = np.sum(norm_psd[np.logical_and(np.logical_and(mt!=0, mv==0), mh!=0)])
    E6 = np.sum(norm_psd[np.logical_and(np.logical_and(mt!=0, mv!=0), mh==0)])
    E7 = np.sum(norm_psd[np.logical_and(np.logical_and(mt!=0, mv==0), mh==0)])
    E8 = np.sum(norm_psd[np.logical_and(np.logical_and(mt==0, mv!=0), mh==0)])
    E9 = np.sum(norm_psd[np.logical_and(np.logical_and(mt==0, mv==0), mh!=0)])

    # Solve linear system for variances
    Ev = np.array([E3, E4, E5, E6, E7, E8, E9])
    
    if numeric_invert:
        mat = (T*V*H) * np.array(
            [
            [0,     0,      0,      0,      0,      0,      (T-1)*(V-1)*(H-1)],
            [0,     0,      0,      0,      0,      T*(V-1)*(H-1),(V-1)*(H-1)],
            [0,     0,      0,      0,      V*(T-1)*(H-1),      0,      (T-1)*(H-1)],
            [0,     0,      0,      H*(T-1)*(V-1),      0,      0,      (T-1)*(V-1)],
            [(T-1)*(V)*(H),     0,      0,      (T-1)*H,      (T-1)*V,      0,      (T-1)],
            [0,     (V-1)*(T)*(H),      0,      (V-1)*H,      0,      (V-1)*T,      (V-1)],
            [0,     0,      (H-1)*(T)*(V),      0,      (H-1)*V,      (H-1)*T,      (H-1)]
            ]
        )
        var_psd = np.matmul(np.linalg.inv(mat), Ev)
    else:
        inv_mat = np.array(
            [
                [1/(H**2*T*V**2*(H - 1)*(T - 1)*(V - 1)), 0, -1/(H**2*T*V**2*(H - 1)*(T - 1)), -1/(H**2*T*V**2*(T - 1)*(V - 1)), 1/(H**2*T*V**2*(T - 1)), 0, 0],
                [1/(H**2*T**2*V*(H - 1)*(T - 1)*(V - 1)), -1/(H**2*T**2*V*(H - 1)*(V - 1)), 0, -1/(H**2*T**2*V*(T - 1)*(V - 1)), 0, 1/(H**2*T**2*V*(V - 1)), 0],
                [1/(H*T**2*V**2*(H - 1)*(T - 1)*(V - 1)), -1/(H*T**2*V**2*(H - 1)*(V - 1)), -1/(H*T**2*V**2*(H - 1)*(T - 1)), 0, 0, 0, 1/(H*T**2*V**2*(H - 1))],
                [  -1/(H**2*T*V*(H - 1)*(T - 1)*(V - 1)), 0, 0, 1/(H**2*T*V*(T - 1)*(V - 1)), 0, 0, 0],
                [  -1/(H*T*V**2*(H - 1)*(T - 1)*(V - 1)), 0, 1/(H*T*V**2*(H - 1)*(T - 1)), 0, 0, 0, 0],
                [  -1/(H*T**2*V*(H - 1)*(T - 1)*(V - 1)), 1/(H*T**2*V*(H - 1)*(V - 1)), 0, 0, 0, 0,0],
                [      1/(H*T*V*(H - 1)*(T - 1)*(V - 1)), 0, 0, 0, 0, 0, 0]
            ]
            )
        var_psd = np.matmul(inv_mat, Ev)

    t, v, h, tv, th, vh, tvh = var_psd
    res = t, v, h, tv, th, vh, tvh, np.sum(var_psd)
    return res + (NAMES + ("tot",), ) if names else res
