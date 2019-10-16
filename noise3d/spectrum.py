import numpy as np

from .opr import dt, dv, dh, idt, idv, idh
from .opr import n_s, n_t, n_v, n_h, n_tv, n_th, n_vh, n_tvh
from .opr import n_dt, n_dv, n_dh, n_dtdv, n_dtdh, n_dvdh


def compute_spectrum_correction_matrix(T, V, H):
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



#import sympy as sp
#
#T, V, H = sp.symbols("T V H")
#
#sp_mat = 1/(T*V*H)*sp.Matrix([
#        [V*H*(T-1), 0, 0, H*(T-1), V*(T-1), 0, T-1],
#        [0, T*H*(V-1), 0, H*(V-1), 0, T*(V-1), V-1],
#        [0, 0, T*V*(H-1), 0, V*(H-1), T*(H-1), H-1],
#        [0, 0, 0, H*(T-1)*(V-1), 0, 0, (T-1)*(V-1)],
#        [0, 0, 0, 0, V*(T-1)*(H-1), 0, (T-1)*(H-1)],
#        [0, 0, 0, 0, 0, T*(V-1)*(H-1), (V-1)*(H-1)],
#        [0, 0, 0, 0, 0,             0, (T-1)*(V-1)*(H-1)],
#    ])
##
#inv = sp_mat.inv()
#print(inv)



def dft_noise(seq):
    return np.fft.fft(seq)


def compute_meas_psd(seq):
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



def compute_psd(ns):
    T, V, H = ns.shape
    correction_mat = compute_spectrum_correction_matrix(T, V, H)
    inv_mat = np.linalg.inv(correction_mat)
    #print(inv_mat.shape)
    psd_m = np.asarray(compute_meas_psd(ns))
    #print(all_spectrum_meas.shape)
    vec = np.einsum("ij,jklm->iklm", inv_mat, psd_m)
    # verifs : print(psd_m[-1] == (T-1)*(V-1)*(H-1)/(T*V*H)*vec[-1])
    return vec


def compute_var_m(seq):
    T, V, H = seq.shape
    vec_psd_m = np.asarray(compute_meas_psd(seq))
    return np.sum(vec_psd_m, axis=(1,2,3))/(T*V*H)**2 # **2 : 1 pour la convention DFT, 1 pour passer en puissance

def compute_var(seq):
    T, V, H = seq.shape
    vec_psd  = compute_psd(seq)
    return np.sum(vec_psd, axis=(1,2,3))/(T*V*H)**2 # **2 : 1 pour la convention DFT, 1 pour passer en puissance
