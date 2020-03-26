import numpy as np

import noise3d
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



class SpectrumDemo():
    
    def __init__(self, T=100, V=100, H=100, sigma=1, zero_mean=False):
        self.T = T
        self.V = V
        self.H = H
        self.sigma = sigma
        seq = noise3d.genseq.genseq_tvh(self.T,
                                        self.V,
                                        self.H,
                                        0,
                                        self.sigma)
        if zero_mean:
            seq = seq - np.mean(seq)
        self.seq = seq
        self._compute_psds()
    def _compute_psds(self):
        self.computed_psds = noise3d.spectrum.compute_psd(self.seq)
        self.psd = np.sum(self.computed_psds, axis=0)
        
        # Compute 3d fft
        self.dft = np.fft.fftn(self.seq, axes=(0, 1, 2))
        # take module2
        self.mod2 = np.real(np.conjugate(self.dft)*self.dft)
        
    def DFT_property_origin_value_vs_sum(self):
        """Compare the sum of the sequence with the square root of
        squared module of Fourier Transform of seq: 
            - sum(raw_seq)
            - np.sqrt(mod2[0, 0, 0])
        with mod2 = real(conj(dft(raw_seq))*dft(raw_seq))
        
        According to DFT properties, the origin value of the mod2 is the squared sum of the input signal : 
        
            mod2(0) = |\sum_{n}x_n e^{0}|^2 = |\sum_{n}x_n|^2
        
        """
        print("Sum of raw_seq : np.sum(seq):")
        print(np.sum(self.seq))
        print("Square root of mod2 of seq dft :")
        print(np.sqrt(self.mod2[0, 0, 0]))
        
    def DFT_property_mean_mod2_vs_sum_of_squared(self):
        """Compare the mean of the module and the sum of the squared :
            - np.sum(seq**2)
            - np.mean(mod2)
        
        According to DFT properties, the mean of the module2 is equal to the sum of the squared input signal.
        """
        print("Sum of squared raw_seq : np.sum(raw_seq**2)")
        print(np.sum(self.seq**2))
        print("Mean of mod2 : ")
        print(np.mean(self.mod2))
        
    def DFT_property_relation_to_variance(self):
        """Compare : 
            - Mean of mod2
            - Sum of squared input
            - N * variance of input
            
        In addition to mean(Mod2) == Sum(seq**2) in all case,
        if the input is 0-mean, it is also equal to N * variance(seq).
        
        Other way to put it : if the signal is 0-mean, 
        the variance is directly the sum of squared signal, 
        which is always the mean of mod2.
        """
        if not np.mean(self.seq)==0:
            raw_seq_0mean = self.seq - np.mean(self.seq)
        else:
            raw_seq_0mean = self.seq
        
        dft_0mean = np.fft.fftn(raw_seq_0mean, axes=(0, 1, 2))
        mod2_0mean = np.real(np.conjugate(dft_0mean)*dft_0mean)
        print("Remove mean value of raw_seq...")
        print("Sum of squared raw_seq : np.sum(raw_seq**2)")
        print(np.sum(raw_seq_0mean**2))
        print("Mean of mod2 : ")
        print(np.mean(mod2_0mean))
        print("N times Variance of seq")
        print(raw_seq_0mean.size * np.var(raw_seq_0mean))
        
    def DFT_property_Parseval(self):
        """
        The mean of the module is the sum of squared signal.
        
        If the signal is 0-mean, the variance can be used, as : 
            var(seq) = 1/N sum x_n^2
        """
        print("Mean of mod2")
        print(np.mean(self.mod2))
        print("Sum of squared signal")
        print(np.sum(self.seq**2))
        print("N times Variance of 0-mean seq")
        print(self.seq.size * np.var(self.seq-np.mean(self.seq)))
        
    def sums_of_psds(self):
        """
        Compare the sum of the DFT with the sum of
        the overall psd computed.
        
        """
        sum1 = np.sum(self.mod2)
        sum2 = np.sum(self.psd)
        print("Sum of mod2")
        print(sum1)
        print("Sum of psd")
        print(sum2)
        print(f"Error : {(sum1/sum2-1)}")
        
    def vars_of_pdsd(self):
        """Compare the variances of the input seq dft mod2 and 
        the recomputed psd. 
        If the sums are equals, then the 
        variance should be as well since the input seq must
        be 0-mean."""
        print(np.var(self.mod2))
        print(np.var(self.psd))
        
    def origin_value_close_to_zero(self):
        if not np.isclose(np.mean(self.seq), 0):
            raise ValueError("The input seq must be 0 mean for this comparison to work. This is because the way the 7 psds are computed first remove the mean.")
        print(self.psd[0, 0, 0])
        print(self.mod2[0, 0,0])

    def histograms(self, bins=100):
        """Overlay the 2 histograms"""
        import matplotlib.pyplot as plt
        plt.hist(self.psd.flatten(), bins=bins,
                 alpha=0.5, edgecolor="k",
                label="recomputed psd")
        plt.hist(self.mod2.flatten(), bins=bins,
                 alpha=0.5, edgecolor="k",
                 label="mod2")
        plt.legend()
        
    def diff_hist(self, bins=100):
        import matplotlib.pyplot as plt
        plt.hist(self.psd.flatten()-self.mod2.flatten(), bins=bins)