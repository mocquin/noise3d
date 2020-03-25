import numpy as np

from . import noise
from . import genseq

def var_t_err_tvh(T, V, H, var):
    """Compute the expected variance of t noise for a tvh sequence."""
    return var/(H*V)*(1 - 1/T)

def var_v_err_tvh(T, V, H, var):
    """Compute the expected variance of v noise for a tvh sequence."""
    return var/(H*T)*(1 - 1/V)


def var_h_err_tvh(T, V, H, var):
    """Compute the expected variance of h noise for a tvh sequence."""
    return var/(T*V)*(1 - 1/H)


def var_tv_err_tvh(T, V, H, var):
    """Compute the expected variance of tv noise for a tvh sequence."""
    return var/H*(1 - 1/T - 1/V + 1/(T*V))


def var_th_err_tvh(T, V, H, var):
    """Compute the expected variance of th noise for a tvh sequence."""
    return var/V*(1 - 1/T - 1/H + 1/(T*H))


def var_vh_err_tvh(T, V, H, var):
    """Compute the expected variance of vh noise for a tvh sequence."""
    return var/T*(1 - 1/V - 1/H + 1/(V*H))


def var_tvh_err_tvh(T, V, H, var):
    """Compute the expected variance of tvh noise for a tvh sequence."""
    return var*(1 - 1/T - 1/V - 1/H + 1/(T*V) + 1/(T*H) + 1/(V*H) - 1/(T*V*H))


class BiasedEstimatorTVH(object):
    """This class's purpose is to demonstrate the biasness of 
    the standard estimators Works only on tvh sequence.
    """
    def __init__(self, V=100, H=100, nb_matrix=100, sigma_tvh=1):
        
        self.t_min = 10
        self.t_max = 300
        self.delta_t = 5
        self.V = V
        self.H = H
        
        # Nb of iteration to average to improve estimation ("MonteCarlo")
        self.nb_matrix = nb_matrix
        
        # We choose a tvh sequence with std of 1
        self.sigma_tvh = sigma_tvh
    
    def setup(self):
        
        self.ech_t = np.arange(self.t_min, self.t_max, self.delta_t)
        
        # Setting up stack results
        self.res_var_tot = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_t = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_v = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_h = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_tv = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_th = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_vh = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)
        self.res_var_tvh = np.zeros((self.nb_matrix, len(self.ech_t)), dtype=np.float64)

    def compute_results(self):
        
        # "Monte Carlo" loop
        for k in range(1, self.nb_matrix+1):
            i = k-1
            # generate samples
            data_total = genseq.genseq_tvh(self.t_max, self.V, self.H, 0, self.sigma_tvh)
            j = 0
            # For all T-length
            for t in self.ech_t:
                # sub-sampling data on T axis
                data = data_total[0:t]
                # Compute the variances with given samples
                vt, vv, vh, vtv, vth, vvh, vtvh, vtot  = noise.get_all_3d_noise_var_fast(data)
                # Store results
                self.res_var_t[i, j] = vt
                self.res_var_v[i, j] = vv
                self.res_var_h[i, j] = vh
                self.res_var_tv[i, j] = vtv
                self.res_var_th[i, j] = vth
                self.res_var_vh[i, j] = vvh
                self.res_var_tvh[i, j] = vtvh
                self.res_var_tot[i, j] = vtot
                j = j+1

        # at this point, all data is stored in the res_var_ arrays.
        # Averaging results over all Monte Carlos
        self.mean_var_tot = np.mean(self.res_var_tot, axis=0, dtype=np.float64)
        self.mean_var_t = np.mean(self.res_var_t, axis=0, dtype=np.float64)
        self.mean_var_v = np.mean(self.res_var_v, axis=0, dtype=np.float64)
        self.mean_var_h = np.mean(self.res_var_h, axis=0, dtype=np.float64)
        self.mean_var_tv = np.mean(self.res_var_tv, axis=0, dtype=np.float64)
        self.mean_var_th = np.mean(self.res_var_th, axis=0, dtype=np.float64)
        self.mean_var_vh = np.mean(self.res_var_vh, axis=0, dtype=np.float64)
        self.mean_var_tvh = np.mean(self.res_var_tvh, axis=0, dtype=np.float64)
        
    def display_results(self):
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 4, figsize=(14,6))
        # Computed variances labels
        var_labels = ('var_tot', 
                      'var_t', 'var_v', 'var_h',
                      'var_tv', 'var_th', 'var_vh',
                      "var_tvh")
        # Error labels
        error_labels = ("", 
                        'var_t_err_tvh', 'var_v_err_tvh', "var_h_err_tvh",
                        'var_tv_err_tvh', 'var_th_err_tvh', 'var_vh_err_tvh',
                        'var_tvh_err_tvh')
        # Computed variances values
        res = (self.mean_var_tot,
               self.mean_var_t, self.mean_var_v, self.mean_var_h,
               self.mean_var_tv, self.mean_var_th, self.mean_var_vh,
               self.mean_var_tvh)
        # Error functions
        error_funcs = (None,
                       var_t_err_tvh, var_v_err_tvh, var_h_err_tvh, 
                       var_tv_err_tvh, var_th_err_tvh, var_vh_err_tvh,
                       var_tvh_err_tvh)
        
        
        # Plotting 
        for ax, var_label, error_label, res_var, err_func in zip(axes.flatten(),
                                                                   var_labels,
                                                                   error_labels,
                                                                   res, 
                                                                   error_funcs):
            ax.plot(self.ech_t, res_var, label=var_label)
            if var_label != "var_tot":
                # no error function to plot on total variance
                ax.plot(self.ech_t, err_func(self.ech_t, self.V, self.H, self.sigma_tvh**2), "o", label=error_label)
            ax.legend()

        plt.tight_layout()
