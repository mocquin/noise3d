import numpy as np

import matplotlib.pyplot as plt

from . import noise
from . import opr
from . import spectrum as ns

import scipy.stats as st

ORDER = [-1, 1, 2, 5, 0, 3, 4, 6]
TITLES = ["tot", "v", "h", "vh", "t", "tv", "th", "tvh"]


def analyse_normal_seq(data, mu_X=None, sigma_X=None, law_X="normal", alpha=0.05, print_CI=False, ddof=1):

    d_count = data.size
    n = d_count
    d_min = np.min(data)
    d_max = np.max(data)
    d_median = np.median(data)
    d_mean = np.mean(data)
    d_var = np.var(data, ddof=1)
    d_std = np.std(data, ddof=1)
    
    
    # Moyenne
    ## Si X suit une loi normale
    if law_X=="normal":
        # Dans tous les cas, Xbar suit une loi symétrique
        alpha_Xbar_2 = alpha/2

        # Si on connait la variance de X
        if sigma_X is not None:
            # On calcule l'écart type de la loi de Xbar
            sigma_Xbar = sigma_X/np.sqrt(n)
            
            # Y=(Xbar-mu)/sigma_xbar suit alors une loi normale centrée réduite
            
            # On calcul la positions des bornes pour une loi normale centrée 
            # réduite Y telle que l'aire entre entre les bornes soit de proba_Xbar%

            # On utilise la percent point function, telle que ppf(proba) renvoi la borne max 
            zscore_CI_p = st.norm.ppf(1 - alpha_Xbar_2)
            # Comme on a une loi normale centrée réduite, la loi est symmétrique donc
            zscore_CI_m = - zscore_CI_p
        
            # On traduit les bornes en terme de X que mu soit dans Xbar +- sigma_Xbar à proba_Xbar %
            ci_Xbar_p = zscore_CI_p * sigma_xbar
            ci_Xbar_m = zscore_CI_m * sigma_xbar
            
            # Au final, on sait que mu est dans l'interval Xbar+-ci à proba_Xbar%
            if print_CI:
                print("\sigma_X known, using z-score to compute CI for Xmean")
                print("\mu is within {}+/-{} at {}%".format(d_mean, ci_Xbar_p, alpha))
            
        # Si on ne connait pas sigma_X
        if sigma_X is None:
            # On doit estimer sigma_X, et alors Xbar suit une loi de student t
            s_X = np.std(data, ddof=ddof)
            s_Xbar = s_X/np.sqrt(n)

            df_Xbar = n-1
            tscore_CI_p = st.t.ppf(1 - alpha_Xbar_2, df=df_Xbar)
            # Comme on a une loi de student, elle est symétrique
            tscore_CI_m = - tscore_CI_p
            
            # On traduit les bornes en terme de X que mu soit dans Xbar +- sigma_Xbar à proba_Xbar %
            ci_Xbar_p = tscore_CI_p * s_Xbar
            ci_Xbar_m = tscore_CI_m * s_Xbar
            if print_CI:
                print("\sigma_{X} unknow, estimating it and using Student's t distribution for Xmean")
                print("\mu is within {}+-{} at {}%".format(d_mean, ci_Xbar_p, alpha))
    

    
    # Variance
    ## Suit une loi du Ki2 à df=n-1 dof
    if law_X=="normal":
        # Si on connait la moyenne de X
        if mu_X is not None:
            # s2 suit une loi du ki2 à N degrés
            s2 = np.mean(np.abs(data - mu_X)**2) * n/(n-ddof)
            
            # On calcule les bornes de la variable réduite
            kiscore_CI_var_p = st.chi2.ppf(1 - alpha, df=n)
            kiscore_CI_var_m = st.chi2.ppf(alpha, df=n)
            
            # On traduit en terme de variance
            ci_var_p = s2 *(n-ddof) / kiscore_CI_var_m
            ci_var_m = s2 *(n-ddof) / kiscore_CI_var_p
            if print_CI:
                print("\mu_X known, using chi2 score with df=n to compute CI for var")
                print("Var_x is wihtin {}+-{}/{} at {}%".format(s2, ci_var_m, ci_var_p, alpha))

            
        # Si on ne connait pas la moyenne de X
        if mu_X is None:
            # s2 suit une loi du ki2 à N-1 degrés
            s2 = np.var(data, ddof=ddof)
            
            # On calcule les bornes de la variable réduite
            kiscore_CI_var_p = st.chi2.ppf(1 - alpha, df=n)
            kiscore_CI_var_m = st.chi2.ppf(alpha, df=n)
            
            # On traduit en terme de variance
            ci_var_p = s2 *(n-ddof) / kiscore_CI_var_m
            ci_var_m = s2 *(n-ddof) / kiscore_CI_var_p
            if print_CI:
                print("\mu_X not known, using chi2 score with df=n-1 to compute CI for var")
                print("Var_x is wihtin {}+-{}/{} at {}%".format(s2, ci_var_m, ci_var_p, alpha))
        
    results = {
        "min":d_min,
        "max":d_max,
        "median":d_median,
        "mean":d_mean,
        "var":d_var,
        "std":d_std,
        "count":d_count,
        "ci_Xbar_m":ci_Xbar_m,
        "ci_Xbar_p":ci_Xbar_p,
        "ci_var_m":ci_var_m,
        "ci_var_p":ci_var_p,
    }
    return results

    # visual 
    ## histogram
    ## qqplot
    ## ppplot



def histo_seq(ax, seq, fit=True, stats_on=True, stats_print=False, print_CI=False, nbin=10, density=True):
    vals = seq.flatten()
    ech = np.linspace(np.min(seq), np.max(seq), nbin)
    val, bins, patchs = ax.hist(vals, bins=ech, density=density)
    
    mean = np.mean(vals)
    std = np.std(vals)
    
    if fit:
        ech_gauss = noise.gauss(ech, mean, std)
        ax.plot(ech, ech_gauss)

    if stats_print:
        
        results = analyse_normal_seq(seq, print_CI=print_CI)
        print("Count : {}".format(results["count"]))
        print("Min/Max : {}/{}".format(results["min"], results["max"]))
        print("Mean : {}".format(results["mean"]))
        print("Median : {}".format(results["median"]))
        print("Var : {}".format(results["var"]))
        print("Std : {}".format(results["std"]))

    if stats_on:
        results = analyse_normal_seq(seq, print_CI=print_CI)
        ax.errorbar(mean, max(val)/2, yerr=None, xerr=std, solid_capstyle='projecting', capsize=5)
        text = "Count : {}\nMin/Max : {:.3f}/{:.3f}\nMean : {:.3f}\nMedian : {:.3f}\nVar : {:.3f}\nStd : {:.3f}".format(
            results["count"], 
            results["min"], 
            results["max"],
            results["mean"],
            results["median"],
            results["var"],
            results["std"])
        
        ax.annotate(text, xy=(0.99, 0.99), xycoords='axes fraction', xytext=(0.99, 0.99), textcoords='axes fraction',
                   horizontalalignment='right', verticalalignment='top')
    return ax


def display_7seq(seqs, fit=True, stats_on=True, stats_print=False, print_CI=False, nbin=10, density=True, figsize=(8.0, 6.0)):
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    top_left_ax = axes[0]
    top_left_val = seqs[0]
    axes = axes[1:]
    seqs = seqs[1:]
    for seq, ax in zip(seqs, axes):
        histo_seq(ax, seq, fit=fit, stats_on=stats_on, stats_print=stats_print, print_CI=print_CI, nbin=nbin, density=density)
    fig = plt.gcf()
    return fig

def display_8seq(seq3d, fit=True, stats_on=True, stats_print=False, print_CI=False, nbin=10, density=True, figsize=(8.0, 6.0), samex=False):
    seqs_brute = opr.get_all_3d_noise_seq_fast(seq3d)
    #seqs_brute = opr.get_all_3d_noise_seq(seq3d)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    xmin = np.min(seqs_brute[-1])
    xmax = np.max(seqs_brute[-1])

    seqs_brute = [seqs_brute[i] for i in ORDER]

    
    for seq, ax, title in zip(seqs_brute, axes, TITLES):
        histo_seq(ax, seq, fit=fit, stats_on=stats_on, stats_print=stats_print, print_CI=print_CI, nbin=nbin, density=density)
        ax.set_title(title)
        if samex:
            ax.set_xlim(xmin, xmax)
    fig = plt.gcf()
    fig.tight_layout()
    return fig



def noise_resume(seq, pc=False, method="fast"):
    T, V, H = seq.shape
    if method == "fast":
        noises = noise.get_all_3d_noise_var_fast(seq)
    elif method == "classic":
        noises = noise.get_all_3D_noise_var(seq)
    elif method == "classic matrix":
        noises = noise.get_all_3d_classic_var_matrix(seq)
    elif method == "corrected":
        noises = noise.get_all_3d_corrected_var_matrix(seq)
    else:
        raise ValueError("Specify method among fast, classic, classic matrix, and corrected")
    
    if pc: 
        noises = tuple(np.array(noises)/noises[-1])

    var_t, var_v, var_h, var_tv, var_th, var_vh, var_tvh, var_tot = noises
        
    sep = "-------------------------------------------------\n"
    
    high_lvl = "Mean : {:.3f} | Var : {:.3f} | Sum : {:.3f}\n".format(np.mean(seq), np.var(seq), np.sum(seq))
    shape = "T : {} | V : {} |  H : {} | Max error : {:.1%}\n".format(T, V, H, ((1-1/T-1/V-1/H)-1))
    method = "---------------- {:10} ---------------------\n".format(method)
    res_1 = '$\sigma^2_tot$ : {:6.3f} | $\sigma^2_v$ :  {:6.3f} | $\sigma^2_h$ :  {:6.3f} | $\sigma^2_vh$ :  {:6.3f}\n'.format(var_tot, var_v, var_h, var_vh)
    res_2 = '$\sigma^2_t$ : {:8.3f} | $\sigma^2_tv$ : {:6.3f} | $\sigma^2_th$ : {:6.3f} | $\sigma^2_tvh$ : {:6.3f} \n'.format(var_t, var_tv, var_th, var_tvh)
    string_resume = sep + high_lvl + shape + method + res_1 + res_2 + sep
    
    return string_resume
    #print("-------------------------------------------------")
    #print("Mean : {:.3f} | Var : {:.3f} | Sum : {:.3f}".format(np.mean(seq), np.var(seq), np.sum(seq)))
    #print("T : {} | V : {} |  H : {} | Max error : {:.1%}".format(T, V, H, ((1-1/T-1/V-1/H)-1)))
    #print("---------------- {:10} ---------------------".format(method))
    #print('$\sigma^2_tot$ : {:6.3f} | $\sigma^2_v$ :  {:6.3f} | $\sigma^2_h$ :  {:6.3f} | $\sigma^2_vh$ :  {:6.3f}'.format(stot, sv, sh, svh))
    #print('$\sigma^2_t$ : {:8.3f} | $\sigma^2_tv$ : {:6.3f} | $\sigma^2_th$ : {:6.3f} | $\sigma^2_tvh$ : {:6.3f} '.format(st, stv, sth, stvh))
    #print("-------------------------------------------------")

    
### Display spectrum
def disp_spectrum(seq, xt=0, xv=0, xh=0, figsize=(4,2), share_scale=True):

    vec_psds = ns.compute_psd(seq)
    t, v, h, tv, th, vh, tvh = vec_psds
    
    vmin_vec = np.min(vec_psds)
    vmax_vec = np.max(vec_psds)

    # Spectres 1D
    psd_t_1D = t[:, xv, xh]
    psd_v_1D = v[xt, :, xh]
    psd_h_1D = h[xt, xv, :]
    # Spectres 2D 
    psd_tv_2D = tv[:, :, xh]
    psd_th_2D = th[:, xv, :]
    psd_vh_2D = vh[xt, :, :]
    # Spectre 3D
    psd_tvh_3D = tvh

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.ravel().tolist()
    # 1D spectrum
    axes[0].plot(psd_t_1D)
    axes[0].set_xlabel("t")
    axes[1].plot(psd_v_1D)
    axes[1].set_xlabel("v")
    axes[2].plot(psd_h_1D)
    axes[2].set_xlabel("h")
    
    if share_scale:
        axes[0].set_ylim((vmin_vec, vmax_vec))
        axes[1].set_ylim((vmin_vec, vmax_vec))
        axes[2].set_ylim((vmin_vec, vmax_vec))
    
    # 2D spectrum
    if share_scale:
        axes[3].imshow(psd_tv_2D, vmin=vmin_vec, vmax=vmax_vec)
        axes[4].imshow(psd_th_2D, vmin=vmin_vec, vmax=vmax_vec)
        axes[5].imshow(psd_vh_2D, vmin=vmin_vec, vmax=vmax_vec)
    else:
        axes[3].imshow(psd_tv_2D)
        axes[4].imshow(psd_th_2D)
        axes[5].imshow(psd_vh_2D)

    axes[3].set_xlabel("t")
    axes[3].set_ylabel("v")
    axes[4].set_xlabel("t")
    axes[4].set_ylabel("h")
    axes[5].set_xlabel("v")
    axes[5].set_ylabel("h")
    
    if share_scale:
        axes[6].imshow(psd_tvh_3D[xt, :, :], vmin=vmin_vec, vmax=vmax_vec)
        axes[7].imshow(psd_tvh_3D[:, xv, :], vmin=vmin_vec, vmax=vmax_vec)
        im = axes[8].imshow(psd_tvh_3D[:, :, xh], vmin=vmin_vec, vmax=vmax_vec)
        fig.colorbar(im, ax=axes)
    else:
        axes[6].imshow(psd_tvh_3D[xt, :, :])#, vmin=vmin, vmax=vmax)
        axes[7].imshow(psd_tvh_3D[:, xv, :])#, vmin=vmin, vmax=vmax)
        im = axes[8].imshow(psd_tvh_3D[:, :, xh])#, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        fig.colorbar(im, ax=axes)

    return fig, axes




### Display images
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volumes, title_list=TITLES, vmin="default", vmax="default", cmap="gray", interpolation=None):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots(nrows=2, ncols=4, squeeze=True)
    if vmin=="default" and vmax=="default":
        vmin = np.min(volumes)
        vmax = np.max(volumes)
    for myax, volume, title in zip(ax.flatten(), volumes, title_list):
        myax.volume = volume
        myax.index = 0
        myax.seq_name = title
        myax.imshow(volume[myax.index], vmin=vmin, vmax=vmax, interpolation=interpolation, cmap=cmap)
        myax.set_title(title + "[{}]".format(myax.index))
        myax.axis("off")
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', process_key)
    fig.suptitle("3D noise sequences")


def process_key(event):
    fig = event.canvas.figure
    for ax in fig.axes:
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()
        plt.tight_layout()


def previous_slice(ax):
    ax.index = (ax.index - 1) % ax.volume.shape[0]  # wrap around using %
    ax.images[0].set_array(ax.volume[ax.index])
    ax.set_title(ax.seq_name + "[{}]".format(ax.index))


def next_slice(ax):
    ax.index = (ax.index + 1) % ax.volume.shape[0]
    ax.images[0].set_array(ax.volume[ax.index])
    ax.set_title(ax.seq_name + "[{}]".format(ax.index))

    
def display_spectrums():
    pass
