import numpy as np

import matplotlib.pyplot as plt

import noise3d
from . import noise
from . import opr
from . import stats
#from . import spectrum as ns

import scipy.stats as st

# ORDER is used to re-order the sequences to match TITLES (for display)
ORDER = [-1, 1, 2, 5, 0, 3, 4, 6]
TITLES = ["tot", "v", "h", "vh", "t", "tv", "th", "tvh"]


def gauss(x, mu, sigma):
    """Helper gaussian function"""
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)


def histo_seq(seq, ax=None, fit=True, stats_on=True, stats_print=False,
              print_CI=False, nbin=10,
              density=True):
    """
    Plot histrogram of seq with additional infos.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    # Compute histogram
    vals = seq.flatten()
    ech = np.linspace(np.min(seq), np.max(seq), nbin)
    val, bins, patchs = ax.hist(vals, bins=ech, density=density)
    # Compute basic stats
    mean = np.mean(vals)
    std = np.std(vals)
    
    # Add a gaussian fit
    if fit:
        ech_gauss = gauss(ech, mean, std)
        ax.plot(ech, ech_gauss)

    # If stats are needed
    if stats_print or stats_on:
        results = stats.basic_stats(seq)
        
    # Print stats in standard output
    if stats_print:
        print("Count : {}".format(results["count"]))
        print("Min/Max : {}/{}".format(results["min"], results["max"]))
        print("Mean : {}".format(results["mean"]))
        print("Median : {}".format(results["median"]))
        print("Var : {}".format(results["var"]))
        print("Std : {}".format(results["std"]))

    # Add stat on plot
    if stats_on:
        ax.errorbar(mean,
                    max(val)/2,
                    yerr=None,
                    xerr=std,
                    solid_capstyle='projecting',
                    capsize=5)
        text = "Count : {}\nMin/Max : {:.3f}/{:.3f}\nMean : {:.3f}\nMedian : {:.3f}\nVar : {:.3f}\nStd : {:.3f}".format(
            results["count"], 
            results["min"], 
            results["max"],
            results["mean"],
            results["median"],
            results["var"],
            results["std"])
        
        ax.annotate(text, xy=(0.99, 0.99),
                    xycoords='axes fraction', 
                    xytext=(0.99, 0.99), 
                    textcoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='top')

    return ax


#def display_7seq(seqs, fit=True, stats_on=True,
#                 stats_print=False, print_CI=False,
#                 nbin=50, density=True, figsize=(8.0, 6.0), samex=False):
#    """
#    Loop over 7 axes and plot the noise histogram.
#    """
#    # create plots
#    fig, axes = plt.subplots(2, 4, figsize=figsize)
#    axes = axes.flatten()
#    # Separate the first ("tot") seq from the other 7 noises
#    top_left_ax = axes[0]
#    top_left_val = seqs[0]
#    # noises
#    axes = axes[1:]
#    seqs = seqs[1:]
#    seqs = [seqs[i] for i in ORDER]
#    # Bounds
#    xmin = np.min(seqs)
#    xmax = np.max(seqs)
#
#    # looping over noises
#    for seq, ax, title in zip(seqs, axes, TITLES[1:]):
#        # plot histograms
#        histo_seq(seq, ax=ax, fit=fit, stats_on=stats_on,
#                  stats_print=stats_print, print_CI=print_CI,
#                  nbin=nbin, density=density)
#        ax.set_title(title)
#        if samex:
#            ax.set_xlim(xmin, xmax)
#    fig = plt.gcf()
#    fig.tight_layout()
#    return fig

def _compute_noises(seq, method="fast"):
    # Choose method to compute noises
    if method == "fast":
        return noise.get_all_3d_noise_var_fast(seq)
    elif method == "classic":
        return noise.get_all_3D_noise_var(seq)
    elif method == "classic matrix":
        return noise.get_all_3d_classic_var_matrix(seq)
    elif method == "corrected":
        return noise.get_all_3d_corrected_var_matrix(seq)
    else:
        raise ValueError("Specify method among fast, classic, classic matrix, and corrected")

def display_8seq(seqs, extract=False, fit=True, stats_on=True, 
                 stats_print=False, print_CI=False,
                 nbin=50, density=True, figsize=(8.0, 6.0),
                 samex=False):
    """Display all 8 histograms.
    Input seqs can be :
         - 1 raw 3d sequence, with extract=True
         - 8 sequences in a list/tuple (t, v, h, tv, th, vh, tvh, tot)
    """
    # analyze sequences 
    if extract:
        seqs = noise3d.opr.get_all_3d_noise_seq_fast(seqs, names=True)
        names = seqs[-1]
        seqs = seqs[:-1]
    # define plots
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    # Compute histgram bounds if samex
    if samex:
        xmin = np.min(seqs[-1])
        xmax = np.max(seqs[-1])
    # reorder sequences
    seqs = [seqs[i] for i in ORDER]
    # for all sequences
    for seq, ax, title in zip(seqs, axes, TITLES):
        histo_seq(seq, ax=ax, fit=fit, stats_on=stats_on,
                  stats_print=stats_print,
                  print_CI=print_CI, nbin=nbin, 
                  density=density)
        # setting title
        ax.set_title(title)
        # seetting x axis limits
        if samex:
            ax.set_xlim(xmin, xmax)
    fig = plt.gcf()
    fig.tight_layout()
    return fig



def noise_resume(seq, pc=False, method="fast"):
    """
    Compute all noise infos in a string
    """
    T, V, H = seq.shape
    # compute noise
    noises = _compute_noises(seq, method=method)
    # express all noise in % of total variance
    if pc: 
        noises = tuple(np.array(noises)/noises[-1])

    # split noises tuple
    var_t, var_v, var_h, var_tv, var_th, var_vh, var_tvh, var_tot = noises
        
    sep = "-------------------------------------------------\n"
    
    # general infos
    high_lvl = "Mean : {:.3f} | Var : {:.3f} | Sum : {:.3f}\n".format(np.mean(seq), np.var(seq), np.sum(seq))
    shape = "T : {} | V : {} |  H : {} | Max error : {:.1%}\n".format(T, V, H, ((1-1/T-1/V-1/H)-1))
    method = "---------------- {:10} ---------------------\n".format(method)
    # line of noise
    res_1 = '$\sigma^2_tot$ : {:6.3f} | $\sigma^2_v$ :  {:6.3f} | $\sigma^2_h$ :  {:6.3f} | $\sigma^2_vh$ :  {:6.3f}\n'.format(var_tot, var_v, var_h, var_vh)
    # line 2 of noise
    res_2 = '$\sigma^2_t$ : {:8.3f} | $\sigma^2_tv$ : {:6.3f} | $\sigma^2_th$ : {:6.3f} | $\sigma^2_tvh$ : {:6.3f} \n'.format(var_t, var_tv, var_th, var_tvh)
    #concatenate
    string_resume = sep + high_lvl + shape + method + res_1 + res_2 + sep
    return string_resume


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
