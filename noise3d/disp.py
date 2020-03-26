import numpy as np

import matplotlib.pyplot as plt

import noise3d
from . import noise
from . import opr
from . import stats
from . import spectrum
#from . import spectrum as ns

#import scipy.stats as st

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
    noises = noise.get_noise_vars(seq, method=method)
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
def disp_spectrum(seq, 
                  xt=0, xv=0, xh=0,
                  figsize=(4,2),
                  share_scale=True):
    from matplotlib import cm
    # Compute spectrums
    vec_psds = spectrum.compute_psd(seq)
    t, v, h, tv, th, vh, tvh = vec_psds
    
    # get bounds
    vmin_vec = np.min(vec_psds)
    vmax_vec = np.max(vec_psds)

    # Extract 1D spectrums
    psd_t_1D = t[:, xv, xh]
    psd_v_1D = v[xt, :, xh]
    psd_h_1D = h[xt, xv, :]
    # Extract 2D spectrums
    psd_tv_2D = tv[:, :, xh]
    psd_th_2D = th[:, xv, :]
    psd_vh_2D = vh[xt, :, :]
    # Extract 3D spectrum
    psd_tvh_3D = tvh

    # create plots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.ravel().tolist()
        
    # Plot 1D spectrum
    #axes[0].plot(psd_t_1D)
    axes[0].scatter(np.arange(len(psd_t_1D)), psd_t_1D, c=cm.viridis(psd_t_1D/np.max(psd_t_1D)), edgecolor="none")
    axes[0].set_xlabel("t")
    axes[0].set_title("t noise")
    #axes[1].plot(psd_v_1D)
    axes[1].scatter(np.arange(len(psd_v_1D)), psd_v_1D, c=cm.viridis(psd_v_1D/np.max(psd_v_1D)), edgecolor="none")
    axes[1].set_xlabel("v")
    axes[1].set_title("v noise")
    #axes[2].plot(psd_h_1D)
    axes[2].scatter(np.arange(len(psd_h_1D)), psd_h_1D, c=cm.viridis(psd_h_1D/np.max(psd_h_1D)), edgecolor="none")
    axes[2].set_xlabel("h")
    axes[2].set_title("h noise")
    if share_scale:
        axes[0].set_ylim((vmin_vec, vmax_vec))
        axes[1].set_ylim((vmin_vec, vmax_vec))
        axes[2].set_ylim((vmin_vec, vmax_vec))
    
    # Plot 2D spectrum
    if share_scale:
        axes[3].imshow(psd_tv_2D, vmin=vmin_vec, vmax=vmax_vec)
        axes[4].imshow(psd_th_2D, vmin=vmin_vec, vmax=vmax_vec)
        axes[5].imshow(psd_vh_2D, vmin=vmin_vec, vmax=vmax_vec)
    else:
        axes[3].imshow(psd_tv_2D)
        axes[4].imshow(psd_th_2D)
        axes[5].imshow(psd_vh_2D)
    axes[3].set_title("tv noise")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("v")
    axes[4].set_title("th noise")
    axes[4].set_xlabel("t")
    axes[4].set_ylabel("h")
    axes[5].set_title("vh noise")
    axes[5].set_xlabel("v")
    axes[5].set_ylabel("h")
    
    # Plot 3D spectrum
    if share_scale:
        axes[6].imshow(psd_tvh_3D[xt, :, :],
                       vmin=vmin_vec, vmax=vmax_vec)
        axes[7].imshow(psd_tvh_3D[:, xv, :],
                       vmin=vmin_vec, vmax=vmax_vec)
        im = axes[8].imshow(psd_tvh_3D[:, :, xh],
                            vmin=vmin_vec, vmax=vmax_vec)
    else:
        axes[6].imshow(psd_tvh_3D[xt, :, :])
        axes[7].imshow(psd_tvh_3D[:, xv, :])
        im = axes[8].imshow(psd_tvh_3D[:, :, xh])
    
    # Titles and labels
    axes[6].set_title(f"tvh noise - vh plane (t={xt})")
    axes[6].set_xlabel("v")
    axes[6].set_ylabel("h")
    axes[7].set_title(f"tvh noise - th plane (v={xv})")
    axes[7].set_xlabel("t")
    axes[7].set_ylabel("h")
    axes[8].set_title(f"tvh noise - tv plane (h={xh})")
    axes[8].set_xlabel("t")
    axes[8].set_ylabel("v")

    #fig.suptitle("Noise Power Spectrum Density")
    plt.tight_layout()
    fig.colorbar(im, ax=axes)

    return fig, axes




class SequenceViewer(object):
    """Allow to view 8 3d volumes.
    Use j and k keys to change slice on first axis (typicaly temporal).
    The 8 volumes must have same shapes.
    """
    
    def __init__(self, volumes, title_list=TITLES,
                 vmin="default", vmax="default",
                 cmap="gray", interpolation=None):
        self._remove_keymap_conflicts({'j', 'k'})
        
        fig, ax = plt.subplots(nrows=2, ncols=4, squeeze=True)
        self.fig = fig
        if vmin=="default" and vmax=="default":
            vmin = np.min(np.asarray(volumes))
            vmax = np.max(np.asarray(volumes))
        
        # reorder seqs to match classic display order 
        volumes = [volumes[i] for i in ORDER]
        
        for myax, volume, title in zip(ax.flatten(), volumes, title_list):
            myax.volume = volume
            myax.index = 0
            myax.seq_name = title
            im = myax.imshow(volume[myax.index], vmin=vmin, vmax=vmax, interpolation=interpolation, cmap=cmap)
            myax.set_title(title + "[{}]".format(myax.index))
            myax.axis("off")
        #fig.canvas.draw()
        plt.tight_layout()
        fig.canvas.mpl_connect('key_press_event', self._process_key)
        #fig.colorbar(im, ax=ax)
        fig.suptitle("3D noise sequences\n(use j and k)")
        
    
    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    
    def _process_key(self, event):
        """Widget API"""
        fig = event.canvas.figure
        for ax in fig.axes:
            if event.key == 'j':
                self._change_slice(ax, "prev")
            elif event.key == 'k':
                self._change_slice(ax, "next")
            fig.canvas.draw()
            plt.tight_layout()
    
    # Doesnt work as it is
    #def change_all_slice(self, next_or_prev="next"):
    #    """Python API : viewer.change_all_slice()"""
    #    fig = self.fig
    #    for ax in fig.axes:
    #        self._change_slice(ax, next_or_prev=next_or_prev)
    #        fig.canvas.draw()
    #        plt.tight_layout()
    
    def _change_slice(self, ax, next_or_prev="next"):
        # update ax.index
        if next_or_prev == "next":
            ax.index = (ax.index + 1) % ax.volume.shape[0] # wrap around using %
        elif next_or_prev == "prev":
            ax.index = (ax.index - 1) % ax.volume.shape[0] # wrap around using %
        else:
            raise ValueError("next or prev not ", next_or_prev)
        # update data
        ax.images[0].set_array(ax.volume[ax.index])
        # update titles
        ax.set_title(ax.seq_name + "[{}]".format(ax.index))
        #plt.show()


def idea_to_display_3d_seq():
    """Starter for an interactive 3D display of a sequence"""
    #%matplotlib qt

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    N = 4
    volume = np.random.rand(N, N, N)
    filled = np.ones((N, N, N), dtype=np.bool)
    
    # repeating values 3 times for grayscale
    colors = np.repeat(volume[:, :, :, np.newaxis], 3, axis=3)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.voxels(filled, facecolors=colors, edgecolors='k')
    plt.show()
