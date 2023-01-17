import numpy as np
import matplotlib.pyplot as plt

"""
Methods to do make 1d and 2d histograms of the train vs test vs generated data distributions.
Code copy-pasted from Theo

plot_obs makes a 1d histogram with error extraplots and stuff

plot_deta_dphi makes a 2d heatmap of DeltaEta vs DeltaPhi
(Currently this one assumes a 16dim input and has the Eta and Phi channels hardcoded)
"""

#Constants

netcolor = '#3b528b'
truthcolor = '#e41a1c'
traincolor = '#1a8507'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

LABEL_XGEN = "Gen."
LABEL_XTRAIN = "Train"
LABEL_TRUTH = "True"

def plot_obs(pp, obs_train, obs_test, obs_predict, name,n_epochs, range=[0, 100], num_bins=60,FONTSIZE=14, weights=None, n_jets=2):
        fig1, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1, 1], 'hspace' : 0.00})

        y_t, x_t = np.histogram(obs_test, bins=num_bins, range=range, density=True)
        if weights is not None:
            y_g, x_g = np.histogram(obs_predict, bins=num_bins, range=range, density=True, weights=weights)
        else:
            y_g, x_g = np.histogram(obs_predict, bins=num_bins, range=range, density=True)
        y_tr, x_tr = np.histogram(obs_train, bins=num_bins, range=range, density=True)
        y_tr, y_g, y_t = y_tr/np.sum(y_tr), y_g/np.sum(y_g), y_t/np.sum(y_t)

        fig1.suptitle(f"After training for {n_epochs + 1} epochs for {n_jets} jets")

        #Histogram
        axs[0].step(x_t[:num_bins], y_t, label=LABEL_TRUTH, color=truthcolor, linewidth=1.0, where='mid')
        axs[0].step(x_g[:num_bins], y_g,label=LABEL_XGEN, color=netcolor, linewidth=1.0, where='mid')
        axs[0].step(x_tr[:num_bins], y_tr,label=LABEL_XTRAIN, color=traincolor, linewidth=1.0, where='mid')

        axs[0].legend(loc='upper right', frameon=False)
        axs[0].set_ylabel(r'$\frac{\mathrm{d} \sigma}{\mathrm{d} {%s}}$ [pb/GeV]' % name, fontsize = FONTSIZE)
        if "p_{T" in name:
            axs[0].set_yscale('log')
        y_rel = y_g/y_t
        y_rel[np.isnan(y_rel)==True] = 1
        y_rel[y_rel==np.inf] = 1

        w_gs, w_ts = 1/len(obs_predict), 1/len(obs_test)
        y_g_abs, y_t_abs = y_g/w_gs, y_t/w_ts
        diff_stat =  100 * w_gs/w_ts * np.sqrt(y_g_abs * (y_g_abs + y_t_abs)/((y_t_abs+1e-5)**3))
        diff_stat[np.isnan(diff_stat)==True] = 1
        diff_stat[diff_stat==np.inf] = 1

        #Ratio Panel
        axs[1].step(x_t[:num_bins], y_rel, linewidth=1.0, where='mid', color=netcolor)
        axs[1].step(x_t[:num_bins], y_rel + diff_stat/100, color=netcolor, alpha=0.5, label='$+- stat$', linewidth=0.5, where='mid')
        axs[1].step(x_t[:num_bins], y_rel - diff_stat/100, color=netcolor, alpha=0.5, linewidth=0.5, where='mid')
        axs[1].fill_between(x_t[:num_bins], y_rel - diff_stat/100, y_rel + diff_stat/100, facecolor=netcolor, alpha = 0.3, step = 'mid')
        axs[1].set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{True}}$', fontsize = FONTSIZE)
        axs[1].set_yticks([0.95,1,1.05])
        axs[1].set_ylim([0.9,1.1])
        axs[1].axhline(y=1, c='black', ls='--', lw=0.7)
        axs[1].axhline(y=1.2, c='black', ls='dotted', lw=0.5)
        axs[1].axhline(y=0.8, c='black', ls='dotted', lw=0.5)
        plt.xlabel(r'${%s}$' % name, fontsize = FONTSIZE)

        #Relative Error
        axs[2].set_ylabel(r'$\delta [\%]$', fontsize = FONTSIZE)
        y_diff = np.fabs((y_rel - 1)) * 100

        markers, caps, bars = axs[2].errorbar(x_g[:num_bins], y_diff, yerr=diff_stat, ecolor=netcolor, color=netcolor, elinewidth=0.5, linewidth=0,  fmt='.', capsize=2)

        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]

        axs[2].set_ylim((0.05,20))
        axs[2].set_yscale('log')
        axs[2].set_yticks([0.1, 1.0, 10.0])
        axs[2].set_yticklabels([r'$0.1$', r'$1.0$', "$10.0$"])
        axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

        axs[2].axhline(y=1.0,linewidth=0.5, linestyle='--', color='grey')
        axs[2].axhspan(0, 1.0, facecolor='#cccccc', alpha=0.3)

        #Lower panels train data
        y_rel = y_tr/y_t
        y_rel[np.isnan(y_rel)==True] = 1
        y_rel[y_rel==np.inf] = 1

        w_trs, w_ts = 1/len(obs_train), 1/len(obs_test)
        y_tr_abs, y_t_abs = y_tr/w_trs, y_t/w_ts
        diff_stat =  100 * w_trs/w_ts * np.sqrt(y_tr_abs * (y_tr_abs + y_t_abs)/((y_t_abs+1e-8)**3))

        axs[1].step(x_t[:num_bins], y_rel, linewidth=1.0, where='mid', color=traincolor)
        axs[1].step(x_t[:num_bins], y_rel + diff_stat/100, color=traincolor, alpha=0.5, label='$+- stat$', linewidth=0.5, where='mid')
        axs[1].step(x_t[:num_bins], y_rel - diff_stat/100, color=traincolor, alpha=0.5, linewidth=0.5, where='mid')
        axs[1].fill_between(x_t[:num_bins], y_rel - diff_stat/100, y_rel + diff_stat/100, facecolor=traincolor, alpha = 0.3, step = 'mid')

        y_diff = np.fabs((y_rel - 1)) * 100

        markers, caps, bars = axs[2].errorbar(x_tr[:num_bins], y_diff, yerr=diff_stat, ecolor=traincolor, color=traincolor, elinewidth=0.5, linewidth=0,  fmt='.', capsize=2)

        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]

        plt.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
        plt.close()


def delta_phi(y, idx1, idx2):
    # return y[:,idx1] - y[:,idx2]
    dphi = np.abs(y[:,idx1] - y[:,idx2])
    return np.where(dphi > np.pi, 2*np.pi - dphi, dphi)


def delta_eta(y, idx1, idx2):
    return y[:,idx1] - y[:, idx2]
    # return np.abs(y[:,idx1] - y[:,idx2])


def delta_r(y,  idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14):
    return np.sqrt(delta_phi(y, idx_phi1, idx_phi2)**2 + delta_eta(y, idx_eta1, idx_eta2)**2)


def plot_deta_dphi(pp, data_train, data_test, data_generated, n_epochs, idx_phi1=9, idx_eta1=10, idx_phi2=13,
                   idx_eta2=14, n_jets=2):
    if idx_phi1 == 9 and idx_phi2 == 13:
        i = 1
        j = 2
    elif idx_phi1 == 9 and idx_phi2 == 17:
        i = 1
        j = 3
    elif idx_phi1 == 13 and idx_phi2 == 17:
        i = 2
        j = 3
    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(1, 3, 1)
    dphi = data_train[:, idx_phi1] - data_train[:, idx_phi2]
    deta = data_train[:, idx_eta1] - data_train[:, idx_eta2]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]], rasterized=True)
    plt.xlabel(f'$\Delta \eta{i,j}$')
    plt.ylabel(f'$\Delta \phi{i,j}$')
    plt.title('train')

    fig.add_subplot(1, 3, 2)
    dphi = data_test[:,idx_phi1] - data_test[:, idx_phi2]
    deta = data_test[:, idx_eta1] - data_test[:, idx_eta2]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]], rasterized=True)
    plt.xlabel(f'$\Delta \eta{i,j}$')
    plt.ylabel(f'$\Delta \phi{i,j}$')
    plt.title('test')
    fig.add_subplot(1, 3, 3)
    dphi = data_generated[:, idx_phi1] - data_generated[:, idx_phi2]
    deta = data_generated[:, idx_eta1] - data_generated[:, idx_eta2]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]], rasterized=True)
    plt.xlabel(f'$\Delta \eta{i,j}$')
    plt.ylabel(f'$\Delta \phi{i,j}$')
    plt.title('generated')
    fig.suptitle(f"After training for {n_epochs + 1} epochs for {n_jets} jets")
    plt.savefig(pp, format="pdf")
    plt.close()