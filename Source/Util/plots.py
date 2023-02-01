import numpy as np
import matplotlib.pyplot as plt
import warnings

"""
Methods to do make 1d and 2d histograms of the train vs test vs generated data distributions.
Code copy-pasted from Theo

plot_obs makes a 1d histogram with error extraplots and stuff

plot_deta_dphi makes a 2d heatmap of DeltaEta vs DeltaPhi
(Currently this one assumes a 16dim input and has the Eta and Phi channels hardcoded)
"""

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_obs(pp, obs_train, obs_test, obs_predict, name, bins=60, range=None, unit=None, weight_samples=None,
                             save_dict=None, predict_weights=None, n_epochs=None, n_jets=None):
        '''
        Up-to-date plotting function from Theo (binn branch of precision-enthusiasts repo)
        slightly modified (removed save_dict option and renamed parameters to match our earlier version)
        Note that one cannot specify the range, we let it be chosen by matplotlib
        :pp: name of file where the plot should be stored
        :obs_train: training data
        :obs_test: test data
        :obs_predict: predicted data
        :bins: Bins to be used for the histogram. Can be either a number (then bins are created equidistantly)
                or an array of the bin edges, as taken by np.histogram(..., bins=bins)
        :range: Range to be used for the histogram. Optional parameter, if not specified then the range is chosen from the data
        :name: Name of the variable to be histogrammed (goes into xlabel)
        :unit: Unit of the variable to be histogrammed (goes into xlabel)
        :weight_samples: Integer that specifies how many different choices of weights have been used (for Bayesian models)
                If weight_samples!=None, assume that obs_predict has the form [dataset1, dataset2, ...]
        :predict_weights: Weights of the predicted events (for e.g. discriminator reweighting)
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            y_t,  bins = np.histogram(obs_test, bins=bins, range=range) #generate bin array if needed
            y_tr, _ = np.histogram(obs_train, bins=bins)
            if weight_samples is None:
                y_g,  _ = np.histogram(obs_predict, bins=bins, weights=predict_weights)
                hists = [y_t, y_g, y_tr]
                hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]
            else:
                obs_predict = obs_predict.reshape(weight_samples,
                        len(obs_predict)//weight_samples)
                hist_weights = (None if predict_weights is None
                                else predict_weights.reshape(obs_predict.shape))
                hists_g = np.array([np.histogram(obs_predict[i,:], bins=bins,
                                                 weights=hist_weights[i,:])[0]
                                    for i in range(weight_samples)])
                hists = [y_t, np.mean(hists_g, axis=0), y_tr]
                hist_errors = [np.sqrt(y_t), np.std(hists_g, axis=0), np.sqrt(y_tr)]
            integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
            scales = [1 / integral if integral != 0. else 1. for integral in integrals]

            FONTSIZE = 14
            labels = ["True", "Model", "Train"]
            colors = ["#e41a1c", "#3b528b", "#1a8507"]
            dup_last = lambda a: np.append(a, a[-1])

            if weight_samples is None:
                fig1, axs = plt.subplots(3, 1, sharex=True,
                        gridspec_kw={"height_ratios" : [4, 1, 1], "hspace" : 0.00})
            else:
                fig1, axs = plt.subplots(5, 1, sharex=True,
                        gridspec_kw={"height_ratios" : [4, 1, 1, 1, 1], "hspace" : 0.00})
            if n_epochs is not None:
                if n_jets is not None:
                     fig1.suptitle(f"After training for {n_epochs+1} epochs for {n_jets} jets")
                else:
                     fig1.suptitle(f"After training for {n_epochs+1} epochs")

            for y, y_err, scale, label, color in zip(hists, hist_errors, scales,
                                                     labels, colors):
               
                    
                axs[0].step(bins, dup_last(y) * scale, label=label, color=color,
                        linewidth=1.0, where="post")
                axs[0].step(bins, dup_last(y + y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
                axs[0].step(bins, dup_last(y - y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
                axs[0].fill_between(bins, dup_last(y - y_err) * scale,
                        dup_last(y + y_err) * scale, facecolor=color,
                        alpha=0.3, step="post")

                if label == "True": continue

                ratio = (y * scale) / (hists[0] * scales[0])
                ratio_err = np.sqrt((y_err / y)**2 + (hist_errors[0] / hists[0])**2)
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
                axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
                axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
                axs[1].fill_between(bins, dup_last(ratio - ratio_err),
                        dup_last(ratio + ratio_err), facecolor=color, alpha=0.3, step="post")

                delta = np.fabs(ratio - 1) * 100
                delta_err = ratio_err * 100

                markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:])/2, delta,
                        yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                        linewidth=0, fmt=".", capsize=2)
                [cap.set_alpha(0.5) for cap in caps]
                [bar.set_alpha(0.5) for bar in bars]

            if weight_samples is not None:
                mu = hists[1] * scales[1]
                sigma = hist_errors[1] * scales[1]
                train = hists [0] * scales[0]
                axs[3].step(bins, dup_last(sigma /mu) , label=label, color="#3b528b",
                        linewidth=1.0, where="post")
                axs[4].step(bins, dup_last(np.abs(train - mu )/mu) , label=label,
                        color="#3b528b", linewidth=1.0, where="post")

            axs[0].legend(loc="upper right", frameon=False)
            axs[0].set_ylabel("normalized", fontsize = FONTSIZE)
            if "p_{T" in name:
                axs[0].set_yscale("log")

            axs[1].set_ylabel(r"$\frac{\mathrm{True}}{\mathrm{Model}}$",
                    fontsize = FONTSIZE)
            axs[1].set_yticks([0.95,1,1.05])
            axs[1].set_ylim([0.9,1.1])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
            plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                    fontsize = FONTSIZE)

            axs[2].set_ylim((0.05,20))
            axs[2].set_yscale("log")
            axs[2].set_yticks([0.1, 1.0, 10.0])
            axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

            axs[2].axhline(y=1.0,linewidth=0.5, linestyle="--", color="grey")
            axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            axs[2].set_ylabel(r"$\delta [\%]$", fontsize = FONTSIZE)
            
            if weight_samples is not None:
                axs[3].set_yscale("log")
                axs[3].set_ylabel(r"$\frac{\sigma_{\mathrm{INN}}}{\mu_{\mathrm{INN}}}$",
                        fontsize = 10)

                axs[4].set_yscale("log")
                axs[4].set_ylabel(r"$\frac{\mu_{\mathrm{INN}}-\mu_{\mathrm{True}}}" +
                                  r"{\mu_{\mathrm{INN}}}$", fontsize = 10)

            plt.savefig(pp, bbox_inches="tight", format="pdf", pad_inches=0.05)
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

def get_R(data):
        '''Calculate radius of samples following a hypersphere distribution'''
        return np.sum(data ** 2, axis=1) ** .5
def get_xsum(data):
        '''Calculate radius of samples following a hypersphere distribution'''
        return np.sum(data, axis=1)
