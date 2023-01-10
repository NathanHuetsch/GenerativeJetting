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


LABEL_XGEN = "Generated"
LABEL_XTRAIN = "Train"
LABEL_TRUTH = "Truth"


def plot_obs(pp, obs_train, obs_test, obs_predict, name, range=[0, 100], num_bins=60, FONTSIZE=14, weights=None):
        fig1, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1, 1], 'hspace' : 0.00})

        y_t, x_t = np.histogram(obs_test, bins=num_bins, range=range, density=True)
        if not weights is None:
            y_g, x_g = np.histogram(obs_predict, bins=num_bins, range=range, density=True, weights=weights)
        else:
            y_g, x_g = np.histogram(obs_predict, bins=num_bins, range=range, density=True)
        y_tr, x_tr = np.histogram(obs_train, bins=num_bins, range=range, density=True)
        y_tr, y_g, y_t = y_tr/np.sum(y_tr), y_g/np.sum(y_g), y_t/np.sum(y_t)


        #Histogram
        axs[0].step(x_t[:num_bins], y_t, label=LABEL_TRUTH, color=truthcolor, linewidth=1.0, where='mid')
        axs[0].step(x_g[:num_bins], y_g,label=LABEL_XGEN, color=netcolor, linewidth=1.0, where='mid')
        axs[0].step(x_tr[:num_bins], y_tr,label=LABEL_XTRAIN, color=traincolor, linewidth=1.0, where='mid')

        axs[0].legend(loc='upper right', frameon=False)
        axs[0].set_ylabel(r'$\frac{\mathrm{d} \sigma}{\mathrm{d} {%s}}$' % name, fontsize = FONTSIZE)
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


def plot_deta_dphi(file_name, data_train, data_test, data_generated):
    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(1, 3, 1)
    dphi = data_train[:, 9] - data_train[:, 13]
    deta = data_train[:, 10] - data_train[:, 14]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
    plt.xlabel('$\Delta \eta$')
    plt.ylabel('$\Delta \phi$')
    plt.title('train')

    fig.add_subplot(1, 3, 2)
    dphi = data_test[:, 9] - data_test[:, 13]
    deta = data_test[:, 10] - data_test[:, 14]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
    plt.xlabel('$\Delta \eta$')
    plt.ylabel('$\Delta \phi$')
    plt.title('test')

    fig.add_subplot(1, 3, 3)
    dphi = data_generated[:, 9] - data_generated[:, 13]
    deta = data_generated[:, 10] - data_generated[:, 14]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    plt.hist2d(deta, dphi, bins=100, range=[[-5, 5], [-np.pi, np.pi]])
    plt.xlabel('$\Delta \eta$')
    plt.ylabel('$\Delta \phi$')
    plt.title('generated')

    plt.savefig(file_name)
    plt.close()


def observable_histogram(pp, obs_train, obs_predict,
                         bins=60, log_scale=False, weight_samples=None,
                         weights=None, save_dict=None):

    y_tr, bins = np.histogram(obs_train, bins=bins, range=[0, 8])

    if not (weight_samples is None):
        obs_predict = obs_predict.reshape(weight_samples,
                                          len(obs_predict) // weight_samples)
        hists_g = np.array([np.histogram(obs_predict[i, :], bins=bins)[0]
                            for i in range(weight_samples)])
        hists = [y_tr, np.mean(hists_g, axis=0)]
        hist_errors = [np.sqrt(y_tr), np.std(hists_g, axis=0)]
    elif not (weights is None):
        y_r, x_r = np.histogram(obs_predict, bins=bins, weights=weights, range=[0, 8])
        y_g, x_g = np.histogram(obs_predict, bins=bins, range=[0, 8])
        hists = [y_tr, y_g, y_r]
        hist_errors = [np.sqrt(y_tr), np.sqrt(y_g), np.sqrt(y_r)]
    else:
        y_g, x_g = np.histogram(obs_predict, bins=bins)
        hists = [y_tr, y_g]
        hist_errors = [np.sqrt(y_tr), np.sqrt(y_g)]

    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0. else 1. for integral in integrals]

    FONTSIZE = 14
    labels = ["Train", "Generated"]
    colors = ["#1a8507", "#3b528b"]
    if not (weights is None):
        labels.append("Reweighted")
        colors.append("#8b3b78")
    dup_last = lambda a: np.append(a, a[-1])
    fig1, axs = plt.subplots(3 + int(not (weights is None)), 1, sharex=True,
                             gridspec_kw={
                                 "height_ratios": [4, 1, 1] + [1 for i in range(int(not (weights is None)))],
                                 "hspace": 0.00})

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

        if label == "Train": continue

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
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

        markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                              yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                              linewidth=0, fmt=".", capsize=2)
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]

    axs[0].legend(loc="lower right", frameon=False)
    axs[0].set_ylabel("normalized", fontsize=FONTSIZE)
    if log_scale:
        axs[0].set_yscale("log")

    axs[1].set_ylabel(r"$\frac{\mathrm{True}}{\mathrm{Model}}$",
                      fontsize=FONTSIZE)
    axs[1].set_yticks([0.95, 1, 1.05])
    axs[1].set_ylim([0.9, 1.1])
    axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
    axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
    plt.xlabel("dR",
               fontsize=FONTSIZE)

    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                       2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 2.0, facecolor="#cccccc", alpha=0.3)

    if not (weights is None):
        weight = y_r / y_g
        disc = weight / (1 + weight)
        axs[3].step(bins[:-1], weight, color=colors[-1], linewidth=1.0)
        axs[3].set_ylabel("D(x)", fontsize=FONTSIZE)
        axs[3].set_yticks([0.0, 0.5, 1.0, 1.5, 2])
        axs[3].axhline(y=0.5, linewidth=0.5, linestyle="--", color="black")

    plt.savefig(pp)
    plt.close()