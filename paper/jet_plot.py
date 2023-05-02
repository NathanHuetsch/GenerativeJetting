import numpy as np
import os, sys, time
sys.path.append(os.getcwd()) #can we do better?
from Source.Util.util import load_params

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import matplotlib.font_manager as font_manager

from Source.Util.plots import plot_obs, delta_r
from Source.Util.physics import get_M_ll

### setup matplotlib
font_dir = ['paper/bitstream-charter-ttf/Charter/']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'

model_name = "jetGPT"
bins = 60


obs_names = ["p_{T,l1}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1}",
                          "p_{T,l2}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2}",
                          "p_{T,j1}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1}",
                          "p_{T,j2}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2}",
                          "p_{T,j3}", "\phi_{j3}", "\eta_{j3}", "\mu_{j3}"]

def corner_text(ax, text,horizontal_pos,vertical_pos, fontsize):
    ax.text(
        x=0.95 if horizontal_pos == "right" else 0.05,
        y=0.95 if vertical_pos == "top" else 0.05,
        s=text,
        horizontalalignment=horizontal_pos,
        verticalalignment=vertical_pos,
        transform=ax.transAxes,
        fontsize= fontsize
    )
    # Dummy line for automatic legend placement
    plt.plot(
        0.8 if horizontal_pos == "right" else 0.2,
        0.8 if vertical_pos == "top" else 0.2,
        transform=ax.transAxes,
        color="none"
    )

def plot_paper(pp, obs_train, obs_test, obs_predict, name, bins=60, weight_samples=1,
               predict_weights=None, unit=None, range=None, error_range=None, n_jets=None, y_ticks=None):

    y_t, bins = np.histogram(obs_test, bins=bins, range=range)
    y_tr, _ = np.histogram(obs_train, bins=bins)

    if weight_samples == 1:
        y_g, _ = np.histogram(obs_predict, bins=bins, weights=predict_weights)
        hists = [y_t, y_g, y_tr]
        hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]
    else:
        obs_predict = obs_predict.reshape(weight_samples,
                                              len(obs_predict) // weight_samples)
        hist_weights = (weight_samples * [None] if predict_weights is None
                            else predict_weights.reshape(obs_predict.shape))
        hists_g = np.array([np.histogram(obs_predict[i, :], bins=bins,
                                             weights=hist_weights[i])[0]
                                for i in np.arange(weight_samples)])
        hists = [y_t, np.mean(hists_g, axis=0), y_tr]
        hist_errors = [np.sqrt(y_t), np.std(hists_g, axis=0), np.sqrt(y_tr)]
    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0. else 1. for integral in integrals]

    FONTSIZE = 16
    labels = ["True", model_name, "Train"]
    colors = ["black","#A52A2A", "#0343DE"]
    dup_last = lambda a: np.append(a, a[-1])

    fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
    fig1.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5, rect=(0.08, 0.08, 1., 1.))

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

    axs[0].legend(loc="center right", frameon=False, fontsize=FONTSIZE)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    if "p_{T" in name:
        axs[0].set_yscale("log")

    axs[1].set_ylabel(r"$\frac{\mathrm{%s}}{\mathrm{True}}$" % model_name,
                          fontsize=FONTSIZE)
    axs[1].set_yticks(y_ticks)
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
    axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)

    plt.xlim((range[0]+0.1,range[1]-0.1))

    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[0].tick_params(axis="both", labelsize=FONTSIZE)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE)
    axs[2].tick_params(axis="both", labelsize=FONTSIZE)

    corner_text(axs[0],f"Z+{n_jets} exclusive",horizontal_pos="right",vertical_pos="top", fontsize=FONTSIZE)

    plt.savefig(pp, format="pdf")
    plt.close()

path = "AutoRegGMMcond_4342"
iterations = 10
weights=None
plot_weights=None

if True:
    with warnings.catch_warnings() and PdfPages(f"paper/jet/{path}.pdf") as out:
        plot_train = np.load(f"paper/jet/{path}_train_1.npy")
        plot_test = np.load(f"paper/jet/{path}_test_1.npy")
        plot_samples = np.load(f"paper/jet/{path}_sample_1.npy")
        obs_train = plot_train[:, 8]
        obs_test = plot_test[:, 8]
        obs_generated = plot_samples[:, 8]
        # Get the name and the range of the observable
        obs_name = obs_names[8]
        obs_range = [17,157]

        # Create the plot
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=obs_range,
                 weight_samples=iterations,
                 error_range=[0.83,1.17],
                 n_jets=1,
                 y_ticks=[0.9,1,1.1])

        obs_name = "M_{\mu \mu}"
        obs_range = [79, 104]
        data_train = get_M_ll(plot_train)
        data_test = get_M_ll(plot_test)
        data_generated = get_M_ll(plot_samples)
        plot_paper(pp=out,
                 obs_train=data_train,
                 obs_test=data_test,
                 obs_predict=data_generated,
                 name=obs_name,
                 range=obs_range,
                 bins=60,
                 weight_samples=iterations,
                 predict_weights=weights,
                 error_range=[0.71,1.29],
                 n_jets=1,
                 y_ticks=[0.8,1,1.2])

        plot_train = np.load(f"paper/jet/{path}_train_2.npy")
        plot_test = np.load(f"paper/jet/{path}_test_2.npy")
        plot_samples = np.load(f"paper/jet/{path}_sample_2.npy")

        obs_train = plot_train[:, 12]
        obs_test = plot_test[:, 12]
        obs_generated = plot_samples[:, 12]
        # Get the name and the range of the observable
        obs_name = obs_names[12]
        obs_range = [17,82]

        # Create the plot
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=obs_range,
                 weight_samples=iterations,
                 error_range=[0.83,1.17],
                 n_jets=2,
                 y_ticks=[0.9,1,1.1])

        obs_name = "\Delta R_{j_1 j_2}"
        obs_train = delta_r(plot_train)
        obs_test = delta_r(plot_test)
        weights = plot_weights
        obs_generated = delta_r(plot_samples)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins= 55,
                 weight_samples=iterations,
                 predict_weights=weights,
                 error_range = [0.83,1.17],
                 n_jets=2,
                 y_ticks=[0.9,1,1.1])

        plot_train = np.load(f"paper/jet/{path}_train_3.npy")
        plot_test = np.load(f"paper/jet/{path}_test_3.npy")
        plot_samples = np.load(f"paper/jet/{path}_sample_3.npy")
        
        obs_name = "\Delta R_{j_1 j_3}"
        obs_train = delta_r(plot_train, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples, idx_phi1=9, idx_eta1=10, idx_phi2=17,
                                idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=iterations,
                 predict_weights=weights,
                 error_range = [0.83,1.17],
                 n_jets=3,
                 y_ticks=[0.9,1,1.1])
        obs_name = "\Delta R_{j_2 j_3}"
        obs_train = delta_r(plot_train, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples, idx_phi1=13, idx_eta1=14, idx_phi2=17,
                                idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=iterations,
                 predict_weights=weights,
                 error_range = [0.83,1.17],
                 n_jets=3,
                 y_ticks=[0.9,1,1.1])
