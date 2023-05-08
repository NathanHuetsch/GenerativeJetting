import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from matplotlib.backends.backend_pdf import PdfPages
import os, sys
os.chdir("..")
from Source.Models.autoregGMM import AutoRegGMM
from Source.Models.autoregBinned import AutoRegBinned
from Source.Util.simulateToyData import ToySimulator
from Source.Util.util import load_params, get, get_device
from Source.Experiments import toy
import matplotlib as mpl
import matplotlib.font_manager as font_manager
font_dir = ['paper/bitstream-charter-ttf/Charter/']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.font_manager.findSystemFonts(fontpaths='scipostphys-matplotlib', fontext='ttf')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
plt.rcParams["figure.figsize"] = (9,7)

path = sys.argv[1]
toy_type = sys.argv[2]
data_split = 0.6


data = np.load("/remote/gpu07/huetsch/data/2dGaussSphere.npy")
n_data = len(data)
cut1 = int(n_data * data_split)
data_train = data[:cut1]
data_test = data[cut1:]

mus = []
sigmas = []
for i in range(0,10):
    path_path = path + f"/run_0{i}/"
    mu_path = path_path + "sigmamu_R_mu.npy"
    sigma_path = path_path + "sigmamu_R_sigma.npy"
    mu = np.load(mu_path)
    sigma = np.load(sigma_path)
    mus.append(mu)
    sigmas.append(sigma)

mus = np.array(mus)
sigmas = np.array(sigmas)
def plot_paper(out, obs_train, obs_test, obs_predict, name, bins=60, range=None,unit=None, ymaxAbs=1., ymaxRel=1.,):

    with PdfPages(out) as pp:
        y_t, bins = np.histogram(obs_test, bins=bins, range=range)
        y_tr, _ = np.histogram(obs_train, bins=bins)
        mus, sigmas = obs_predict[0] , obs_predict[1]

        mu = np.mean(mus, axis=0)
        sigma = np.mean(sigmas, axis=0)

        std = np.std(sigmas, axis=0)

        hists = [y_t, mu, y_tr]
        hist_errors = [np.sqrt(y_t), sigma , np.sqrt(y_tr)]
        integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
        scales = [1 / integral if integral != 0. else 1. for integral in integrals]

        FONTSIZE = 30
        labels = ["True", "CFM", "Train"]
        colors = ["black","#A52A2A", "#0343DE"]
        dup_last = lambda a: np.append(a, a[-1])

        fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
        fig1.tight_layout(pad=0.9, w_pad=0.9, h_pad=0.5, rect=(0.1, 0.08, 1, 1))

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

            axs[1].step(bins, dup_last(ratio), linewidth=3.0, where="post", color=color)
            axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
            axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
            axs[1].fill_between(bins, dup_last(ratio - ratio_err),
                                dup_last(ratio + ratio_err), facecolor=color, alpha=0.25, step="post")

            delta = np.fabs(ratio - 1) * 100
            delta_err = ratio_err * 100

            markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                                  yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                                  linewidth=0, fmt=".", capsize=2, markersize=10)
            [cap.set_alpha(0.5) for cap in caps]
            [bar.set_alpha(0.5) for bar in bars]

        #axs[0].legend(loc="lower right", frameon=False, fontsize=FONTSIZE)
        for line in axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE).get_lines():
            line.set_linewidth(3.0)
        axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

        axs[1].set_ylabel(r"$\frac{\mathrm{CFM}}{\mathrm{True}}$",
                          fontsize=FONTSIZE)
        y_ticks = [0.9, 1, 1.1]
        axs[1].set_yticks(y_ticks)
        axs[1].set_ylim([0.81, 1.19])
        axs[1].axhline(y=y_ticks[1], c="black", ls="--", lw=0.7)
        axs[1].axhline(y=y_ticks[2], c="black", ls="dotted", lw=0.5)
        axs[1].axhline(y=y_ticks[0], c="black", ls="dotted", lw=0.5)
        plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)
        #plt.xlim((range[0] + 0.01, range[1] - 0.01))
        plt.xlim(0.65, 1.35)

        axs[2].set_ylim((0.05, 20))
        axs[2].set_yscale("log")
        axs[2].set_yticks([0.1, 1.0, 10.0])
        axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
        axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

        axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
        axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
        axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

        axs[0].tick_params(axis="both", labelsize=FONTSIZE-6)
        axs[1].tick_params(axis="both", labelsize=FONTSIZE-6)
        axs[2].tick_params(axis="both", labelsize=FONTSIZE-6)

        plt.savefig(pp, format="pdf")
        plt.close()

        fig2, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.00})
        #fig2, axs = plt.subplots(1, 1)
        fig2.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0, rect=(0.1, 0.08, 1, 1))

        axs[0].set_ylabel(r"$\sigma$", fontsize=FONTSIZE)
        #axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
        #               fontsize=FONTSIZE)

        axs[0].step(bins, dup_last(hist_errors[1] * scales[1]), color=colors[1],linewidth=3.0, where="post")

        axs[0].step(bins, dup_last(hist_errors[1] - std) * scales[1], color=colors[1],
                    alpha=0.5, linewidth=0.5, where="post")
        axs[0].step(bins, dup_last(hist_errors[1] + std) * scales[1], color=colors[1],
                    alpha=0.5, linewidth=0.5, where="post")
        axs[0].fill_between(bins, dup_last(hist_errors[1] - std) * scales[1],
                            dup_last(hist_errors[1] + std) * scales[1], facecolor=colors[1],
                            alpha=0.3, step="post")

        axs[0].set_ylim(0., ymaxAbs)
        axs[1].set_yticks([0, 0.02, 0.04])


        #plt.savefig(pp, format="pdf")
        #plt.close()

        #fig3, axs = plt.subplots(1, 1)
        #fig3.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.07, 0.06, 0.99, 0.95))

        axs[1].set_ylabel(r"$\sigma / \mu$", fontsize=FONTSIZE)
        #axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
        #               fontsize=FONTSIZE)

        axs[1].step(bins, dup_last(hist_errors[1] / hists[1]), color=colors[1],linewidth=3.0, where="post")
        axs[1].step(bins, dup_last((hist_errors[1] - std)/hists[1]) * scales[1], color=colors[1],
                 alpha=0.5, linewidth=0.5, where="post")
        axs[1].step(bins, dup_last((hist_errors[1] + std)/hists[1]) * scales[1], color=colors[1],
                 alpha=0.5, linewidth=0.5, where="post")
        axs[1].fill_between(bins, dup_last((hist_errors[1] - std)/hists[1]) * scales[1],
                         dup_last((hist_errors[1] + std)/hists[1]) * scales[1], facecolor=colors[1],
                         alpha=0.3, step="post")

        axs[1].set_ylim(0., ymaxRel)
        axs[1].set_yticks([0, 0.05, 0.1])

        axs[0].tick_params(axis="both", labelsize=FONTSIZE-6)
        axs[1].tick_params(axis="both", labelsize=FONTSIZE-6)

        fig2.align_labels()
        plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)
        #plt.xlim((range[0] + 0.01, range[1] - 0.01))
        plt.xlim(0.65, 1.35)

        plt.savefig(pp, format="pdf")
        plt.close()


# %%
if toy_type == "ramp":
    plot_paper(f"{path}paper_plots_new.pdf", data_train[:, 1], data_test[:, 1],
               [mus,sigmas], "x_1", ymaxAbs=.05, ymaxRel=.05, range=[.1, .9])

if toy_type == "gauss_sphere":
    R_train, _ = ToySimulator.getSpherical(data_train)
    R_test, _ = ToySimulator.getSpherical(data_test)
    plot_paper(f"{path}paper_plots_new.pdf", R_train, R_test, [mus,sigmas],
               "R", ymaxAbs=.05, ymaxRel=.15, range=[0.5, 1.5])