import numpy as np
import os, sys, time
sys.path.append(os.getcwd()) #can we do better?
from Source.Util.util import load_params

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")
import matplotlib.font_manager as font_manager

from Source.Util.plots import plot_obs, delta_r, delta_phi
from Source.Util.physics import get_M_ll

### setup matplotlib
import matplotlib.font_manager as font_manager
font_dir = ['paper/bitstream-charter-ttf/Charter/']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
plt.rcParams["figure.figsize"] = (9,9)

model_name = "AT"
n_bins = 60
y_ticks = [0.8, 1., 1.2]
error_range = [0.71,1.29]

path = "AutoRegGMMcond_4630"
n_samples = 1000000
iterations = 30
full = False

#path = "ensemble"
#iterations = 7

obs_names = ["p_{T,l1} \mathrm{[GeV]}", "\phi_{l1}", "\eta_{l1}", "\mu_{l1} \mathrm{[GeV]}",
                          "p_{T,l2} \mathrm{[GeV]}", "\phi_{l2}", "\eta_{l2}", "\mu_{l2} \mathrm{[GeV]}",
                          "p_{T,j1} \mathrm{[GeV]}", "\phi_{j1}", "\eta_{j1}", "\mu_{j1} \mathrm{[GeV]}",
                          "p_{T,j2} \mathrm{[GeV]}", "\phi_{j2}", "\eta_{j2}", "\mu_{j2} \mathrm{[GeV]}",
                          "p_{T,j3} \mathrm{[GeV]}", "\phi_{j3}", "\eta_{j3}", "\mu_{j3} \mathrm{[GeV]}"]


obs_ranges = [[0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50],
                           [0.5, 150], [-4, 4], [-6, 6], [0, 50]]

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

def plot_paper(pp, obs_train, obs_test, obs_predict, name, bins=n_bins, weight_samples=1,
               predict_weights=None, unit=None, range=None, error_range=None, n_jets=None, y_ticks=None,
               inclusive=False):

    y_t, bins = np.histogram(obs_test, bins=bins, range=range)
    y_tr, _ = np.histogram(obs_train, bins=bins)

    if weight_samples == 1:
        if inclusive:
            obs_predict = np.concatenate(obs_predict)
        y_g, _ = np.histogram(obs_predict, bins=bins, weights=predict_weights)
        hists = [y_t, y_g, y_tr]
        hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]
    else:
        if inclusive:
            for i,_ in enumerate(obs_predict):
                obs_predict[i] = obs_predict[i].reshape(weight_samples, len(obs_predict[i])//weight_samples)
            obs_predict = np.concatenate(obs_predict, axis=1)
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

    FONTSIZE = 30
    TICKLABELSIZE = FONTSIZE-3
    labels = ["Truth", model_name, "Train"]
    colors = ["black","#A52A2A","#0343DE"]
    dup_last = lambda a: np.append(a, a[-1])

    fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
    #fig1.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5, rect=(0.1, 0.1, 1., 1.))

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

        if label == "Truth": continue

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
                                dup_last(ratio + ratio_err), facecolor=color, alpha=0.3, step="post")

        delta = np.fabs(ratio - 1) * 100
        delta_err = ratio_err * 100

        markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                                  yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                                  linewidth=0, fmt=".", capsize=2, markersize=10)
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]

    #axs[0].legend(loc="center right", frameon=False, fontsize=FONTSIZE)
    for line in axs[0].legend(loc="center right", frameon=False, fontsize=FONTSIZE).get_lines():
            line.set_linewidth(3.0)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    if "p_{T" in name:
        axs[0].set_yscale("log")

    axs[1].set_ylabel(r"$\frac{\mathrm{%s}}{\mathrm{Truth}}$" % model_name,
                          fontsize=FONTSIZE)
    axs[1].set_yticks(y_ticks)
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
    axs[1].axhline(y=y_ticks[0], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=y_ticks[2], c="black", ls="dotted", lw=0.5)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)

    axs[2].set_xlim((range[0]+0.1,range[1]-0.1))
    if range[1]==8: #for deltaR
        axs[2].set_xticks(np.arange(1, 8))

    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[0].tick_params(axis="both", labelsize=TICKLABELSIZE)
    axs[1].tick_params(axis="both", labelsize=TICKLABELSIZE)
    axs[2].tick_params(axis="both", labelsize=TICKLABELSIZE)

    if n_jets is not None:
        corner_text(axs[0],f"Z+{n_jets} jet exclusive",horizontal_pos="right",vertical_pos="top", fontsize=FONTSIZE)
    else:
        corner_text(axs[0],f"Z+jets inclusive",horizontal_pos="right",vertical_pos="top", fontsize=FONTSIZE)

    plt.savefig(pp, format="pdf", bbox_inches="tight")
    plt.close()

    #gc.collect()
    print(f"Finished {name} for {n_jets} jets")

def doPlots(path, n_samples=1000000, iterations=30, full=False):
    pdfname = f"paper/jet/{path}.pdf" if not full else f"paper/jet/{path}_full.pdf"
    with PdfPages(pdfname) as out:
        plot_train_1 = np.load(f"paper/jet/{path}_train_1.npy")
        plot_test_1 = np.load(f"paper/jet/{path}_test_1.npy")
        plot_samples_1 = np.load(f"paper/jet/{path}_sample_1.npy")
        weights_1=None
        
        obs_train = plot_train_1[:, 8]
        obs_test = plot_test_1[:, 8]
        obs_generated = plot_samples_1[:, 8]
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
                predict_weights=weights_1,
                 error_range=error_range,
                 n_jets=1,
                 y_ticks=y_ticks)
        
        obs_name = "M_{\mu \mu} \mathrm{[GeV]}"
        obs_range = [79, 104]
        data_train = get_M_ll(plot_train_1)
        data_test = get_M_ll(plot_test_1)
        data_generated = get_M_ll(plot_samples_1)
        plot_paper(pp=out,
                 obs_train=data_train,
                 obs_test=data_test,
                 obs_predict=data_generated,
                 name=obs_name,
                 range=obs_range,
                 weight_samples=iterations,
                 predict_weights=weights_1,
                 error_range=error_range,
                 n_jets=1,
                 y_ticks=y_ticks)

        plot_train_2 = np.load(f"paper/jet/{path}_train_2.npy")
        plot_test_2 = np.load(f"paper/jet/{path}_test_2.npy")
        plot_samples_2 = np.load(f"paper/jet/{path}_sample_2.npy")
        weights_2=None

        obs_train = plot_train_2[:, 12]
        obs_test = plot_test_2[:, 12]
        obs_generated = plot_samples_2[:, 12]
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
                 error_range=error_range,
                 predict_weights=weights_2,
                 n_jets=2,
                 y_ticks=y_ticks)

        obs_name = "\Delta R_{j1, j2}"
        obs_train = delta_r(plot_train_2)
        obs_test = delta_r(plot_test_2)
        obs_generated = delta_r(plot_samples_2)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 weight_samples=iterations,
                 predict_weights=weights_2,
                 error_range = [0.71,1.29],
                 n_jets=2,
                 y_ticks=y_ticks)

        plot_train_3 = np.load(f"paper/jet/{path}_train_3.npy")
        plot_test_3 = np.load(f"paper/jet/{path}_test_3.npy")
        plot_samples_3 = np.load(f"paper/jet/{path}_sample_3.npy")
        weights_3=None

        obs_name = "\Delta R_{j1, j2}"
        obs_train = delta_r(plot_train_3)
        obs_test = delta_r(plot_test_3)
        obs_generated = delta_r(plot_samples_3)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 weight_samples=iterations,
                 predict_weights=weights_3,
                 error_range = [0.71,1.29],
                 n_jets=3,
                 y_ticks=y_ticks)
        
        obs_name = "\Delta R_{j1, j3}"
        obs_train = delta_r(plot_train_3, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test_3, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples_3, idx_phi1=9, idx_eta1=10, idx_phi2=17,
                                idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 weight_samples=iterations,
                 predict_weights=weights_3,
                 error_range = error_range,
                 n_jets=3,
                 y_ticks=y_ticks)
        obs_name = "\Delta R_{j2, j3}"
        obs_train = delta_r(plot_train_3, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test_3, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples_3, idx_phi1=13, idx_eta1=14, idx_phi2=17,
                                idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 weight_samples=iterations,
                 predict_weights=weights_3,
                 error_range = error_range,
                 n_jets=3,
                 y_ticks=y_ticks)
        '''
        #inclusive plot
        obs_name = "\sum_i p_{T,j_i}"
        obs_range = [15,250]

        obs_train = plot_train_1[:,8]
        obs_train = np.append(obs_train, np.sum(plot_train_2[:,[8,12]], axis=-1))
        obs_train = np.append(obs_train, np.sum(plot_train_3[:,[8,12,16]], axis=-1))

        obs_test = plot_test_1[:,8]
        obs_test = np.append(obs_test, np.sum(plot_test_2[:,[8,12]], axis=-1))
        obs_test = np.append(obs_test, np.sum(plot_test_3[:,[8,12,16]], axis=-1))

        samples_1 = plot_samples_1[:,8]
        samples_2 = plot_samples_2[:,[8,12]].sum(axis=-1)
        samples_3 = plot_samples_3[:,[8,12,16]].sum(axis=-1)

        n1 = round(len(plot_train_2)/len(plot_train_1) * n_samples) * iterations
        n2 = round(len(plot_train_3)/len(plot_train_1) * n_samples) * iterations
        obs_generated = [samples_1, samples_2[:n1], samples_3[:n2]]

        weights_inclusive=None
        
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=obs_range,
                 weight_samples=iterations,
                 predict_weights=weights_inclusive,
                 error_range = error_range,
                 n_jets=None, inclusive=True,
                 y_ticks=y_ticks)

        if not full:
            return 0

        
        #individual components
        for n_jets in [1,2,3]:
            plot_channels = np.array([i for i in range(n_jets * 4 + 8) if i not in [1, 3, 7]]).tolist()
            for _, channel in enumerate(plot_channels):
                obs_train = eval(f"plot_train_{n_jets}")[:, channel]
                obs_test = eval(f"plot_test_{n_jets}")[:, channel]
                obs_generated = eval(f"plot_samples_{n_jets}")[:, channel]
                # Get the name and the range of the observable
                obs_name = obs_names[channel]
                obs_range = obs_ranges[channel]
                # Create the plot
                plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             range=obs_range,
                             n_jets=n_jets,
                             weight_samples=iterations,
                             predict_weights=eval(f"weights_{n_jets}"),
                             error_range=error_range,
                             y_ticks=y_ticks)

        
        #Delta R
        # 1-jet
        obs_name = "\Delta R_{l_1 l_2}"
        obs_train = delta_r(plot_train_1, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        obs_test = delta_r(plot_test_1, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        obs_generated = delta_r(plot_samples_1, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=1,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_1)

        obs_name = "\Delta R_{l_1 j_1}"
        obs_train = delta_r(plot_train_1, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        obs_test = delta_r(plot_test_1, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        obs_generated = delta_r(plot_samples_1, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=1,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_1)

        obs_name = "\Delta R_{l_2 j_1}"
        obs_train = delta_r(plot_train_1, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        obs_test = delta_r(plot_test_1, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        obs_generated = delta_r(plot_samples_1, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=1,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_1)

        # 2-jet
        obs_name = "\Delta R_{l_1 l_2}"
        obs_train = delta_r(plot_train_2, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        obs_test = delta_r(plot_test_2, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        obs_generated = delta_r(plot_samples_2, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_2)

        obs_name = "\Delta R_{l_1 j_1}"
        obs_train = delta_r(plot_train_2, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        obs_test = delta_r(plot_test_2, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        obs_generated = delta_r(plot_samples_2, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_2)

        obs_name = "\Delta R_{l_2 j_1}"
        obs_train = delta_r(plot_train_2, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        obs_test = delta_r(plot_test_2, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        obs_generated = delta_r(plot_samples_2, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=2,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_2)

        obs_name = "\Delta R_{l_1 j_2}"
        obs_train = delta_r(plot_train_2, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        obs_test = delta_r(plot_test_2, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        obs_generated = delta_r(plot_samples_2, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                                 y_ticks=y_ticks, error_range=error_range,
                                 n_jets=2,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights_2)

        obs_name = "\Delta R_{l_2 j_2}"
        obs_train = delta_r(plot_train_2, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        obs_test = delta_r(plot_test_2, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        obs_generated = delta_r(plot_samples_2, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                 n_jets=2,
                                 range=[0, 8],
                                 weight_samples=iterations,
                             predict_weights=weights_2)

        # 3-jet
        obs_name = "\Delta R_{l_1 l_2}"
        obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_3)

        obs_name = "\Delta R_{l_1 j_1}"
        obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_3)

        obs_name = "\Delta R_{l_2 j_1}"
        obs_train = delta_r(plot_train_3, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        obs_test = delta_r(plot_test_3, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        obs_generated = delta_r(plot_samples_3, idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
        plot_paper(pp=out,
                             obs_train=obs_train,
                             obs_test=obs_test,
                             obs_predict=obs_generated,
                             name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                             n_jets=3,
                             range=[0, 8],
                             weight_samples=iterations,
                             predict_weights=weights_3)

        obs_name = "\Delta R_{l_1 j_2}"
        obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights_3)

        obs_name = "\Delta R_{l_2 j_2}"
        obs_train = delta_r(plot_train_3, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        obs_test = delta_r(plot_test_3, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        obs_generated = delta_r(plot_samples_3, idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights_3)
                
        obs_name = "\Delta R_{l_1 j_3}"
        obs_train = delta_r(plot_train_3, idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test_3, idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples_3, idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
        plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights_3)

        obs_name = "\Delta R_{l_2 j_3}"
        obs_train = delta_r(plot_train_3, idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test_3, idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples_3, idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
        plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                 n_jets=3,
                                 range=[0, 8],
                                 weight_samples=iterations,
                                 predict_weights=weights_3)
        '''
        # angular differences
        #def makeAngle(angle):
        #    return (angle + np.pi)%(2*np.pi) - np.pi
        # 1-jet
        differences = [[2, 6], [2, 10], [6, 10], [5, 9]]
        for channels in differences:
            channel1 = channels[0]
            channel2 = channels[1]
            obs_name = obs_names[channel1] + " - " + obs_names[channel2]

            if channel1%2==0: #is eta
                obs_range = [-8., 8.]
                obs_train = plot_train_1[:, channel1] - plot_train_1[:, channel2]
                obs_test = plot_test_1[:, channel1] - plot_test_1[:, channel2]
                obs_generated = plot_samples_1[:, channel1] - plot_samples_1[:, channel2]
            else: #is phi
                obs_range=[-np.pi, np.pi]
                obs_train = delta_phi(plot_train_1, channel1, channel2)
                obs_test = delta_phi(plot_test_1, channel1, channel2)
                obs_generated = delta_phi(plot_samples_1, channel1, channel2)
            plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                 n_jets=1,
                                   range=obs_range,
                                 weight_samples=iterations,
                                 predict_weights=weights_1)
        # 2-jet
        differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13], [1,5], [1,9], [1,13]]
        for channels in differences:
            channel1 = channels[0]
            channel2 = channels[1]
            obs_name = obs_names[channel1] + " - " + obs_names[channel2]

            if channel1%2==0: #is eta
                obs_range = [-8., 8.]
                obs_train = plot_train_2[:, channel1] - plot_train_2[:, channel2]
                obs_test = plot_test_2[:, channel1] - plot_test_2[:, channel2]
                obs_generated = plot_samples_2[:, channel1] - plot_samples_2[:, channel2]
            else: #is phi
                obs_range=[-np.pi, np.pi]
                obs_train = delta_phi(plot_train_2, channel1, channel2)
                obs_test = delta_phi(plot_test_2, channel1, channel2)
                obs_generated = delta_phi(plot_samples_2, channel1, channel2)
            plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                                range=obs_range,
                                 n_jets=2,
                                 weight_samples=iterations,
                                 predict_weights=weights_2)
        # 3-jet
        differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13],
                               [2, 18], [6, 18], [10, 18], [14, 18], [5, 17], [9, 17], [13, 17]]
        for channels in differences:
            channel1 = channels[0]
            channel2 = channels[1]
            obs_name = obs_names[channel1] + " - " + obs_names[channel2]

            if channel1%2==0: #is eta    
                obs_range = [-8., 8.]        
                obs_train = plot_train_3[:, channel1] - plot_train_3[:, channel2]
                obs_test = plot_test_3[:, channel1] - plot_test_3[:, channel2]
                obs_generated = plot_samples_3[:, channel1] - plot_samples_3[:, channel2]
            else: #is phi
                obs_range=[-np.pi, np.pi]
                obs_train = delta_phi(plot_train_3, channel1, channel2)
                obs_test = delta_phi(plot_test_3, channel1, channel2)
                obs_generated = delta_phi(plot_samples_3, channel1, channel2)
            plot_paper(pp=out,
                                 obs_train=obs_train,
                                 obs_test=obs_test,
                                 obs_predict=obs_generated,
                                 name=obs_name,
                             y_ticks=y_ticks, error_range=error_range,
                               range=obs_range,
                                 n_jets=3,
                                 weight_samples=iterations,
                                 predict_weights=weights_3)

doPlots("AutoRegGMMcond_4630", full=True)
