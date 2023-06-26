import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from matplotlib.backends.backend_pdf import PdfPages
import os, sys

from Source.Models.autoregGMM import AutoRegGMM
from Source.Models.autoregBinned import AutoRegBinned
from Source.Util.simulateToyData import ToySimulator
from Source.Util.util import get, get_device, magic_trafo, inverse_magic_trafo, load_params
from Source.Experiments import z1,z2,z3
from Source.Util.plots import plot_obs, delta_r
from Source.Util.physics import get_M_ll
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
plt.rcParams["figure.figsize"] = (9,9)
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

    FONTSIZE = 30
    labels = ["True", "CFM", "Train"]
    colors = ["black","#A52A2A", "#0343DE"]
    dup_last = lambda a: np.append(a, a[-1])

    fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
    fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

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

    for line in axs[0].legend(loc="center right", frameon=False, fontsize=FONTSIZE).get_lines():
        line.set_linewidth(3.0)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    names = ["p_{T,l1}", "\mu_{l1}","p_{T,l2}", "\mu_{l2}","p_{T,j1}", "\mu_{j1}",
                          "p_{T,j2}","\mu_{j2}","p_{T,j3}", "\mu_{j3}"]

    if name in names:
        axs[0].set_yscale("log")

    axs[1].set_ylabel(r"$\frac{\mathrm{CFM}}{\mathrm{True}}$",
                          fontsize=FONTSIZE)
    axs[1].set_yticks(y_ticks)
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=y_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=y_ticks[2], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=y_ticks[0], c="black", ls="dotted", lw=0.5)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)
    if range:
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

    axs[0].tick_params(axis="both", labelsize=FONTSIZE-3)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE-3)
    axs[2].tick_params(axis="both", labelsize=FONTSIZE-3)

    corner_text(axs[0],f"Z+{n_jets} jet exclusive",horizontal_pos="right",vertical_pos="top", fontsize=FONTSIZE)

    plt.savefig(pp, format="pdf", bbox_inches='tight')
    plt.close()



path = sys.argv[1]

params = load_params(os.path.join(path, "paramfile.yaml"))

params["warm_start"] = True
params["warm_start_path"] = path
params['train'] = False

params['redirect_console'] = True
params['plot_loss'] = False

params["plot"] = True
params['iterations'] = 30
params['n_samples'] = 1000000
#params['iterations'] = 10
params['batch_size_sample'] = 100000
params['save_samples'] = True


n_jets = get(params,"n_jets",3)
if n_jets == 1:
    experiment = z1.Z1_Experiment(params)

elif n_jets == 2:
    experiment = z2.Z2_Experiment(params)
elif n_jets == 3:
    experiment = z3.Z3_Experiment(params)
else:
    experiment = None

params["sample"] = True
experiment.full_run()

plot_train = []
plot_test = []
plot_samples = []
plot_weights = []
weights = None
unit = "GeV"

if n_jets !=3:
    for i in range(n_jets, 4):
        plot_train_jets = experiment.model.data_train[experiment.model.data_train[:, 0] == i]
        plot_train_jets = plot_train_jets[:,1:]
        plot_train.append(plot_train_jets)

        plot_test_jets = experiment.model.data_test[experiment.model.data_test[:, 0] == i]
        plot_test_jets = plot_test_jets[:,1:]
        plot_test.append(plot_test_jets)

        plot_samples_jets = experiment.samples[experiment.samples[:, 0] == i]
        plot_samples_jets = plot_samples_jets[:,1:]
        plot_samples.append(plot_samples_jets)

        if get(params, "magic_transformation", False):
            if n_jets == 2:
                deltaR12 = delta_r(plot_samples_jets, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
                weights = inverse_magic_trafo(deltaR12)
        plot_weights.append(weights)

else:
    plot_train.append(experiment.model.data_train)
    plot_test.append(experiment.model.data_test)
    plot_samples.append(experiment.samples)

    if get(params, "magic_transformation", False):

        deltaR12 = delta_r(experiment.samples, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
        deltaR13 = delta_r(experiment.samples, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        deltaR23 = delta_r(experiment.samples, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        weights12 = inverse_magic_trafo(deltaR12)
        weights13 = inverse_magic_trafo(deltaR13)
        weights23 = inverse_magic_trafo(deltaR23)
        weights = weights12 * weights13 * weights23

    plot_weights.append(weights)


with PdfPages(f"{path}/paper_plots.pdf") as out:
    for i, channel in enumerate(experiment.model.params["plot_channels"]):
        obs_train = plot_train[0][:, channel]
        obs_test = plot_test[0][:, channel]
        obs_generated = plot_samples[0][:, channel]
        # Get the name and the range of the observable
        obs_name = experiment.model.obs_names[channel]
        obs_range = experiment.model.obs_ranges[channel]
        obs_unit = experiment.obs_units[channel]

        weights = plot_weights[0]
        print(obs_generated.shape)
        print(channel)
        # Create the plot
        plot_paper(pp=out,
                    obs_train=obs_train,
                    obs_test=obs_test,
                    obs_predict=obs_generated,
                    name=obs_name,
                    range=obs_range,
                    weight_samples=experiment.model.iterations,
                    error_range=[0.71,1.29],
                    n_jets=n_jets,
                    y_ticks=[0.8,1,1.2],
                    unit = obs_unit,
                    predict_weights=weights)

    obs_name = "M_{\mu \mu}"
    obs_range = [79, 104]
    data_train = get_M_ll(plot_train[0])
    data_test = get_M_ll(plot_test[0])
    data_generated = get_M_ll(plot_samples[0])
    plot_paper(pp=out,
             obs_train=data_train,
             obs_test=data_test,
             obs_predict=data_generated,
             name=obs_name,
             range=obs_range,
             bins=60,
             weight_samples=experiment.model.iterations,
             predict_weights=weights,
             error_range=[0.71,1.29],
             n_jets=n_jets,
             y_ticks=[0.8,1,1.2],
             unit = unit)

    obs_name = "\Delta R_{l1 l2}"
    obs_train = delta_r(plot_train[0], idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
    obs_test = delta_r(plot_test[0], idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
    obs_generated = delta_r(plot_samples[0], idx_phi1=1, idx_eta1=2, idx_phi2=5, idx_eta2=6)
    weights = plot_weights[0]
    plot_paper(pp=out,
             obs_train=obs_train,
             obs_test=obs_test,
             obs_predict=obs_generated,
             name=obs_name,
             n_jets=n_jets,
             range=[0, 8],
             weight_samples=experiment.model.iterations,
             predict_weights=weights,
             error_range=[0.71, 1.29],
             y_ticks=[0.8, 1, 1.2])

    obs_name = "\Delta R_{l1 j1}"
    obs_train = delta_r(plot_train[0], idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
    obs_test = delta_r(plot_test[0], idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
    obs_generated = delta_r(plot_samples[0], idx_phi1=1, idx_eta1=2, idx_phi2=9, idx_eta2=10)
    plot_paper(pp=out,
             obs_train=obs_train,
             obs_test=obs_test,
             obs_predict=obs_generated,
             name=obs_name,
             n_jets=n_jets,
             range=[0, 8],
             weight_samples=experiment.model.iterations,
             predict_weights=weights,
             error_range=[0.71, 1.29],
             y_ticks=[0.8, 1, 1.2])

    obs_name = "\Delta R_{l2 j1}"
    obs_train = delta_r(plot_train[0], idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
    obs_test = delta_r(plot_test[0], idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
    obs_generated = delta_r(plot_samples[0], idx_phi1=5, idx_eta1=6, idx_phi2=9, idx_eta2=10)
    plot_paper(pp=out,
             obs_train=obs_train,
             obs_test=obs_test,
             obs_predict=obs_generated,
             name=obs_name,
             n_jets= n_jets,
             range=[0, 8],
             weight_samples=experiment.model.iterations,
             predict_weights=weights,
             error_range=[0.71, 1.29],
             y_ticks=[0.8, 1, 1.2])

    if n_jets == 1:
        differences = [[2, 6], [2, 10], [6, 10], [5, 9], [1, 5], [1, 9]]
    elif n_jets == 2:
        differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13], [1, 5], [1, 9],
                       [1, 13]]
    else:
        differences = [[2, 6], [2, 10], [2, 14], [6, 10], [6, 14], [10, 14], [5, 9], [5, 13], [9, 13],
                       [2, 18], [6, 18], [10, 18], [14, 18], [5, 17], [9, 17], [13, 17], [1, 5], [1, 9], [1, 13],
                       [1, 17]]
    for channels in differences:
        channel1 = channels[0]
        channel2 = channels[1]
        obs_name = experiment.model.obs_names[channel1] + " - " + experiment.model.obs_names[channel2]
        obs_train = plot_train[0][:, channel1] - plot_train[0][:, channel2]
        obs_test = plot_test[0][:, channel1] - plot_test[0][:, channel2]
        obs_generated = plot_samples[0][:, channel1] - plot_samples[0][:, channel2]
        weights = plot_weights[0]
        if channels in [[1, 5], [1, 9], [1, 13], [1, 17], [5, 9], [5, 13], [5, 17],
                        [9, 13], [9, 17], [13, 17]]:
            obs_train = (obs_train + np.pi) % (2 * np.pi) - np.pi
            obs_test = (obs_test + np.pi) % (2 * np.pi) - np.pi
            obs_generated = (obs_generated + np.pi) % (2 * np.pi) - np.pi
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 n_jets= n_jets,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range=[0.71, 1.29],
                 y_ticks=[0.8, 1, 1.2])

    if n_jets > 1:
        obs_name = "\Delta R_{j1 j2}"
        obs_train = delta_r(plot_train[0])
        obs_test = delta_r(plot_test[0])
        weights = plot_weights[0]
        obs_generated = delta_r(plot_samples[0])
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins= 55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range = [0.71,1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8,1,1.2])

        obs_name = "\Delta R_{l1 j2}"
        obs_train = delta_r(plot_train[0], idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        obs_test = delta_r(plot_test[0], idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        obs_generated = delta_r(plot_samples[0], idx_phi1=1, idx_eta1=2, idx_phi2=13, idx_eta2=14)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range=[0.71, 1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8, 1, 1.2])

        obs_name = "\Delta R_{l2 j2}"
        obs_train = delta_r(plot_train[0], idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        obs_test = delta_r(plot_test[0], idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        obs_generated = delta_r(plot_samples[0], idx_phi1=5, idx_eta1=6, idx_phi2=13, idx_eta2=14)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range=[0.71, 1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8, 1, 1.2])

    if n_jets > 2:
        obs_name = "\Delta R_{j1 j3}"
        obs_train = delta_r(plot_train[0], idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test[0], idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples[0], idx_phi1=9, idx_eta1=10, idx_phi2=17,
                                idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range = [0.71,1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8,1,1.2])
        obs_name = "\Delta R_{j2 j3}"
        obs_train = delta_r(plot_train[0], idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test[0], idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples[0], idx_phi1=13, idx_eta1=14, idx_phi2=17,
                                idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range = [0.71,1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8,1,1.2])

        obs_name = "\Delta R_{l1 j3}"
        obs_train = delta_r(plot_train[0], idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test[0], idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples[0], idx_phi1=1, idx_eta1=2, idx_phi2=17, idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range=[0.71, 1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8, 1, 1.2])

        obs_name = "\Delta R_{l2 j3}"
        obs_train = delta_r(plot_train[0], idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
        obs_test = delta_r(plot_test[0], idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
        obs_generated = delta_r(plot_samples[0], idx_phi1=5, idx_eta1=6, idx_phi2=17, idx_eta2=18)
        plot_paper(pp=out,
                 obs_train=obs_train,
                 obs_test=obs_test,
                 obs_predict=obs_generated,
                 name=obs_name,
                 range=[0, 8],
                 bins=55,
                 weight_samples=experiment.model.iterations,
                 predict_weights=weights,
                 error_range=[0.71, 1.29],
                 n_jets=n_jets,
                 y_ticks=[0.8, 1, 1.2])






