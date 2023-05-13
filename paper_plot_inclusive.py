import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, sys

from Source.Util.util import get, get_device, magic_trafo, inverse_magic_trafo, load_params
from Source.Experiments import z1,z2,z3
from Source.Util.plots import delta_r
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


def plot_paper(pp, obs_train_exclusive, obs_test_exclusive, obs_predict_exclusive, name, bins=60, weight_samples=1,
               predict_weights=None, unit=None, range=None, error_range=None, n_jets=None, y_ticks=None):
    obs_train = np.concatenate(obs_train_exclusive)
    obs_test = np.concatenate(obs_test_exclusive)

    y_t, bins = np.histogram(obs_test, bins=bins, range=range)
    y_tr, _ = np.histogram(obs_train, bins=bins)

    if weight_samples == 1:
        obs_predict = np.concatenate(obs_predict_exclusive)
        y_g, _ = np.histogram(obs_predict, bins=bins, weights=predict_weights)
        hists = [y_t, y_g, y_tr]
        hist_errors = [np.sqrt(y_t), np.sqrt(y_g), np.sqrt(y_tr)]
    else:
        for i,_ in enumerate(obs_predict_exclusive):
            predict_weights[i] = predict_weights[i].reshape(weight_samples,
                                                            len(obs_predict_exclusive[i]) // weight_samples)
            predict_weights[i] = predict_weights[i]/predict_weights[i].mean()

            obs_predict_exclusive[i] = obs_predict_exclusive[i].reshape(weight_samples,
                                              len(obs_predict_exclusive[i]) // weight_samples)

        obs_predict = np.concatenate(obs_predict_exclusive, axis=1)

        hist_weights = np.concatenate(predict_weights, axis=1)

        hists_g = np.array([np.histogram(obs_predict[i, :], bins=bins,
                                             weights=hist_weights[i])[0]
                                for i in np.arange(weight_samples)])
        hists = [y_t, np.mean(hists_g, axis=0), y_tr]
        hist_errors = [np.sqrt(y_t), np.std(hists_g, axis=0), np.sqrt(y_tr)]
    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0. else 1. for integral in integrals]

    FONTSIZE = 30
    labels = ["True", "DDPM", "Train"]
    colors = ["black", "#A52A2A", "#0343DE"]
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

    if "p_{T" in name:
        axs[0].set_yscale("log")

    axs[1].set_ylabel(r"$\frac{\mathrm{DDPM}}{\mathrm{True}}$",
                      fontsize=FONTSIZE)
    axs[1].set_yticks(y_ticks)
    axs[1].set_ylim(error_range)
    axs[1].axhline(y=y_ticks[1], c="black", ls="--", lw=0.7)
    axs[1].axhline(y=y_ticks[2], c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=y_ticks[0], c="black", ls="dotted", lw=0.5)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
               fontsize=FONTSIZE)

    plt.xlim((range[0] + 0.1, range[1] - 0.1))

    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                       2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[0].tick_params(axis="both", labelsize=FONTSIZE - 3)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE - 3)
    axs[2].tick_params(axis="both", labelsize=FONTSIZE - 3)

    corner_text(axs[0], f"Z+jets inclusive", horizontal_pos="right", vertical_pos="top", fontsize=FONTSIZE)

    plt.savefig(pp, format="pdf", bbox_inches='tight')
    plt.close()


root = "/remote/gpu05/palacios/GenerativeJetting/"
path_1 = root + sys.argv[1]
path_2 = root + sys.argv[2]
path_3 = root + sys.argv[3]

load_samples = sys.argv[4]

print(f"load samples set to {load_samples}")

params_1 = load_params(os.path.join(path_1, "paramfile.yaml"))
params_2 = load_params(os.path.join(path_2, "paramfile.yaml"))
params_3 = load_params(os.path.join(path_3, "paramfile.yaml"))

plot_train = []
plot_test = []
plot_samples = []
plot_weights = []
weights = None
unit = "GeV"
iterations = 30

for i in range(1,4):
    params = globals()[f"params_{i}"]
    path = globals()[f"path_{i}"]

    params["warm_start"] = True
    params["warm_start_path"] = path
    params['train'] = False

    params['redirect_console'] = False
    params['plot_loss'] = False

    params["plot"] = False
    params['iterations'] = iterations
    params['n_samples'] = 1000000

    params['batch_size_sample'] = 50000
    params['save_samples'] = True


    n_jets = get(params,"n_jets",3)
    if n_jets == 1:
        experiment_1 = z1.Z1_Experiment(params)

    elif n_jets == 2:
        experiment_2 = z2.Z2_Experiment(params)
    elif n_jets == 3:
        experiment_3 = z3.Z3_Experiment(params)
    else:
        experiment_1 = None
        experiment_2 = None
        experiment_3 = None

    if load_samples:
        params["sample"] = False
        globals()[f"experiment_{i}"].full_run()
        globals()[f"samples_{i}"] = []

        for j in range(0,30):
            globals()[f"samples_{i}"].append(np.load(path+f"samples/samples_final_{j}.npy"))

        globals()[f"experiment_{i}"].samples = np.concatenate(globals()[f"samples_{i}"])
    else:
        params["sample"] = True
        globals()[f"experiment_{i}"].full_run()

    if i < 3:

        plot_train_jets = globals()[f"experiment_{i}"].model.data_train_raw[globals()[f"experiment_{i}"].model.data_train_raw[:, 0] == i]
        plot_train_jets = plot_train_jets[:, 1:]

        plot_test_jets = globals()[f"experiment_{i}"].model.data_test_raw[globals()[f"experiment_{i}"].model.data_test_raw[:, 0] == i]
        plot_test_jets = plot_test_jets[:, 1:]

        plot_samples_jets = globals()[f"experiment_{i}"].samples[globals()[f"experiment_{i}"].samples[:, 0] == i]
        plot_samples_jets = plot_samples_jets[:, 1:]



        if i == 2 :
            if get(params, "magic_transformation", False):
                deltaR12 = delta_r(plot_samples_jets, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
                weights = inverse_magic_trafo(deltaR12)

            plot_weights.append(weights)

            plot_train_jets = plot_train_jets[:,8] + plot_train_jets[:,12]
            plot_test_jets = plot_test_jets[:, 8] + plot_test_jets[:, 12]
            plot_samples_jets = plot_samples_jets[:, 8] + plot_samples_jets[:, 12]
        else:
            plot_train_jets = plot_train_jets[:, 8]
            plot_test_jets = plot_test_jets[:, 8]
            plot_samples_jets =plot_samples_jets[:, 8]

            plot_weights.append(np.ones_like(plot_samples_jets))


    else:
        plot_train_jets = globals()[f"experiment_{i}"].model.data_train_raw
        plot_test_jets = (globals()[f"experiment_{i}"].model.data_test_raw)
        plot_samples_jets = (globals()[f"experiment_{i}"].samples)

        if get(params, "magic_transformation", False):
            deltaR12 = delta_r(plot_samples_jets, idx_phi1=9, idx_eta1=10, idx_phi2=13, idx_eta2=14)
            deltaR13 = delta_r(plot_samples_jets, idx_phi1=9, idx_eta1=10, idx_phi2=17, idx_eta2=18)
            deltaR23 = delta_r(plot_samples_jets, idx_phi1=13, idx_eta1=14, idx_phi2=17, idx_eta2=18)
            weights12 = inverse_magic_trafo(deltaR12)
            weights13 = inverse_magic_trafo(deltaR13)
            weights23 = inverse_magic_trafo(deltaR23)
            weights =  weights12 * weights13 * weights23

        plot_train_jets = plot_train_jets[:, 8] + plot_train_jets[:, 12] + plot_train_jets[:, 16]
        plot_test_jets = plot_test_jets[:, 8] + plot_test_jets[:, 12] + plot_test_jets[:, 16]
        plot_samples_jets = plot_samples_jets[:, 8] + plot_samples_jets[:, 12] + plot_samples_jets[:, 16]
        plot_weights.append(weights)

    plot_train.append(plot_train_jets)
    plot_samples.append(plot_samples_jets)
    plot_test.append(plot_test_jets)

c_1 = len(plot_train[1])/len(plot_train[0])
c_2 = len(plot_train[2])/len(plot_train[0])
N_1 = round(c_1 * len(plot_samples[0])/iterations) * iterations
N_2 = round(c_2 * len(plot_samples[0])/iterations) * iterations

plot_samples[1] = plot_samples[1][:N_1]
plot_samples[2] = plot_samples[2][:N_2]

plot_weights[1] = plot_weights[1][:N_1]
plot_weights[2] = plot_weights[2][:N_2]



obs_name = "\sum_i p_{T,ji}"
obs_range = [15,250]

# Create the plot
with PdfPages(root + f"runs/paper_plots.pdf") as out:
    plot_paper(pp=out,
          obs_train_exclusive=plot_train,
          obs_test_exclusive=plot_test,
          obs_predict_exclusive=plot_samples,
          name=obs_name,
          range=obs_range,
          weight_samples=iterations,
          error_range=[0.71,1.29],
          n_jets=1,
          y_ticks=[0.8,1,1.2],
          predict_weights=plot_weights,
          unit = unit)