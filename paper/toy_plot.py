import numpy as np
import os, sys, time
sys.path.append(os.getcwd()) #can we do better?
from Source.Util.util import load_params

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import matplotlib.font_manager as font_manager

### setup matplotlib
font_dir = ['paper/bitstream-charter-ttf/Charter/']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
plt.rcParams["figure.figsize"] = (9,7)

#ramp (GMM, Binned, NN), sphere (GMM, Binned, NN)
yminAbsArr = [[0., 0., 0.], [0., 0., 0.]]
ymaxAbsArr = [[.059, .034, .5], [.039, .029, .5]]
yminRelArr = [[0., 0., 0.], [0., 0., 0.]]
ymaxRelArr = [[.045, .034, .2], [.079, .055, 1.]]
iShowArr = [[5, 5, 5], [5, 5, 5]]
model_name = "AT"

def imodel(model_type):
    if model_type == "AutoRegGMM":
        return 0
    elif model_type == "AutoRegBinned":
        return 1
    elif model_type == "AutoRegNN":
        return 2

def plot(path):
    histograms = np.load(f"paper/toy/{path}.npy")
    params = load_params(f"runs/{path}_1/paramfile.yaml")
    toy_type = params["toy_type"]
    model_type = params["model"]
    
    if toy_type == "ramp":
        name = "x_2"
        unit=None
        yminAbs = yminAbsArr[0][imodel(model_type)]
        ymaxAbs = ymaxAbsArr[0][imodel(model_type)]
        yminRel = yminRelArr[0][imodel(model_type)]
        ymaxRel = ymaxRelArr[0][imodel(model_type)]
        iShow = iShowArr[0][imodel(model_type)]
        x_range = [.1,.9]
        legend_loc = "lower right"
    elif toy_type == "gauss_sphere":
        name = "R"
        unit=None
        yminAbs = yminAbsArr[1][imodel(model_type)]
        ymaxAbs = ymaxAbsArr[1][imodel(model_type)]
        yminRel = yminRelArr[1][imodel(model_type)]
        ymaxRel = ymaxRelArr[1][imodel(model_type)]
        iShow = iShowArr[1][imodel(model_type)]
        x_range = [.65, 1.35] #[.5, 1.5]
        legend_loc = "upper right"

    with warnings.catch_warnings() and PdfPages(f"paper/toy/{path}.pdf") as pp:
        warnings.simplefilter("ignore", RuntimeWarning)
    
        bins = histograms[0,:,0]
        hists = [histograms[2,:,0], histograms[iShow,:,0], histograms[1,:,0]]
        hist_errors = [histograms[2,:,1], histograms[iShow,:,1], histograms[1,:,1]]
        
        FONTSIZE = 30
        TICKLABELSIZE = FONTSIZE -3
        labels = ["Truth", model_name, "Train"]
        colors = ["black","#A52A2A", "#0343DE"]
        y_ticks = [.9, 1., 1.1]

        fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
        fig1.tight_layout(rect=(0.1, 0.08, 1., 1.))

        for y, y_err, label, color in zip(hists, hist_errors, labels, colors):

            axs[0].step(bins, y, label=label, color=color,
                        linewidth=1.0, where="post")
            axs[0].step(bins, y+y_err, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
            axs[0].step(bins, y - y_err, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
            axs[0].fill_between(bins, y - y_err,
                                y + y_err, facecolor=color,
                                alpha=0.3, step="post")

            if label == "Truth": continue

            ratio = y / hists[0]
            ratio_err = np.sqrt( (y_err / y)**2 + (hist_errors[0] / hists[0])**2)
            ratio_isnan = np.isnan(ratio)
            ratio[ratio_isnan] = 1.
            ratio_err[ratio_isnan] = 0.
                       
            axs[1].step(bins, ratio, linewidth=3.0, where="post", color=color)
            axs[1].step(bins, ratio + ratio_err, color=color, alpha=0.5,
                        linewidth=0.5, where="post")
            axs[1].step(bins, ratio - ratio_err, color=color, alpha=0.5,
                        linewidth=0.5, where="post")
            axs[1].fill_between(bins, ratio - ratio_err,
                                ratio + ratio_err, facecolor=color, alpha=0.3, step="post")

            delta = np.fabs(ratio[:-1] - 1) * 100
            delta_err = ratio_err[:-1] * 100

            markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                                  yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                                  linewidth=0, fmt=".", capsize=2, markersize=10)
            [cap.set_alpha(0.5) for cap in caps]
            [bar.set_alpha(0.5) for bar in bars]

        #axs[0].legend(loc=legend_loc, frameon=False, fontsize=FONTSIZE)
        for line in axs[0].legend(loc=legend_loc, frameon=False, fontsize=FONTSIZE).get_lines():
            line.set_linewidth(3.0)
        axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

        axs[1].set_ylabel(r"$\frac{\mathrm{%s}}{\mathrm{Truth}}$" % model_name,
                          fontsize=FONTSIZE)
        axs[1].set_yticks(y_ticks)
        axs[1].set_ylim(.81, 1.19)
        axs[1].axhline(y=y_ticks[1], c="black", ls="--", lw=0.7)
        axs[1].axhline(y=y_ticks[0], c="black", ls="dotted", lw=0.5)
        axs[1].axhline(y=y_ticks[2], c="black", ls="dotted", lw=0.5)
        plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)

        axs[2].set_ylim((0.05, 20))
        axs[2].set_yscale("log")
        axs[2].set_yticks([0.1, 1.0, 10.0])
        axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
        axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

        axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
        axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
        axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

        axs[0].set_xlim(x_range)

        axs[0].tick_params(axis="both", labelsize=TICKLABELSIZE)
        axs[1].tick_params(axis="both", labelsize=TICKLABELSIZE)
        axs[2].tick_params(axis="both", labelsize=TICKLABELSIZE)

        plt.savefig(pp, format="pdf")
        plt.close()

        fig2, ax = plt.subplots(2,1, sharex=True,
                                gridspec_kw={"height_ratios": [1,1], "hspace": 0.})
        fig2.tight_layout(rect=(0.1, 0.08, 1., 1.))
        
        axs = ax[0]

        axs.set_ylabel(r"$\sigma$", fontsize=FONTSIZE)
        axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                       fontsize=FONTSIZE)

        abs_unc = histograms[4,:,0]
        abs_unc_unc = histograms[4,:,1]
        abs_unc_i = histograms[5:,:,1]
    
        axs.step(bins, abs_unc, color=colors[1], linewidth=3.0, where="post", label=labels[1])
        axs.step(bins, abs_unc + abs_unc_unc, color=colors[1], linewidth=1.0, where="post", alpha=.5)
        axs.step(bins, abs_unc - abs_unc_unc, color=colors[1], linewidth=1.0, where="post", alpha=.5)
        axs.fill_between(bins, abs_unc + abs_unc_unc, abs_unc-+ abs_unc_unc,
                         step="post", alpha=.3, facecolor=colors[1])
        axs.step(bins, hist_errors[2], linewidth=3.0, where="post", color=colors[2], label=labels[2])
        #for i in range(len(histograms[:,0,1])-5):
        #    axs.step(bins, abs_unc_i[i, :], color="orange", linewidth=1.0, where="post")
        axs.set_ylim(yminAbs, ymaxAbs)
        #axs.legend(loc="upper left", frameon=False, fontsize=FONTSIZE)

        axs = ax[1]
        axs.set_ylabel(r"$\sigma / p$", fontsize=FONTSIZE)
        axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                       fontsize=FONTSIZE)

        rel_unc = histograms[4,:,0] / histograms[3,:,0]
        rel_unc[np.isnan(rel_unc)] = 0.
        rel_unc_unc = histograms[4,:,1] / histograms[3,:,0]
        rel_unc_unc[np.isnan(rel_unc_unc)] = 0.
        rel_unc_i = histograms[5:,:,1] / histograms[5:,:,0]
        
        axs.step(bins, rel_unc, color=colors[1], linewidth=3.0, where="post", label=labels[1])
        axs.step(bins, rel_unc + rel_unc_unc, color=colors[1], linewidth=1.0, where="post", alpha=.5)
        axs.step(bins, rel_unc - rel_unc_unc, color=colors[1], linewidth=1.0, where="post", alpha=.5)
        axs.fill_between(bins, rel_unc + rel_unc_unc, rel_unc - rel_unc_unc,
                         color=colors[1], alpha=0.3, step="post")
        axs.step(bins, hist_errors[2] / hists[2], linewidth=3.0, where="post", color=colors[2], label=labels[2])
        #for i in range(len(histograms[:,0,1])-5):
        #    axs.step(bins, rel_unc_i[i, :], color="orange", linewidth=1.0, where="post")
        axs.set_ylim(yminRel, ymaxRel)
        #axs.legend(loc="upper right", frameon=False, fontsize=FONTSIZE)

        ax[0].set_xlim(x_range)
        ax[0].tick_params(axis="both", labelsize=TICKLABELSIZE)
        ax[1].tick_params(axis="both", labelsize=TICKLABELSIZE)

        plt.savefig(pp, format="pdf")
        plt.close()
        
plot("paper_GMM_ramp6")
plot("paper_GMM_sphere6")
plot("paper_Binned_ramp6")
plot("paper_Binned_ramp7")
plot("paper_Binned_ramp8")
plot("paper_Binned_sphere6")

'''
plot("paper_GMM_ramp4")
plot("paper_GMM_sphere5")
plot("paper_Binned_ramp4")
plot("paper_Binned_sphere4")

#plot("paper_NN_ramp2")
#plot("paper_NN_sphere2")
'''
