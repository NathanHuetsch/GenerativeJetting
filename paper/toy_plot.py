import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot(toy_type):
    histograms = np.load(f"paper/toy/histograms_{toy_type}.npy")
    if toy_type == "ramp":
        name = "x_1"
        unit=None
        ymaxAbs = .07
        ymaxRel = .05
    elif toy_type == "gauss_sphere":
        name = "R"
        unit=None
        ymaxAbs = .1
        ymaxRel = .1

    #for i in range(61):
    #    print("{0:.2e} {1:.2e} {2:.2e}".format(histograms[3,i,1], histograms[4,i,1], histograms[5,i,1]))

    with warnings.catch_warnings() and PdfPages("paper/toy/plot_ramp.pdf") as pp:
        warnings.simplefilter("ignore", RuntimeWarning)

        bins = histograms[0,:,0]
        hists = [histograms[2,:,0], histograms[4,:,0], histograms[1,:,0]]
        hist_errors = [histograms[2,:,1], histograms[4,:,1], histograms[1,:,1]]

        hists_unc = [histograms[2,:,0], histograms[3,:,0], histograms[1,:,0]]
        hist_errors_unc = [histograms[2,:,1], histograms[3,:,1], histograms[1,:,1]]
        
        FONTSIZE = 14
        labels = ["True", "Model", "Train"]
        colors = ["#e41a1c", "#3b528b", "#1a8507"]

        fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.00})
        fig1.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.07, 0.06, 0.99, 0.95))

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

            if label == "True": continue

            ratio = y / hists[0]
            ratio_err = np.sqrt( (y_err / y)**2 + (hist_errors[0] / hists[0])**2)
            ratio_isnan = np.isnan(ratio)
            ratio[ratio_isnan] = 1.
            ratio_err[ratio_isnan] = 0.
                       
            axs[1].step(bins, ratio, linewidth=1.0, where="post", color=color)
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
                                                  linewidth=0, fmt=".", capsize=2)
            [cap.set_alpha(0.5) for cap in caps]
            [bar.set_alpha(0.5) for bar in bars]

        axs[0].legend(loc="upper left", frameon=False, fontsize=FONTSIZE)
        axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

        axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{True}}$",
                          fontsize=FONTSIZE)
        axs[1].set_yticks([0.95, 1, 1.05])
        axs[1].set_ylim([0.9, 1.1])
        axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
        axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
        axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)
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

        plt.savefig(pp, format="pdf")
        plt.close()

        '''
        fig2, axs = plt.subplots(1, 1)
        fig2.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.07, 0.06, 0.99, 0.95))

        axs.set_ylabel("Absolute uncertainty", fontsize=FONTSIZE)
        axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                       fontsize=FONTSIZE)

        axs.step(bins, hist_errors_unc[1], color=colors[1])
        axs.fill_between(bins, hist_errors_unc[1]+histograms[0,:,1], hist_errors_unc[1]-histograms[0,:,1],
                         step="post", alpha=.3, color=colors[1])
        axs.set_ylim(0., ymaxAbs)

        plt.savefig(pp, format="pdf")
        plt.close()

        fig3, axs = plt.subplots(1, 1)
        fig3.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.07, 0.06, 0.99, 0.95))

        axs.set_ylabel("Relative uncertainty", fontsize=FONTSIZE)
        axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                       fontsize=FONTSIZE)

        rel_unc = hist_errors_unc[1] / hists_unc[1]
        rel_unc[np.isnan(rel_unc)] = 0.
        rel_unc_unc = histograms[0,:,1] / hists_unc[1]
        rel_unc_unc[np.isnan(rel_unc_unc)] = 0.
        
        axs.step(bins, rel_unc, color=colors[1])
        axs.fill_between(bins, rel_unc + rel_unc_unc, rel_unc - rel_unc_unc,
                         color=colors[1], alpha=0.3, step="post")
        axs.set_ylim(0., ymaxRel)
        '''

        fig2, ax = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios": [1,1], "hspace": 0.})
        
        axs = ax[0]
        fig2.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.07, 0.06, 0.99, 0.95))

        axs.set_ylabel("Absolute uncertainty", fontsize=FONTSIZE)
        axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                       fontsize=FONTSIZE)

        axs.step(bins, hist_errors_unc[1], color=colors[1])
        axs.fill_between(bins, hist_errors_unc[1]+histograms[0,:,1], hist_errors_unc[1]-histograms[0,:,1],
                         step="post", alpha=.3, color=colors[1])
        axs.set_ylim(0., ymaxAbs)

        axs = ax[1]
        axs.set_ylabel("Relative uncertainty", fontsize=FONTSIZE)
        axs.set_xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                       fontsize=FONTSIZE)

        rel_unc = hist_errors_unc[1] / hists_unc[1]
        rel_unc[np.isnan(rel_unc)] = 0.
        rel_unc_unc = histograms[0,:,1] / hists_unc[1]
        rel_unc_unc[np.isnan(rel_unc_unc)] = 0.
        
        axs.step(bins, rel_unc, color=colors[1])
        axs.fill_between(bins, rel_unc + rel_unc_unc, rel_unc - rel_unc_unc,
                         color=colors[1], alpha=0.3, step="post")
        axs.set_ylim(0., ymaxRel)

        plt.savefig(pp, format="pdf")
        plt.close()

plot("ramp")
#plot("gauss_sphere")
