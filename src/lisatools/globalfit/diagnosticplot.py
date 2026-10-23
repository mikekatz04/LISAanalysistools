"""Diagnostic plotting utilities (corner plots, log-likelihood traces, styling).

Plot producers that read a chain reader (``get_chain`` / ``get_log_like``,
e.g. a :class:`~lisatools.globalfit.hdfbackend.GFHDFBackend` or an eryn
sampler): :func:`plot_loglikelihood`, :func:`base_branch_plots` and the
per-branch ``produce_*_plots`` (dispatched by :data:`all_branches_functions`),
plus :class:`DiagnosticPlotter`, an eryn ``update_fn`` callback that runs them
every ``plot_every`` iterations.

Not on the production path: no global-fit module imports this one, and the
running fit plots through eryn's ``PlotContainer``
(:meth:`~lisatools.globalfit.run.GlobalFit.make_plot_container`,
``MAKE_DIAGNOSTIC_PLOTS`` / ``PLOT_ITERATIONS``), not through
:class:`DiagnosticPlotter`. The LATW tutorial 01 lists the producers.

Style side effects: the producers decorated with
:func:`with_custom_plotting_style` call :func:`set_plotting_style` first and
leave those ``matplotlib.rcParams`` in place (LaTeX text only when a
``latex`` executable is found). Importing the module needs ``chainconsumer``,
``corner``, ``seaborn`` and ``pandas``.
"""

import os
import shutil

import chainconsumer
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lisatools.globalfit.hdfbackend import GFHDFBackend

import logging

logger = logging.getLogger(__name__)

#: Interactive debug hooks below are guarded by this (see
#: globalfit/moves/addremovemove.py). A bare breakpoint() in shipped
#: code hangs or BdbQuit-kills any non-interactive run.
DEBUG_MODE = False


def set_plotting_style(background_color: str = "white", front_color: str = "black") -> None:
    """
    Set the plotting style for matplotlib.

    Mutates the global ``matplotlib.rcParams``: serif Computer Modern text
    (``text.usetex`` only when ``latex`` is on the ``PATH``), font sizes,
    grid and tick styling, ``figure.figsize`` ``(7, 7)``, tight bounding
    boxes, and the colours of :func:`set_colors`.

    Args:
        background_color: Background color for the plots. Default
            ``"white"``.
        front_color: Main color for the plots. Default ``"black"``.

    Raises:
        ValueError: The two colours are equal.
    """
    if background_color == front_color:
        raise ValueError("Background color and main color cannot be the same.")

    # text settings -- only enable LaTeX rendering when latex is actually
    # installed, else matplotlib crashes on machines without it (fall back to
    # mathtext). Applies to every global-fit plot (grid-gen, debug, diag).
    _has_latex = shutil.which("latex") is not None
    mpl.rcParams["text.usetex"] = _has_latex  # Use LaTeX for text rendering
    if _has_latex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"  # AMS math
    mpl.rcParams["font.family"] = "serif"  # Use serif font for text
    mpl.rcParams["font.serif"] = ["Computer Modern Roman"]  # Use Computer Modern font for LaTeX
    mpl.rcParams["font.weight"] = "medium"

    # set colors
    set_colors(background_color, front_color)
    # Set the style for matplotlib plots
    mpl.rcParams["grid.color"] = "#d3d3d3"  # Light gray for grid lines
    mpl.rcParams["grid.linestyle"] = "--"  # Dashed grid lines
    mpl.rcParams["grid.linewidth"] = 0.5  # Thinner grid
    mpl.rcParams["lines.linewidth"] = 1.5  # Thicker lines
    mpl.rcParams["font.size"] = 20
    mpl.rcParams["figure.dpi"] = 100  # Set the figure resolution
    mpl.rcParams["figure.figsize"] = (7, 7)  # Set default figure size
    mpl.rcParams["savefig.bbox"] = "tight"
    # mpl.rcParams['savefig.transparent'] = True

    # set ticks size
    fontsize = 12
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["axes.formatter.limits"] = [-2, 4]
    mpl.rcParams["axes.titlesize"] = 15
    mpl.rcParams["axes.labelsize"] = 15
    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["xtick.minor.size"] = 3
    mpl.rcParams["ytick.major.size"] = 5
    mpl.rcParams["ytick.minor.size"] = 3


def set_colors(background_color="white", front_color="black"):
    """
    Set the colors for the plot.

    Writes the text, label, edge, tick, title, legend, grid and line colours
    (``front_color``) and the axes / figure / legend face colours
    (``background_color``) into the global ``matplotlib.rcParams``.

    Args:
        background_color (str): The background color of the plot. Default is 'white'.
        front_color (str): The foreground color of the plot. Default is 'black'.
    """

    mpl.rcParams["text.color"] = front_color
    mpl.rcParams["axes.labelcolor"] = front_color
    mpl.rcParams["axes.edgecolor"] = front_color
    mpl.rcParams["xtick.color"] = front_color
    mpl.rcParams["ytick.color"] = front_color
    mpl.rcParams["axes.facecolor"] = background_color
    mpl.rcParams["figure.facecolor"] = background_color
    mpl.rcParams["legend.facecolor"] = background_color
    # mpl.rcParams['legend.edgecolor'] = front_color
    mpl.rcParams["axes.titlecolor"] = front_color
    mpl.rcParams["legend.labelcolor"] = front_color
    mpl.rcParams["grid.color"] = front_color
    mpl.rcParams["lines.color"] = front_color


def override_plotting_style(custom_style: dict) -> None:
    """
    Override the default plotting style with a custom style dictionary.

    Each entry is written into ``matplotlib.rcParams``; then
    :func:`set_colors` reapplies the colours from the resulting
    ``axes.facecolor`` / ``text.color``.

    Args:
        custom_style: Dictionary containing custom style parameters
            (``rcParams`` keys).
    """
    for key, value in custom_style.items():
        mpl.rcParams[key] = value
    set_colors(
        mpl.rcParams.get("axes.facecolor", "white"),
        mpl.rcParams.get("text.color", "black"),
    )


def restore_default_plotting_style() -> None:
    """
    Restore the default matplotlib plotting style.

    ``rcParams.update(rcParamsDefault)``, then :func:`set_colors` with the
    default face / text colours.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    set_colors(
        mpl.rcParams.get("axes.facecolor", "white"),
        mpl.rcParams.get("text.color", "black"),
    )


# make a decorator to set the plotting style for a function
def with_custom_plotting_style(func):
    """
    Decorator to set the plotting style for a function.

    The wrapper calls :func:`set_plotting_style` with its defaults before
    every call and does NOT restore the previous ``rcParams`` afterwards.
    The wrapper does not copy the wrapped function's name or docstring.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Decorated function with plotting style set.
    """

    def wrapper(*args, **kwargs):
        set_plotting_style()
        return func(*args, **kwargs)

    return wrapper


# make a decorator to set the plotting style to default for a function
def with_default_plotting_style(func):
    """
    Decorator to set the default plotting style for a function.

    The wrapper copies ``rcParams``, applies
    :func:`restore_default_plotting_style`, calls ``func`` and restores the
    copy (not restored if ``func`` raises).

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Decorated function with default plotting style set.
    """

    def wrapper(*args, **kwargs):
        current_style = mpl.rcParams.copy()
        restore_default_plotting_style()
        result = func(*args, **kwargs)
        mpl.rcParams.update(current_style)
        return result

    return wrapper


@with_custom_plotting_style
def plot_loglikelihood(reader, discard=0, save_dir="./"):
    """Plot per-walker log-likelihood traces and a per-walker facet grid.

    Uses the cold chain (temperature 0). Writes
    ``<save_dir>loglikelihood_evolution.png`` (all walkers) and
    ``<save_dir>loglikelihood_facetgrid.png`` (one panel per walker, the
    log-likelihood minus the per-step walker mean). Applies the
    :func:`set_plotting_style` style first (decorator).

    Args:
        reader: HDF backend or any reader with ``get_log_like(discard=)``.
        discard: Burn-in steps to discard. Default ``0``.
        save_dir: Directory to write the resulting PNGs into, used as a
            string prefix (keep the trailing ``/``). Default ``"./"``.

    Example:
        .. code-block:: python

            from lisatools.globalfit.diagnosticplot import plot_loglikelihood
            from lisatools.globalfit.hdfbackend import GFHDFBackend

            plot_loglikelihood(GFHDFBackend("run_store.h5"), discard=100, save_dir="plots/")
    """
    logl = reader.get_log_like(discard=discard)[:, 0]

    nsteps, nwalkers = logl.shape
    plt.figure()
    for j in range(nwalkers):
        plt.plot(logl[:, j], color=f"C{j % 10}", alpha=0.8)
    plt.xlabel("Sampler Iteration")
    plt.ylabel("Log-Likelihood")
    save_file = save_dir + "loglikelihood_evolution.png"
    plt.savefig(save_file)
    plt.close()

    # plot a facet grid of loglikelihood evolution  for each walker
    # Reshape: logl is (nsteps, nwalkers), need to flatten properly
    mean_logl = np.mean(logl, axis=1)
    facet_logl = logl - mean_logl[:, np.newaxis]

    step = np.tile(range(nsteps), nwalkers)
    walker = np.int32(np.repeat(range(nwalkers), nsteps))

    df = pd.DataFrame(
        np.c_[facet_logl.flat, step, walker],
        columns=[r"$\Delta \log\mathcal{L}$", "step", "walker"],
    )

    # Initialize a grid of plots with an Axes for each walker
    grid = sns.FacetGrid(df, col="walker", hue="walker", palette="tab20c", col_wrap=9, height=1.5)

    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "step", r"$\Delta \log\mathcal{L}$", marker=".")

    grid.refline(
        y=0, linestyle=":"
    )  # Add a horizontal reference line at y=0 ~ average loglikelihood at each step

    plt.savefig(save_dir + "loglikelihood_facetgrid.png")
    plt.close()


@with_custom_plotting_style
def base_branch_plots(
    chain, key, labels, save_dir="./", plot_trace=True, plot_corner=True, truths=None
):
    """Generic per-branch trace + corner plotting helper.

    Writes ``<save_dir><key>_traceplot.png`` (one panel per parameter, every
    walker and leaf) and ``<save_dir><key>_corner.png`` (a ``chainconsumer``
    corner of all rows with finite values, leaves pooled). Applies the
    :func:`set_plotting_style` style first (decorator).

    Args:
        chain: Branch chain of shape ``(nsteps, nwalkers, nleaves, ndim)``.
        key: Branch name used to build output filenames.
        labels: Per-dimension axis labels.
        save_dir: Directory to write PNGs into (a string prefix). Default
            ``"./"``.
        plot_trace: If ``True``, render a per-dimension trace plot.
        plot_corner: If ``True``, render a corner plot.
        truths: Optional injection truths overlaid on the plots (one per
            dimension).
    """
    nsteps, nwalkers, nleaves, ndim = chain.shape

    # chain = chain.reshape((nsteps, nwalkers, ndim))

    if plot_trace:
        fig, axes = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
        for i in range(ndim):
            for j in range(nwalkers):
                axes[i].plot(chain[:, j, :, i], color=f"C{j % 10}", alpha=0.3)
            if truths is not None:
                axes[i].axhline(truths[i], color="red", linestyle="--")
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("Sampler Iteration")
        savename = save_dir + f"{key}_traceplot.png"
        plt.savefig(savename)
        plt.close()

    if plot_corner:
        savename = save_dir + f"{key}_corner.png"
        chain = chain.reshape((-1, ndim))
        chain = chain[np.isfinite(chain).all(axis=1)]  # remove rows with NaNs or infs
        # fig = corner.corner(chain, labels=labels, truths=truths)
        df = pd.DataFrame(chain, columns=labels)
        C = chainconsumer.ChainConsumer()
        C.add_chain(chain=chainconsumer.Chain(samples=df, name=f"{key} posterior"))
        if truths is not None:
            C.add_truth(chainconsumer.Truth(location=dict(zip(labels, truths)), name="Injection"))

        plot_config = {
            "serif": True,
            "usetex": shutil.which("latex") is not None,
            "spacing": 1.5,
        }
        C.set_plot_config(chainconsumer.PlotConfig(**plot_config))
        fig = C.plotter.plot()
        fig.savefig(savename)
        plt.close()


def produce_mbh_plots(chain=None, reader=None, discard=0, save_dir="./", truths=None):
    """Plot the MBH chain (trace + corner) with default parameter labels.

    The 11 labels follow a legacy basis order (``log m1``, ``q``, ``a1``,
    ``a2``, ``d_L``, ``Phi0``, ``cos iota``, ``beta``, ``sin lambda``,
    ``psi``, ``t_c``) that does not match the stock ICRS MBH sampling basis
    ``(logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi, alpha, sin_delta,
    t_plunge)``: labels 0 and 7-9 are wrong for stock chains.

    Args:
        chain (np.ndarray, optional): ``(nsteps, nwalkers, nleaves, 11)``;
            read from ``reader`` (branch ``"mbh"``, cold chain) when
            ``None``.
        reader (optional): Chain reader with ``get_chain(discard=)``.
        discard (int): Burn-in steps when reading. Default ``0``.
        save_dir (str): Output prefix (see :func:`base_branch_plots`).
            Default ``"./"``.
        truths (array-like, optional): Values overlaid on the plots.
    """
    labels = [
        r"$\log (m_1 / M_\odot)$",
        r"$q$",
        r"$a_1$",
        r"$a_2$",
        r"$d_L$",
        r"$\Phi_{0}$",
        r"$\cos(\iota)$",
        r"$\beta$",
        r"$\sin(\lambda)$",
        r"$\psi$",
        r"$t_c$",
    ]

    if chain is None:
        chain = reader.get_chain(discard=discard)["mbh"][:, 0]

    base_branch_plots(chain, key="mbh", labels=labels, save_dir=save_dir, truths=truths)


def produce_gb_plots(chain=None, reader=None, discard=0, save_dir="./"):
    """Placeholder for GB diagnostic plots (no-op for now).

    Takes no ``truths`` argument, so the :class:`DiagnosticPlotter` call
    (which passes one) raises ``TypeError`` there, which it logs.
    """
    pass


def produce_emri_plots(chain=None, reader=None, discard=0, save_dir="./", truths=None):
    """Plot the EMRI chain (trace + corner) with default parameter labels.

    The 12 labels match the stock EMRI sampling basis (``logm1, m2, a, p0,
    e0, dist [Gpc], cos qS, phiS, cos qK, phiK, Phi_phi0, Phi_r0``).

    Args:
        chain (np.ndarray, optional): ``(nsteps, nwalkers, nleaves, 12)``;
            read from ``reader`` (branch ``"emri"``, cold chain) when
            ``None``.
        reader (optional): Chain reader with ``get_chain(discard=)``.
        discard (int): Burn-in steps when reading. Default ``0``.
        save_dir (str): Output prefix. Default ``"./"``.
        truths (array-like, optional): Values overlaid on the plots.
    """
    labels = [
        r"$\log (m_1 / M_\odot)$",
        r"$m_2 / M_\odot$",
        r"$a$",
        r"$p_0$",
        r"$e_0$",
        r"$d_L / \rm{Gpc}$",
        r"$\cos(\theta_S)$",
        r"$\phi_S$",
        r"$\cos(\theta_K)$",
        r"$\phi_K$",
        r"$\Phi_{\phi_0}$",
        r"$\Phi_{r_0}$",
    ]

    if chain is None:
        chain = reader.get_chain(discard=discard)["emri"][:, 0]

    base_branch_plots(chain, key="emri", labels=labels, save_dir=save_dir, truths=truths)


def produce_psd_plots(chain=None, reader=None, discard=0, save_dir="./", truths=None):
    """Plot the PSD chain (trace + corner) with default parameter labels.

    Labels ``S_oms_A, S_acc_A, S_oms_E, S_acc_E``, truncated to the chain's
    ``ndim`` (at most 4 parameters).

    Args:
        chain (np.ndarray, optional): ``(nsteps, nwalkers, nleaves, ndim)``;
            read from ``reader`` (branch ``"psd"``, cold chain) when ``None``.
        reader (optional): Chain reader with ``get_chain(discard=)``.
        discard (int): Burn-in steps when reading. Default ``0``.
        save_dir (str): Output prefix. Default ``"./"``.
        truths (array-like, optional): Values overlaid on the plots.
    """
    labels = [
        r"$S_{\mathrm{oms}_A}$",
        r"$S_{\mathrm{acc}_A}$",
        r"$S_{\mathrm{oms}_E}$",
        r"$S_{\mathrm{acc}_E}$",
    ]

    if chain is None:
        chain = reader.get_chain(discard=discard)["psd"][:, 0]

    base_branch_plots(
        chain,
        key="psd",
        labels=labels[: chain.shape[-1]],
        save_dir=save_dir,
        truths=truths,
    )


def produce_galfor_plots(chain=None, reader=None, discard=0, save_dir="./"):
    """Placeholder for galactic-foreground diagnostic plots (no-op for now).

    Like :func:`produce_gb_plots`, it takes no ``truths`` argument.
    """
    pass


#: Branch names :class:`DiagnosticPlotter` has a producer for.
all_branches = ["mbh", "gb", "emri", "psd", "galfor"]
#: ``{branch: produce_*_plots}`` dispatch table of :class:`DiagnosticPlotter`.
all_branches_functions = dict(
    zip(
        all_branches,
        [
            produce_mbh_plots,
            produce_gb_plots,
            produce_emri_plots,
            produce_psd_plots,
            produce_galfor_plots,
        ],
    )
)


class DiagnosticPlotter:
    """Sampler callback that periodically writes per-branch diagnostic plots.

    Matches eryn's ``update_fn(i, state, sampler)`` hook of
    :class:`~eryn.ensemble.EnsembleSampler`. Not wired into the stock global
    fits (they plot through eryn's ``PlotContainer``); usable with a plain
    eryn sampler. Construction creates the plot folder.

    Args:
        curr: Global-fit info object exposing ``source_info`` and
            ``general_info``. The folder is
            ``general_info.main_file_path`` with
            ``"parameter_estimation_main.h5"`` replaced by
            ``"diagnostic_plots/"`` (a path without that legacy file name is
            used unchanged, so pass a store named that way). Injection
            truths are read from ``source_info[branch].injection``.
        plot_every (int): Iteration cadence at which plots are regenerated.
            Default ``10``.

    Attributes:
        curr: Constructor argument.
        plot_every (int): Constructor argument.
        savedir (str): Plot folder.
        truths (dict): ``{branch: injection or None}``.
    """

    def __init__(self, curr, plot_every=10):
        self.curr = curr  # information holder
        self.plot_every = plot_every
        savedir = self.curr.general_info.main_file_path.replace(
            "parameter_estimation_main.h5", "diagnostic_plots/"
        )
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.savedir = savedir

        self.truths = {}
        for source in self.curr.source_info.keys():
            self.truths[source] = (
                self.curr.source_info[source].injection
                if hasattr(self.curr.source_info[source], "injection")
                else None
            )

        logger.info("Saving diagnostic plots to %s", self.savedir)

    def __call__(self, iteration, last_sample, sampler):
        """Print per-move acceptance and write diagnostic plots every ``plot_every`` iterations.

        On iterations ``> 0`` that are multiples of ``plot_every``: logs each
        move's cold acceptance fraction, then, discarding the first 10 % of
        ``sampler.iteration``, runs the producer of every branch in
        :data:`all_branches` (EMRI with its injection truths, PSD with the
        fixed truths ``[7.9e-12, 2.4e-15, 7.9e-12, 2.4e-15]``; producer
        errors are logged, not raised) and :func:`plot_loglikelihood`.

        Args:
            iteration (int): Iteration counter passed by the sampler.
            last_sample: Current state (unused).
            sampler: The eryn sampler (also the chain reader).
        """
        if iteration > 0 and iteration % self.plot_every == 0:

            for move in sampler.moves:
                logger.info(
                    f"{move.__class__.__name__} acceptance fraction: {np.mean(move.acceptance_fraction[0]):.4f}"
                )
            discard = int(0.1 * sampler.iteration)

            for branch in sampler.branch_names:
                if branch in all_branches:
                    if branch == "emri":
                        truths_here = self.truths["emri"]
                    elif branch == "psd":
                        truths_here = [7.9e-12, 2.4e-15, 7.9e-12, 2.4e-15]
                    else:
                        truths_here = None
                    try:
                        all_branches_functions[branch](
                            reader=sampler,
                            discard=discard,
                            save_dir=self.savedir,
                            truths=truths_here,
                        )
                    except Exception as e:
                        logger.error(f"Error producing plots for branch {branch}: {e}")

            plot_loglikelihood(sampler, discard=discard, save_dir=self.savedir)


# Developer snippet with a hard-coded store path of an old run; not part of the API.
if __name__ == "__main__":
    filepath = "/data/asantini/packages/LISAanalysistools/global_fit_output/psd_separate_9th_try_parameter_estimation_main.h5"
    reader = GFHDFBackend(filepath)
    discard = 0

    logl = reader.get_log_like(discard=discard)[:, 0]
    if DEBUG_MODE:  # stray dev hook; off by default
        breakpoint()
    walkers_keep = np.where((logl > 0.9 * np.max(logl)).all(axis=0))[0]
    print(f"Keeping {len(walkers_keep)} walkers out of {logl.shape[1]} based on loglikelihood cut.")

    # produce_mbh_plots(reader=reader, discard=0, save_dir='./')
    chain = reader.get_chain(discard=discard)["psd"][:, 0, walkers_keep]
    produce_psd_plots(chain=chain, reader=reader, discard=discard, save_dir="./")

    # chain = reader.get_chain(discard=discard)["emri"][:, 0, walkers_keep]
    # produce_emri_plots(chain=chain, reader=reader, discard=discard, save_dir='./')
    # plot_loglikelihood(reader, discard=discard, save_dir='./')p axs
