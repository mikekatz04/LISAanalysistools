import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chainconsumer
import corner
from lisatools.globalfit.hdfbackend import GFHDFBackend
import pandas as pd

def set_plotting_style(background_color: str = 'white', front_color: str = 'black') -> None:
    """
    Set the plotting style for matplotlib.

    Args:
        background_color (str): Background color for the plots.
        front_color (str): Main color for the plots.

    """
    if background_color == front_color:
        raise ValueError("Background color and main color cannot be the same.")
   
    # text settings
    mpl.rcParams['text.usetex'] = True  # Use LaTeX for text rendering
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Use AMS math package
    mpl.rcParams['font.family'] = 'serif'  # Use serif font for text
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Use Computer Modern font for LaTeX
    mpl.rcParams['font.weight'] = 'medium'
    
    # set colors
    set_colors(background_color, front_color)
    # Set the style for matplotlib plots
    mpl.rcParams['grid.color'] = '#d3d3d3'  # Light gray for grid lines
    mpl.rcParams['grid.linestyle'] = '--'  # Dashed grid lines
    mpl.rcParams['grid.linewidth'] = 0.5  # Thinner grid
    mpl.rcParams['lines.linewidth'] = 1.5  # Thicker lines
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['figure.dpi'] = 100  # Set the figure resolution   
    mpl.rcParams['figure.figsize'] = (7, 7)  # Set default figure size
    mpl.rcParams['savefig.bbox'] = 'tight'
    #mpl.rcParams['savefig.transparent'] = True

    # set ticks size
    fontsize = 12
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['axes.formatter.limits'] = [-2, 4]
    mpl.rcParams['axes.titlesize'] = 15
    mpl.rcParams['axes.labelsize'] = 15
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.minor.size'] = 3


def set_colors(background_color='white', front_color='black'):
    """
    Set the colors for the plot.

    Args:
        background_color (str): The background color of the plot. Default is 'white'.
        front_color (str): The foreground color of the plot. Default is 'black'.
    """

    mpl.rcParams['text.color'] = front_color
    mpl.rcParams['axes.labelcolor'] = front_color
    mpl.rcParams['axes.edgecolor'] = front_color
    mpl.rcParams['xtick.color'] = front_color
    mpl.rcParams['ytick.color'] = front_color
    mpl.rcParams['axes.facecolor'] = background_color
    mpl.rcParams['figure.facecolor'] = background_color
    mpl.rcParams['legend.facecolor'] = background_color
    #mpl.rcParams['legend.edgecolor'] = front_color
    mpl.rcParams['axes.titlecolor'] = front_color
    mpl.rcParams['legend.labelcolor'] = front_color
    mpl.rcParams['grid.color'] = front_color
    mpl.rcParams['lines.color'] = front_color


def override_plotting_style(custom_style: dict) -> None:
    """
    Override the default plotting style with a custom style dictionary.

    Args:
        custom_style (dict): Dictionary containing custom style parameters.
    """
    for key, value in custom_style.items():
        mpl.rcParams[key] = value
    set_colors(mpl.rcParams.get('axes.facecolor', 'white'), mpl.rcParams.get('text.color', 'black'))

def restore_default_plotting_style() -> None:
    """
    Restore the default matplotlib plotting style.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    set_colors(mpl.rcParams.get('axes.facecolor', 'white'), mpl.rcParams.get('text.color', 'black'))

# make a decorator to set the plotting style for a function
def with_custom_plotting_style(func):
    """
    Decorator to set the plotting style for a function.

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
def plot_loglikelihood(reader, discard=0, save_dir='./'):
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

    df = pd.DataFrame(np.c_[facet_logl.flat, step, walker],
                      columns=[r"$\Delta \log\mathcal{L}$", "step", "walker"])
    
    # Initialize a grid of plots with an Axes for each walker
    grid = sns.FacetGrid(df, col="walker", hue="walker", palette="tab20c",
                     col_wrap=9, height=1.5)
    
    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "step", r"$\Delta \log\mathcal{L}$", marker=".")

    grid.refline(y=0, linestyle=":") # Add a horizontal reference line at y=0 ~ average loglikelihood at each step

    plt.savefig(save_dir + "loglikelihood_facetgrid.png")
    plt.close()

@with_custom_plotting_style
def base_branch_plots(chain, key, labels, save_dir='./', plot_trace=True, plot_corner=True, truths=None):

    nsteps, nwalkers, nleaves, ndim = chain.shape

    chain = chain.reshape((nsteps, nwalkers, ndim)) 

    if plot_trace:
        fig, axes = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
        for i in range(ndim):
            for j in range(nwalkers):
                axes[i].plot(chain[:, j, i], color=f"C{j % 10}", alpha=0.3)
            if truths is not None:
                axes[i].axhline(truths[i], color='red', linestyle='--')
            axes[i].set_ylabel(labels[i])
        axes[-1].set_xlabel("Sampler Iteration")
        savename = save_dir + f"{key}_traceplot.png"
        plt.savefig(savename)
        plt.close()

    if plot_corner:
        savename = save_dir + f"{key}_corner.png"
        chain = chain.reshape((-1, ndim))
        #fig = corner.corner(chain, labels=labels, truths=truths)
        df = pd.DataFrame(chain, columns=labels)
        C = chainconsumer.ChainConsumer()
        C.add_chain(chain=chainconsumer.Chain(samples=df, name=f"{key} posterior"))
        if truths is not None:
            C.add_truth(chainconsumer.Truth(location=dict(zip(labels, truths)), name="Injection"))

        plot_config = {
            "serif": True,
            "usetex": True,
            "spacing": 1.5,
        }
        C.set_plot_config(
            chainconsumer.PlotConfig(**plot_config)
        )
        fig = C.plotter.plot()
        fig.savefig(savename)
        plt.close()

def produce_mbh_plots(chain=None, reader=None, discard=0, save_dir='./'):
    pass

def produce_gb_plots(chain=None, reader=None, discard=0, save_dir='./'):
    pass


def produce_emri_plots(chain=None, reader=None, discard=0, save_dir='./'):
    labels = [
        r'$\log (m_1 / M_\odot)$',
        r'$m_2 / M_\odot$',
        r'$a$',
        r'$p_0$',
        r'$e_0$',
        r'$d_L / \rm{Gpc}$',
        r'$\cos(\theta_S)$',
        r'$\phi_S$',
        r'$\cos(\theta_K)$',
        r'$\phi_K$',
        r'$\Phi_{\phi_0}$',
        r'$\Phi_{r_0}$'
    ]
    #todo remove hardcoded
    truths = [np.log(1e6), 10.0, 0.9, 4.181726507479478, 0.5, 1.3714627528304648, np.cos(0.3), 0.3, np.cos(1.0471975511965976), 1.0471975511965976, 1.5707963267948966, 3.141592653589793]
    
    if chain is None:
        chain = reader.get_chain(discard=discard)["emri"][:, 0]
    
    base_branch_plots(chain, key="emri", labels=labels, save_dir=save_dir, truths=truths)

def produce_psd_plots(chain=None, reader=None, discard=0, save_dir='./'):
    labels = [r"$S_{\mathrm{oms}_A}$", r"$S_{\mathrm{acc}_A}$", r"$S_{\mathrm{oms}_E}$", r"$S_{\mathrm{acc}_E}$"]
    truths = [7.9e-12, 2.4e-15, 7.9e-12, 2.4e-15]
    if chain is None:
        chain = reader.get_chain(discard=discard)["psd"][:, 0]

    base_branch_plots(chain, key="psd", labels=labels, save_dir=save_dir, truths=truths)

def produce_galfor_plots(chain=None, reader=None, discard=0, save_dir='./'):
    pass

all_branches = ["mbh", "gb", "emri", "psd", "galfor"]
all_branches_functions = dict(zip(all_branches, [produce_mbh_plots, produce_gb_plots, produce_emri_plots, produce_psd_plots, produce_galfor_plots]))

class DiagnosticPlotter:
    def __init__(self, curr, plot_every=10):
        self.curr = curr # information holder
        self.plot_every = plot_every
        savedir = self.curr.general_info.main_file_path.replace('parameter_estimation_main.h5', 'diagnostic_plots/')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.savedir = savedir
        
        print("Saving diagnostic plots to ", self.savedir)

    def __call__(self, iteration, last_sample, sampler):
        if iteration > 0 and iteration % self.plot_every == 0:
            discard = int(0.1 * sampler.iteration)

            for branch in sampler.branch_names:
                if branch in all_branches:
                    all_branches_functions[branch](reader=sampler, discard=discard, save_dir=self.savedir)

            plot_loglikelihood(sampler, discard=discard, save_dir=self.savedir)

    

if __name__ == "__main__":
    filepath = '/data/asantini/packages/LISAanalysistools/global_fit_output/psd_separate_1st_try_parameter_estimation_main.h5'
    reader = GFHDFBackend(filepath)
    produce_psd_plots(reader=reader, discard=0, save_dir='./')
    #produce_emri_plots(reader=reader, discard=0, save_dir='./')
    plot_loglikelihood(reader, discard=0, save_dir='./')