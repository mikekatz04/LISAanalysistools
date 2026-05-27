from lisatools.composite_sensitivity import (
        InstrumentNoise,
        GalacticForeground,
        SGWB,
        CompositeSensitivityMatrix
)
from lisatools import domains
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from matplotlib.colors import LogNorm

def scalo(ax, arr, wdm, *, log=False, vmin=None, vmax=None, cmap="viridis", title="", logy=False):
    """pcolormesh scalogram (freq on y [log], time on x) with physical axes."""
    a = np.abs(np.asarray(arr))  # (Nf, Nt)
    t_days = np.asarray(wdm.t_arr_edges, dtype=float) / 86400.0  # (Nt+1,)
    f_mHz = np.asarray(wdm.f_arr_edges, dtype=float) * 1e3       # (Nf+1,)
    # log y-axis needs positive edges; the DC layer (f=0) is masked anyway, so
    # nudge any non-positive edge below the lowest real layer and crop it via ylim
    f_lo = f_mHz[f_mHz > 0].min()
    f_mHz = f_mHz.copy()
    f_mHz[f_mHz <= 0] = f_lo * 1e-3
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
    im = ax.pcolormesh(
        t_days, f_mHz, a, norm=norm,
        vmin=None if log else vmin, vmax=None if log else vmax,
        cmap=cmap, shading="flat",
    )
    if logy:
        ax.set_yscale("log")
    ax.set_ylim(f_lo, float(f_mHz[-1]))
    ax.set_xlabel("time [days]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_title(title)
    return im


if __name__ == '__main__':
    # Nt=338 is roughly one month of data
    Nf = 1536
    Nt = 338*12
    dt = 5

    settings = domains.WDMSettings(Nf=Nf,
                                Nt=Nt,
                                dt=dt,
                                min_freq=3e-4,
                                max_freq=2e-2,
                                force_backend="cpu")

    glass_modulation = np.loadtxt("./modulation.dat")
    modulation = np.array([
        [glass_modulation[:,1], glass_modulation[:,4], glass_modulation[:,5]],
        [glass_modulation[:,4], glass_modulation[:,2], glass_modulation[:,6]],
        [glass_modulation[:,5], glass_modulation[:,6], glass_modulation[:,3]],
    ])
    modulation_interped = interp1d(glass_modulation[:,0], modulation)(settings.t_arr)
    components = [
            InstrumentNoise(
                model='sangria'
                ),
            GalacticForeground(
                foreground_params = [
                    3.26651613e-44, # amp
                    2.09278117e-03, # fk
                    1.18300266e00,  # alpha
                    3.01430978e03,  # slope 1
                    2.95774596e03,  # slope 2
                    ],
                modulation = modulation_interped,
                model='sangria', # why is this here??
                ),
            SGWB(
                sgwb_params = [-8.45, 2./3.],
                stochastic_fn = "PowerLawSGWB",
                modulation = None,
                model='sangria', # why is this here??
                ),
            ]
    sensmat = CompositeSensitivityMatrix(settings, components)

    # generate fake data
    XXcov = sensmat.sens_mat[0,0,:]
    XYcov = sensmat.sens_mat[0,1,:]
    ZZcov = sensmat.sens_mat[2,2,:]
    fig, ax = plt.subplots()
    scalo(ax, XXcov, settings, log=True, logy=True)
    plt.show()
    fig, ax = plt.subplots()
    scalo(ax, np.abs(XYcov), settings, log=True, logy=True)
    plt.show()
    fig, ax = plt.subplots()
    scalo(ax, np.abs(ZZcov), settings, log=True, logy=True)
    plt.show()
