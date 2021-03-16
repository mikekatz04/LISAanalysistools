import warnings

import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

try:
    from tdi import *

    tdi_available = True

except (ModuleNotFoundError, ImportError):
    tdi_available = False
    warnings.warn("No tdi module available.")


def cornish_lisa_psd(f, sky_averaged=False, use_gpu=False):
    """PSD from https://arxiv.org/pdf/1803.01944.pdf

    Power Spectral Density for the LISA detector assuming it has been active for a year.
    I found an analytic version in one of Niel Cornish's paper which he submitted to the arXiv in
    2018. I evaluate the PSD at the frequency bins found in the signal FFT.

    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf

    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    if sky_averaged:
        sky_averaging_constant = 20 / 3

    else:
        sky_averaging_constant = 1.0  # set to one for one source

    L = 2.5 * 10 ** 9  # Length of LISA arm
    f0 = 19.09 * 10 ** (-3)  # transfer frequency

    # Optical Metrology Sensor
    Poms = ((1.5e-11) * (1.5e-11)) * (1 + xp.power((2e-3) / f, 4))

    # Acceleration Noise
    Pacc = (
        (3e-15)
        * (3e-15)
        * (1 + (4e-4 / f) * (4e-4 / f))
        * (1 + xp.power(f / (8e-3), 4))
    )

    # constants for Galactic background after 1 year of observation
    alpha = 0.171
    beta = 292
    k = 1020
    gamma = 1680
    f_k = 0.00215

    # Galactic background contribution
    Sc = (
        9e-45
        * xp.power(f, -7 / 3)
        * xp.exp(-xp.power(f, alpha) + beta * f * xp.sin(k * f))
        * (1 + xp.tanh(gamma * (f_k - f)))
    )

    # PSD
    PSD = (sky_averaging_constant) * (
        (10 / (3 * L * L))
        * (Poms + (4 * Pacc) / (xp.power(2 * np.pi * f, 4)))
        * (1 + 0.6 * (f / f0) * (f / f0))
        + Sc
    )

    return PSD


def get_sensitivity(f, sens_fn="lisasens", return_type="PSD", *args, **kwargs):
    """Generic sensitivity generator

    Same interface to many sensitivity curves.

    Args:
        f (1D double np.ndarray): Array containing frequency  values.
        sens_fn (str, optional): String that represents the name of the desired
            SNR function. Options are "cornish_lisa_psd" or any sensitivity
            function found in tdi.py from the MLDC gitlab. Default is the
            LISA sensitivity from the tdi.py.
        return_type (str, optional): Described the desired output. Choices are ASD,
            PSD, or char_strain (characteristic strain). Default is ASD.

        *args (list or tuple, optional): Any additional arguments for the sensitivity function.
        **kwargs (dict, optional): Keyword arguments to pass to sensitivity function.

    """

    try:
        sensitivity = globals()[sens_fn]
    except KeyError:
        raise ValueError("{} sensitivity is not available.".format(sens_fn))

    PSD = sensitivity(f, *args, **kwargs)

    if return_type == "PSD":
        return PSD

    elif return_type == "ASD":
        return PSD ** (1 / 2)

    elif return_type == "char_strain":
        return (f * PSD) ** (1 / 2)

    else:
        raise ValueError("return_type must be PSD, ASD, or char_strain.")
