import numpy as np
from scipy.stats import uniform
from lisatools.sensitivity import get_sensitivity


def generate_noise_fd(freqs, df, **sensitivity_kwargs):
    norm = 0.5 * (1.0 / df) ** 0.5
    psd = get_sensitivity(freqs, **sensitivity_kwargs)
    noise_to_add = psd ** (1 / 2) * (
        np.random.normal(0, norm, len(freqs))
        + 1j * np.random.normal(0, norm, len(freqs))
    )

    return noise_to_add


def uniform_dist(min, max):
    if min > max:
        temp = min
        min = max
        max = temp

    mean = (max + min) / 2.0
    sig = max - min
    dist = uniform(min, sig)
    return dist
