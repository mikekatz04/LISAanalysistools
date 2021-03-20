import numpy as np
from lisatools.sensitivity import get_sensitivity


def generate_noise_fd(freqs, df, **sensitivity_kwargs):
    norm = 0.5 * (1.0 / df) ** 0.5
    psd = get_sensitivity(freqs, **sensitivity_kwargs)
    noise_to_add = psd ** (1 / 2) * (
        np.random.normal(0, norm, len(freqs))
        + 1j * np.random.normal(0, norm, len(freqs))
    )

    return noise_to_add


def AET(X, Y, Z):
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )
