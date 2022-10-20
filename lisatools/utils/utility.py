import numpy as np
from lisatools.sensitivity import get_sensitivity


def generate_noise_fd(freqs, df, *sensitivity_args, func=None, **sensitivity_kwargs):

    if func is None:
        func = get_sensitivity

    norm = 0.5 * (1.0 / df) ** 0.5
    psd = func(freqs, *sensitivity_args, **sensitivity_kwargs)
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


def searchsorted2d_vec(a,b, xp=None, gpu=None, **kwargs):
    if xp is None:
        xp = np
    else:
        try:
            xp.cuda.runtime.setDevice(gpu)
        except AttributeError:
            pass

    m,n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num*xp.arange(a.shape[0])[:,None]
    p = xp.searchsorted( (a+r).ravel(), (b+r).ravel(), **kwargs).reshape(m,-1)

    out = p - n*(xp.arange(m)[:,None])
    try:
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return out
    