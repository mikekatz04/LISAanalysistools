from __future__ import annotations
from typing import Tuple
import typing
import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

    pass


def get_array_module(arr: np.ndarray | cp.ndarray) -> object:
    """Return array library of an array (np/cp).

    Args:
        arr: Numpy or Cupy array.

    """
    if isinstance(arr, np.ndarray):
        return np
    elif isinstance(arr, cp.ndarray):
        return cp
    else:
        raise ValueError("arr must be a numpy or cupy array.")


def generate_noise_fd(N: int, df: float, *sensitivity_args: typing.Any, func: typing.Optional[typing.Callable]=None, **sensitivity_kwargs: typing.Any) -> np.ndarray[float]:
    """Generate noise directly in the Frequency Domain. 
    
    Args:
        N: Number of points in frequency domain waveform assuming frequencies greater than
            or equal to zero.
        df: Frequency domain bin size (``1 / (N * dt)``). 
        sensitivity_args: Arguments for ``func``.
        func: Function for generating the sensitivity curve as a function of frequency. (default: :func:`lisatools.sensitivity.get_sensivity`)
        sensitivity_kwargs: Keyword arguments for ``func``. 

    Returns:
        An instance of generated noise in the frequency domain. 


    """
    if not isinstance(N, int):
        raise ValueError(f"N must be an integer. See documentation for more information.")

    if not isinstance(df, float):
        raise ValueError(f"N must be an integer. See documentation for more information.")

    if func is None:
        # TODO: make this better
        from lisatools.sensitivity import get_sensitivity
        func = get_sensitivity

    freqs = np.arange(N) * df

    norm = 0.5 * (1.0 / df) ** 0.5
    psd = func(freqs, *sensitivity_args, **sensitivity_kwargs)
    noise_realization = psd ** (1 / 2) * (
        np.random.normal(0, norm, len(freqs))
        + 1j * np.random.normal(0, norm, len(freqs))
    )
    return noise_realization


def AET(
    X: float | np.ndarray, Y: float | np.ndarray, Z: float | np.ndarray
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Transform to AET from XYZ

    .. math::

        A = (Z - X) / \\sqrt(2)

    .. math::

        E = (X - 2Y + Z) / \\sqrt(6)

    .. math::

        T = (X + Y + Z) / \\sqrt(3)

    Args:
        X: X-channel information.
        Y: Y-channel information.
        Z: Z-channel information.

    Returns:
        A, E, T Channels.

    """
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


# def searchsorted2d_vec(a, b, xp=None, gpu=None, **kwargs):
#     if xp is None:
#         xp = np
#     else:
#         try:
#             xp.cuda.runtime.setDevice(gpu)
#         except AttributeError:
#             pass

#     m, n = a.shape
#     max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
#     r = max_num * xp.arange(a.shape[0])[:, None]
#     p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)

#     out = p - n * (xp.arange(m)[:, None])
#     try:
#         xp.cuda.runtime.deviceSynchronize()
#     except AttributeError:
#         pass

#     return out

