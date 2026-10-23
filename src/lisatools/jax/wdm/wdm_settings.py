"""JAX port of ``WDMSettings`` / ``WDMSettingsWrap``.

Pure-Python POD holder mirroring the C++ ``WDMSettingsWrap`` data
container (``lisatools/cutils/binding_wdm_settings.hpp``; the
``WDMSettings`` struct lives in ``domains.hpp``). Holds the 9 scalar
ints/doubles that describe the WDM grid and the active band; constructor
signature matches the C++ wrapper exactly so the JAX backend slot can be a
drop-in replacement.
"""
from __future__ import annotations


class WDMSettingsWrapJAX:
    """Mirror of the C++ ``WDMSettingsWrap`` data container.

    Constructor signature matches the C++ wrapper exactly. Every argument is
    stored under its own name (cast to ``float`` / ``int``), plus the derived
    active-band sizes.

    Args:
        layer_df: WDM frequency-layer width [Hz].
        layer_dt: WDM time-pixel width [s].
        Nf: Number of frequency layers.
        Nt: Number of time pixels.
        num_channel: Number of TDI channels.
        ind_min_t: First time pixel of the active band.
        ind_max_t: Last time pixel of the active band (inclusive).
        ind_min_f: First frequency layer of the active band.
        ind_max_f: Last frequency layer of the active band (inclusive).

    Attributes:
        Nf_active (int): ``ind_max_f - ind_min_f + 1``.
        Nt_active (int): ``ind_max_t - ind_min_t + 1``.

    Example:
        .. code-block:: python

            from lisatools.jax.wdm.wdm_settings import WDMSettingsWrapJAX  # needs jax

            s = WDMSettingsWrapJAX(1e-4, 5000.0, 256, 128, 3, 0, 127, 10, 20)
            s.Nf_active, s.Nt_active   # (11, 128)
    """

    def __init__(self,
                 layer_df: float, layer_dt: float,
                 Nf: int, Nt: int, num_channel: int,
                 ind_min_t: int, ind_max_t: int,
                 ind_min_f: int, ind_max_f: int):
        self.layer_df = float(layer_df)
        self.layer_dt = float(layer_dt)
        self.Nf = int(Nf)
        self.Nt = int(Nt)
        self.num_channel = int(num_channel)
        self.ind_min_t = int(ind_min_t)
        self.ind_max_t = int(ind_max_t)
        self.ind_min_f = int(ind_min_f)
        self.ind_max_f = int(ind_max_f)
        self.Nf_active = self.ind_max_f - self.ind_min_f + 1
        self.Nt_active = self.ind_max_t - self.ind_min_t + 1
