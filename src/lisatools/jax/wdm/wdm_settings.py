"""JAX port of ``WDMSettings`` / ``WDMSettingsWrap`` (TDIonTheFly.hh).

Pure-Python POD holder mirroring the C++ ``WDMSettingsWrap`` data
container. Holds the 9 scalar ints/doubles that describe the WDM grid
and the active band; constructor signature matches the C++ wrapper
exactly so the JAX backend slot can be a drop-in replacement.
"""
from __future__ import annotations


class WDMSettingsWrapJAX:
    """Mirror of the C++ ``WDMSettingsWrap`` data container.

    Constructor signature matches the C++ wrapper exactly:
        (layer_df, layer_dt,
         Nf, Nt, num_channel,
         ind_min_t, ind_max_t, ind_min_f, ind_max_f)
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
