# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import warnings

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import GroupStretchMove
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds
from .gbmultipletryrj import GBMutlipleTryRJ

from ...diagnostic import inner_product


__all__ = ["GBGroupStretchMove"]


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBGroupStretchMove(GroupStretchMove, GBMutlipleTryRJ):
    """Generate Revesible-Jump proposals for GBs with try-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`ProbDistContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb_args,
        gb_kwargs,
        start_ind_limit=10,
        *args,
        **kwargs
    ):

        self.name = "gbgroupstretch"
        self.start_ind_limit = start_ind_limit
        GBMutlipleTryRJ.__init__(self, *gb_args, **gb_kwargs)
        GroupStretchMove.__init__(self, *args, **kwargs)

    