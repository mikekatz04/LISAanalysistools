"""Back-compat shim -- moved to :mod:`lisatools.globalfit.warmstart.proposal`.

User ruling 2026-09-14: all the warm-start code lives in
``lisatools.globalfit.warmstart``. Import from there; this module
re-exports the old names so existing imports keep working (the public names
of :mod:`lisatools.globalfit.warmstart.proposal`, plus ``CIRCULAR_COLS``,
``COLUMN_NAMES`` and :class:`~lisatools.globalfit.warmstart.proposal.WarmStartComponents`
explicitly). Only tests import it; the run path imports the new location.
"""

from ..globalfit.warmstart.proposal import *  # noqa: F401,F403
from ..globalfit.warmstart.proposal import (  # noqa: F401
    CIRCULAR_COLS,
    COLUMN_NAMES,
    WarmStartComponents,
)
