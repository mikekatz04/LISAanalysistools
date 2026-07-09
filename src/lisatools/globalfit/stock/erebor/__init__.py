"""Erebor stock global-fit family.

This package holds the Erebor recipe's building blocks (per-branch
``*Settings``/``*Setup`` pairs, transform containers, shared infrastructure)
plus the registered stock run variants.

Primary interface::

    from lisatools.globalfit.stock import erebor

    erebor.get_stock_options()          # available variants + descriptions
    fit = erebor.get_stock("gb_no_fg", nwalkers=16)
    fit = erebor.gb_no_fg(nwalkers=16)  # same thing via the module attribute

Everything importable from the old ``stock/erebor.py`` module remains
importable from here.
"""

from __future__ import annotations

# --- back-compat: the full public surface of the old erebor.py module ------
from ...engine import GeneralSetup, Settings, Setup  # noqa: F401
from ..base import StockRegistry
from .emri import EMRIHDFBackend, EMRISettings, EMRISetup, EMRIState  # noqa: F401
from .gb import GBHDFBackend, GBSettings, GBSetup, GBState  # noqa: F401
from .mbh import MBHHDFBackend, MBHSettings, MBHSetup, MBHState  # noqa: F401
from .noise import (  # noqa: F401
    EqualArmlengthOrbits,
    GalForSettings,
    GalForSetup,
    PSDSettings,
    PSDSetup,
    get_galfor_erebor_settings,
)
from .sobbh import SOBBHHDFBackend, SOBBHSettings, SOBBHSetup, SOBBHState  # noqa: F401
from .transforms import (  # noqa: F401
    LISA_to_SSB,
    SSB_to_LISA,
    beta_to_cosqS,
    cosqS_to_beta,
    f_ms_to_s,
    f_s_to_ms,
    gpc_to_mpc,
    m1_m2_to_mT_Q,
    m1_m2_to_mT_q,
    make_emri_transform_container,
    make_gb_transform_container,
    make_mbh_transform_container,
    make_sobbh_transform_container,
    mbh_dist_trans,
    mpc_to_gpc,
    mT_Q,
    mT_q,
    ten_to_the_x,
)

# --- stock variant registry -------------------------------------------------
_registry = StockRegistry("erebor")

get_stock = _registry.get
get_stock_options = _registry.options


def __getattr__(name):
    # Defined for nicer error messages on unknown attributes; the variant
    # module attributes (gb_no_fg, ...) are real module globals set below.
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Stock options: {_registry.names()}."
    )
