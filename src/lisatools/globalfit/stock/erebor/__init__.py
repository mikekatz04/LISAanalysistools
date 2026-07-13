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
from .stochastic import (  # noqa: F401
    SGWBSettings,
    SGWBSetup,
    get_sgwb_erebor_settings,
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

# --- stock family base + variant registry ----------------------------------
from .fit import EreborFit, EreborGeneralSettings  # noqa: F401

_registry = StockRegistry("erebor")

get_stock = _registry.get
get_stock_options = _registry.options

from .variants.all_sources import AllSourcesGlobalFit  # noqa: E402
from .variants.full_year_combined import FullYearCombinedGlobalFit  # noqa: E402
from .variants.gb_no_fg import GBNoForegroundGlobalFit  # noqa: E402
from .variants.lite import (  # noqa: E402
    AllSourcesLiteGlobalFit,
    FullYearCombinedLiteGlobalFit,
    GBNoForegroundLiteGlobalFit,
    NoiseOnlyLiteGlobalFit,
    NoiseSGWBLiteGlobalFit,
)
from .variants.noise import NoiseOnlyGlobalFit, NoiseSGWBGlobalFit  # noqa: E402

for _cls in (
    GBNoForegroundGlobalFit,
    AllSourcesGlobalFit,
    FullYearCombinedGlobalFit,
    NoiseOnlyGlobalFit,
    NoiseSGWBGlobalFit,
    GBNoForegroundLiteGlobalFit,
    AllSourcesLiteGlobalFit,
    FullYearCombinedLiteGlobalFit,
    NoiseOnlyLiteGlobalFit,
    NoiseSGWBLiteGlobalFit,
):
    _registry.register(_cls)

#: Names of the registered stock options (mirrors the sensitivity.py idiom).
__stock_globalfit_options__ = _registry.names()

# Module-level callable default instances: lightweight, so the defaults live
# in memory. ``erebor.gb_no_fg(**overrides)`` CLONES the defaults into a
# fresh fit (the primary pattern); mutating ``erebor.gb_no_fg`` itself
# changes the session-wide default for later clones — deliberate, but shared.
# Every full model has a ``*_lite`` laptop-smoke twin (same machinery, all
# knobs turned down; equivalently ``erebor.<name>(lite=True)``).
gb_no_fg = GBNoForegroundGlobalFit()
all_sources = AllSourcesGlobalFit()
full_year_combined = FullYearCombinedGlobalFit()
noise_only = NoiseOnlyGlobalFit()
noise_sgwb = NoiseSGWBGlobalFit()
gb_no_fg_lite = GBNoForegroundLiteGlobalFit()
all_sources_lite = AllSourcesLiteGlobalFit()
full_year_combined_lite = FullYearCombinedLiteGlobalFit()
noise_only_lite = NoiseOnlyLiteGlobalFit()
noise_sgwb_lite = NoiseSGWBLiteGlobalFit()


def __getattr__(name):
    # Nicer error messages on unknown attributes; the variant module
    # attributes (gb_no_fg, ...) are real module globals set above.
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Stock options: {_registry.names()}."
    )
