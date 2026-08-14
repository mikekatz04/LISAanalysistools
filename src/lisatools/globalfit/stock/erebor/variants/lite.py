"""Lite (laptop-smoke) twins of every Erebor stock variant.

Each registered stock model gets a ``*_lite`` twin whose ONLY difference is
the lite preset: every knob turned down to a small, CPU-friendly smoke
configuration (short WDM grid, ~3 iterations, 4 walkers x 2 temps, few GB
repeat proposals, CPU compute). Two equivalent spellings::

    fit = erebor.all_sources_lite()          # the registered twin
    fit = erebor.all_sources(lite=True)      # the kwarg on the full model

Both apply the same per-variant override table (:data:`ALL_SOURCES_LITE`,
...), so lite <-> heavy conversion is just knob changes: scale a lite fit UP
by assigning production values (``fit.general.tobs_target = YRSID_SI``,
``fit.gb.num_repeat_proposals = 100``, ...), or scale the full model DOWN
with ``lite=True`` / individual assignments. Precedence when the preset is
applied: explicit kwarg > env var > lite preset > class default.

Env vars overrule the lite preset: each preset table has a companion
``*_LITE_ENV`` map (dotted path -> env-var name) so that a set env var (e.g.
``NWALKERS=16``, ``USE_GPU=1``) keeps its construction-time value instead of
being stamped down by the preset. See :meth:`StockGlobalFit.apply_lite` /
:meth:`StockGlobalFit.lite_env_vars`.

This module is also the single home of the preset tables — they are
attached to the FULL variant classes here (rather than edited into each
variant module) so the ``lite=True`` kwarg works on those classes too.
"""

from __future__ import annotations

from .all_sources import AllSourcesGlobalFit
from .full_year_combined import FullYearCombinedGlobalFit
from .gb_no_fg import GBNoForegroundGlobalFit
from .noise import NoiseOnlyGlobalFit, NoiseSGWBGlobalFit
from .vgb import VGBGlobalFit

__all__ = [
    "ALL_SOURCES_LITE",
    "FULL_YEAR_COMBINED_LITE",
    "GB_NO_FG_LITE",
    "NOISE_ONLY_LITE",
    "NOISE_SGWB_LITE",
    "VGB_LITE",
    "AllSourcesLiteGlobalFit",
    "FullYearCombinedLiteGlobalFit",
    "GBNoForegroundLiteGlobalFit",
    "NoiseOnlyLiteGlobalFit",
    "NoiseSGWBLiteGlobalFit",
    "VGBLiteGlobalFit",
]

# ---------------------------------------------------------------------------
# Preset tables (dotted fit paths -> lite values) + their env-var maps
# ---------------------------------------------------------------------------
#
# Each ``*_LITE`` table has a companion ``*_LITE_ENV`` map giving, for every
# preset path that is ALSO env-backed, the env-var name that overrules it (a
# set env var keeps its construction-time value; the preset is skipped for
# that path). Paths with no env var (e.g. fixed-grid ``general.nf``/``nt``)
# are simply absent from the map — the preset always applies to them.

_COMMON_LITE = {
    # A few iterations is enough for a smoke: it exercises the whole pipeline
    # and shows the output shapes. Kept low so the heaviest lite fit
    # (all_sources, six branches incl. MBH phentax) finishes and stays within
    # a laptop's memory even inside a notebook kernel. Scale up for real runs.
    "general.num_iterations": 3,
    "general.nwalkers": 4,
    # per-branch ladders (general.ntemps is retired: the engine runs
    # cold-chain only; entries for branches a variant lacks are skipped)
    "gb.ntemps": 2,
    "vgb.ntemps": 2,
    "mbh.ntemps": 2,
    "emri.ntemps": 2,
    "sobbh.ntemps": 2,
    "psd.ntemps": 2,
    "galfor.ntemps": 2,
    "sgwb.ntemps": 2,
    # lite defaults to a CPU smoke, but compute device is environmental (not a
    # size knob): ``USE_GPU`` is in _COMMON_LITE_ENV, so ``USE_GPU=1`` keeps
    # the env-resolved value and this default is skipped. An explicit
    # ``use_gpu=`` kwarg is applied after the preset and still beats both.
    "general.use_gpu": False,
}
_COMMON_LITE_ENV = {
    "general.num_iterations": "NUM_ITERATIONS",
    "general.nwalkers": "NWALKERS",
    "gb.ntemps": "GB_NTEMPS",
    "vgb.ntemps": "VGB_NTEMPS",
    "mbh.ntemps": "MBH_NTEMPS",
    "emri.ntemps": "EMRI_NTEMPS",
    "sobbh.ntemps": "SOBBH_NTEMPS",
    "psd.ntemps": "PSD_NTEMPS",
    "galfor.ntemps": "GALFOR_NTEMPS",
    "sgwb.ntemps": "SGWB_NTEMPS",
    "general.use_gpu": "USE_GPU",
}

# all_sources: fixed small grid — nf=720, nt=180 (the validated smoke shape;
# ~7.5 d at dt=5 synthetic, ~3.75 d at the mojito dt=2.5 flip) — plus a
# narrow GB band (small sub-band count) and few repeat proposals.
# NOTE: even lite, the six-branch fit peaks at ~4 GB (per-branch waveform
# generation + WDM transforms), so it wants >8 GB RAM to run comfortably in a
# notebook kernel; the single-purpose lite variants (gb_no_fg_lite,
# noise_only_lite) run on any laptop.
ALL_SOURCES_LITE = {
    **_COMMON_LITE,
    "general.nf": 720,
    "general.nt": 180,
    "general.plot_iterations": 10,
    "gb.min_freq": 1.0e-3,
    "gb.max_freq": 1.0e-2,
    "gb.num_repeat_proposals": 2,
}
ALL_SOURCES_LITE_ENV = {
    **_COMMON_LITE_ENV,
    # general.nf / general.nt are the fixed-grid override (not env-backed).
    "general.plot_iterations": "PLOT_ITERATIONS",
    "gb.min_freq": "GB_MIN_FREQ",
    "gb.max_freq": "GB_MAX_FREQ",
    "gb.num_repeat_proposals": "GB_NUM_REPEAT_PROPOSALS",
}

# gb_no_fg already runs a narrow band; lite shortens the span + proposals.
GB_NO_FG_LITE = {
    **_COMMON_LITE,
    "general.tobs_target": 14 * 86400.0,
    "gb.num_repeat_proposals": 2,
}
GB_NO_FG_LITE_ENV = {
    **_COMMON_LITE_ENV,
    "general.tobs_target": "TOBS_TARGET",
    "gb.num_repeat_proposals": "GB_NUM_REPEAT_PROPOSALS",
}

# full_year_combined: a month instead of a year (nwalkers/ntemps drop from
# the variant's 6x3 to the common 4x2).
FULL_YEAR_COMBINED_LITE = {
    **_COMMON_LITE,
    "general.tobs_target": 30 * 86400.0,
}
FULL_YEAR_COMBINED_LITE_ENV = {
    **_COMMON_LITE_ENV,
    "general.tobs_target": "TOBS_TARGET",
}

# vgb: fixed-dimensional leaves need no RJ machinery — the lite twin just shortens
# the span (matching gb_no_fg_lite) on top of the common smoke shape.
VGB_LITE = {
    **_COMMON_LITE,
    "general.tobs_target": 14 * 86400.0,
}
VGB_LITE_ENV = {
    **_COMMON_LITE_ENV,
    "general.tobs_target": "TOBS_TARGET",
}

# noise fits: quarter-length time grid on the stock 768-layer band.
NOISE_ONLY_LITE = {
    **_COMMON_LITE,
    "general.nf": 768,
    "general.nt": 256,
}
# general.nf / general.nt are not env-backed on the noise variants.
NOISE_ONLY_LITE_ENV = dict(_COMMON_LITE_ENV)
NOISE_SGWB_LITE = dict(NOISE_ONLY_LITE)
NOISE_SGWB_LITE_ENV = dict(NOISE_ONLY_LITE_ENV)


def _attach(cls, table: dict, env_map: dict) -> None:
    """Give ``cls`` (and its lite twin, by inheritance) the preset table + env map.

    The env map (dotted path -> env-var name) is consulted by
    :meth:`StockGlobalFit.apply_lite`: a path whose env var is set keeps its
    construction-time (env-resolved) value and the preset is skipped for it,
    so env vars overrule the preset (``USE_GPU`` included — no longer a
    special case).
    """

    def lite_overrides(self) -> dict:
        return dict(table)

    def lite_env_vars(self) -> dict:
        return dict(env_map)

    lite_overrides.__doc__ = (
        f"Laptop-smoke preset for ``{cls.option_name}`` (see "
        "``stock/erebor/variants/lite.py``)."
    )
    lite_env_vars.__doc__ = (
        f"Preset-path -> env-var map for ``{cls.option_name}`` (env vars "
        "overrule the lite preset; see ``stock/erebor/variants/lite.py``)."
    )
    cls.lite_overrides = lite_overrides
    cls.lite_env_vars = lite_env_vars


_attach(AllSourcesGlobalFit, ALL_SOURCES_LITE, ALL_SOURCES_LITE_ENV)
_attach(GBNoForegroundGlobalFit, GB_NO_FG_LITE, GB_NO_FG_LITE_ENV)
_attach(FullYearCombinedGlobalFit, FULL_YEAR_COMBINED_LITE, FULL_YEAR_COMBINED_LITE_ENV)
_attach(NoiseOnlyGlobalFit, NOISE_ONLY_LITE, NOISE_ONLY_LITE_ENV)
_attach(NoiseSGWBGlobalFit, NOISE_SGWB_LITE, NOISE_SGWB_LITE_ENV)
_attach(VGBGlobalFit, VGB_LITE, VGB_LITE_ENV)


# ---------------------------------------------------------------------------
# Registered lite twins
# ---------------------------------------------------------------------------


class AllSourcesLiteGlobalFit(AllSourcesGlobalFit):
    """``all_sources`` with the lite preset applied at construction."""

    option_name = "all_sources_lite"
    description = (
        "Laptop-smoke twin of all_sources: the same six branches and "
        "machinery on a small WDM grid (nf=720, nt=180), 3 iterations, "
        "4 walkers x 2 temps, narrow GB band, CPU. Scale any knob back up "
        "to reach the full model (and vice-versa via all_sources(lite=True))."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class GBNoForegroundLiteGlobalFit(GBNoForegroundGlobalFit):
    """``gb_no_fg`` with the lite preset applied at construction."""

    option_name = "gb_no_fg_lite"
    description = (
        "Laptop-smoke twin of gb_no_fg: two-week span, 3 iterations, "
        "4 walkers x 2 temps, 2 GB repeat proposals, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class FullYearCombinedLiteGlobalFit(FullYearCombinedGlobalFit):
    """``full_year_combined`` with the lite preset applied at construction."""

    option_name = "full_year_combined_lite"
    description = (
        "Laptop-smoke twin of full_year_combined: one month instead of a "
        "year, 3 iterations, 4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class NoiseOnlyLiteGlobalFit(NoiseOnlyGlobalFit):
    """``noise_only`` with the lite preset applied at construction."""

    option_name = "noise_only_lite"
    description = (
        "Laptop-smoke twin of noise_only: quarter-length time grid, "
        "3 iterations, 4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class NoiseSGWBLiteGlobalFit(NoiseSGWBGlobalFit):
    """``noise_sgwb`` with the lite preset applied at construction."""

    option_name = "noise_sgwb_lite"
    description = (
        "Laptop-smoke twin of noise_sgwb: quarter-length time grid, "
        "3 iterations, 4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class VGBLiteGlobalFit(VGBGlobalFit):
    """``vgb`` with the lite preset applied at construction."""

    option_name = "vgb_lite"
    description = (
        "Laptop-smoke twin of vgb: two-week span, 3 iterations, "
        "4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)
