"""Lite (laptop-smoke) twins of every Erebor stock variant.

Each registered stock model gets a ``*_lite`` twin whose ONLY difference is
the lite preset: every knob turned down to a small, CPU-friendly smoke
configuration (short WDM grid, ~10 iterations, 4 walkers x 2 temps, few GB
repeat proposals, CPU compute). Two equivalent spellings::

    fit = erebor.all_sources_lite()          # the registered twin
    fit = erebor.all_sources(lite=True)      # the kwarg on the full model

Both apply the same per-variant override table (:data:`ALL_SOURCES_LITE`,
...), so lite <-> heavy conversion is just knob changes: scale a lite fit UP
by assigning production values (``fit.general.tobs_target = YRSID_SI``,
``fit.gb.num_repeat_proposals = 100``, ...), or scale the full model DOWN
with ``lite=True`` / individual assignments. Precedence when the preset is
applied: explicit kwarg > lite preset > env var > class default.

This module is also the single home of the preset tables — they are
attached to the FULL variant classes here (rather than edited into each
variant module) so the ``lite=True`` kwarg works on those classes too.
"""

from __future__ import annotations

from .all_sources import AllSourcesGlobalFit
from .full_year_combined import FullYearCombinedGlobalFit
from .gb_no_fg import GBNoForegroundGlobalFit
from .noise import NoiseOnlyGlobalFit, NoiseSGWBGlobalFit

__all__ = [
    "ALL_SOURCES_LITE",
    "FULL_YEAR_COMBINED_LITE",
    "GB_NO_FG_LITE",
    "NOISE_ONLY_LITE",
    "NOISE_SGWB_LITE",
    "AllSourcesLiteGlobalFit",
    "FullYearCombinedLiteGlobalFit",
    "GBNoForegroundLiteGlobalFit",
    "NoiseOnlyLiteGlobalFit",
    "NoiseSGWBLiteGlobalFit",
]

# ---------------------------------------------------------------------------
# Preset tables (dotted fit paths -> lite values)
# ---------------------------------------------------------------------------

_COMMON_LITE = {
    # A few iterations is enough for a smoke: it exercises the whole pipeline
    # and shows the output shapes. Kept low so the heaviest lite fit
    # (all_sources, six branches incl. MBH phentax) finishes and stays within
    # a laptop's memory even inside a notebook kernel. Scale up for real runs.
    "general.num_iterations": 3,
    "general.nwalkers": 4,
    "general.ntemps": 2,
    "general.use_gpu": False,  # lite == CPU smoke; pass use_gpu=True to override
}

# all_sources: fixed small grid — nf=720, nt=180 (the validated smoke shape;
# ~7.5 d at dt=5 synthetic, ~3.75 d at the mojito dt=2.5 flip) — plus a
# narrow GB band (small sub-band count) and few repeat proposals.
ALL_SOURCES_LITE = {
    **_COMMON_LITE,
    "general.nf": 720,
    "general.nt": 180,
    "general.plot_iterations": 10,
    "gb.min_freq": 1.0e-3,
    "gb.max_freq": 1.0e-2,
    "gb.num_repeat_proposals": 2,
}

# gb_no_fg already runs a narrow band; lite shortens the span + proposals.
GB_NO_FG_LITE = {
    **_COMMON_LITE,
    "general.tobs_target": 14 * 86400.0,
    "gb.num_repeat_proposals": 2,
}

# full_year_combined: a month instead of a year (nwalkers/ntemps drop from
# the variant's 6x3 to the common 4x2).
FULL_YEAR_COMBINED_LITE = {
    **_COMMON_LITE,
    "general.tobs_target": 30 * 86400.0,
}

# noise fits: quarter-length time grid on the stock 768-layer band.
NOISE_ONLY_LITE = {
    **_COMMON_LITE,
    "general.nf": 768,
    "general.nt": 256,
}
NOISE_SGWB_LITE = dict(NOISE_ONLY_LITE)


def _attach(cls, table: dict) -> None:
    """Give ``cls`` (and its lite twin, by inheritance) the preset table."""

    def lite_overrides(self) -> dict:
        return dict(table)

    lite_overrides.__doc__ = (
        f"Laptop-smoke preset for ``{cls.option_name}`` (see "
        "``stock/erebor/variants/lite.py``)."
    )
    cls.lite_overrides = lite_overrides


_attach(AllSourcesGlobalFit, ALL_SOURCES_LITE)
_attach(GBNoForegroundGlobalFit, GB_NO_FG_LITE)
_attach(FullYearCombinedGlobalFit, FULL_YEAR_COMBINED_LITE)
_attach(NoiseOnlyGlobalFit, NOISE_ONLY_LITE)
_attach(NoiseSGWBGlobalFit, NOISE_SGWB_LITE)


# ---------------------------------------------------------------------------
# Registered lite twins
# ---------------------------------------------------------------------------


class AllSourcesLiteGlobalFit(AllSourcesGlobalFit):
    """``all_sources`` with the lite preset applied at construction."""

    option_name = "all_sources_lite"
    description = (
        "Laptop-smoke twin of all_sources: the same six branches and "
        "machinery on a small WDM grid (nf=720, nt=180), 10 iterations, "
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
        "Laptop-smoke twin of gb_no_fg: two-week span, 10 iterations, "
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
        "year, 10 iterations, 4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class NoiseOnlyLiteGlobalFit(NoiseOnlyGlobalFit):
    """``noise_only`` with the lite preset applied at construction."""

    option_name = "noise_only_lite"
    description = (
        "Laptop-smoke twin of noise_only: quarter-length time grid, "
        "10 iterations, 4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)


class NoiseSGWBLiteGlobalFit(NoiseSGWBGlobalFit):
    """``noise_sgwb`` with the lite preset applied at construction."""

    option_name = "noise_sgwb_lite"
    description = (
        "Laptop-smoke twin of noise_sgwb: quarter-length time grid, "
        "10 iterations, 4 walkers x 2 temps, CPU."
    )

    def __init__(self, **knobs):
        knobs.setdefault("lite", True)
        super().__init__(**knobs)
