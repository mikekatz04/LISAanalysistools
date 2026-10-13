"""Run identity for the monitor pages: banner label, run kind and arm-cache tag.

The page generator (``_generator.py``, and its lean twin under
``scripts/diagnostics``) names the run it is drawing from the STORE DIRECTORY
name: the banner reads ``RUN_LABEL``, the per-run completeness curve is cached
as ``gf_arm_<ARM_TAG>.npz`` in the page directory, and every ``gf_arm_*.npz``
found there is overlaid as a comparison arm. A name the ladder does not
recognise used to fall through to ``("3-Month", "3mo")`` -> arm tag ``v2`` --
the WRONG banner and, worse, a silent overwrite of the shared ``gf_arm_v2.npz``
cache. That happened four times (1yr 2026-08-22, v7 08-26, v8 09-02, 3mo v9
09-28), each fixed by one more hard-coded branch.

2026-10-07 (the 9mo v9 run): the ladder lives here, every historical branch
kept VERBATIM (their arm caches already exist on the cluster pages, so their
tags must not move), plus ONE generic rule for anything else that carries a
``<N>mo_v<K>`` / ``<N>yr_v<K>`` token: ``gf_prod_9mo_v9`` -> ``("9-Month v9",
"9mo_v9")``, ``gf_prod_3mo_v9_2gpu`` -> ``("3-Month v9", "3mo_v9")``,
``gf_prod_1yr_v9`` -> ``("1-Year v9", "1yr_v9")``. The plain ``3mo`` / ``v2``
fall-through remains only for a name with no version token at all (the
original 3-month v2 run).
"""
from __future__ import annotations

import re

__all__ = ["run_identity", "arm_tag"]

#: ``<N>mo_v<K>`` / ``<N>yr_v<K>`` anywhere in the name (``(?<![0-9])`` keeps
#: ``gf_prod_23mo_v1`` from matching as ``3mo``).
_VARIANT = re.compile(r"(?<![0-9])(\d+)(mo|yr)_v(\d+)")

#: The flagship series (2026-10-11, scripts/flagship): ``gf_prod_<N>mo_<tag>``
#: with a series tag from ``_SERIES_TAGS`` -> ``("<N>-Month <tag>",
#: "<N>mo_<tag>")``. Checked FIRST so ``gf_prod_6mo_flagship`` never lands on
#: the plain ``6mo`` arm cache of the live 6mo v9 run, and ``gf_prod_12mo_...``
#: / ``gf_prod_24mo_...`` never fall through to the 3-month default. The 9mo
#: keeps its live name (``gf_prod_9mo_v9``, the generic variant rule). A
#: renamed series (runs.py ``SERIES``) must be added here.
_SERIES_TAGS = ("flagship",)
_SERIES = re.compile(r"(?<![0-9])(\d+)(mo|yr)_(" + "|".join(_SERIES_TAGS) + r")(?![a-z0-9])")

#: Run kinds whose arm cache predates the per-kind naming (2026-08-22) and
#: keeps its historical file name.
_LEGACY_ARM_TAGS = {"3mo_v3": "v3", "3mo_v4": "v4", "3mo": "v2"}


def _generic(m: "re.Match[str]") -> tuple[str, str]:
    n, unit, v = m.group(1), m.group(2), m.group(3)
    return f"{n}-{'Month' if unit == 'mo' else 'Year'} v{v}", f"{n}{unit}_v{v}"


def run_identity(base: str) -> tuple[str, str]:
    """``(RUN_LABEL, RUN_KIND)`` for a run directory basename.

    The historical branches come first and are unchanged (see the module
    docstring); the generic ``<N>mo_v<K>`` rule claims any other versioned
    name before the plain-3-month fall-through. A ``1yr_v<K>`` name other
    than v8 takes the generic rule too (the bare ``1yr`` branch is the v5
    1-year run, whose store carries no version token).
    """
    base = str(base)
    m = _SERIES.search(base)
    if m is not None:
        n, unit, tag = m.group(1), m.group(2), m.group(3)
        return f"{n}-{'Month' if unit == 'mo' else 'Year'} {tag}", f"{n}{unit}_{tag}"
    if "23mo" in base:
        return "23-Month", "23mo"
    if "6mo" in base:
        # kept as the plain 6-Month identity: the 6mo v8 and v9 pages have
        # been writing gf_arm_6mo.npz since 2026-08-15, and renaming the tag
        # would leave that file behind as a duplicate overlay arm
        return "6-Month", "6mo"
    if base.endswith("_v4") or "3mo_v4" in base:
        return "3-Month v4", "3mo_v4"
    if base.endswith("_v3") or "3mo_v3" in base:
        # The v3 A/B carries the same Tobs as v2, so the label has to come
        # from the VARIANT or the two pages are indistinguishable in a
        # browser tab -- which is the whole point of running them side by side.
        return "3-Month v3", "3mo_v3"
    if "1yr_v8" in base or (base.endswith("_v8") and "1yr" in base):
        # 2026-09-03: the v8 lineage 1-yr run (submit_gf_1yr_v8.sh). Its own
        # arm cache + banner so it never collides with the v5 1-yr page.
        return "1-Year v8", "1yr_v8"
    m = _VARIANT.search(base)
    if "1yr" in base:
        if m is not None and m.group(2) == "yr":
            return _generic(m)              # 1yr_v9 and later
        # 2026-08-22: without this branch the 1-yr page fell through to the
        # 3-Month banner AND (worse) the v2 ARM_TAG, clobbering the shared
        # gf_arm_v2.npz cache with 1-yr data.
        return "1-Year v5", "1yr_v5"
    if base.endswith("_v5") or "3mo_v5" in base:
        return "3-Month v5", "3mo_v5"
    if base.endswith("_v6") or "3mo_v6" in base:
        return "3-Month v6", "3mo_v6"
    if base.endswith("_v7") or "3mo_v7" in base:
        # 2026-08-26: same trap as the 1-yr branch above.
        return "3-Month v7", "3mo_v7"
    if "10walker" in base and ("3mo_v8" in base or base.endswith("_v8")):
        # 2026-09-05: the v8 3-month 10-WALKER twin (gf_prod_3mo_v8_10walkers).
        # Own arm tag + banner so its overlay curve reads "3mo_v8_10w",
        # distinct from the 24-walker "3mo_v8_24w" arm -- both otherwise
        # collide on the plain "3mo_v8" tag and clobber each other's cache.
        return "3-Month v8 · 10 Walkers", "3mo_v8_10w"
    if base.endswith("_v8") or "3mo_v8" in base:
        return "3-Month v8", "3mo_v8"
    if m is not None:
        # the generic rule (2026-10-07): 9mo v9, 3mo v9, and whatever comes next
        return _generic(m)
    return "3-Month", "3mo"


def arm_tag(run_kind: str, override: str | None = None) -> str:
    """The ``gf_arm_<tag>.npz`` tag for ``run_kind`` (``override`` wins when set).

    Every run kind gets its OWN arm cache (2026-08-22): the old {v3, v4}-else-v2
    map sent v5, v6 and the 1-yr run all to ``gf_arm_v2.npz``, silently
    clobbering the shared v2 arm. The three legacy kinds keep their historical
    file names.
    """
    if override:
        return str(override)
    return _LEGACY_ARM_TAGS.get(str(run_kind), str(run_kind))
