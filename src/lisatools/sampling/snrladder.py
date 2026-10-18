"""Per-leaf temperature ladders keyed to each leaf's own SNR (default OFF).

User ruling 2026-10-18 (Mike). The EMRI ladders went to a uniform top
temperature so the LOUDEST leaf (EMRI 3, SNR 50) sees an effective SNR ~2 at
the top rung ("let's go until it is at SNR ~2"): the valleys between the
discrete posterior islands scale with SNR^2, so a hot rung can hop them only
when the valley has shrunk to a few nats. A uniform ``T_max`` is right for
ONE SNR only: at 625 the SNR 12-15 leaves have prior-like top rungs and the
SNR 20-35 leaves are hotter than they need. "While that runs, you can add an
option (default off) to set the ladder by SNR."

The rule, per leaf ``k`` of an MBH / EMRI / SOBBH branch (the add/remove
family, one ladder per leaf, ``betas_all[k]``)::

    T_k = clip((SNR_k / top_snr)^2, tmax_min, tmax)
    betas_all[k] = make_ladder(ndim, ntemps, Tmax=T_k)

so the top rung sees the leaf at an effective SNR ``SNR_k / sqrt(T_k) =
top_snr`` unless the clip bites. ``top_snr`` is ``{BRANCH}_LADDER_TOP_SNR``
(0 = off; the words on / true / yes take :data:`LADDER_TOP_SNR_DEFAULT` =
2.0), the floor ``{BRANCH}_LADDER_TMAX_MIN`` (default 4) and the ceiling the
existing ``{BRANCH}_LADDER_TMAX`` (unset: the top temperature of the uncapped
``make_ladder(ndim, ntemps)``). With top SNR 2, floor 4 and ceiling 625:
EMRI 3 (SNR 50) -> 625 (the ceiling), EMRI 6 (SNR 34) -> 289, EMRI 0 (SNR
21) -> 110, an SNR 12 leaf -> 36, a prior-only leaf (SNR < 2) -> 4.

The rung COUNT stays the branch's (the store freezes ``betas_all``'s
``ntemps`` axis), so the spacing varies per leaf: the ceiling's leaf gets
make_ladder's ~25 % swap spacing at the launcher count (``make_ladder(12,
Tmax=625)`` = 11 rungs, ratio 1.90), and a leaf below the ceiling a DENSER
ladder (smaller ratio, higher swap acceptance). The log line names each
leaf's ratio.

``SNR_k`` is the leaf's OPTIMAL SNR ``sqrt(<h|h>)`` at the max-lnL cold
walker (the walker the ``walker_max`` eigen table is built at), from the
per-walker per-leaf inner products the move records on its sub-state at
the end of each leaf visit (``sub.d_h`` / ``sub.h_h``, ``(nwalkers,
nleaves)``; stored per row under ``sub_backend/<branch>``). Only when
``<h|h>`` is unusable at that walker does it fall back to the detection SNR
at the likelihood maximum, where ``<d|h> = <h|h>`` makes ``<d|h> /
sqrt(<h|h>) = sqrt(<d|h>)``.

Everything here is a pure function of plain values (no state, no RNG); the
move (``ResidualAddOneRemoveOneMove._maybe_update_snr_ladders``), the
builder (``recipe.resolve_snr_keyed_ladders``) and the store migration
(``scripts/fstat_proposal/rerung_store_branches.py --top-snr``) share it.
"""

from __future__ import annotations

import math
import typing

import numpy as np

#: the top SNR when ``{BRANCH}_LADDER_TOP_SNR`` is switched on by a word
#: (on / true / yes) rather than a number (ruling 2026-10-18: SNR ~2)
LADDER_TOP_SNR_DEFAULT = 2.0
#: ``{BRANCH}_LADDER_TMAX_MIN``: the floor of a leaf's top temperature
LADDER_TMAX_MIN_DEFAULT = 4.0
#: relative SNR change since a leaf's ladder was built that triggers a
#: refresh (``{BRANCH}_LADDER_TOP_SNR_REFRESH``)
LADDER_SNR_REFRESH_CHANGE = 0.2
#: lnL at or below this is the out-of-domain sentinel, never a maximum
_SENTINEL = -1e299

_WORDS_ON = ("on", "true", "yes")
_WORDS_OFF = ("off", "false", "no", "")


def top_snr_value(raw) -> float:
    """Cast of ``{BRANCH}_LADDER_TOP_SNR`` (a module-level function: the
    settings tree must pickle).

    A number is the top SNR (``<= 0`` = off); the words on / true / yes are
    :data:`LADDER_TOP_SNR_DEFAULT`; off / false / no / an empty string (and
    ``None``) are off. ``"1"`` is the number 1, not "on". Anything else
    raises (a mistyped knob must not silently do nothing).
    """
    if raw is None:
        return 0.0
    if isinstance(raw, (bool, np.bool_)):
        return LADDER_TOP_SNR_DEFAULT if bool(raw) else 0.0
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in _WORDS_ON:
            return LADDER_TOP_SNR_DEFAULT
        if low in _WORDS_OFF:
            return 0.0
        raw = low
    val = float(raw)
    if not math.isfinite(val):
        raise ValueError(f"LADDER_TOP_SNR must be a finite number or on/off, got {raw!r}")
    return val if val > 0.0 else 0.0


def ceiling_tmax(ndim, ntemps, ladder_tmax=None) -> float:
    """The clip's ceiling: ``ladder_tmax`` when set, else the top temperature
    of the uncapped ``make_ladder(ndim, ntemps)`` (eryn's ~25 % spacing)."""
    if ladder_tmax is not None:
        val = float(ladder_tmax)
        if math.isfinite(val):
            return val
    from eryn.moves.tempering import make_ladder

    betas = np.asarray(make_ladder(int(ndim), ntemps=int(ntemps)), dtype=float)
    return float(1.0 / betas[-1])


def check_clip(tmax_min, tmax_ceiling) -> typing.Tuple[float, float]:
    """``(floor, ceiling)`` as floats; refuses a floor <= 1 or above the ceiling."""
    lo, hi = float(tmax_min), float(tmax_ceiling)
    if not (math.isfinite(lo) and lo > 1.0):
        raise ValueError(f"LADDER_TMAX_MIN must be a temperature > 1, got {tmax_min!r}")
    if not (math.isfinite(hi) and hi >= lo):
        raise ValueError(
            f"the SNR-keyed ladder's ceiling (LADDER_TMAX, or the uncapped make_ladder "
            f"top) {tmax_ceiling!r} is below its floor LADDER_TMAX_MIN {tmax_min!r}; "
            "set {BRANCH}_LADDER_TMAX")
    return lo, hi


def snr_tmax(snr, top_snr, tmax_min, tmax_ceiling):
    """``(T_leaf, T_raw, clip)`` for one leaf.

    ``T_raw = (snr / top_snr)^2`` and ``T_leaf = clip(T_raw, tmax_min,
    tmax_ceiling)``; ``clip`` is ``"floor"``, ``"ceiling"`` or ``""``. A
    missing / non-finite / non-positive ``snr`` (no source) has ``T_raw = 0``
    and lands on the floor.
    """
    top = float(top_snr)
    if not top > 0.0:
        raise ValueError(f"top_snr must be > 0 when the SNR-keyed ladder is on, got {top_snr!r}")
    lo, hi = check_clip(tmax_min, tmax_ceiling)
    s = float("nan") if snr is None else float(snr)
    raw = (s / top) ** 2 if (math.isfinite(s) and s > 0.0) else 0.0
    if raw < lo:
        return lo, raw, "floor"
    if raw > hi:
        return hi, raw, "ceiling"
    return raw, raw, ""


def snr_keyed_ladder(ndim, ntemps, tmax) -> np.ndarray:
    """``make_ladder(ndim, ntemps, Tmax=tmax)``: geometric from 1 to ``1 / tmax``."""
    from eryn.moves.tempering import make_ladder

    return np.asarray(make_ladder(int(ndim), ntemps=int(ntemps), Tmax=float(tmax)), dtype=float)


def leaf_snr(d_h, h_h, cold_ll=None):
    """``(snr, walker, kind)`` of one leaf from its per-walker inner products.

    ``walker`` is the max-lnL cold walker among those with a usable
    measurement (``cold_ll``: the leaf's cold-rung lnL per walker, the
    sub-state's ``log_like[leaf, 0, :]``; without it the largest ``<h|h>``).
    ``kind`` is ``"optimal"`` (``sqrt(<h|h>)``) or ``"detection"`` (the
    fallback ``sqrt(<d|h>)``, the detection SNR at the maximum, used only
    when that walker's ``<h|h>`` is unusable). No usable walker:
    ``(nan, None, None)``.
    """
    hh = np.asarray(h_h, dtype=float).ravel()
    dh = np.asarray(d_h, dtype=float).ravel() if d_h is not None else np.full(hh.shape, np.nan)
    if dh.shape != hh.shape:
        dh = np.full(hh.shape, np.nan)
    ok_hh = np.isfinite(hh) & (hh > 0.0)
    ok_dh = np.isfinite(dh) & (dh > 0.0)
    usable = ok_hh | ok_dh
    if not np.any(usable):
        return float("nan"), None, None
    score = None
    if cold_ll is not None:
        ll = np.asarray(cold_ll, dtype=float).ravel()
        if ll.shape == hh.shape:
            good = usable & np.isfinite(ll) & (ll > _SENTINEL)
            if np.any(good):
                score = np.where(good, ll, -np.inf)
    if score is None:
        score = np.where(usable, np.where(ok_hh, hh, dh), -np.inf)
    w = int(np.argmax(score))
    if ok_hh[w]:
        return float(np.sqrt(hh[w])), w, "optimal"
    return float(np.sqrt(dh[w])), w, "detection"


def leaf_alive(inds, leaf) -> bool:
    """Whether ``leaf`` carries a source on any cold walker (``inds`` ``(nt, nw, nl)``)."""
    if inds is None:
        return True
    arr = np.asarray(inds)
    if arr.ndim != 3 or int(leaf) >= arr.shape[-1]:
        return True
    return bool(np.any(arr[0, :, int(leaf)]))


def leaf_snrs(d_h, h_h, log_like=None, nleaves=None):
    """``[(snr, walker, kind), ...]`` per leaf from sub-state-shaped arrays.

    ``d_h`` / ``h_h`` ``(nwalkers, nleaves)``; ``log_like`` the per-leaf
    ladder ``(nleaves, ntemps, nwalkers)`` (its cold rung picks the walker)
    or ``None``. Missing arrays give ``nan`` for every leaf.
    """
    if h_h is None:
        n = int(nleaves or 0)
        return [(float("nan"), None, None)] * n
    hh = np.asarray(h_h, dtype=float)
    if hh.ndim != 2:
        n = int(nleaves or 0)
        return [(float("nan"), None, None)] * n
    nl = int(nleaves) if nleaves is not None else int(hh.shape[1])
    dh = None if d_h is None else np.asarray(d_h, dtype=float)
    ll = None if log_like is None else np.asarray(log_like, dtype=float)
    out = []
    for leaf in range(nl):
        if leaf >= hh.shape[1]:
            out.append((float("nan"), None, None))
            continue
        cold = None
        if ll is not None and ll.ndim == 3 and leaf < ll.shape[0] and ll.shape[2] == hh.shape[0]:
            cold = ll[leaf, 0, :]
        d_col = dh[:, leaf] if (dh is not None and dh.shape == hh.shape) else None
        out.append(leaf_snr(d_col, hh[:, leaf], cold))
    return out


def substate_leaf_snrs(sub, nleaves=None):
    """:func:`leaf_snrs` read off a per-leaf ladder sub-state (``d_h`` / ``h_h`` /
    ``log_like``); every leaf ``nan`` when the sub-state recorded nothing."""
    if nleaves is None:
        nleaves = getattr(sub, "nleaves_max", None)
    return leaf_snrs(getattr(sub, "d_h", None), getattr(sub, "h_h", None),
                     getattr(sub, "log_like", None), nleaves)


def snr_changed(snr, ref, frac=LADDER_SNR_REFRESH_CHANGE) -> bool:
    """``|snr - ref| > frac * ref`` (both finite, ``ref > 0``)."""
    if snr is None or ref is None:
        return False
    s, r = float(snr), float(ref)
    if not (math.isfinite(s) and math.isfinite(r) and r > 0.0):
        return False
    return abs(s - r) > float(frac) * r


def describe_clip(clip, tmax_min, tmax_ceiling, tmax) -> str:
    """``"ceiling 625 -> 625.0"`` / ``"floor 4 -> 4.0"`` / ``"within [4, 625]"``."""
    if clip == "ceiling":
        return f"ceiling {float(tmax_ceiling):g} -> {float(tmax):.1f}"
    if clip == "floor":
        return f"floor {float(tmax_min):g} -> {float(tmax):.1f}"
    return f"within [{float(tmax_min):g}, {float(tmax_ceiling):g}]"


def rung_ratio(betas) -> float:
    """Adjacent-temperature ratio of a geometric ladder: ``T_max^(1 / (ntemps - 1))``.

    The rung COUNT is frozen per branch by the store (``betas_all`` has a
    fixed ``ntemps`` axis), so a leaf whose ``T_max`` sits below the ceiling
    gets a DENSER ladder than the ceiling's: a smaller ratio, a higher swap
    acceptance than make_ladder's ~25 % (harmless; it costs nothing extra).
    """
    b = np.asarray(betas, dtype=float).ravel()
    if b.size < 2 or not b[-1] > 0.0:
        return 1.0
    return float((1.0 / b[-1]) ** (1.0 / (b.size - 1)))


def make_ladder_count(ndim, tmax) -> int:
    """The rung count ``make_ladder(ndim, Tmax=tmax)`` picks for its ~25 % swap
    spacing (eryn: ``int(log(Tmax) / log(tstep) + 2)``)."""
    from eryn.moves.tempering import make_ladder

    return int(np.size(make_ladder(int(ndim), Tmax=float(tmax))))


def describe_leaf(tag, leaf, snr, tmax, raw, clip, top_snr, tmax_min, tmax_ceiling, betas,
                  why, kind="optimal", walker=None) -> str:
    """The per-leaf log line of a build / reset / refresh::

        [EMRI_LADDER] leaf 3: SNR 50.2 -> T_max 630.0 (top SNR 2.0, ceiling 625 -> 625.0),
        ladder [1 ... 0.0016], ratio 1.90 over 11 rungs (fresh store; optimal SNR at walker 1)

    The ratio is the leaf's adjacent-temperature ratio (:func:`rung_ratio`).
    """
    b = np.asarray(betas, dtype=float).ravel()
    s = float("nan") if snr is None else float(snr)
    if math.isfinite(s):
        head = f"SNR {s:.1f} -> T_max {float(raw):.1f}"
        where = f"{kind} SNR" + ("" if walker is None else f" at walker {int(walker)}")
    else:
        head = f"no source -> T_max {float(tmax_min):g}"
        where = "no SNR"
    return (f"{tag} leaf {int(leaf)}: {head} (top SNR {float(top_snr):.1f}, "
            f"{describe_clip(clip, tmax_min, tmax_ceiling, tmax)}), "
            f"ladder [1 ... {float(b[-1]):.3g}], ratio {rung_ratio(b):.2f} over {b.size} rungs "
            f"({why}; {where})")
