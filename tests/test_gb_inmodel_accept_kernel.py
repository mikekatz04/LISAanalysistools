"""Fused GB in-model gate/accept kernels (``GB_INMODEL_ACCEPT_KERNEL``).

The in-model repeat step is launch-bound: the pre-score gate chain and the
post-score accept/bookkeeping chain together pay ~110-150 separate
array-library launches per repeat around ONE real scoring call. The two
backend entry points in ``cutils/gf_routing_kernels.cu`` collapse that into 3
launches. This file is the correctness harness for that swap.

Three layers:

* **REFERENCE** -- ``_ref_gate_compact`` / ``_ref_accept_apply`` are pure-numpy
  transliterations of the chain as it stands in
  ``gbspecialstretch._run_in_model_repeats``, written in the same vectorized
  style as the original so a reviewer can diff them against it line by line.
  They are the definition of "correct" for the kernels.
* **KERNEL vs REFERENCE** -- synthetic-input equivalence, skipped until the
  backend module carrying the kernels is built. Exact equality is demanded
  (``assert_array_equal``), not ``allclose``: every operation in the chain is
  either integer or a same-order floating-point expression, so a difference
  means a real divergence, not rounding.
* **KNOB-OFF** -- active right now: the default is OFF, the call site with the
  knob off still reproduces the straight-line accept chain bit-for-bit, and
  the fused path stands down cleanly when it cannot run.
"""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _inmodel_accept_kernel_on,
    _inmodel_trace_knobs_active,
)

# The harness that drives the REAL ``_run_in_model_repeats`` on numpy, plus
# the pre-2026-08-15 straight-line reference it is checked against. Reused
# here so the knob-ON path is measured against the same yardstick the
# de-synced python path already passes.
from tests.test_inmodel_repeats import (
    _build_problem,
    _FakeBuffer,
    _make_move,
    _reference_run,
)

LL_FLOOR = -1e300
BAD_LL = -1e299
BAD_LOGP = -1e229


def _backend_kernels():
    """``(backend, ok)`` -- whether the built backend carries the kernels."""
    try:
        from lisatools import get_backend

        backend = get_backend("cpu")
    except Exception:  # pragma: no cover - backend not built in this env
        return None, False
    ok = (
        getattr(backend, "gb_inmodel_gate_compact", None) is not None
        and getattr(backend, "gb_inmodel_accept_apply", None) is not None
    )
    return backend, ok


_BACKEND, _HAVE_KERNELS = _backend_kernels()
_SKIP_MSG = (
    "backend module has no gb_inmodel_* routing kernels yet "
    "(rebuild lisatools to activate these)"
)


# ==========================================================================
# REFERENCE IMPLEMENTATIONS (pure numpy, transliterated from the live chain)
# ==========================================================================
def _ref_cap_cell(band, f_hz, cap_band_lo, cap_band_step, cap_divisor,
                  cap_stagger, num_cap_cells):
    """Numpy twin of ``GBSpecialBase._cap_cell_index``."""
    if cap_divisor == 1:
        return band.astype(np.int32)
    sub = np.floor(
        (f_hz - cap_band_lo[band]) / cap_band_step[band]
        + (0.5 if cap_stagger else 0.0)
    )
    if cap_stagger:
        cell = band.astype(np.int64) * cap_divisor + sub.astype(np.int64)
        return np.clip(cell, 0, num_cap_cells - 1).astype(np.int32)
    sub = np.clip(sub, 0, cap_divisor - 1).astype(np.int64)
    return (band.astype(np.int64) * cap_divisor + sub).astype(np.int32)


def _ref_cap_members(band, f_hz, cap_band_lo, cap_band_step, cap_edges,
                     cap_edge_ext, cap_divisor, cap_stagger, num_cap_cells,
                     overlap_on):
    """Numpy twin of ``GBSpecialBase._cap_cell_members``."""
    p = _ref_cap_cell(band, f_hz, cap_band_lo, cap_band_step, cap_divisor,
                      cap_stagger, num_cap_cells)
    if not overlap_on:
        return p, p.copy(), np.zeros(p.shape, dtype=np.int32)
    low = f_hz < (cap_edges[p] + cap_edge_ext[p])
    high = f_hz > (cap_edges[p + 1] - cap_edge_ext[p + 1])
    nb = np.where(low, p - 1, np.where(high, p + 1, p)).astype(np.int32)
    hn = low | high
    # End-edge extensions are 0 by construction, so this clamp is inert; it
    # mirrors the kernel's identical defensive guard.
    inside = (nb >= 0) & (nb < num_cap_cells)
    nb = np.where(inside, nb, p).astype(np.int32)
    return p, nb, (hn & inside).astype(np.int32)


def _ref_cap_flat(t, w, cell, nwalkers, num_cap_cells):
    return (t.astype(np.int64) * nwalkers + w) * num_cap_cells + cell


def _ref_gate_compact(cfg):
    """Pure-numpy twin of ``gb_inmodel_gate_compact``.

    ``cfg`` is the same bag of arrays the kernel takes. Returns a dict with
    the mutated ``new_logp`` plus every output buffer.
    """
    new_logp = cfg["new_logp"].copy()
    new = cfg["new"]
    n_sub = new.shape[0]
    row_map = cfg["row_map"]
    row_map = np.arange(n_sub) if row_map is None else row_map.astype(np.int64)
    curr_sl = cfg["curr"][row_map]
    trust_counts = np.zeros(3, dtype=np.int64)
    dg_count = np.zeros(1, dtype=np.int64)

    # ---- f0 window (gbspecialstretch: the ``self._f0_col is not None`` block)
    if cfg["window_on"]:
        fc, df = cfg["f0_col"], cfg["df"]
        n4 = cfg["n4"].astype(np.int64)
        new_bin = np.abs(new[:, fc] / 1e3 / df).astype(int)
        new_logp[
            (np.abs(new[:, fc] / 1e3 - curr_sl[:, fc] / 1e3) / df).astype(int)
            > n4
        ] = -np.inf
        new_logp[new_bin < cfg["lo_bin"].astype(np.int64) - n4] = -np.inf
        new_logp[new_bin > cfg["hi_bin"].astype(np.int64) + n4] = -np.inf

    # ---- sig-het trust region
    pc = cfg.get("pc")
    if pc is not None:
        amp, f0, fdot = cfg["anchor"]
        damp = np.abs(np.log(np.abs(pc[:, 0]) / amp[row_map]))
        drift = (
            2.0 * np.pi * np.abs(pc[:, 1] - f0[row_map]) * cfg["trust_Tobs"]
            + np.pi * np.abs(pc[:, 2] - fdot[row_map]) * cfg["trust_Tobs"] ** 2
        )
        rej_a = damp > cfg["trust_dlna"][row_map]
        rej_p = drift > cfg["trust_dphase"][row_map]
        new_logp[rej_a | rej_p] = -np.inf
        trust_counts[0] += rej_a.sum()
        trust_counts[1] += rej_p.sum()
        trust_counts[2] += (rej_a | rej_p).sum()

    # ---- cap drift gate veto (+ the covering-cell stash)
    cur_cells = np.zeros(3 * n_sub, dtype=np.int32)
    new_cells = np.zeros(3 * n_sub, dtype=np.int32)
    if cfg["dg_on"]:
        fc = cfg["f0_col"]
        t, w, b = cfg["t"], cfg["w"], cfg["b"]
        ov = cfg["overlap_on"]
        args = (cfg["cap_band_lo"], cfg["cap_band_step"], cfg["cap_edges"],
                cfg["cap_edge_ext"], cfg["cap_divisor"], cfg["cap_stagger"],
                cfg["num_cap_cells"], ov)
        c_p, c_nb, c_hn = _ref_cap_members(b, curr_sl[:, fc] / 1e3, *args)
        n_p, n_nb, n_hn = _ref_cap_members(b, new[:, fc] / 1e3, *args)
        cur_cells[:n_sub], cur_cells[n_sub:2 * n_sub] = c_p, c_nb
        cur_cells[2 * n_sub:] = c_hn
        new_cells[:n_sub], new_cells[n_sub:2 * n_sub] = n_p, n_nb
        new_cells[2 * n_sub:] = n_hn

        counts, cap = cfg["dg_counts"], cfg["dg_cap"]
        nw, ncc = cfg["cap_nwalkers"], cfg["num_cap_cells"]
        if ov:
            veto = np.zeros(n_sub, dtype=bool)
            ones = np.ones(n_sub, dtype=bool)
            for cell, memb in ((n_p, ones), (n_nb, n_hn.astype(bool))):
                foreign = (
                    memb & (cell != c_p)
                    & (~c_hn.astype(bool) | (cell != c_nb))
                )
                flat = _ref_cap_flat(t, w, cell, nw, ncc)
                veto = veto | (
                    foreign & (cap[cell] >= 0) & (counts[flat] >= cap[cell])
                )
        else:
            cross = n_p != c_p
            flat = _ref_cap_flat(t, w, n_p, nw, ncc)
            veto = cross & (cap[n_p] >= 0) & (counts[flat] >= cap[n_p])
        new_logp[veto] = -np.inf
        dg_count[0] += veto.sum()

    keep = ~np.isinf(new_logp)
    keep_idx = np.where(keep)[0]
    keep_pos = np.full(n_sub, -1, dtype=np.int32)
    keep_pos[keep_idx] = np.arange(keep_idx.size, dtype=np.int32)
    return {
        "new_logp": new_logp,
        "keep_flag": keep.astype(np.uint8),
        "keep_idx": keep_idx.astype(np.int64),
        "keep_pos": keep_pos,
        "n_keep": keep_idx.size,
        "cur_cells": cur_cells,
        "new_cells": new_cells,
        "trust_counts": trust_counts,
        "dg_count": dg_count,
    }


def _ref_accept_apply(cfg):
    """Pure-numpy twin of ``gb_inmodel_accept_apply``.

    Mutates copies of the tracked state and returns them, so a caller can
    compare against the kernel's in-place writes.
    """
    n_sub = cfg["new"].shape[0]
    row_map = cfg["row_map"]
    row_map = np.arange(n_sub) if row_map is None else row_map.astype(np.int64)
    curr = cfg["curr"].copy()
    ll_ref = cfg["ll_ref"].copy()
    curr_prior = cfg["curr_prior"].copy()
    ll_change_log = cfg["ll_change_log"].copy()
    prop1 = cfg["prop1"].copy()
    acc1 = cfg["acc1"].copy()
    dg_counts = cfg["dg_counts"].copy() if cfg["dg_on"] else None
    sorter_dh = None if cfg["sorter_dh"] is None else cfg["sorter_dh"].copy()
    sorter_hh = None if cfg["sorter_hh"] is None else cfg["sorter_hh"].copy()
    warn = np.zeros(1, dtype=np.int64)
    kind = np.zeros(2, dtype=np.int64)

    keep_idx = cfg["keep_idx"]
    n_keep = keep_idx.size
    new_ll = np.full(n_sub, LL_FLOOR)
    hh = cfg["h_h"]
    dh = cfg["d_h"]
    if n_keep:
        new_ll[keep_idx] = cfg["scored"]
        if dh is not None:
            opt = np.sqrt(np.maximum(hh, 0.0))
            viol = opt < cfg["snr_limit"]
            if cfg["snr_detected"]:
                viol = viol | ((dh / np.maximum(opt, 1e-300)) < cfg["snr_limit"])
            new_ll[keep_idx] = np.where(viol, LL_FLOOR, new_ll[keep_idx])
    delta = new_ll - ll_ref[row_map]

    lnpdiff = (
        cfg["beta"] * delta
        + (cfg["new_logp"] - curr_prior[row_map])
        + cfg["factors"]
    )
    accept_pre = lnpdiff >= np.log(cfg["u"])
    bad = (new_ll <= BAD_LL) | (cfg["new_logp"] <= BAD_LOGP)
    warn[0] += ((accept_pre & bad) & (cfg["beta"] != 0.0)).sum()
    accept = accept_pre & ~bad

    t, w, b = cfg["t"], cfg["w"], cfg["b"]
    prop1[t, w, b] += 1
    kind[0] += accept.sum()
    if cfg["cold"] is not None:
        kind[1] += (accept & cfg["cold"].astype(bool)).sum()

    rows = row_map[accept]
    curr[rows] = cfg["new"][accept]
    ll_ref[rows] = new_ll[accept]
    curr_prior[rows] = cfg["new_logp"][accept]
    ll_change_log[t[accept], w[accept], b[accept]] += delta[accept]
    acc1[t[accept], w[accept], b[accept]] += 1

    if sorter_dh is not None and dh is not None and n_keep:
        dh_full = np.empty(n_sub)
        hh_full = np.empty(n_sub)
        dh_full[keep_idx] = dh
        hh_full[keep_idx] = hh
        ids = cfg["ids"]
        sorter_dh[ids] = np.where(accept, dh_full, sorter_dh[ids])
        sorter_hh[ids] = np.where(accept, hh_full, sorter_hh[ids])

    if cfg["dg_on"]:
        n_sub_ = n_sub
        cc, nc = cfg["cur_cells"], cfg["new_cells"]
        c_p, c_nb = cc[:n_sub_], cc[n_sub_:2 * n_sub_]
        c_hn = cc[2 * n_sub_:].astype(bool)
        n_p, n_nb = nc[:n_sub_], nc[n_sub_:2 * n_sub_]
        n_hn = nc[2 * n_sub_:].astype(bool)
        nw, ncc = cfg["cap_nwalkers"], cfg["num_cap_cells"]
        if cfg["overlap_on"]:
            ones = np.ones(n_sub_, dtype=bool)

            def _in_cur(c):
                return (c == c_p) | (c_hn & (c == c_nb))

            def _in_new(c):
                return (c == n_p) | (n_hn & (c == n_nb))

            for cell, memb, sign, covered in (
                (n_p, ones, 1, _in_cur), (n_nb, n_hn, 1, _in_cur),
                (c_p, ones, -1, _in_new), (c_nb, c_hn, -1, _in_new),
            ):
                wgt = accept & memb & ~covered(cell)
                np.add.at(dg_counts, _ref_cap_flat(t, w, cell, nw, ncc),
                          (sign * wgt.astype(np.int32)).astype(np.int32))
        else:
            wgt = (accept & (n_p != c_p)).astype(np.int32)
            np.add.at(dg_counts, _ref_cap_flat(t, w, n_p, nw, ncc), wgt)
            np.add.at(dg_counts, _ref_cap_flat(t, w, c_p, nw, ncc), -wgt)

    return {
        "new_ll": new_ll, "delta": delta, "lnpdiff": lnpdiff,
        "accept_pre": accept_pre.astype(np.uint8),
        "accept": accept.astype(np.uint8),
        "curr": curr, "ll_ref": ll_ref, "curr_prior": curr_prior,
        "ll_change_log": ll_change_log, "prop1": prop1, "acc1": acc1,
        "dg_counts": dg_counts, "sorter_dh": sorter_dh,
        "sorter_hh": sorter_hh, "warn": warn, "kind": kind,
    }


# ==========================================================================
# Synthetic problem builder shared by the kernel-vs-reference tests
# ==========================================================================
def _synth(seed=7, n_sub=17, n_block=None, ndim=5, ntemps=3, nwalkers=4,
           num_bands=6, window=True, trust=False, dg=False, overlap=False,
           parity=False, detected=False, cap_divisor=1, stagger=False):
    """One fully-populated argument bag for both entry points."""
    rng = np.random.RandomState(seed)
    n_block = n_sub if n_block is None else n_block
    row_map = None
    if parity:
        # A parity half: n_sub distinct rows of a wider block, ascending.
        n_block = max(n_block, 2 * n_sub)
        row_map = np.sort(
            rng.choice(n_block, size=n_sub, replace=False)
        ).astype(np.int32)

    curr = rng.uniform(-1.0, 1.0, (n_block, ndim))
    curr[:, 1] = rng.uniform(2.95, 3.05, n_block)  # f0, in "mHz" units
    new = curr[np.arange(n_sub) if row_map is None else row_map].copy()
    new += rng.normal(0.0, 2e-4, new.shape)

    new_logp = np.where(rng.rand(n_sub) < 0.15, -np.inf, 0.0)
    df = 1e-5
    n4 = np.full(n_sub, 16, dtype=np.int32)
    f_bins = np.abs(new[:, 1] / 1e3 / df).astype(np.int64)
    lo_bin = (f_bins - rng.randint(0, 40, n_sub)).astype(np.int32)
    hi_bin = (f_bins + rng.randint(0, 40, n_sub)).astype(np.int32)

    t = rng.randint(0, ntemps, n_sub).astype(np.int32)
    w = rng.randint(0, nwalkers, n_sub).astype(np.int32)
    # Serial-within-band: one picked row per (temp, walker, band) cell. Give
    # every row its own band so the ledger scatter-adds cannot collide.
    b = np.arange(n_sub, dtype=np.int32) % num_bands
    uniq = {}
    for i in range(n_sub):
        while (int(t[i]), int(w[i]), int(b[i])) in uniq:
            w[i] = (w[i] + 1) % nwalkers
            if (int(t[i]), int(w[i]), int(b[i])) not in uniq:
                break
            t[i] = (t[i] + 1) % ntemps
        uniq[(int(t[i]), int(w[i]), int(b[i]))] = i

    cfg = {
        "new": np.ascontiguousarray(new),
        "curr": np.ascontiguousarray(curr),
        "row_map": row_map,
        "new_logp": np.ascontiguousarray(new_logp),
        "n4": n4, "lo_bin": lo_bin, "hi_bin": hi_bin,
        "f0_col": 1, "ndim": ndim, "df": df, "window_on": window,
        "t": t, "w": w, "b": b,
        "cold": (t == 0).astype(np.uint8),
        "ids": np.arange(n_sub, dtype=np.int32),
        "n_sub": n_sub, "n_block": n_block,
        "ntemps": ntemps, "nwalkers": nwalkers, "num_bands": num_bands,
        "cap_nwalkers": nwalkers,
        "trust_Tobs": 0.0,
        "dg_on": False, "overlap_on": False,
        "cap_divisor": 1, "cap_stagger": 0, "num_cap_cells": num_bands,
        "cap_band_lo": np.zeros(0), "cap_band_step": np.zeros(0),
        "cap_edges": np.zeros(0), "cap_edge_ext": np.zeros(0),
        "dg_counts": np.zeros(0, dtype=np.int32),
        "dg_cap": np.zeros(0, dtype=np.int32),
        "pc": None, "anchor": None,
        "trust_dlna": None, "trust_dphase": None,
    }

    if trust:
        # Scales chosen so BOTH arms of the gate straddle their threshold --
        # an all-pass or all-reject fixture would exercise nothing.
        # |dlnA| ~ 5% vs a 0.05 gate; 2*pi*df0*Tobs ~ O(1) rad and
        # pi*dfdot*Tobs^2 ~ O(1) rad vs a 2.0 rad gate.
        Tobs = 7.86e6
        pc = np.empty((n_sub, 3))
        pc[:, 0] = rng.uniform(1e-22, 5e-22, n_sub)
        pc[:, 1] = new[:, 1] / 1e3
        pc[:, 2] = rng.normal(0.0, 1e-17, n_sub)
        anchor = [rng.uniform(1e-22, 5e-22, n_block) for _ in range(3)]
        rows = np.arange(n_sub) if row_map is None else row_map
        anchor[0][rows] = np.abs(pc[:, 0]) * np.exp(
            rng.normal(0.0, 0.05, n_sub))
        anchor[1][rows] = pc[:, 1] + rng.normal(
            0.0, 1.0 / (2.0 * np.pi * Tobs), n_sub)
        anchor[2][rows] = pc[:, 2] + rng.normal(
            0.0, 1.0 / (np.pi * Tobs ** 2), n_sub)
        cfg["pc"] = np.ascontiguousarray(pc)
        cfg["anchor"] = [np.ascontiguousarray(a) for a in anchor]
        cfg["trust_dlna"] = np.full(n_block, 0.05)
        cfg["trust_dphase"] = np.full(n_block, 2.0)
        cfg["trust_Tobs"] = Tobs

    if dg:
        num_cap_cells = num_bands * cap_divisor
        band_lo = 2.9e-3 + 2e-5 * np.arange(num_bands)
        band_step = np.full(num_bands, 2e-5 / cap_divisor)
        edges = np.concatenate([
            band_lo[0] + (2e-5 / cap_divisor) * np.arange(num_cap_cells),
            [band_lo[0] + 2e-5 * num_bands],
        ])
        ext = np.zeros(num_cap_cells + 1)
        if overlap:
            ext[1:-1] = 0.25 * (2e-5 / cap_divisor)
        counts = rng.randint(0, 3, ntemps * nwalkers * num_cap_cells)
        cap = rng.randint(-1, 3, num_cap_cells)
        cfg.update({
            "dg_on": True, "overlap_on": overlap,
            "cap_divisor": cap_divisor, "cap_stagger": int(stagger),
            "num_cap_cells": num_cap_cells,
            "cap_band_lo": np.ascontiguousarray(band_lo),
            "cap_band_step": np.ascontiguousarray(band_step),
            "cap_edges": np.ascontiguousarray(edges),
            "cap_edge_ext": np.ascontiguousarray(ext),
            "dg_counts": counts.astype(np.int32),
            "dg_cap": cap.astype(np.int32),
        })
        # Keep every f0 inside the modelled cap grid so the cell arithmetic
        # is exercised rather than saturating on the clip. new and curr get
        # INDEPENDENT draws so most rows cross a cell boundary -- which is
        # what the gate polices.
        span = 2e-5 * num_bands
        for arr in (cfg["new"], cfg["curr"]):
            arr[:, 1] = 1e3 * (band_lo[0] + span * rng.rand(arr.shape[0]))
        # Those independent draws would otherwise be annihilated by the +-N/4
        # step window before the cap gate ever sees them. Re-derive the window
        # around the NEW f0 and open it wide: the window stage still runs (and
        # is still compared against the reference), it just stops dominating.
        f_bins = np.abs(cfg["new"][:, 1] / 1e3 / df).astype(np.int64)
        cfg["n4"] = np.full(n_sub, 10 ** 7, dtype=np.int32)
        cfg["lo_bin"] = (f_bins - 5000).astype(np.int32)
        cfg["hi_bin"] = (f_bins + 5000).astype(np.int32)

    # post-score inputs
    cfg["ll_ref"] = np.ascontiguousarray(rng.normal(0.0, 10.0, n_block))
    cfg["curr_prior"] = np.zeros(n_block)
    cfg["beta"] = np.ascontiguousarray(
        rng.choice([0.0, 0.3, 1.0], n_sub).astype(np.float64))
    cfg["factors"] = np.ascontiguousarray(rng.normal(0.0, 0.2, n_sub))
    cfg["u"] = np.ascontiguousarray(rng.uniform(1e-9, 1.0, n_sub))
    cfg["ll_change_log"] = np.zeros((ntemps, nwalkers, num_bands))
    cfg["prop1"] = np.zeros((ntemps, nwalkers, num_bands), dtype=np.int64)
    cfg["acc1"] = np.zeros((ntemps, nwalkers, num_bands), dtype=np.int64)
    cfg["sorter_dh"] = np.full(n_sub + 5, np.nan)
    cfg["sorter_hh"] = np.full(n_sub + 5, np.nan)
    cfg["snr_limit"] = 4.0
    cfg["snr_detected"] = detected
    return cfg


def _call_gate(backend, cfg):
    """Invoke the kernel with the argument order the binding declares."""
    n_sub, n_block = cfg["n_sub"], cfg["n_block"]
    E64, E32, EI64 = np.zeros(0), np.zeros(0, np.int32), np.zeros(0, np.int64)
    out = {
        "new_logp": cfg["new_logp"].copy(),
        "keep_flag": np.zeros(n_sub, np.uint8),
        "keep_idx": np.zeros(n_sub, np.int64),
        "keep_pos": np.full(n_sub, -1, np.int32),
        "n_keep": np.zeros(1, np.int64),
        "cur_cells": np.zeros(3 * n_sub, np.int32),
        "new_cells": np.zeros(3 * n_sub, np.int32),
        "trust_counts": np.zeros(3, np.int64),
        "dg_count": np.zeros(1, np.int64),
    }
    trust = cfg["pc"] is not None
    backend.gb_inmodel_gate_compact(
        out["new_logp"], out["keep_flag"], out["keep_idx"], out["n_keep"],
        out["cur_cells"], out["new_cells"], out["keep_pos"],
        out["trust_counts"] if trust else EI64,
        out["dg_count"] if cfg["dg_on"] else EI64,
        cfg["new"], cfg["curr"],
        E32 if cfg["row_map"] is None else cfg["row_map"],
        cfg["n4"] if cfg["window_on"] else E32,
        cfg["lo_bin"] if cfg["window_on"] else E32,
        cfg["hi_bin"] if cfg["window_on"] else E32,
        cfg["f0_col"], cfg["ndim"], cfg["df"], int(cfg["window_on"]),
        cfg["pc"] if trust else E64, 3 if trust else 0,
        cfg["anchor"][0] if trust else E64,
        cfg["anchor"][1] if trust else E64,
        cfg["anchor"][2] if trust else E64,
        cfg["trust_dlna"] if trust else E64,
        cfg["trust_dphase"] if trust else E64,
        cfg["trust_Tobs"],
        int(cfg["dg_on"]), int(cfg["overlap_on"]),
        cfg["t"], cfg["w"], cfg["b"],
        cfg["cap_band_lo"], cfg["cap_band_step"],
        cfg["cap_edges"], cfg["cap_edge_ext"],
        cfg["dg_counts"], cfg["dg_cap"],
        cfg["cap_divisor"], cfg["cap_stagger"], cfg["num_cap_cells"],
        cfg["cap_nwalkers"], n_sub, n_block,
    )
    return out


def _call_accept(backend, cfg, gate, scored, dh, hh):
    n_sub, n_block = cfg["n_sub"], cfg["n_block"]
    E64, E32, EI64 = np.zeros(0), np.zeros(0, np.int32), np.zeros(0, np.int64)
    n_keep = int(gate["n_keep"][0]) if "n_keep" in gate else gate["n_keep"]
    n_keep = int(n_keep)
    keep_idx = gate["keep_idx"][:n_keep]
    out = {
        "new_ll": np.full(n_sub, LL_FLOOR),
        "delta": np.zeros(n_sub),
        "lnpdiff": np.zeros(n_sub),
        "accept_pre": np.zeros(n_sub, np.uint8),
        "accept": np.zeros(n_sub, np.uint8),
        "curr": cfg["curr"].copy(),
        "ll_ref": cfg["ll_ref"].copy(),
        "curr_prior": cfg["curr_prior"].copy(),
        "ll_change_log": cfg["ll_change_log"].copy(),
        "prop1": cfg["prop1"].copy(),
        "acc1": cfg["acc1"].copy(),
        "dg_counts": cfg["dg_counts"].copy(),
        "sorter_dh": cfg["sorter_dh"].copy(),
        "sorter_hh": cfg["sorter_hh"].copy(),
        "warn": np.zeros(1, np.int64),
        "kind": np.zeros(2, np.int64),
    }
    backend.gb_inmodel_accept_apply(
        out["new_ll"], out["delta"], out["lnpdiff"], out["accept_pre"],
        out["accept"], out["curr"], out["ll_ref"], out["curr_prior"],
        scored if n_keep else E64, keep_idx if n_keep else EI64,
        gate["keep_pos"], n_keep,
        dh if (dh is not None and n_keep) else E64,
        hh if (hh is not None and n_keep) else E64,
        1, 1, cfg["snr_limit"], int(cfg["snr_detected"]),
        cfg["new"], cfg["new_logp_gated"], cfg["factors"], cfg["beta"],
        cfg["u"], E32 if cfg["row_map"] is None else cfg["row_map"],
        cfg["ndim"], cfg["t"], cfg["w"], cfg["b"], cfg["cold"],
        out["ll_change_log"], out["prop1"], out["acc1"],
        cfg["nwalkers"], cfg["num_bands"],
        out["warn"], out["kind"], out["sorter_dh"], out["sorter_hh"],
        cfg["ids"], int(cfg["dg_on"]), int(cfg["overlap_on"]),
        out["dg_counts"], gate["cur_cells"], gate["new_cells"],
        cfg["num_cap_cells"], n_sub, n_block,
    )
    return out


def _score(cfg, gate, rng):
    """Synthetic scored likelihoods + <d|h>/<h|h> for the kept rows."""
    n_keep = int(gate["n_keep"][0]) if hasattr(gate["n_keep"], "__len__") \
        else int(gate["n_keep"])
    if n_keep == 0:
        return np.zeros(0), None, None
    scored = np.ascontiguousarray(rng.normal(0.0, 5.0, n_keep))
    hh = np.ascontiguousarray(rng.uniform(1.0, 400.0, n_keep))
    dh = np.ascontiguousarray(hh * rng.uniform(0.5, 1.5, n_keep))
    return scored, dh, hh


# ==========================================================================
# KERNEL vs REFERENCE
# ==========================================================================
@unittest.skipUnless(_HAVE_KERNELS, _SKIP_MSG)
class GateCompactKernelTest(unittest.TestCase):
    def _check(self, **kw):
        cfg = _synth(**kw)
        got = _call_gate(_BACKEND, cfg)
        want = _ref_gate_compact(cfg)
        np.testing.assert_array_equal(got["new_logp"], want["new_logp"])
        np.testing.assert_array_equal(got["keep_flag"], want["keep_flag"])
        n_keep = int(got["n_keep"][0])
        self.assertEqual(n_keep, want["n_keep"])
        np.testing.assert_array_equal(got["keep_idx"][:n_keep],
                                      want["keep_idx"])
        np.testing.assert_array_equal(got["keep_pos"], want["keep_pos"])
        np.testing.assert_array_equal(got["trust_counts"],
                                      want["trust_counts"])
        np.testing.assert_array_equal(got["dg_count"], want["dg_count"])
        if cfg["dg_on"]:
            np.testing.assert_array_equal(got["cur_cells"], want["cur_cells"])
            np.testing.assert_array_equal(got["new_cells"], want["new_cells"])
        # the compaction must be ASCENDING -- downstream gathers depend on it
        self.assertTrue(np.all(np.diff(got["keep_idx"][:n_keep]) > 0))
        return cfg, got, want

    def test_window_only(self):
        cfg, _, want = self._check(window=True)
        # the window must actually reject something beyond the prior
        self.assertLess(want["n_keep"], cfg["n_sub"])

    def test_window_off_keeps_prior_rejections_only(self):
        cfg, got, want = self._check(window=False)
        np.testing.assert_array_equal(
            got["keep_flag"].astype(bool), ~np.isinf(cfg["new_logp"]))

    def test_trust_region(self):
        _, _, want = self._check(trust=True, window=True)
        self.assertGreater(int(want["trust_counts"][2]), 0)

    def test_parity_half_row_map(self):
        self._check(parity=True, window=True, trust=True)

    def test_cap_gate_partition(self):
        _, _, want = self._check(dg=True, cap_divisor=4, window=True)
        self.assertGreater(int(want["dg_count"][0]), 0)

    def test_cap_gate_staggered(self):
        self._check(dg=True, cap_divisor=4, stagger=True, window=True)

    def test_cap_gate_overlap(self):
        self._check(dg=True, overlap=True, cap_divisor=4, window=True)

    def test_cap_gate_divisor_one_overlap(self):
        # divisor 1 WITH overlap is a live 2026-08-26 configuration: the cell
        # index short-circuits to the band but membership still widens.
        self._check(dg=True, overlap=True, cap_divisor=1, window=True)

    def test_everything_at_once(self):
        self._check(window=True, trust=True, dg=True, overlap=True,
                    cap_divisor=4, parity=True)

    def test_all_rejected_compacts_to_zero(self):
        cfg = _synth(window=False)
        cfg["new_logp"][:] = -np.inf
        got = _call_gate(_BACKEND, cfg)
        self.assertEqual(int(got["n_keep"][0]), 0)
        np.testing.assert_array_equal(got["keep_pos"],
                                      np.full(cfg["n_sub"], -1, np.int32))

    def test_large_batch_spans_many_scan_tiles(self):
        # The compaction is a single-block tiled scan; only a batch wider than
        # one tile exercises the running-offset carry between tiles.
        self._check(n_sub=1031, num_bands=1031, window=True, seed=11)


@unittest.skipUnless(_HAVE_KERNELS, _SKIP_MSG)
class AcceptApplyKernelTest(unittest.TestCase):
    def _check(self, seed=3, **kw):
        cfg = _synth(seed=seed, **kw)
        gate = _call_gate(_BACKEND, cfg)
        cfg["new_logp_gated"] = gate["new_logp"]
        rng = np.random.RandomState(seed + 1000)
        scored, dh, hh = _score(cfg, gate, rng)
        got = _call_accept(_BACKEND, cfg, gate, scored, dh, hh)

        n_keep = int(gate["n_keep"][0])
        ref_cfg = dict(cfg)
        ref_cfg["new_logp"] = gate["new_logp"]
        ref_cfg["keep_idx"] = gate["keep_idx"][:n_keep]
        ref_cfg["scored"] = scored
        ref_cfg["d_h"] = dh
        ref_cfg["h_h"] = hh
        ref_cfg["cur_cells"] = gate["cur_cells"]
        ref_cfg["new_cells"] = gate["new_cells"]
        want = _ref_accept_apply(ref_cfg)

        for key in ("new_ll", "delta", "lnpdiff", "curr", "ll_ref",
                    "curr_prior", "ll_change_log", "prop1", "acc1",
                    "sorter_dh", "sorter_hh", "warn", "kind"):
            np.testing.assert_array_equal(got[key], want[key],
                                          err_msg=f"mismatch in {key}")
        np.testing.assert_array_equal(got["accept"], want["accept"])
        np.testing.assert_array_equal(got["accept_pre"], want["accept_pre"])
        if cfg["dg_on"]:
            np.testing.assert_array_equal(got["dg_counts"], want["dg_counts"])
        return cfg, got, want

    def test_plain(self):
        cfg, got, _ = self._check()
        n_acc = int(got["accept"].sum())
        self.assertGreater(n_acc, 0)
        self.assertLess(n_acc, cfg["n_sub"])

    def test_snr_clamp_detected(self):
        _, got, want = self._check(detected=True)
        self.assertTrue(np.any(got["new_ll"] <= BAD_LL))

    def test_parity_half_row_map(self):
        self._check(parity=True)

    def test_cap_occupancy_partition(self):
        self._check(dg=True, cap_divisor=4)

    def test_cap_occupancy_overlap(self):
        self._check(dg=True, overlap=True, cap_divisor=4)

    def test_trust_and_cap_together(self):
        self._check(trust=True, dg=True, overlap=True, cap_divisor=4,
                    parity=True, detected=True)

    def test_no_kept_rows(self):
        cfg = _synth(window=False)
        cfg["new_logp"][:] = -np.inf
        gate = _call_gate(_BACKEND, cfg)
        cfg["new_logp_gated"] = gate["new_logp"]
        got = _call_accept(_BACKEND, cfg, gate, np.zeros(0), None, None)
        # every row floors, nothing is accepted, and the proposal counter
        # still ticks once per row (the python does the same)
        np.testing.assert_array_equal(got["new_ll"],
                                      np.full(cfg["n_sub"], LL_FLOOR))
        self.assertEqual(int(got["accept"].sum()), 0)
        self.assertEqual(int(got["prop1"].sum()), cfg["n_sub"])

    def test_no_snr_arrays_skips_the_clamp(self):
        cfg = _synth(seed=5)
        gate = _call_gate(_BACKEND, cfg)
        cfg["new_logp_gated"] = gate["new_logp"]
        n_keep = int(gate["n_keep"][0])
        rng = np.random.RandomState(99)
        scored = np.ascontiguousarray(rng.normal(0.0, 5.0, n_keep))
        got = _call_accept(_BACKEND, cfg, gate, scored, None, None)
        np.testing.assert_array_equal(
            got["new_ll"][gate["keep_idx"][:n_keep]], scored)
        # with no <d|h>/<h|h> the sorter stash must be untouched
        np.testing.assert_array_equal(got["sorter_dh"], cfg["sorter_dh"])


@unittest.skipUnless(_HAVE_KERNELS, _SKIP_MSG)
class InModelKernelEndToEndTest(unittest.TestCase):
    """Knob ON through the REAL ``_run_in_model_repeats``.

    The yardstick is ``_reference_run`` -- the pre-2026-08-15 straight-line
    accept chain -- which the knob-OFF path already reproduces bit-for-bit.
    Passing it with the knob ON means the fused kernels are a drop-in.
    """

    def test_matches_the_straight_line_reference(self):
        n_rep, seed = 40, 2026
        picked, band_sorter, band_temps, ntemps, nwalkers, n = _build_problem()
        move = _make_move(n_rep)
        buf = _FakeBuffer(n)
        ll_change = np.zeros((ntemps, nwalkers, n))
        prop = np.zeros((2, ntemps, nwalkers, n), dtype=int)
        acc = np.zeros_like(prop)
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_ACCEPT_KERNEL": "1"}):
            np.random.seed(seed)
            move._run_in_model_repeats(
                None, band_sorter, buf, band_temps, picked,
                ll_change, prop, acc,
            )
        picked_ref, _, band_temps_ref, _, _, _ = _build_problem()
        coords0 = _build_problem()[1].coords
        ref = _reference_run(
            picked_ref, coords0, band_temps_ref, n_rep, seed,
            n_src=len(band_sorter.inds),
        )
        np.testing.assert_array_equal(
            band_sorter.coords[picked["ids"]], ref["curr"])
        np.testing.assert_array_equal(ll_change, ref["ll_change"])
        np.testing.assert_array_equal(prop, ref["prop"])
        np.testing.assert_array_equal(acc, ref["acc"])
        self.assertEqual(move._im_kind_counts["fake"], ref["kind"])
        np.testing.assert_array_equal(move._sorter_dh, ref["sorter_dh"])
        np.testing.assert_array_equal(move._sorter_hh, ref["sorter_hh"])
        self.assertGreater(int(acc[1].sum()), 0)
        self.assertLess(int(acc[1].sum()), int(prop[1].sum()))


# ==========================================================================
# KNOB-OFF behaviour -- active in every environment, built or not
# ==========================================================================
class InModelAcceptKernelKnobTest(unittest.TestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_ACCEPT_KERNEL", None)
            self.assertFalse(_inmodel_accept_kernel_on())

    def test_knob_resolution(self):
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_ACCEPT_KERNEL": "1"}):
            self.assertTrue(_inmodel_accept_kernel_on())
        for value in ("0", "", "true", "yes"):
            with mock.patch.dict(os.environ,
                                 {"GB_INMODEL_ACCEPT_KERNEL": value}):
                self.assertFalse(_inmodel_accept_kernel_on())

    def test_trace_knobs_disarm_the_kernel_path(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_TRACE", None)
            os.environ.pop("GB_JUMP_TRACE", None)
            self.assertFalse(_inmodel_trace_knobs_active())
        with mock.patch.dict(os.environ, {"GB_JUMP_TRACE": "1"}):
            self.assertTrue(_inmodel_trace_knobs_active())
        with mock.patch.dict(os.environ, {"GB_INMODEL_TRACE": "5"}):
            self.assertTrue(_inmodel_trace_knobs_active())
        # a malformed value must not take down the sampler
        with mock.patch.dict(os.environ, {"GB_INMODEL_TRACE": "junk"}):
            self.assertFalse(_inmodel_trace_knobs_active())

    def test_block_setup_returns_none_when_knob_is_off(self):
        move = _make_move(3)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_ACCEPT_KERNEL", None)
            self.assertIsNone(
                move._imk_block_setup([], None, None, None, None, None, None,
                                      None, None)
            )

    def test_block_setup_stands_down_for_the_traces(self):
        """A trace knob must beat the accept knob, not race it."""
        move = _make_move(3)
        n = 4
        curr = np.zeros((n, 4))
        with mock.patch.dict(
            os.environ,
            {"GB_INMODEL_ACCEPT_KERNEL": "1", "GB_JUMP_TRACE": "1"},
        ):
            self.assertIsNone(move._imk_block_setup(
                [], curr, np.zeros(n), np.zeros(n),
                np.zeros((1, 1, n)),
                np.zeros((2, 1, 1, n), dtype=np.int64),
                np.zeros((2, 1, 1, n), dtype=np.int64),
                None, None,
            ))

    def test_block_setup_falls_back_on_a_bad_layout(self):
        """A float32 ledger must degrade to the python chain, not crash."""
        move = _make_move(3)
        n = 4
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_ACCEPT_KERNEL": "1"}):
            os.environ.pop("GB_JUMP_TRACE", None)
            os.environ.pop("GB_INMODEL_TRACE", None)
            self.assertIsNone(move._imk_block_setup(
                [], np.zeros((n, 4)), np.zeros(n), np.zeros(n),
                np.zeros((1, 1, n), dtype=np.float32),
                np.zeros((2, 1, 1, n), dtype=np.int64),
                np.zeros((2, 1, 1, n), dtype=np.int64),
                None, None,
            ))

    def test_helpers_are_present_on_the_move(self):
        for name in ("_imk_block_setup", "_imk_gate", "_imk_accept",
                     "_imk_halves", "_imk_rebuild_halves", "_imk_real_1d"):
            self.assertTrue(hasattr(GBSpecialStretchMove, name), name)

    def test_backend_dataclass_carries_the_kernel_fields(self):
        from lisatools.cutils import LISAToolsBackendMethods

        fields = {f.name for f in
                  LISAToolsBackendMethods.__dataclass_fields__.values()}
        self.assertIn("gb_inmodel_gate_compact", fields)
        self.assertIn("gb_inmodel_accept_apply", fields)


class KnobOffCallSiteUnchangedTest(unittest.TestCase):
    """With the knob off the repeat loop must be the historical chain.

    This is the regression net for the branch that was threaded through
    ``_run_in_model_repeats``: same accept decisions, same tracked state,
    same counters as the straight-line reference.
    """

    def test_knob_off_reproduces_the_straight_line_reference(self):
        n_rep, seed = 40, 2026
        picked, band_sorter, band_temps, ntemps, nwalkers, n = _build_problem()
        move = _make_move(n_rep)
        buf = _FakeBuffer(n)
        ll_change = np.zeros((ntemps, nwalkers, n))
        prop = np.zeros((2, ntemps, nwalkers, n), dtype=int)
        acc = np.zeros_like(prop)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_ACCEPT_KERNEL", None)
            np.random.seed(seed)
            move._run_in_model_repeats(
                None, band_sorter, buf, band_temps, picked,
                ll_change, prop, acc,
            )
        picked_ref, _, band_temps_ref, _, _, _ = _build_problem()
        coords0 = _build_problem()[1].coords
        ref = _reference_run(
            picked_ref, coords0, band_temps_ref, n_rep, seed,
            n_src=len(band_sorter.inds),
        )
        np.testing.assert_array_equal(
            band_sorter.coords[picked["ids"]], ref["curr"])
        np.testing.assert_array_equal(ll_change, ref["ll_change"])
        np.testing.assert_array_equal(prop, ref["prop"])
        np.testing.assert_array_equal(acc, ref["acc"])
        self.assertEqual(move._im_kind_counts["fake"], ref["kind"])
        np.testing.assert_array_equal(move._sorter_dh, ref["sorter_dh"])
        np.testing.assert_array_equal(move._sorter_hh, ref["sorter_hh"])


if __name__ == "__main__":
    unittest.main()
