"""F-stat birth-grid fitting: comb scan -> peak selection -> stacked grids.

The compute core of the offline grid prep
(``scripts/fstat_proposal/plot_fstat_proposal_mojito.py``), lifted into the
library so it can run **inside** a sampler move as well as from a script.
Nothing here knows about ``gb_info`` / ``general_info`` / plotting: every
consumer passes the handful of scalars it needs plus an injectable

    call_fstat(params_phys_9col) -> (N (n, 4), M_upper (n, 10))

so the same code serves the offline script (which calls
``gb_wdm_comp.get_fstat_ll_wdm`` directly) and the in-move fit (which routes
through ``_RoutedBandEngine.route_fstat_ll`` against the current residual).

Checkpointing: every expensive sweep funnels through
:func:`chunked_fstat_sweep` / :func:`run_stacked_peak_sweep`, so
checkpointing those two row cursors makes the whole ~hours-scale prep
restartable. Rerunning the same fit resumes each unfinished sweep from its
last saved row; a fingerprint of the sweep's actual inputs (salted with
``fingerprint_extra``) guards against resuming across changed knobs or
across refit epochs. Cadence: ``FSTAT_CKPT_SECS`` (default 300).

All ``FSTAT_*`` env knobs keep flowing through
:func:`lisatools.sampling.fstat_proposal.fstat_knob`, so an in-run fit
inherits the offline knob set unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import time
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ckpt_fingerprint",
    "ckpt_load",
    "ckpt_save",
    "ckpt_resume",
    "ckpt_clear",
    "ckpt_secs",
    "chunked_fstat_sweep",
    "comb_peak_window",
    "comb_sweep_mask",
    "select_comb_peaks",
    "CombLevelSpec",
    "run_comb_level",
    "comb_part_path",
    "save_comb_part",
    "load_comb_part",
    "assemble_comb_level",
    "clear_comb_parts",
    "run_comb_scan",
    "run_stacked_peak_sweep",
    "StageBGroupSpec",
    "run_stage_b_group",
    "write_stacked_npz",
    "split_box_range",
    "stage_b_part_path",
    "save_stage_b_part",
    "load_stage_b_part",
    "assemble_stage_b_group",
    "clear_stage_b_parts",
    "run_stacked_stage_b",
    "run_fstat_grid_fit",
    "comb_cache_usable",
    "stacked_grid_path",
    "stage_b_complete",
    "stacked_cache_at_floor",
    "build_gb_birth_distribution",
    "sighet_fstat_ref_margin_hz",
    "build_sighet_call_fstat",
    "enumerate_center_nodes",
    "run_center_sweep",
    "build_fstat_center_table",
    "GRID_BASENAME",
    "CENTER_TABLE_BASENAME",
]

#: Cache basename. The offline prep writes ``<base>_comb.npz`` /
#: ``<base>_peaks_stacked.npz``; keeping the same suffixes means a prebuilt
#: offline grid drops straight into an epoch directory.
GRID_BASENAME = "fstat_grid.npz"

#: Per-epoch F-stat CENTER table (the RJ distance-birth proposal's
#: ``(phi0, iota, psi, ln A_max, sigma, ln SNR)`` vs f0), written next to the
#: grids in the epoch cache dir and checkpoint-loaded exactly like them.
CENTER_TABLE_BASENAME = "fstat_centers.npz"


def _fmt_secs(s: float) -> str:
    """Compact duration for progress lines: 42s / 7.2m / 1.8h."""
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


# The host-transfer helper already exists in fstat_proposal -- import it
# rather than keeping a second copy.
from lisatools.sampling.fstat_proposal import _host as _to_host  # noqa: E402
from lisatools.sampling.fstat_proposal import fdot_axis_on  # noqa: E402
from lisatools.sampling.gb_observable_basis import (  # noqa: E402
    fdot_axis_bounds,
    mc_floor_for_fdot,
    n_fdot_nodes,
)


def _windowed_max(a, w, xp):
    """Sliding max over ``[i - w, i + w]`` via O(log w) shifted-max folds
    (xp-generic; no scipy)."""
    n = a.shape[0]
    pad = xp.full(w, -xp.inf, dtype=a.dtype)
    b = xp.concatenate([pad, a, pad])
    width = 2 * w + 1
    r = b.copy()
    done = 1
    while done < width:
        step = min(done, width - done)
        r[: r.shape[0] - step] = xp.maximum(r[: r.shape[0] - step], r[step:])
        done += step
    return r[:n]


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def ckpt_fingerprint(*arrays, extra: str = "") -> str:
    """Cheap content hash of a sweep's inputs (subsampled), salted by ``extra``.

    ``extra`` is where the caller puts anything that must invalidate a
    resume without being an array -- the sweep label, and (for the in-move
    fit) the refit epoch, so a later epoch can never resume another epoch's
    rows.

    A row provider that never materializes its block (:class:`_CombRows`)
    exposes ``_ckpt_flat_sample()`` returning ``(shape, the same subsampled
    flat values)``, so a lazy provider hashes BIT-IDENTICALLY to the dense
    array it replaces -- an in-flight sweep's progress file stays valid
    across the lazy switch.
    """
    h = hashlib.md5(extra.encode())
    for a in arrays:
        sampler = getattr(a, "_ckpt_flat_sample", None)
        if sampler is not None:
            shape, sample = sampler()
        else:
            a = np.ascontiguousarray(_to_host(a))
            shape = a.shape
            sample = a.reshape(-1)[::max(1, a.size // 257)]
        h.update(str(shape).encode())
        h.update(np.asarray(sample).tobytes())
    return h.hexdigest()


#: Raw float64 payload written alongside the ``.progress.npz`` header. The
#: header used to carry the F rows themselves, which made every cadence save
#: rewrite the WHOLE completed prefix -- O(n^2) bytes over a sweep (at 23-mo
#: comb density, ~250 GB of writes + the matching device->host copies for one
#: level). The payload file is append-only, so a save costs only the rows
#: produced since the previous one.
_CKPT_PAYLOAD_SUFFIX = ".progress.dat"
_CKPT_HEADER_SUFFIX = ".progress.npz"


def _ckpt_payload_rows(path: str) -> int:
    try:
        return os.path.getsize(path) // 8
    except OSError:
        return -1


def ckpt_load(ckpt: Optional[str], n: int, fingerprint: str):
    """Returns ``(host F array or None, completed row count)``.

    Reads either payload layout: the append-only ``.progress.dat`` written
    by :func:`ckpt_save`, or a legacy header that still carries ``F`` inline
    (a progress file written before the incremental payload landed -- an
    in-flight sweep must keep resuming).
    """
    if not ckpt:
        return None, 0
    path = ckpt + _CKPT_HEADER_SUFFIX
    if not os.path.exists(path):
        return None, 0
    try:
        d = np.load(path, allow_pickle=False)
        if str(d["fingerprint"]) != fingerprint or int(d["n"]) != n:
            logger.info("[ckpt] %s: inputs changed -> restarting this sweep",
                        os.path.basename(path))
            return None, 0
        done = int(d["done"])
        if "F" in d.files:                       # legacy inline payload
            F = np.array(d["F"])
        else:
            dat = ckpt + _CKPT_PAYLOAD_SUFFIX
            have = _ckpt_payload_rows(dat)
            if have < done:
                logger.info("[ckpt] %s: payload holds %d of %d rows -> "
                            "restarting this sweep", os.path.basename(dat),
                            max(have, 0), done)
                return None, 0
            F = np.fromfile(dat, dtype=np.float64, count=done)
        if F.size < done:
            logger.info("[ckpt] %s: payload short (%d < %d) -> restarting "
                        "this sweep", os.path.basename(path), F.size, done)
            return None, 0
        logger.info("[ckpt] resuming %s at %d/%d rows",
                    os.path.basename(ckpt), done, n)
        return F, done
    except Exception as exc:  # unreadable/partial file -> recompute
        logger.info("[ckpt] %s: unreadable (%r) -> restarting this sweep",
                    os.path.basename(path), exc)
        return None, 0


def ckpt_save(ckpt: Optional[str], F_rows, done: int, n: int,
              fingerprint: str, start: Optional[int] = None) -> None:
    """Append ``F_rows`` -- the rows ``[start, done)`` -- and stamp the header.

    ``start=None`` means ``F_rows`` is the whole completed prefix
    ``[0, done)`` (the plain "write everything I have" call; also how a
    resumed legacy payload is normalized into the new layout). Otherwise the
    payload file must already hold exactly ``start`` rows -- a longer file
    (rows appended by a run that died before its header landed) is truncated
    back to the header's cursor first, so the payload never disagrees with
    the count the header advertises.

    The header is written to a temp file and ``os.replace``d, so the cursor
    only ever advances after the rows behind it are on disk.
    """
    if not ckpt:
        return
    path = ckpt + _CKPT_HEADER_SUFFIX
    dat = ckpt + _CKPT_PAYLOAD_SUFFIX
    tmp = ckpt + ".progress.tmp.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = np.ascontiguousarray(np.asarray(F_rows, dtype=np.float64).ravel())
    s = int(done) - int(rows.size) if start is None else int(start)
    if s <= 0:
        with open(dat, "wb") as f:
            rows.tofile(f)
    else:
        have = _ckpt_payload_rows(dat)
        if have > s:
            os.truncate(dat, s * 8)
        elif have < s:
            # The payload no longer holds the rows this append continues
            # from. In practice that means ANOTHER process cleared or
            # rewrote this checkpoint under us (2026-09-16: a second job
            # fitting the same epoch in the same directory finished its comb
            # scan and ran ``ckpt_clear`` while this one was mid-sweep; the
            # 6mo relaunch died on this line 9 minutes in). The sweep's
            # result lives in memory and does not need the checkpoint, so a
            # hole must NOT be fatal: skip this write, leave the header where
            # it was (``ckpt_load`` already restarts a sweep whose payload
            # is shorter than its header), and keep computing. Writing a
            # payload with a hole is still refused -- that would be a
            # silently corrupt resume.
            logger.warning(
                "[ckpt] %s: payload holds %d rows but this append starts at %d "
                "(cleared or rewritten by another process?) -- skipping this "
                "checkpoint write; the sweep continues from memory and a "
                "later resume restarts it", os.path.basename(dat),
                max(have, 0), s,
            )
            return
        with open(dat, "ab") as f:
            rows.tofile(f)
    np.savez(tmp, done=done, n=n, fingerprint=fingerprint)
    os.replace(tmp, path)


def ckpt_resume(ckpt: Optional[str], n: int, fingerprint: str):
    """:func:`ckpt_load` + payload normalization.

    Converts a legacy inline-``F`` progress file into the append-only
    payload layout on the spot, so the sweep that resumes it can append from
    its cursor like any other.
    """
    F_prev, start = ckpt_load(ckpt, n, fingerprint)
    if (ckpt and F_prev is not None and start > 0
            and _ckpt_payload_rows(ckpt + _CKPT_PAYLOAD_SUFFIX) < start):
        ckpt_save(ckpt, F_prev[:start], start, n, fingerprint, start=0)
    return F_prev, start


def ckpt_clear(parts_dir: Optional[str], prefix: str) -> None:
    if not parts_dir or not os.path.isdir(parts_dir):
        return
    suffixes = (_CKPT_HEADER_SUFFIX, _CKPT_PAYLOAD_SUFFIX, ".progress.tmp.npz")
    for fn in os.listdir(parts_dir):
        if fn.startswith(prefix) and fn.endswith(suffixes):
            try:
                os.remove(os.path.join(parts_dir, fn))
            except OSError:
                pass


def ckpt_secs() -> float:
    return float(os.environ.get("FSTAT_CKPT_SECS", "300"))


# --------------------------------------------------------------------------
# sig-het call_fstat builder (FSTAT_USE_SIGHET=1)
# --------------------------------------------------------------------------

def sighet_fstat_ref_margin_hz(n_knots: int, Tobs: float) -> float:
    """Max safe |f0_candidate - f0_reference| for the sig-het F-stat path.

    The limiter is the knot-Nyquist ceiling of the fixed-knot COMPLEX
    resample: the residual beat ``e^{2 pi i df0 tau}`` (analytically
    restored at the knots before the linear-complex spline) must stay
    resolvable by K uniform knots, so the margin scales as K and 1/Tobs.

    Measured anchors (``gb_sighet_f0_tolerance.py``, K = 128): the v5
    scorer's tiered-allowance crossing sits between the 57.8-rad last-pass
    and the 140-rad first-fail sweep points (2026-08-11 rerun; the earlier
    2.9e-6 Hz @ 90 d anchor = 141.7 rad was the optimistic edge of the same
    bracket). This returns the LAST-PASS point -- 57.8 rad, i.e. 9.2 beat
    cycles = (K/2)/7 -- as the margin: 1.18e-6 Hz at Tobs = 7.776e6 s
    (90 d), scaled by K/128 and 7.776e6/Tobs. The default reference spacing
    equals this margin, so the worst bucket displacement (spacing/2) keeps
    a further 2x headroom under the measured last-pass.

    The sweeps also showed the error is VIOLENTLY NON-MONOTONIC past the
    envelope (a displaced point can look fine and its neighbor be off by
    thousands), so consumers must HARD-assert every displacement against
    this margin -- never trust a lucky point.
    """
    return 1.18e-6 * (float(n_knots) / 128.0) * (7.776e6 / float(Tobs))


def build_sighet_call_fstat(sighet_comp, wdm_holder, *, xp, Tobs: float,
                            f0_lims_hz, data_index: int = 0,
                            noise_index: Optional[int] = None,
                            spacing_hz: Optional[float] = None,
                            ref_block: Optional[int] = None,
                            fstat_mode: Optional[int] = None):
    """Reference-bucketing ``call_fstat`` over the sig-het F-stat kernel.

    Returns a ``call_fstat(params_phys_9col) -> (N (n, 4), M_upper (n, 10))``
    closure that scores through
    ``GBSignalHetComputations.get_fstat_ll_wdm`` against SHARED heterodyne
    references instead of the chunked per-candidate heterodyne
    (``GBWDMComputations.get_fstat_ll_wdm``): drop-in for
    :func:`chunked_fstat_sweep` / :func:`run_fstat_grid_fit`.

    Reference layout: an f0 comb across ``f0_lims_hz`` at ``spacing_hz``
    (env ``FSTAT_SIGHET_REF_SPACING_HZ``; default = the measured margin
    :func:`sighet_fstat_ref_margin_hz`, so the worst bucket displacement
    spacing/2 sits at HALF the measured envelope -- 2x safety). A spacing
    wider than twice the margin is REFUSED at build time. References carry
    fdot = 0 and a fixed canonical sky (the v5 derotation absorbs the
    candidate's df0/dfdot analytically, and the measured sky arm holds
    across the full sphere); their extrinsics are forced to the basis frame
    by ``setup_fstat_references`` (the amplitude-scale rule).

    Memory: the compact per-reference stash is built in CONTIGUOUS BLOCKS
    of ``ref_block`` references (env ``FSTAT_SIGHET_REF_BLOCK``, default
    512), one block resident at a time -- each incoming batch is grouped by
    block, and a block rebuild only happens when the sweep crosses a block
    boundary (the comb/stage-B sweeps are f0-ordered per segment, so
    thrash is bounded). Every row is HARD-asserted to sit within the margin
    of its bucketed reference (the kernel-side assert in
    ``setup_fstat_references(assert_max_df0=...)``).
    """
    n_knots = int(sighet_comp._g.get("v4_knots", 0))
    if n_knots <= 0:
        raise ValueError(
            "build_sighet_call_fstat needs a sig-het comp constructed with "
            "v4_knots > 0 (the fixed-knot scorer).")
    margin = sighet_fstat_ref_margin_hz(n_knots, Tobs)
    env_sp = os.environ.get("FSTAT_SIGHET_REF_SPACING_HZ", "").strip()
    if spacing_hz is None:
        spacing_hz = float(env_sp) if env_sp else margin
    spacing_hz = float(spacing_hz)
    if spacing_hz > 2.0 * margin:
        raise ValueError(
            f"FSTAT_SIGHET_REF_SPACING_HZ={spacing_hz:.3e} puts the worst "
            f"bucket displacement {0.5 * spacing_hz:.3e} Hz beyond the "
            f"measured sig-het envelope {margin:.3e} Hz (K={n_knots}, "
            f"Tobs={Tobs:.3e}); the scorer is non-monotonic past it.")
    if ref_block is None:
        ref_block = int(os.environ.get("FSTAT_SIGHET_REF_BLOCK", "512"))
    ref_block = max(1, int(ref_block))

    f0_lo = float(f0_lims_hz[0])
    f0_hi = float(f0_lims_hz[-1])
    node_f0 = np.arange(f0_lo - spacing_hz, f0_hi + 2.0 * spacing_hz,
                        spacing_hz)
    n_nodes = len(node_f0)
    n_blocks = (n_nodes + ref_block - 1) // ref_block
    logger.info(
        "[sighet-fstat] %d references over [%.4e, %.4e] Hz, spacing %.3e Hz "
        "(margin %.3e Hz, K=%d, Tobs=%.3e s), %d block(s) of <= %d",
        n_nodes, node_f0[0], node_f0[-1], spacing_hz, margin, n_knots, Tobs,
        n_blocks, ref_block)

    state = {"block": -1, "stash": None}

    def _ensure_block(b: int):
        # Residency is only valid while the COMP still holds the stash THIS
        # closure built: the record lives on the closure but the stash lives
        # on the comp, so a second scorer sharing the comp (the multidev
        # check mode's pinned shadow; any future concurrent fit) silently
        # replaces it -- without the identity check this closure would score
        # against a foreign block's references (the kernel df0 assert turns
        # that into a hard error; this guard makes it a rebuild instead).
        if (state["block"] == b
                and getattr(sighet_comp, "_fstat", None) is not None
                and sighet_comp._fstat is state["stash"]):
            return
        lo = b * ref_block
        hi = min(lo + ref_block, n_nodes)
        refs = np.zeros((hi - lo, 9))
        refs[:, 1] = node_f0[lo:hi]
        # canonical sky; extrinsics are overridden to the basis frame by
        # setup_fstat_references itself.
        refs[:, 7] = 0.0
        refs[:, 8] = 0.0
        sighet_comp.setup_fstat_references(
            refs, wdm_holder, data_index=data_index,
            noise_index=noise_index,
            assert_max_df0=0.5 * spacing_hz * (1.0 + 1e-9))
        state["block"] = b
        # Strong ref on purpose: while scorers share a comp, each keeps at
        # most its own last-built block alive alongside the comp's resident
        # one (a rebuild replaces both refs). Lanes never share in
        # production, so the extra block only exists in the check mode.
        state["stash"] = sighet_comp._fstat
        # ONE block is resident at a time (each is ~GBs), so re-entering an
        # evicted block REBUILDS it. Track unique builds and rebuilds
        # separately -- an earlier version summed them into one "built k/N"
        # counter, which read as 387/36 under cyclic eviction.
        seen = state.setdefault("seen", set())
        if b in seen:
            state["rebuilds"] = state.get("rebuilds", 0) + 1
            if state["rebuilds"] % 256 == 0:
                logger.info(
                    "[sighet-fstat] %d reference-block rebuilds (single-"
                    "resident cache; heavy rebuilding means the sweep's row "
                    "order is not f0-local).", state["rebuilds"])
        else:
            seen.add(b)
            step = max(1, n_blocks // 10)
            if len(seen) % step == 0 or len(seen) == n_blocks:
                logger.info(
                    "[sighet-fstat] reference blocks built: %d/%d unique "
                    "(%d rebuilds)", len(seen), n_blocks,
                    state.get("rebuilds", 0))

    def call_fstat(params):
        # Bucket on the HOST copy of the f0 column. The comb / center sweeps
        # assemble their rows on the host, so reading f0 off an uploaded
        # device copy meant an H2D followed immediately by a synchronizing
        # D2H of the same column -- one hard device sync per batch, for
        # numbers the caller already had.
        p_in = params if hasattr(params, "shape") else np.asarray(params)
        if getattr(p_in, "ndim", 2) == 1:
            p_in = p_in[None, :]
        f0_h = _to_host(p_in[:, 1])
        n_rows = int(f0_h.shape[0])
        k = np.clip(np.round((f0_h - node_f0[0]) / spacing_hz).astype(int),
                    0, n_nodes - 1)
        worst = np.abs(f0_h - node_f0[k]).max() if n_rows else 0.0
        if worst > 0.5 * spacing_hz * (1.0 + 1e-9):
            raise RuntimeError(
                f"[sighet-fstat] candidate f0 {worst:.3e} Hz from its "
                f"bucketed reference (> spacing/2 = "
                f"{0.5 * spacing_hz:.3e}); the reference comb does not "
                "cover the requested f0 range.")
        if n_rows == 0:
            return (xp.zeros((0, 4), dtype=xp.float64),
                    xp.zeros((0, 10), dtype=xp.float64))
        p = xp.asarray(p_in, dtype=xp.float64)
        blocks = k // ref_block
        if int(blocks.min()) == int(blocks.max()):
            # SINGLE-BLOCK FAST PATH -- the production case. Both sweeps
            # order rows f0-locally (node-major comb, f0-sorted stage-B
            # boxes) and a block spans ``ref_block`` references, so a batch
            # almost always sits inside one block. Scoring it whole skips
            # the (n, 4) / (n, 10) scratch pair, three uploads of the row
            # index, a device gather of the rows and two scatters back --
            # all of which reduce to the identity when there is one group.
            # ``xp.asarray(..., float64)`` keeps the dtype contract the
            # scratch buffers used to impose (a no-op view when the kernel
            # already returns float64).
            b0 = int(blocks[0])
            _ensure_block(b0)
            di = xp.asarray((k - b0 * ref_block).astype(np.int32))
            N_b, M_b = sighet_comp.get_fstat_ll_wdm(
                p, None, data_index=di, fstat_mode=fstat_mode)
            return (xp.asarray(N_b, dtype=xp.float64),
                    xp.asarray(M_b, dtype=xp.float64))
        N_all = xp.zeros((n_rows, 4), dtype=xp.float64)
        M_all = xp.zeros((n_rows, 10), dtype=xp.float64)
        for b in np.unique(blocks):
            sel = np.where(blocks == b)[0]
            _ensure_block(int(b))
            sel_d = xp.asarray(sel)
            di = xp.asarray((k[sel] - int(b) * ref_block).astype(np.int32))
            N_b, M_b = sighet_comp.get_fstat_ll_wdm(
                p[sel_d], None, data_index=di, fstat_mode=fstat_mode)
            N_all[sel_d] = N_b
            M_all[sel_d] = M_b
        return N_all, M_all

    return call_fstat


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def chunked_fstat_sweep(call_fstat: Callable, params, *, xp, label: str = "",
                        ckpt: Optional[str] = None,
                        fingerprint_extra: str = ""):
    """ONE chunked kernel stream over a pre-assembled physical params array.

    The batching invariant: every F-stat computation sweeps a single
    concatenated parameter set spanning ALL sub-bands / sky points / peak
    boxes at once, in ``FSTAT_BATCH``-row calls -- never a per-band /
    per-sky / per-peak kernel loop. Returns F on ``xp``.
    """
    from lisatools.sampling.fstat_proposal import compute_fstat, fstat_knob

    fp = (ckpt_fingerprint(params, extra=label + fingerprint_extra)
          if ckpt else "")
    # Never upload the full set: the comb reaches ~1.2e9 rows at 23-mo
    # density (~85 GB, more than any device), so rows stay where the caller
    # assembled them (host, for the comb) and every call_fstat implementation
    # converts its own FSTAT_BATCH-slice input.
    if not hasattr(params, "shape"):
        params = np.asarray(params)
    n = params.shape[0]
    batch = fstat_knob("FSTAT_BATCH", int)
    F = xp.empty(n, dtype=xp.float64)
    F_prev, start = ckpt_resume(ckpt, n, fp)
    if F_prev is not None and start > 0:
        F[:start] = xp.asarray(F_prev[:start])
    # Cadence + flush cursor. Only the rows produced since the last save are
    # copied back and written (the payload file is append-only), so both the
    # device->host traffic and the disk traffic over a sweep are O(n), not
    # O(n^2 / 2) as they were when every save rewrote the whole prefix.
    ckpt_every = ckpt_secs()
    flushed = start
    if ckpt and start == 0:
        # Write the progress file up front (done=0): its presence confirms
        # checkpointing is armed and where the parts live -- and a very early
        # death still leaves a valid (empty) resume point.
        ckpt_save(ckpt, np.empty(0), 0, n, fp, start=0)
        logger.info("[ckpt] progress file: %s.progress.npz (every %.0fs "
                    "+ on interrupt)", ckpt, ckpt_every)
    t0 = last = last_ckpt = time.time()
    done = start
    try:
        for s in range(start, n, batch):
            e = min(s + batch, n)
            N_arr, M_up = call_fstat(params[s:e])
            F[s:e] = compute_fstat(xp.asarray(N_arr), xp.asarray(M_up))
            done = e
            if time.time() - last > 30:
                last = time.time()
                _el = time.time() - t0
                _rate = (e - start) / max(_el, 1e-9)
                logger.info(
                    "[sweep%s] %d/%d evals (%.1f%%) | %.0f evals/s | "
                    "ETA %s | elapsed %s", label, e, n, 100.0 * e / n,
                    _rate, _fmt_secs((n - e) / max(_rate, 1e-9)),
                    _fmt_secs(_el))
            if ckpt and e < n and time.time() - last_ckpt > ckpt_every:
                last_ckpt = time.time()
                ckpt_save(ckpt, _to_host(F[flushed:e]), e, n, fp,
                          start=flushed)
                flushed = e
                logger.info("[ckpt] saved %d/%d rows", e, n)
    except BaseException:
        # Ctrl-C / SIGTERM / OOM between cadence saves: keep every completed
        # row (SIGKILL is the only death the cadence alone must cover).
        if ckpt and done > flushed:
            ckpt_save(ckpt, _to_host(F[flushed:done]), done, n, fp,
                      start=flushed)
            flushed = done
            logger.info("[ckpt] interrupt: saved %d/%d rows", done, n)
        raise
    if ckpt and start < n:
        ckpt_save(ckpt, _to_host(F[flushed:n]), n, n, fp, start=flushed)
    logger.info("[sweep%s] %d evals in one chunked stream (batch=%d, %.0fs%s)",
                label, n, batch, time.time() - t0,
                f", resumed at {start}" if start else "")
    return F


def comb_peak_window(spacing) -> int:
    """Half-width, in comb nodes, of :func:`select_comb_peaks`' tier-1 window.

    The Doppler envelope (~3e-3 mHz) over the node spacing [mHz], at least
    one node. ONE function because :func:`comb_sweep_mask` has to widen the
    swept set by exactly the window peak selection reads -- two spellings
    could drift and the edge of every kept band would grow spurious peaks.
    """
    return max(1, int(round(3e-3 / spacing)))


def comb_sweep_mask(f0_nodes, band_edges_hz, band_skip, spacing):
    """Which comb nodes stage A must SWEEP when ``band_skip`` bands are shut.

    Returns ``None`` for ``band_skip=None`` (sweep everything, the
    historical path), else a host ``(n_nodes,)`` bool array.

    A node is swept when its band is NOT skipped (user ruling 2026-10-10:
    the shut bands leave the stage-A sweep itself, not only peak selection)
    -- PLUS the nodes of skipped bands that sit within
    :func:`comb_peak_window` of a kept node. That margin is what keeps every
    kept band's peaks exactly what a full sweep would select: tier 1 asks
    whether a node is the max over ``+-window`` nodes and the 3-point test
    reads both neighbours, so a kept node beside a zero-filled skipped one
    would turn into a peak wherever F rises into the skipped band. At the
    9-month spacing the window is ~140 nodes, paid once per kept/skipped
    boundary.

    Band membership is the SAME ``searchsorted`` expression
    :func:`select_comb_peaks` uses, so the sweep and the peak mask can never
    disagree about which band a node belongs to. A node outside the band
    grid is never skipped.
    """
    keep = _comb_keep_mask(f0_nodes, band_edges_hz, band_skip)
    if keep is None or keep.all() or not keep.any():
        return keep
    w = comb_peak_window(spacing)
    return _windowed_max(keep.astype(np.float64), w, np) > 0.0


def _comb_keep_mask(f0_nodes, band_edges_hz, band_skip):
    """Nodes whose OWN band is not skipped (``None`` for no skip set):
    :func:`comb_sweep_mask` before the peak-window margin is added."""
    if band_skip is None:
        return None
    f0 = np.asarray(f0_nodes, dtype=float)
    edges_mHz = np.asarray(band_edges_hz, dtype=float) * 1e3
    nb = int(len(edges_mHz) - 1)
    bs = np.asarray(band_skip, dtype=bool).reshape(-1)
    if bs.shape[0] != nb:
        raise ValueError(
            f"comb_sweep_mask got a band_skip covering {bs.shape[0]} bands "
            f"but the grid has {nb} sub-bands.")
    b = np.searchsorted(edges_mHz, f0, side="right") - 1
    keep = np.ones(f0.shape, dtype=bool)
    inb = (b >= 0) & (b < nb)
    keep[inb] = ~bs[b[inb]]
    return keep


def select_comb_peaks(f0_nodes, F_max, band_edges_hz, spacing, xp,
                      min_F=None, band_skip=None):
    """Vectorized per-sub-band peak selection from the comb's ``F_max(f0)``.

    Two-tier candidate set -> absolute SNR floor + interior-band mask ->
    per-band cap:

    * **Tier 1** (window champion): a node survives iff it carries the max F
      within +-``min_sep`` nodes (the Doppler envelope ~3e-3 mHz).
    * **Tier 2** (local-max rescue): ALSO keep any strict 3-point local
      maximum, gated only by the shared SNR floor -- so a quiet real source
      inside a loud neighbor's Doppler window still gets its own grid.

    ``min_F`` overrides the global ``FSTAT_PEAK_MIN_SNR`` floor. It may be
    a scalar (one floor everywhere, the historical behaviour) or a
    ``(num_sub_bands,)`` array giving each sub-band its OWN floor -- the
    F-stat half of the per-(walker, band) search-stage schedule, built by
    ``fstat_proposal.fstat_band_min_F``. ``None`` (the default) keeps
    ``fstat_peak_min_F()`` exactly as before.

    Note the per-band floor's blast radius is bounded by
    ``FSTAT_PEAKS_PER_BAND``: in a band already at that cap, which keeps
    the TOP peaks by F, loosening the floor changes nothing at all.

    Returns a host ``(N, 4)`` array of ``(f0_mHz, F, node_idx, band_idx)``
    sorted by F descending; only interior sub-bands are kept.
    """
    from lisatools.sampling.fstat_proposal import fstat_knob, fstat_peak_min_F

    f0_nodes = xp.asarray(f0_nodes)
    F_max = xp.asarray(F_max)
    # band_edges is Hz; ALL f0 values here are mHz. Convert ONCE.
    band_edges_mHz = xp.asarray(np.asarray(band_edges_hz, dtype=float) * 1e3)
    num_sub_bands = int(len(band_edges_hz) - 1)
    band_of_node = xp.searchsorted(band_edges_mHz, f0_nodes, side="right") - 1

    min_sep = comb_peak_window(spacing)
    wmax = _windowed_max(F_max, min_sep, xp)
    tier1 = F_max >= wmax
    local3 = xp.zeros(F_max.shape, dtype=bool)
    local3[1:-1] = (F_max[1:-1] > F_max[:-2]) & (F_max[1:-1] > F_max[2:])

    if min_F is None:
        min_F = fstat_peak_min_F()
    # PER-BAND floor: gather each node's own band's threshold. The scalar
    # branch is the historical expression, bit-identically.
    if np.ndim(min_F) == 0:
        min_F_row = min_F
        _min_F_lo = _min_F_hi = float(min_F)
    else:
        _mf = np.asarray(min_F, dtype=float).reshape(-1)
        if _mf.shape[0] != num_sub_bands:
            raise ValueError(
                f"select_comb_peaks got a per-band min_F covering "
                f"{_mf.shape[0]} bands but the grid has {num_sub_bands}. "
                f"A mismatched vector would threshold the wrong bands "
                f"while every index stayed in range."
            )
        min_F_row = xp.asarray(_mf)[band_of_node]
        _min_F_lo, _min_F_hi = float(_mf.min()), float(_mf.max())
    interior = (band_of_node >= 1) & (band_of_node <= num_sub_bands - 2)
    # LEVEL-3 SHUT BANDS PRODUCE NO PEAKS (user ruling 2026-09-28: "no
    # fstat refit for shutdown bands"). ``band_skip`` is
    # ``(num_sub_bands,)`` bool, True only where NO walker could still use
    # a peak: the per-walker valve shut on EVERY walker (see
    # fstat_proposal.fstat_band_skip for why ``all`` and not ``any`` --
    # there is ONE peak catalog shared by every walker), or births OFF
    # under the per-band empty-band valve (gbspecialstretch's
    # ``fstat_band_skip_for`` builds the union). Applied through the SAME
    # ``band_of_node`` lookup the per-band ``min_F`` uses, so the two can
    # never disagree about which band a node belongs to.
    #
    # Since 2026-10-10 stage A does not SWEEP those bands either
    # (:func:`comb_sweep_mask`); their nodes arrive here as F = 0. This
    # mask stays all the same: the sweep does score the nodes of a skipped
    # band that sit within the peak window of a kept one, and those real
    # values must not become peaks.
    _skip_row = None
    if band_skip is not None:
        _bs = np.asarray(band_skip, dtype=bool).reshape(-1)
        if _bs.shape[0] != num_sub_bands:
            raise ValueError(
                f"select_comb_peaks got a band_skip covering "
                f"{_bs.shape[0]} bands but the grid has {num_sub_bands} "
                f"sub-bands.")
        _skip_row = xp.asarray(_bs)[band_of_node]
    cand = (tier1 | local3) & (F_max >= min_F_row) & interior
    if _skip_row is not None:
        cand = cand & ~_skip_row

    idx = xp.where(cand)[0]
    if int(idx.shape[0]) == 0:
        logger.warning("[peaks] NO peaks above F >= %s in the interior "
                       "sub-bands.", min_F)
        return np.empty((0, 4))
    b = band_of_node[idx]
    Fv = F_max[idx]

    # Vectorized per-band cap: sort by (band, -F), within-band rank via the
    # band-group start offsets, keep rank < cap.
    cap = fstat_knob("FSTAT_PEAKS_PER_BAND", int)
    order = xp.lexsort(xp.stack([-Fv, b.astype(xp.float64)]))
    b_sorted = b[order]
    rank = xp.arange(order.shape[0]) - xp.searchsorted(
        b_sorted, b_sorted, side="left"
    )
    idx_keep = idx[order[rank < cap]]

    f0_h = _to_host(f0_nodes[idx_keep])
    F_h = _to_host(F_max[idx_keep])
    b_h = _to_host(band_of_node[idx_keep])
    peaks = np.column_stack([f0_h, F_h, _to_host(idx_keep), b_h])
    peaks = peaks[np.argsort(peaks[:, 1])[::-1]]

    counts = np.bincount(b_h.astype(int), minlength=num_sub_bands)
    if _min_F_lo == _min_F_hi:
        logger.info("[peaks] %d peaks (F >= %.1f ~ SNR %.1f, cap %d/band); "
                    "per-interior-band counts: %s", len(peaks), _min_F_lo,
                    np.sqrt(2 * _min_F_lo), cap,
                    dict(enumerate(counts.tolist())))
    else:
        logger.info("[peaks] %d peaks (PER-BAND floor F %.1f-%.1f ~ SNR "
                    "%.2f-%.2f, cap %d/band); per-interior-band counts: %s",
                    len(peaks), _min_F_lo, _min_F_hi,
                    np.sqrt(2 * _min_F_lo), np.sqrt(2 * _min_F_hi), cap,
                    dict(enumerate(counts.tolist())))
    zero_bands = [k for k in range(1, num_sub_bands - 1) if counts[k] == 0]
    if zero_bands:
        logger.warning("[peaks] interior sub-band(s) %s have ZERO peaks -- "
                       "births there fall to the comb/floor components only.",
                       zero_bands)
    return peaks


def _sky_grid(n):
    """Golden-ratio spread over the sphere -> ``(alpha, sin_delta)``."""
    ks = np.arange(n)
    return ((2.0 * np.pi * ks * 0.6180339887) % (2.0 * np.pi),
            -1.0 + 2.0 * (ks + 0.5) / n)


def sky_nodes_required(f0_mHz, Tobs, *, vc=None, nmin=1, nmax=0):
    """Sky nodes a template at ``f0`` needs, snapped to a power of two.

    THE ONE PLACE the sky density law lives. Stage A (:func:`run_comb_scan`)
    and stage B (:func:`run_stacked_stage_b`) both call it, so the scan and
    the refinement can no longer disagree about how finely the sky has to be
    resolved -- which they did until 2026-09-24, stage A adaptive 16..512 and
    stage B pinned at a fixed 8x8=64, i.e. the REFINEMENT was up to 8x
    coarser than the scan that found the peak.

    THE LAW. A source is localized on the sky by its Doppler ridge: the
    orbital motion sweeps the carrier by ``+- f0 * v/c``, which over ``Tobs``
    is ``f0 * (v/c) * Tobs`` resolvable frequency bins, and the sky is
    two-dimensional, so the number of distinguishable directions goes as the
    SQUARE of that. Hence ``n ~ (f0 * Tobs * v/c)^2`` -- 10 cells at 2 mHz,
    39 at 4 mHz, 155 at 8 mHz, ~1000 at 20 mHz. Snapping up to a power of
    two keeps the level set small so the sweep runs as a few rectangular
    batches rather than one design per node.

    Args:
        f0_mHz: carrier frequency/frequencies [mHz]. Scalar or array.
        Tobs: observation time [s].
        vc: orbital speed in units of c (``FSTAT_SKY_VC``, default 1e-4).
        nmin: floor. Stage A uses 16; stage B uses 64 so the adaptive
            grid is never COARSER than the historical fixed 8x8.
        nmax: cap, or ``0``/``None`` for uncapped.

    Returns:
        ``int`` array (or scalar) of node counts, same shape as ``f0_mHz``.
    """
    if vc is None:
        vc = float(os.environ.get("FSTAT_SKY_VC",
                                  os.environ.get("FSTAT_COMB_SKY_VC", "1e-4")))
    f = np.asarray(f0_mHz, dtype=float)
    req = np.ceil((f * 1e-3 * float(Tobs) * float(vc)) ** 2)
    lvl = (2.0 ** np.ceil(np.log2(np.clip(req, 1.0, None))))
    lvl = np.maximum(lvl, float(nmin))
    if nmax:
        lvl = np.minimum(lvl, float(nmax))
    out = lvl.astype(int)
    return out if out.ndim else int(out)


#: Longitude nodes per latitude node in the stage-B rectangular sky grid.
#:
#: Measured, not assumed. Sweeping every factorization of a node budget over
#: 40k uniform sky directions and scoring the 95th-percentile angle to the
#: nearest node, the optimum sits at 8x8, 12x11, 17x15, 26x20 and 38x27 for
#: budgets 64/128/256/512/1024 -- i.e. very nearly SQUARE, drifting to ~1.4
#: as the budget grows. ``n_sd = round(sqrt(N/1.25))`` reproduces every one
#: of those to within 4% of the optimal p95, so the closed form is used
#: rather than a table.
#:
#: The intuition that longitude should be ~pi times denser than latitude (to
#: make cells isotropic at the equator) is WRONG here, and the measurement is
#: what caught it: equal steps in sin(delta) become ever LARGER steps in the
#: angle delta towards the poles, so latitude, not longitude, sets the
#: worst-covered cell. Aspect ratios of 2.5 and up measurably lose.
STAGEB_SKY_ASPECT = 1.25


def stage_b_sky_axes(n_target):
    """``n_target`` sky nodes as rectangular ``(alpha_ax, sd_ax)`` axes.

    Stage B's grid is a 4-D TENSOR (:class:`StackedFStatProposal4D`), so the
    sky has to stay a separable ``alpha x sin_delta`` product -- a
    golden-ratio spread like stage A's would need a different proposal class
    and a different cache format. This picks the best rectangular
    factorization instead, and fixes the two ways the historical
    ``linspace(0, 2pi, 8) x linspace(-1, 1, 8)`` wasted its own nodes:

    * ``linspace(0, 2*pi, n)`` includes BOTH endpoints, and alpha = 0 and
      alpha = 2*pi are the same longitude -- one duplicated column, i.e.
      ``n_sd`` wasted evaluations. Fixed with ``endpoint=False``.
    * every alpha at ``sin_delta = +-1`` is the same direction (the pole), so
      each pole row collapses to ONE point. 8x8 therefore scored 64 nodes at
      only 44 distinct sky directions -- 31% of the sweep spent re-deriving
      values it already had, and the poles over-weighted in the proposal by
      the collapse.

    The poles are KEPT (``sin_delta`` spans the closed interval): dropping
    them for cell centers measures worse, because a rectangular grid's
    worst-covered point is otherwise the polar cap, whose radius goes only as
    ``sqrt(2/n_sd)``. With the endpoint fix an 8x8 carries 50 distinct
    directions and a 95th-percentile miss of 0.368 rad, against 44 and 0.399
    for the historical grid.

    Returns:
        ``(alpha_ax, sd_ax)``, both float64 and uniformly spaced -- the
        latter is required by :class:`StackedFStatProposal4D`, which stores
        one ``dx`` per axis.
    """
    n = max(int(n_target), 9)
    n_sd = max(4, int(round(np.sqrt(n / STAGEB_SKY_ASPECT))))
    n_alpha = max(4, int(round(n / n_sd)))
    # alpha: half-open, so 0 and 2*pi are not the same node twice.
    alpha_ax = 2.0 * np.pi * np.arange(n_alpha, dtype=float) / n_alpha
    # sin_delta: closed, so both poles carry an explicit node.
    sd_ax = np.linspace(-1.0, 1.0, n_sd)
    return alpha_ax, sd_ax


class _CombRows:
    """One comb sky level's ``(n_nodes * n_sky, 9)`` rows, never materialized.

    :func:`chunked_fstat_sweep` only ever touches a level's parameter block
    through ``.shape`` and a row slice, so the block does not have to exist:
    this builds each ``FSTAT_BATCH`` slice from index arithmetic instead.
    The dense array it replaces is 1.6 GB for the 3-month comb's largest sky
    level and ~81 GB for the 23-month one (~1.1e9 rows x 9 float64) -- host
    memory that has to be reserved for the whole level even though the sweep
    consumes it 4096 rows at a time.

    Row order is unchanged (NODE-MAJOR: f0 slow, sky fast), and every column
    is produced by the same elementwise expression as the dense assembly, so
    slices are bit-identical to the array's:

    * ``repeat(nodes, lv) * 1e-3`` -> ``(nodes * 1e-3)[row // lv]``
    * ``tile(alpha, nn)``          -> ``alpha[row % lv]``
    * ``arcsin(tile(sd, nn))``     -> ``arcsin(sd)[row % lv]``
    * ``get_fdot`` is elementwise in ``(f, Mc)``.

    ``_ckpt_flat_sample`` reproduces exactly the subsample
    :func:`ckpt_fingerprint` would have taken of the dense array, so an
    in-flight comb progress file keeps resuming across this change.
    """

    __slots__ = ("_f0_hz", "_alpha", "_sd_asin", "_mc", "_lv", "shape", "ndim")

    def __init__(self, nodes_mHz, alpha, sin_delta, mc_fix):
        alpha = np.asarray(alpha, dtype=float)
        self._f0_hz = np.asarray(nodes_mHz, dtype=float) * 1e-3
        self._alpha = alpha
        self._sd_asin = np.arcsin(np.asarray(sin_delta, dtype=float))
        self._mc = float(mc_fix)
        self._lv = int(alpha.shape[0])
        self.shape = (int(self._f0_hz.shape[0]) * self._lv, 9)
        self.ndim = 2

    def __len__(self):
        return self.shape[0]

    def _rows(self, idx):
        from gbgpu.utils.utility import get_fdot

        idx = np.asarray(idx)
        out = np.zeros((idx.size, 9))
        f0 = self._f0_hz[idx // self._lv]
        j = idx % self._lv
        out[:, 0] = 1e-22
        out[:, 1] = f0
        out[:, 2] = get_fdot(f=f0, Mc=np.full(idx.size, self._mc))
        out[:, 5] = 0.5 * np.pi
        out[:, 7] = self._alpha[j]
        out[:, 8] = self._sd_asin[j]
        return out

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._rows(np.arange(*key.indices(self.shape[0])))
        return self._rows(np.asarray(key))

    def _ckpt_flat_sample(self):
        size = self.shape[0] * self.shape[1]
        flat = np.arange(0, size, max(1, size // 257))
        rows, cols = np.divmod(flat, self.shape[1])
        uniq, inv = np.unique(rows, return_inverse=True)
        return self.shape, self._rows(uniq)[np.ravel(inv), cols]


@dataclasses.dataclass(frozen=True)
class CombLevelSpec:
    """Everything ONE comb sky level's sweep needs, as host arrays.

    Stage A's unit of parallel work, and the exact counterpart of
    :class:`StageBGroupSpec`. ``a``/``b`` are node indices into the LEVEL's
    own f0-ordered node list (the level assignment is a deterministic closed
    form of f0 alone -- see ``nsky_per_node`` in :func:`run_comb_scan` --
    so every rank derives the identical list with no communication, and a
    sub-range is addressed in the same coordinates the head concatenates the
    partials in). With a ``band_skip`` that list is the level's SWEPT nodes
    only (:func:`comb_sweep_mask`): still f0-ordered, so contiguous ranges
    over it keep the sig-het block locality, and the head ships each rank
    its slice of it, so no rank has to know the skip set.

    The SKY GRID IS NEVER SLICED. ``alpha`` / ``sin_delta`` / ``lv`` /
    ``mc_fix`` define what a ROW MEANS (:class:`_CombRows` lays rows out
    node-major, sky fast), so a rank that sliced them would score a
    different physical template while reporting the same node -- the same
    invariant that keeps ``StageBGroupSpec``'s Mc/alpha/sin-delta axes
    whole.
    """

    li: int
    n_levels: int
    lv: int
    a: int
    b: int
    f0_nodes: np.ndarray
    alpha: np.ndarray
    sin_delta: np.ndarray
    mc_fix: float
    label: str
    ckpt_name: Optional[str]
    parts_dir: Optional[str]
    fingerprint_extra: str

    @property
    def n_nodes(self) -> int:
        return int(self.b) - int(self.a)

    def sub_range(self, a2, b2, *, ckpt_name=None,
                  label=None) -> "CombLevelSpec":
        """This level restricted to ABSOLUTE nodes ``[a2, b2)``.

        ``ckpt_name`` must differ per rank: the sweep's fingerprint hashes
        the SLICED rows (:meth:`_CombRows._ckpt_flat_sample`) plus the
        label, so two ranks' checkpoints are already mutually invalid -- but
        they must not collide on one path.
        """
        a2, b2 = int(a2), int(b2)
        if not (self.a <= a2 <= b2 <= self.b):
            raise ValueError(
                f"sub_range({a2}, {b2}) is outside level {self.li}'s "
                f"node range [{self.a}, {self.b})")
        i0, i1 = a2 - int(self.a), b2 - int(self.a)
        return dataclasses.replace(
            self,
            a=a2,
            b=b2,
            f0_nodes=np.ascontiguousarray(self.f0_nodes[i0:i1]),
            ckpt_name=(self.ckpt_name if ckpt_name is None else ckpt_name),
            label=(self.label if label is None else label),
        )


def run_comb_level(spec: CombLevelSpec, call_fstat: Callable, *, xp):
    """Sweep ONE comb sky level (or one rank's node range of it).

    The single entry point both the serial scan and every compute rank use,
    so a split can never diverge from the serial code by construction --
    stage A's counterpart of :func:`run_stage_b_group`, and for the same
    reason.

    Returns the per-node reduction as THREE host arrays, ``(F_max,
    best_alpha, best_sin_delta)``: the sky axis is maximized away here
    rather than shipped, so a rank replies with 3 x n_nodes float64 instead
    of the ``n_nodes x nsky`` block it scored (at the 6-mo level 6 that is
    362,020 x 3 values instead of 185 million).

    An EMPTY range (more ranks than nodes) short-circuits: the sweep's
    checkpoint layer is not defined at ``n_total == 0``, and there is
    nothing to score.
    """
    nn = spec.n_nodes
    if nn == 0:
        z = np.empty(0, dtype=np.float64)
        return z, z.copy(), z.copy()
    params = _CombRows(spec.f0_nodes, spec.alpha, spec.sin_delta, spec.mc_fix)
    ckpt = (os.path.join(spec.parts_dir, spec.ckpt_name)
            if (spec.parts_dir and spec.ckpt_name) else None)
    Fd = chunked_fstat_sweep(
        call_fstat, params, xp=xp, label=spec.label, ckpt=ckpt,
        fingerprint_extra=spec.fingerprint_extra,
    ).reshape(nn, int(spec.lv))
    kb = _to_host(Fd.argmax(axis=1)).astype(int)
    return _to_host(Fd.max(axis=1)), spec.alpha[kb], spec.sin_delta[kb]


def comb_part_path(parts_dir, li, rank) -> str:
    """``<parts>/comb_l{li}_r{rank}.npy`` -- one rank's slice of one level.

    A separate prefix from the per-level PROGRESS files (``comb_nsky{lv}*``)
    on purpose: :func:`run_comb_scan`'s end-of-scan
    ``ckpt_clear(parts_dir, "comb_")`` sweeps only the ``.progress.*``
    suffixes, so these finished partials are cleared by
    :func:`clear_comb_parts` per level instead -- as soon as the head has
    assembled them, rather than at the end of the whole scan.
    """
    return os.path.join(parts_dir, f"comb_l{int(li)}_r{int(rank)}.npy")


def save_comb_part(parts_dir, li, rank, F_max, best_alpha, best_sin_delta):
    """Write one rank's finished reduction as a ``(3, n)`` float64 block.

    ONE file for the three arrays, so the head's read is one open and one
    sha1 -- they are produced together, consumed together, and a partial
    that lost one of the three would be a grid whose sky answers belong to
    other nodes.

    Written through an open file object and fsynced before the rename, for
    the reasons :func:`save_stage_b_part` spells out: this is a cross-NODE
    data path (a worker writes, replies, and the HEAD reads from another
    node), so the barrier is what NFS close-to-open semantics need.
    """
    os.makedirs(parts_dir, exist_ok=True)
    arr = np.ascontiguousarray(
        np.stack([np.asarray(_to_host(F_max), dtype=np.float64),
                  np.asarray(_to_host(best_alpha), dtype=np.float64),
                  np.asarray(_to_host(best_sin_delta), dtype=np.float64)]))
    path = comb_part_path(parts_dir, li, rank)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path, int(arr.shape[1]), hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def load_comb_part(parts_dir, li, rank):
    return np.load(comb_part_path(parts_dir, li, rank), allow_pickle=False)


def assemble_comb_level(parts_dir, li, n_parts, n_nodes, *, sha1s=None):
    """Concatenate one level's per-rank partials in RANK (== node) order.

    :func:`split_box_range` hands rank ``r`` a contiguous ascending node
    range, so rank-ordered concatenation reproduces the whole-level sweep
    exactly. ``sha1s`` (rank -> digest, as the ranks reported them) is
    verified when given, for the reason
    :func:`assemble_stage_b_group`'s is: a partial that changed between the
    reply and the read is a filesystem fault, and silently fitting on it
    would corrupt the epoch with no symptom.

    Returns ``(F_max, best_alpha, best_sin_delta)``, each ``(n_nodes,)``.
    """
    n_nodes = int(n_nodes)
    if sha1s is not None:
        missing = [r for r in range(int(n_parts)) if sha1s.get(r) is None]
        if missing:
            raise RuntimeError(
                f"comb level {li}: sha1s given but missing/None for rank(s) "
                f"{missing} of {n_parts} -- cannot verify those partials, "
                f"refusing to silently skip verification")
    out = np.empty((3, n_nodes), dtype=np.float64)
    off = 0
    for r in range(int(n_parts)):
        arr = np.ascontiguousarray(load_comb_part(parts_dir, li, r))
        if sha1s is not None:
            got = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
            if got != sha1s[r]:
                raise RuntimeError(
                    f"comb partial l{li} r{r} changed under us: reported "
                    f"sha1 {sha1s[r]}, read {got} "
                    f"({comb_part_path(parts_dir, li, r)})")
        if arr.ndim != 2 or arr.shape[0] != 3 or (
                off + int(arr.shape[1]) > n_nodes):
            raise RuntimeError(
                f"comb level {li}: partial r{r} is {tuple(arr.shape)} at node "
                f"offset {off}, which does not fit (3, {n_nodes})")
        out[:, off:off + int(arr.shape[1])] = arr
        off += int(arr.shape[1])
    if off != n_nodes:
        raise RuntimeError(
            f"comb level {li}: assembled {off} node(s) from {n_parts} "
            f"partials, expected {n_nodes}")
    return out[0], out[1], out[2]


def clear_comb_parts(parts_dir, li, n_parts) -> None:
    """Remove one level's partials (and any leftover temp files)."""
    for r in range(int(n_parts)):
        for path in (comb_part_path(parts_dir, li, r),
                     comb_part_path(parts_dir, li, r) + ".tmp"):
            try:
                os.remove(path)
            except OSError:
                pass


def run_comb_scan(call_fstat: Callable, *, xp, Tobs: float, band_edges_hz,
                  f0_lims_hz, mc_lims, cache_path: Optional[str] = None,
                  fingerprint_extra: str = "", comb_runner=None,
                  band_min_F=None, band_skip=None):
    """Dense-in-f0 F-stat comb scan across the sub-band.

    With months of data the F-stat f0 peaks are ~1/Tobs wide -- far too
    narrow for any feasible 4-D tensor grid to land on. But over the same
    stretch ``fdot*Tobs << 1/Tobs``, so Mc is nearly unmeasurable per band
    and is held FIXED, and sky only enters through the Doppler ridge. The
    right scan is therefore dense in f0 (spacing ~ 1/(2*Tobs)) x a small
    spread of sky points, maximized over sky per f0 node -- assembled into
    ONE parameter set per sky level and swept in a single chunked stream.

    ``comb_runner`` replaces each level's kernel stream with
    ``runner(spec, call_fstat, xp=xp) -> (F_max, best_alpha,
    best_sin_delta)`` (``spec`` is a :class:`CombLevelSpec`). ``None`` is
    the serial path, :func:`run_comb_level`, and is byte-identical to the
    historical code. The multi-rank fit passes a runner that splits the
    level by contiguous NODE range across the compute ranks and
    concatenates the partials -- nodes are INDEPENDENT (the per-node
    reduction reads only that node's own sky block), so rank-ordered
    concatenation reproduces the whole-level sweep exactly.

    ``band_skip`` (``(num_sub_bands,)`` bool, or ``None``) takes the shut
    bands out of the SWEEP, not only out of peak selection (user ruling
    2026-10-10; on 9mo job 753 stage A was 22 of the 23 fit minutes and
    swept every one of 996,301 nodes while 471 of 1232 bands were skipped).
    Each level sweeps only its :func:`comb_sweep_mask` nodes; the rest of
    the node grid keeps its place with ``F_max`` / ``best_alpha`` /
    ``best_sin_delta`` = 0, so ``f0_nodes``, the uniform spacing and every
    output shape are exactly what a full sweep returns. A level with no
    swept node is never dispatched. ``None`` -- and an all-False set --
    sweeps every node through the identical calls.

    Returns ``(f0_nodes_mHz, F_max, peaks, extras)``.
    """
    f0_lo = float(f0_lims_hz[0]) * 1e3
    f0_hi = float(f0_lims_hz[-1]) * 1e3
    spacing = float(os.environ.get("FSTAT_F0_SPACING_MHZ", 0.5 / Tobs * 1e3))
    f0_nodes = np.arange(f0_lo, f0_hi + 0.5 * spacing, spacing)
    mc_lims = list(mc_lims or [0.001, 1.0])
    mc_fix = float(os.environ.get(
        "FSTAT_COMB_MC", 0.5 * (float(mc_lims[0]) + float(mc_lims[-1]))))
    n_nodes = len(f0_nodes)

    # Per-node sky count. FIXED if FSTAT_COMB_NSKY is set (back-compat);
    # otherwise ADAPTIVE via :func:`sky_nodes_required`, the law stage B now
    # shares (see that docstring for why it goes as (f0*Tobs*v/c)^2).
    nsky_min = int(os.environ.get("FSTAT_COMB_NSKY_MIN", "16"))
    nsky_max = int(os.environ.get("FSTAT_COMB_NSKY_MAX", "512"))
    _fixed = os.environ.get("FSTAT_COMB_NSKY", "").strip()
    if _fixed:
        nsky_per_node = np.full(n_nodes, int(_fixed), dtype=int)
    else:
        nsky_per_node = sky_nodes_required(
            f0_nodes, Tobs, nmin=nsky_min, nmax=nsky_max)

    levels = np.unique(nsky_per_node)
    logger.info("[comb] %d f0 nodes; sky %s [%d..%d] over %d level(s); "
                "spacing %.3e mHz = %.2f/Tobs; Mc fixed %.3f", n_nodes,
                "FIXED=" + str(int(levels[0])) if len(levels) == 1
                else "ADAPTIVE",
                int(nsky_per_node.min()), int(nsky_per_node.max()),
                len(levels), spacing, spacing / (1e3 / Tobs), mc_fix)

    # THE SHUT BANDS LEAVE THE SWEEP (user ruling 2026-10-10). ``None``
    # with no band_skip, and then every expression below is the historical
    # one. The LEVELS stay those of the full node grid, so a level's index
    # ``li`` -- its partial-file name -- never depends on the skip set.
    swept = comb_sweep_mask(f0_nodes, band_edges_hz, band_skip, spacing)
    n_dropped = 0 if swept is None else int(n_nodes - int(swept.sum()))
    if swept is not None:
        _full_evals = int(nsky_per_node.sum())
        logger.info(
            "[comb] band_skip: %d of %d bands skipped -> %d of %d f0 nodes "
            "dropped from the sweep (%d swept nodes lie in skipped bands, "
            "inside the +-%d-node peak window of a kept band); %d of the "
            "full sweep's %d evals remain",
            int(np.asarray(band_skip, dtype=bool).sum()),
            int(len(band_edges_hz) - 1), n_dropped, n_nodes,
            int((swept & ~_comb_keep_mask(f0_nodes, band_edges_hz,
                                          band_skip)).sum()),
            comb_peak_window(spacing),
            int(nsky_per_node[swept].sum()), _full_evals)

    parts_dir = cache_path.replace(".npz", "_parts") if cache_path else None
    F_max_host = np.zeros(n_nodes)
    best_al_host = np.zeros(n_nodes)
    best_sd_host = np.zeros(n_nodes)
    _lv_idx = {
        int(lv): np.where((nsky_per_node == lv) if swept is None
                          else ((nsky_per_node == lv) & swept))[0]
        for lv in levels}
    # Per-level eval counts upfront so each level header carries the OVERALL
    # comb fraction (the per-sweep lines only know their own level). Counted
    # over the SWEPT nodes, so the percentages track the real work.
    _lv_evals = {int(lv): int(lv) * int(_lv_idx[int(lv)].size)
                 for lv in levels}
    _grand_total = sum(_lv_evals.values())
    total_evals = 0
    for _li, lv in enumerate(levels):
        idx = _lv_idx[int(lv)]
        nodes = f0_nodes[idx]
        nn = len(nodes)
        logger.info(
            "[comb] level %d/%d (nsky=%d): %d nodes x %d sky = %d evals; "
            "comb overall %.1f%% done, %.1f%% after this level",
            _li + 1, len(levels), int(lv), nn, int(lv), _lv_evals[int(lv)],
            100.0 * total_evals / max(_grand_total, 1),
            100.0 * (total_evals + _lv_evals[int(lv)]) / max(_grand_total, 1))
        if nn == 0:
            # Only reachable with a band_skip: every node of this level is
            # in a skipped band. NOT dispatched -- a multi-rank runner would
            # fan out an empty level, every rank would write an empty
            # partial, and the head would assemble nothing. Its nodes keep
            # F_max = 0 like every other unswept node.
            logger.info("[comb] level %d/%d (nsky=%d): every node is in a "
                        "skipped band -- not swept", _li + 1, len(levels),
                        int(lv))
            continue
        al, sd = _sky_grid(int(lv))
        al = np.asarray(al)
        sd = np.asarray(sd)
        # NODE-MAJOR row order (f0 slow, sky fast). Sky-major (f0 fastest)
        # made every sky point sweep the whole band, so batch-sequential
        # scorers with f0-local state -- the sig-het shared-reference cache
        # keeps ONE ~GB block resident -- rebuilt that state once per block
        # PER SKY POINT (cyclic-eviction worst case: ~nsky x n_blocks
        # rebuilds per level, measured 387+ on the first full-band run).
        # Node-major visits each f0 neighborhood exactly once per level.
        # Ordering changes the checkpoint fingerprint, so pre-existing comb
        # progress files restart cleanly rather than resuming misordered.
        # The block is generated per batch (:class:`_CombRows`), not held:
        # the dense array is 1.6 GB at 3-mo / ~81 GB at 23-mo density for
        # nothing but to be read 4096 rows at a time.
        spec = CombLevelSpec(
            li=_li, n_levels=len(levels), lv=int(lv), a=0, b=nn,
            f0_nodes=nodes, alpha=al, sin_delta=sd, mc_fix=mc_fix,
            label=f":comb.nsky{int(lv)}",
            ckpt_name=f"comb_nsky{int(lv)}", parts_dir=parts_dir,
            fingerprint_extra=fingerprint_extra,
        )
        runner = comb_runner if comb_runner is not None else run_comb_level
        F_lv, al_lv, sd_lv = runner(spec, call_fstat, xp=xp)
        F_max_host[idx] = F_lv
        best_al_host[idx] = al_lv
        best_sd_host[idx] = sd_lv
        total_evals += int(lv) * nn
    logger.info("[comb] total %d F-stat evals across %d sky level(s)%s",
                total_evals, len(levels),
                "" if swept is None else
                f" ({n_dropped} of {n_nodes} f0 nodes dropped from the "
                f"sweep in skipped bands; F_max = 0 there)")

    peaks = select_comb_peaks(f0_nodes, xp.asarray(F_max_host), band_edges_hz,
                              spacing, xp, min_F=band_min_F,
                              band_skip=band_skip)
    for f0p, Fp, _, bi in peaks[:10]:
        logger.info("[comb]   %.5f  %10.2f  band %d", f0p, Fp, int(bi))

    _al_top, _sd_top = _sky_grid(int(nsky_per_node.max()))
    if cache_path:
        comb_cache = cache_path.replace(".npz", "_comb.npz")
        os.makedirs(os.path.dirname(comb_cache), exist_ok=True)
        np.savez(comb_cache, f0_nodes_mHz=f0_nodes, F_max=F_max_host,
                 peaks=peaks[:, (0, 1, 3)] if len(peaks) else np.empty((0, 3)),
                 band_edges=np.asarray(band_edges_hz, dtype=float),
                 F_all=F_max_host[None, :], sky_alpha=_al_top,
                 sky_sin_delta=_sd_top, best_alpha=best_al_host,
                 best_sin_delta=best_sd_host, nsky_per_node=nsky_per_node,
                 # WHAT RESIDUAL THIS SCAN WAS SCORED AGAINST. The same
                 # string that salts the in-flight progress files -- epoch,
                 # GB_FSTAT_GB_FREE, and (the reason it is here) the
                 # reference walker. A FINISHED comb is not a checkpoint, so
                 # nothing used to test it: it was reloaded on file
                 # existence alone. See :func:`comb_cache_usable`.
                 fingerprint_extra=fingerprint_extra,
                 # WHICH NODES WERE SCORED, written only when the sweep
                 # dropped some (so a complete sweep's npz is unchanged,
                 # key set included). The zeros elsewhere are "not
                 # swept", not "no signal": comb_cache_usable refuses to
                 # re-select peaks from this file for a skip set that
                 # needs a node it never scored.
                 **({"node_swept": swept} if n_dropped else {}))
        logger.info("[cache] wrote %s", comb_cache)
        # The comb npz is now the durable artifact; drop the per-level
        # progress files so a later knob change can't resurrect stale rows.
        ckpt_clear(parts_dir, "comb_")

    extras = dict(sky_alpha=_al_top, sky_sin_delta=_sd_top,
                  F_all=F_max_host[None, :], best_alpha=best_al_host,
                  best_sin_delta=best_sd_host, mc_fix=mc_fix,
                  spacing=spacing, nsky_per_node=nsky_per_node,
                  n_nodes_dropped=n_dropped)
    return f0_nodes, F_max_host, peaks, extras


def run_stacked_peak_sweep(call_fstat: Callable, f0_los, f0_dxs, mc_ax,
                           alpha_ax, sd_ax, node_shape, *, xp,
                           ckpt: Optional[str] = None,
                           fingerprint_extra: str = "",
                           fdot_axis: bool = False, c_t: float = 0.0):
    """ONE chunked kernel stream over EVERY peak box of EVERY sub-band.

    ``node_shape`` is ``(K, n_f0, n_1, n_alpha, n_sd)``. Rows are assembled
    per chunk from index arithmetic (so the full ``n_total x 9`` array never
    materializes); the kernel sees a single ``FSTAT_BATCH``-chunked stream.

    Axis 2 has two meanings, selected by ``fdot_axis``:

    * ``False`` (default, byte-identical to the historical path) -- ``mc_ax``
      is a CHIRP MASS axis and ``fdot = fdot_gr(f0, Mc)``. That is the
      ``r = 0`` manifold: the sweep scores only GR-driven chirps, and
      negative ``fdot`` is unreachable.
    * ``True`` -- ``mc_ax`` is an ``fdot`` axis in Hz/s, scored directly.
      Axis 0 then carries ``f_mid`` rather than ``f0``, and the physical
      ``f0 = f_mid - c_t * fdot`` is derived per row. Because ``fdot`` is a
      free axis and ``c_t`` a constant, that is an EXACT unit-determinant
      shear -- it decorrelates the ``(f0, fdot)`` metric at zero cost to
      the measure, which is what licenses the coarser aligned node
      spacing (see :func:`fdot_coherence_width`).
    """
    from gbgpu.utils.utility import get_fdot

    from lisatools.sampling.fstat_proposal import compute_fstat, fstat_knob

    n_total = int(np.prod(node_shape))
    batch = fstat_knob("FSTAT_BATCH", int)
    # THE AXIS FLAG IS PART OF THE FINGERPRINT. The axis VALUES already
    # differ between the two meanings, but relying on that is not enough:
    # a resumed fit whose grid_basis disagrees with the checkpoint would
    # reuse F values scored at entirely different physical templates, and
    # nothing downstream would notice. Same precedent as GB_FSTAT_GB_FREE.
    fp = (ckpt_fingerprint(np.asarray(f0_los), np.asarray(f0_dxs),
                           np.asarray(mc_ax), np.asarray(alpha_ax),
                           np.asarray(sd_ax),
                           extra=(str(node_shape) + fingerprint_extra
                                  + f"|fdot_axis={int(bool(fdot_axis))}"
                                  + f"|c_t={float(c_t):.6e}"))
          if ckpt else "")
    f0_los_d = xp.asarray(f0_los)
    f0_dxs_d = xp.asarray(f0_dxs)
    mc_d = xp.asarray(mc_ax)
    al_d = xp.asarray(alpha_ax)
    sd_d = xp.asarray(sd_ax)
    F_flat = xp.empty(n_total, dtype=xp.float64)
    F_prev, start = ckpt_resume(ckpt, n_total, fp)
    if F_prev is not None and start > 0:
        F_flat[:start] = xp.asarray(F_prev[:start])
    ckpt_every = ckpt_secs()
    flushed = start
    if ckpt and start == 0:
        ckpt_save(ckpt, np.empty(0), 0, n_total, fp, start=0)
        logger.info("[ckpt] progress file: %s.progress.npz (every %.0fs "
                    "+ on interrupt)", ckpt, ckpt_every)
    t0 = last = last_ckpt = time.time()
    done = start
    try:
        for s in range(start, n_total, batch):
            e = min(s + batch, n_total)
            k, i0, i1, i2, i3 = xp.unravel_index(xp.arange(s, e), node_shape)
            pr = xp.zeros((e - s, 9), dtype=xp.float64)
            pr[:, 0] = 1e-22
            if fdot_axis:
                # axis 2 IS fdot; axis 0 carries f_mid, so the physical f0
                # is sheared off it. Order matters: fdot first.
                pr[:, 2] = mc_d[i1]
                pr[:, 1] = ((f0_los_d[k] + i0 * f0_dxs_d[k]) * 1e-3
                            - float(c_t) * pr[:, 2])
            else:
                pr[:, 1] = (f0_los_d[k] + i0 * f0_dxs_d[k]) * 1e-3
                pr[:, 2] = xp.asarray(get_fdot(f=pr[:, 1], Mc=mc_d[i1]))
            pr[:, 5] = 0.5 * np.pi
            pr[:, 7] = al_d[i2]
            pr[:, 8] = xp.arcsin(sd_d[i3])
            N_arr, M_up = call_fstat(pr)
            F_flat[s:e] = compute_fstat(xp.asarray(N_arr), xp.asarray(M_up))
            done = e
            if time.time() - last > 30:
                last = time.time()
                _el = time.time() - t0
                _rate = (e - start) / max(_el, 1e-9)
                logger.info(
                    "[stageB] %d/%d evals (%.1f%%) | %.0f evals/s | ETA %s "
                    "| elapsed %s", e, n_total, 100.0 * e / n_total, _rate,
                    _fmt_secs((n_total - e) / max(_rate, 1e-9)),
                    _fmt_secs(_el))
            if ckpt and e < n_total and time.time() - last_ckpt > ckpt_every:
                last_ckpt = time.time()
                ckpt_save(ckpt, _to_host(F_flat[flushed:e]), e, n_total, fp,
                          start=flushed)
                flushed = e
                logger.info("[ckpt] saved %d/%d rows", e, n_total)
    except BaseException:
        if ckpt and done > flushed:
            ckpt_save(ckpt, _to_host(F_flat[flushed:done]), done, n_total, fp,
                      start=flushed)
            flushed = done
            logger.info("[ckpt] interrupt: saved %d/%d rows", done, n_total)
        raise
    if ckpt and start < n_total:
        ckpt_save(ckpt, _to_host(F_flat[flushed:n_total]), n_total, n_total,
                  fp, start=flushed)
    logger.info("[stageB] %d evals in one chunked stream (batch=%d, %.0fs%s)",
                n_total, batch, time.time() - t0,
                f", resumed at {start}" if start else "")
    return F_flat.reshape(node_shape)


@dataclasses.dataclass(frozen=True)
class StageBGroupSpec:
    """Everything one stage-B Mc group's sweep needs, as host arrays.

    The unit the parallel fit ships and slices. ``a``/``b`` are ABSOLUTE box
    indices into the f0-sorted global box order, so a sub-range is addressed
    in the same coordinates the assembled grid is concatenated in (box is
    the SLOWEST axis of ``node_shape``, :func:`run_stacked_peak_sweep`).

    Everything except ``f0_los``/``f0_dxs``/``node_shape[0]`` is group-wide
    and is NEVER sliced: the Mc/alpha/sin-delta axes, the basis flag and the
    shear coefficient define what a row MEANS, and a rank that sliced them
    would score a different physical template while reporting the same box.
    """

    gi: int
    n_groups: int
    a: int
    b: int
    f0_los: np.ndarray
    f0_dxs: np.ndarray
    mc_ax: np.ndarray
    alpha_ax: np.ndarray
    sd_ax: np.ndarray
    node_shape: tuple
    ckpt_name: Optional[str]
    parts_dir: Optional[str]
    fingerprint_extra: str
    fdot_axis: bool
    c_t: float

    @property
    def n_boxes(self) -> int:
        return int(self.b) - int(self.a)

    def sub_range(self, a2, b2, *, ckpt_name=None) -> "StageBGroupSpec":
        """This group restricted to ABSOLUTE boxes ``[a2, b2)``.

        ``ckpt_name`` must differ per rank: the sweep's fingerprint hashes
        the SLICED inputs plus ``node_shape``, so two ranks' checkpoints are
        already mutually invalid -- but they must not collide on one path.
        """
        a2, b2 = int(a2), int(b2)
        if not (self.a <= a2 <= b2 <= self.b):
            raise ValueError(
                f"sub_range({a2}, {b2}) is outside group {self.gi}'s "
                f"box range [{self.a}, {self.b})")
        i0, i1 = a2 - int(self.a), b2 - int(self.a)
        return dataclasses.replace(
            self,
            a=a2,
            b=b2,
            f0_los=np.ascontiguousarray(self.f0_los[i0:i1]),
            f0_dxs=np.ascontiguousarray(self.f0_dxs[i0:i1]),
            node_shape=(i1 - i0,) + tuple(self.node_shape[1:]),
            ckpt_name=(self.ckpt_name if ckpt_name is None else ckpt_name),
        )


def run_stage_b_group(spec: StageBGroupSpec, call_fstat: Callable, *, xp):
    """Sweep ONE stage-B group (or one rank's box range of it).

    The single entry point both the serial fit and every compute rank use,
    so a split can never diverge from the serial code by construction. An
    EMPTY range (more ranks than boxes) short-circuits: the sweep's
    checkpoint layer is not defined at ``n_total == 0``, and there is
    nothing to score.
    """
    if int(spec.node_shape[0]) == 0:
        return xp.empty(tuple(spec.node_shape), dtype=xp.float64)
    ckpt = (os.path.join(spec.parts_dir, spec.ckpt_name)
            if (spec.parts_dir and spec.ckpt_name) else None)
    return run_stacked_peak_sweep(
        call_fstat, spec.f0_los, spec.f0_dxs, spec.mc_ax, spec.alpha_ax,
        spec.sd_ax, spec.node_shape, xp=xp, ckpt=ckpt,
        fingerprint_extra=spec.fingerprint_extra,
        fdot_axis=spec.fdot_axis, c_t=spec.c_t)


def write_stacked_npz(stacked_path, *, grids_g, mc_ax_g, f0_los, f0_dxs,
                      alpha_ax, sd_ax, grid_basis, grid_c_t, peaks,
                      band_idx, band_edges_mHz, band_edges_hz,
                      group_sizes=None, alpha_ax_g=None, sd_ax_g=None):
    """Write the stage-B cache. ``group_sizes=None`` = the LEGACY 1-group keys.

    ``alpha_ax_g`` / ``sd_ax_g`` are the PER-GROUP sky axes (2026-09-24,
    f0-adaptive sky). When given they are written as ``alpha_ax_g{gi}`` /
    ``sin_delta_ax_g{gi}`` beside the flat ``alpha_ax`` / ``sin_delta_ax``,
    which stay for readers that predate the change -- those flat keys then
    describe GROUP 0 only, so a reader that needs the real axis of group
    ``gi`` must prefer the suffixed key when it exists. The loader
    (``fstat_proposal.stacked_from_cache``) does exactly that.

    Factored out of :func:`run_stacked_stage_b` so the parallel fit's head,
    which assembles each group from per-rank partials, writes through the
    SAME code -- the loader (``fstat_proposal.stacked_from_cache``)
    dispatches on ``"logp_grids" in keys``, so the two formats are not
    interchangeable and must never be produced by two separate writers.

    The parallel fit feeds this function rank-ASSEMBLED concatenations
    rather than whole-group sweeps, so a short or long partial (a dropped
    rank, an off-by-one in :func:`split_box_range`, ...) would otherwise
    write a cache whose box axis silently disagrees with ``f0_los[a:b]`` --
    wrong templates proposed forever, with no error anywhere. Validate the
    shapes before touching disk.
    """
    if len(grids_g) != len(mc_ax_g):
        raise ValueError(
            f"write_stacked_npz: {len(grids_g)} grid(s) but {len(mc_ax_g)} "
            f"Mc axis/axes -- exactly one Mc axis per group is required")
    if group_sizes is not None:
        if len(grids_g) != len(group_sizes):
            raise ValueError(
                f"write_stacked_npz: {len(grids_g)} grid(s) but "
                f"{len(group_sizes)} group_sizes entries")
        for gi, (grid, want) in enumerate(zip(grids_g, group_sizes)):
            # ``grid`` may be a device (cupy) array here -- read ``.shape``
            # directly rather than routing through ``np.asarray``/``_to_host``,
            # which would force a device->host copy of the WHOLE grid (up to
            # ~650 MB) just to look at one integer, or (for cupy specifically)
            # raise outright: cupy's ``__array__`` refuses implicit
            # conversion, so ``np.asarray(cupy_array)`` is not even a slow
            # path here, it is a crash on every GPU production run.
            got = int(grid.shape[0])
            want = int(want)
            if got != want:
                raise ValueError(
                    f"write_stacked_npz: group {gi} grid has {got} box(es) "
                    f"but group_sizes[{gi}] says {want} -- a short or long "
                    f"partial would write a cache whose box axis silently "
                    f"disagrees with f0_los[a:b]")
    total_boxes = sum(int(g.shape[0]) for g in grids_g)
    if total_boxes != len(f0_los):
        raise ValueError(
            f"write_stacked_npz: {total_boxes} box(es) across all groups "
            f"but f0_los has {len(f0_los)} entries")
    if total_boxes != len(f0_dxs):
        raise ValueError(
            f"write_stacked_npz: {total_boxes} box(es) across all groups "
            f"but f0_dxs has {len(f0_dxs)} entries")

    os.makedirs(os.path.dirname(stacked_path), exist_ok=True)
    common = dict(
        f0_los=f0_los, f0_dxs=f0_dxs, alpha_ax=alpha_ax, sin_delta_ax=sd_ax,
        # THE BASIS IS PART OF THE CACHE. Axis 2's VALUES differ between the
        # two meanings, but a consumer that reads them as chirp masses when
        # they are Hz/s gets no error at all -- just births at absurd
        # parameters. Stamp it, and refuse a mismatch on load.
        grid_basis=grid_basis, grid_c_t=float(grid_c_t),
        # THE PEAK THRESHOLD IS PART OF THE CACHE, for the same reason the
        # basis is: the stage-B load path short-circuits selection entirely,
        # so a cache fitted at one FSTAT_PEAK_MIN_SNR and loaded under
        # another keeps the old peak list with no error at all. Stamped
        # here, checked in _check_cached_peak_threshold on load.
        peak_min_F=float(_peak_min_F_now()),
        peak_f0_mHz=peaks[:, 0], peak_F=peaks[:, 1], band_idx=band_idx,
        band_f0_lo=band_edges_mHz[band_idx],
        band_f0_hi=band_edges_mHz[band_idx + 1],
        band_edges=np.asarray(band_edges_hz, dtype=float),
    )
    if group_sizes is None:
        np.savez(stacked_path, logp_grids=_to_host(grids_g[0]),
                 mc_ax=mc_ax_g[0], **common)
        return
    if alpha_ax_g is not None and len(alpha_ax_g) != len(grids_g):
        raise ValueError(
            f"write_stacked_npz: {len(grids_g)} grid(s) but "
            f"{len(alpha_ax_g)} alpha axis/axes -- one sky axis per group "
            f"is required once the sky is f0-adaptive, or the cache would "
            f"describe a different sky grid than the one that was swept")
    # Per-group sky keys are written ONLY when the groups actually differ.
    # When every group shares one sky grid -- any pinned-sky fit, and every
    # configuration that predates the adaptive sky -- the flat alpha_ax /
    # sin_delta_ax already say everything, and emitting redundant suffixed
    # copies would change the key set of a cache whose content is unchanged,
    # invalidating the stage-B golden files and every reader that checks the
    # layout. So: identical sky across groups => byte-identical key set.
    _sky_per_group = False
    if alpha_ax_g is not None:
        for gi in range(len(grids_g)):
            _al = np.asarray(alpha_ax_g[gi], dtype=float)
            _sd = np.asarray(sd_ax_g[gi], dtype=float)
            _want = tuple(int(n) for n in grids_g[gi].shape[3:5])
            if (len(_al), len(_sd)) != _want:
                raise ValueError(
                    f"write_stacked_npz: group {gi} grid has sky shape "
                    f"{_want} but the axes given are {len(_al)}x{len(_sd)} "
                    f"-- births would be placed at sky angles the grid was "
                    f"never scored at")
            if not (np.array_equal(_al, np.asarray(alpha_ax, dtype=float))
                    and np.array_equal(_sd, np.asarray(sd_ax, dtype=float))):
                _sky_per_group = True
    group_arrays = {}
    for gi in range(len(grids_g)):
        group_arrays[f"logp_grids_g{gi}"] = _to_host(grids_g[gi])
        group_arrays[f"mc_ax_g{gi}"] = mc_ax_g[gi]
        if _sky_per_group:
            group_arrays[f"alpha_ax_g{gi}"] = np.asarray(alpha_ax_g[gi],
                                                        dtype=float)
            group_arrays[f"sin_delta_ax_g{gi}"] = np.asarray(sd_ax_g[gi],
                                                            dtype=float)
    np.savez(stacked_path, group_sizes=np.asarray(group_sizes, dtype=int),
             **common, **group_arrays)


def union_stacked_npz(slot_paths, out_path):
    """Union several walkers' stage-B grids into ONE birth cache.

    Each input is a complete stage-B npz fitted against a DIFFERENT walker's
    residual. The union keeps every slot's groups as separate MIXTURE
    COMPONENTS -- it does not merge or re-bin anything -- and writes through
    the same key layout :func:`write_stacked_npz` produces, so
    ``fstat_proposal.stacked_from_cache`` loads it unchanged.

    WHY COMPONENTS AND NOT ONE STACK. Under ``FSTAT_FDOT_AXIS`` a group's
    axis-2 bounds are the INTERSECTION of its boxes' feasible fdot ranges, a
    function of the group's f0 extent -- so two boxes from different walkers
    with the same ``n_Mc`` have DIFFERENT fdot axes and cannot share a
    ``StackedFStatProposal4D``. Keeping each slot's groups intact sidesteps
    that, and it also keeps ``_overlap_depth`` per component instead of
    multiplying the f0-overlap gather width by W.

    NO DEDUPLICATION, deliberately. Two walkers' boxes on the same comb node
    carry different grids (both the residual and the inverse-PSD row differ
    per walker), so any dedup would be lossy rather than a de-duplication --
    and overlapping boxes are priced EXACTLY by
    ``StackedFStatProposal4D.logpdf``, which sums over every box whose f0
    interval contains the point. The standing "never use f0 proximity as a
    duplicate test" rule is satisfied vacuously: nothing here tests identity.

    Refuses on any disagreement in the arrays that must be shared (the sky
    axes, the grid basis, the band edges) rather than silently preferring
    one slot's -- a basis mismatch produces births at absurd parameters with
    no error anywhere.
    """
    slot_paths = [str(p) for p in slot_paths]
    if not slot_paths:
        raise ValueError("union_stacked_npz: no slot paths given")
    files = [np.load(p, allow_pickle=False) for p in slot_paths]
    try:
        first = files[0]
        if "group_sizes" not in first.files:
            raise ValueError(
                "union_stacked_npz: %s is a LEGACY single-group cache; the "
                "union needs the grouped format (group_sizes)" % slot_paths[0])

        # -- arrays every slot must agree on --------------------------------
        # SKY AXES. With one shared sky grid per cache they had to agree
        # across slots, because the format could only hold one. Since the sky
        # became f0-adaptive (2026-09-24) each GROUP carries its own axes and
        # the union keeps every slot's groups as separate, self-describing
        # mixture components -- so there is nothing left to reconcile across
        # slots, and requiring agreement would be wrong twice over: slots
        # legitimately have DIFFERENT group counts (each walker selects its
        # own peaks, so the Mc ladder lands differently), and two slots whose
        # group 3 covers different frequencies SHOULD carry different sky
        # grids. The historical cross-slot check is kept exactly as it was
        # for the uniform-sky case, which is what every pinned-sky and
        # pre-adaptive cache is.
        _per_group_sky = ["alpha_ax_g0" in f.files for f in files]
        _shared = ("band_edges",) if any(_per_group_sky) else (
            "alpha_ax", "sin_delta_ax", "band_edges")
        for key in _shared:
            ref = np.asarray(first[key])
            for p, f in zip(slot_paths[1:], files[1:]):
                got = np.asarray(f[key])
                if got.shape != ref.shape or not np.array_equal(got, ref):
                    raise ValueError(
                        f"union_stacked_npz: {key} differs between "
                        f"{slot_paths[0]} and {p}; the slots were not swept "
                        "on the same grid")
        basis = first["grid_basis"]
        c_t = float(first["grid_c_t"])
        for p, f in zip(slot_paths[1:], files[1:]):
            if str(f["grid_basis"]) != str(basis) or float(f["grid_c_t"]) != c_t:
                raise ValueError(
                    f"union_stacked_npz: grid basis/c_t differs between "
                    f"{slot_paths[0]} and {p} -- axis 2 would mean two "
                    "different things in one cache")

        # -- concatenate the per-box (global) arrays ------------------------
        per_box = ("f0_los", "f0_dxs", "peak_f0_mHz", "peak_F", "band_idx",
                   "band_f0_lo", "band_f0_hi")
        out = {k: np.concatenate([np.asarray(f[k]) for f in files])
               for k in per_box}
        out["alpha_ax"] = np.asarray(first["alpha_ax"])
        out["sin_delta_ax"] = np.asarray(first["sin_delta_ax"])
        out["band_edges"] = np.asarray(first["band_edges"])
        out["grid_basis"] = basis
        out["grid_c_t"] = c_t
        #: which slot each box came from -- the provenance the cell census
        #: and any per-walker diagnostic needs, and cheap to carry
        out["box_slot"] = np.concatenate([
            np.full(int(np.sum(np.asarray(f["group_sizes"]))), i, dtype=np.int32)
            for i, f in enumerate(files)])

        # -- groups, renumbered across the slots ----------------------------
        sizes, gi_out = [], 0
        for f in files:
            gs = [int(x) for x in np.asarray(f["group_sizes"])]
            for gi in range(len(gs)):
                out[f"logp_grids_g{gi_out}"] = f[f"logp_grids_g{gi}"]
                out[f"mc_ax_g{gi_out}"] = f[f"mc_ax_g{gi}"]
                # Per-group sky axes through the renumbering. If ANY slot
                # carries them, EVERY output group gets them -- a uniform-sky
                # slot contributing its flat axes explicitly. Leaving those
                # groups unsuffixed would send the loader to the OUTPUT's flat
                # key, which is slot 0's, and silently rescale their sky.
                if any(_per_group_sky):
                    out[f"alpha_ax_g{gi_out}"] = (
                        f[f"alpha_ax_g{gi}"] if f"alpha_ax_g{gi}" in f.files
                        else f["alpha_ax"])
                    out[f"sin_delta_ax_g{gi_out}"] = (
                        f[f"sin_delta_ax_g{gi}"]
                        if f"sin_delta_ax_g{gi}" in f.files
                        else f["sin_delta_ax"])
                gi_out += 1
            sizes.extend(gs)
        out["group_sizes"] = np.asarray(sizes, dtype=int)

        total = int(sum(sizes))
        if total != len(out["f0_los"]):
            raise ValueError(
                f"union_stacked_npz: {total} box(es) across {len(sizes)} "
                f"group(s) but f0_los has {len(out['f0_los'])} entries")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, **out)
        return total, len(sizes)
    finally:
        for f in files:
            f.close()


def union_cell_keys(band_idx, peak_f0_mHz):
    """Exact ``(band, node)`` keys for the union's peaks -- NOT f0 proximity.

    ``FSTAT_PEAK_WEIGHT_CELLS`` equalizes weight per f0 cell by COUNTING
    boxes in it. Across a union that inverts: a cell holding W copies of one
    contested peak reads as "W peaks" and gets DOWN-weighted -- penalising
    exactly the peaks the union exists to keep. The census must count
    UNIQUE peaks instead.

    The key is exact, not approximate. The comb node grid is
    ``arange(f0_lo, f0_hi + 0.5*spacing, spacing)``, a deterministic function
    of ``Tobs`` and the band edges and therefore IDENTICAL across walkers --
    so the same node yields the same float64 bits in every slot, and
    ``(band_idx, peak_f0_mHz)`` compares equal by value. That is why this is
    not the forbidden "f0 proximity" test: nothing here has a tolerance.

    Returns an integer label per box, equal iff the boxes sit on the same
    comb node of the same band.
    """
    band_idx = np.asarray(band_idx)
    f0 = np.asarray(peak_f0_mHz)
    if band_idx.shape != f0.shape:
        raise ValueError(
            f"union_cell_keys: band_idx has {band_idx.shape} entries but "
            f"peak_f0_mHz has {f0.shape}")
    pairs = np.stack([band_idx.astype(np.int64), f0.view(np.int64)], axis=1)
    _uniq, labels = np.unique(pairs, axis=0, return_inverse=True)
    return labels.astype(np.int64)


def split_box_range(a, b, n_parts):
    """Split boxes ``[a, b)`` into ``n_parts`` CONTIGUOUS, near-equal ranges.

    A pure function of ``(a, b, n_parts)`` -- that is load-bearing: a resume
    must reproduce the same rank -> range map, or a rank would find another
    rank's checkpoint under its own name (the fingerprint would reject it and
    the group would silently restart, which is safe but wastes the sweep).

    CONTIGUITY, not interleaving, is also load-bearing: the sig-het F-stat
    keeps ONE reference block resident and rebuilds it on f0 boundary
    crossings, so an f0-contiguous range divides the block builds
    proportionally, while an interleaved one would rebuild every block on
    every rank (measured 4,600 rebuilds ~ 350 s in the F-ordered-box
    incident, see run_stacked_stage_b's f0-sort comment).

    The first ``(b - a) % n_parts`` ranges get one extra box. With fewer
    boxes than parts the tail ranges come back EMPTY (zero width), which
    :func:`run_stage_b_group` short-circuits.
    """
    a, b, n = int(a), int(b), int(n_parts)
    if n <= 0:
        raise ValueError(f"n_parts must be positive, got {n_parts!r}")
    total = max(b - a, 0)
    base, rem = divmod(total, n)
    out, start = [], a
    for i in range(n):
        width = base + (1 if i < rem else 0)
        out.append((start, start + width))
        start += width
    return out


def stage_b_part_path(parts_dir, gi, rank) -> str:
    """``<parts>/stageb_g{gi}_r{rank}.npy`` -- one rank's slice of one group.

    The ``stageb`` prefix is deliberate: the existing
    ``ckpt_clear(_parts, "stageb")`` at the end of a successful stage B
    already removes the per-rank PROGRESS files by prefix, and these
    partials are cleared by :func:`clear_stage_b_parts` alongside them.
    """
    return os.path.join(parts_dir, f"stageb_g{int(gi)}_r{int(rank)}.npy")


def save_stage_b_part(parts_dir, gi, rank, grid):
    """Write one rank's finished slice as raw float64; return ``(path, n_rows, sha1)``.

    ``np.save`` through an open file object, NOT a path: given a path it
    appends ``.npy`` to whatever it is handed, which would turn the
    write-then-rename temp name into ``....npy.tmp.npy``. The rename is what
    makes the head's read of a partial atomic on a shared filesystem.

    FLUSHED AND FSYNCED BEFORE THE RENAME. This is the branch's one
    cross-NODE data path -- a worker writes here, replies, and the HEAD reads
    the file from another node -- so the usual "the close is the barrier"
    reasoning leans entirely on NFS close-to-open semantics. A bad read is
    loud rather than silent (a short file raises out of ``np.load``, stale
    content is caught by the sha1 ``assemble_stage_b_group`` re-checks), but
    what it costs is a discarded stage-B wall, and the barrier is two lines.
    """
    os.makedirs(parts_dir, exist_ok=True)
    arr = np.ascontiguousarray(_to_host(grid), dtype=np.float64)
    path = stage_b_part_path(parts_dir, gi, rank)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path, int(arr.shape[0]), hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def load_stage_b_part(parts_dir, gi, rank):
    return np.load(stage_b_part_path(parts_dir, gi, rank), allow_pickle=False)


def assemble_stage_b_group(parts_dir, gi, n_parts, node_shape, *, xp,
                           sha1s=None):
    """Concatenate one group's per-rank partials in RANK (== box) order.

    Box is the slowest axis of ``node_shape``
    (:func:`run_stacked_peak_sweep`), and :func:`split_box_range` hands rank
    ``r`` a contiguous ascending range, so rank-ordered concatenation on
    axis 0 reproduces the whole-group sweep exactly. ``sha1s`` (rank ->
    digest, as the ranks reported them) is verified when given: a partial
    that changed between the reply and the read is a filesystem fault, and
    silently fitting on it would corrupt the epoch with no symptom.

    If ``sha1s`` is passed at all, EVERY rank ``0..n_parts-1`` must carry a
    non-``None`` digest in it. A rank missing from the dict (a lost or
    malformed MPI reply) is exactly the failure this check exists to catch
    -- silently skipping verification for that one rank would defeat the
    whole point, so it raises instead of falling back to "unverified".

    Each partial is read STRAIGHT INTO its slice of one preallocated output
    array. A ``parts`` list plus ``np.concatenate`` would hold the whole
    group twice on the host at the moment of the concat -- ~1.3 GB transient
    for the 653 MB production group 1, on top of the device copy -- for no
    gain: the destination is contiguous and the order is known up front.
    """
    node_shape = tuple(int(v) for v in node_shape)
    if sha1s is not None:
        missing = [r for r in range(int(n_parts)) if sha1s.get(r) is None]
        if missing:
            raise RuntimeError(
                f"stage-B group {gi}: sha1s given but missing/None for "
                f"rank(s) {missing} of {n_parts} -- cannot verify those "
                f"partials, refusing to silently skip verification")
    out = np.empty(node_shape, dtype=np.float64)
    off = 0
    for r in range(int(n_parts)):
        arr = np.ascontiguousarray(load_stage_b_part(parts_dir, gi, r))
        if sha1s is not None:
            got = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
            if got != sha1s[r]:
                raise RuntimeError(
                    f"stage-B partial g{gi} r{r} changed under us: reported "
                    f"sha1 {sha1s[r]}, read {got} "
                    f"({stage_b_part_path(parts_dir, gi, r)})")
        if tuple(arr.shape[1:]) != node_shape[1:] or (
                off + int(arr.shape[0]) > node_shape[0]):
            raise RuntimeError(
                f"stage-B group {gi}: partial r{r} is {tuple(arr.shape)} at "
                f"box offset {off}, which does not fit {node_shape}")
        out[off:off + int(arr.shape[0])] = arr
        off += int(arr.shape[0])
    if off != node_shape[0]:
        raise RuntimeError(
            f"stage-B group {gi}: assembled {off} box(es) from {n_parts} "
            f"partials, expected {node_shape[0]} ({node_shape})")
    return xp.asarray(out)


def clear_stage_b_parts(parts_dir, gi, n_parts) -> None:
    """Remove one group's partials (and any leftover temp files)."""
    for r in range(int(n_parts)):
        for path in (stage_b_part_path(parts_dir, gi, r),
                     stage_b_part_path(parts_dir, gi, r) + ".tmp"):
            try:
                os.remove(path)
            except OSError:
                pass


def mc_ladder_levels(n_req):
    """Doubling-ladder quantization of per-box Mc-node requirements.

    Levels come from the ladder ``{3, 6, 12, 24, 48, 96}`` (smallest level
    >= the box's requirement), then a running max keeps them MONOTONE in
    the f0-sorted box order so the resulting groups are contiguous. The
    ladder only PARTITIONS boxes into groups: each group's actual axis
    density is its max RAW requirement, so a single-group fit reproduces
    the historical single-stack node count exactly, and per-box waste is
    bounded by the ladder ratio (< 2x) instead of the full-band f0^{11/3}
    span (~30x at 22 mHz over the low band).
    """
    out = np.empty(len(n_req), dtype=int)
    prev = 0
    for i, r in enumerate(np.asarray(n_req, dtype=int)):
        L = 3
        while L < r:
            L *= 2
        prev = max(prev, min(L, 96))
        out[i] = prev
    return out


def run_stacked_stage_b(call_fstat: Callable, peaks, *, xp, Tobs: float,
                        band_edges_hz, mc_lims, ratio_max=None,
                        cache_path: Optional[str] = None,
                        fingerprint_extra: str = "", epoch=None,
                        sweep_runner=None):
    """Stage B: clamped boxes -> batched sweep(s) -> stacked grids.

    Assemble (host, cheap): every selected peak's 4-D box with its f0 range
    CLAMPED to its sub-band edges (unclamped boxes would propose f0 the
    consumer's per-cell gate rejects). Sweep: one chunked kernel stream per
    Mc GROUP (see :func:`mc_ladder_levels` -- the banded-Mc refactor, user
    ruling 2026-08-26; a single group is the historical one-stream fit).
    Build: one :class:`StackedFStatProposal4D` per group -- a single group
    returns it directly (bit-identical to the historical path, legacy npz
    format included); several return a
    :class:`GroupedStackedFStatProposal` (exact mixture equivalence) and
    the grouped npz format. ``FSTAT_MC_GROUPING=0`` restores the one
    max-f0-sized stack.

    ``sweep_runner`` replaces the per-group kernel stream with
    ``runner(spec, call_fstat, xp=xp) -> grid`` (``spec`` is a
    :class:`StageBGroupSpec`). ``None`` is the serial path,
    :func:`run_stage_b_group`, and is byte-identical to the historical
    code. The multi-rank fit passes a runner that splits the group by
    contiguous box range across the compute ranks and concatenates the
    partials -- box is the SLOWEST axis, so concatenation on axis 0 in box
    order reproduces the whole-group sweep exactly.

    Returns the live proposal (or ``None`` if no peaks).
    """
    from lisatools.sampling.fstat_proposal import (
        GroupedStackedFStatProposal,
        StackedFStatProposal4D,
        fstat_knob,
        fstat_n_axis,
        fstat_n_f0,
        fstat_n_mc,
    )

    if len(peaks) == 0:
        logger.warning("[stageB] no peaks selected; nothing to fit.")
        return None

    half_f0 = fstat_knob("FSTAT_PEAK_HALF_MHZ", float)
    n_f0 = fstat_n_f0(2.0 * half_f0, float(Tobs))
    n_alpha = fstat_n_axis("FSTAT_N_ALPHA")
    n_sd = fstat_n_axis("FSTAT_N_SINDELTA")
    n_fit_env = os.environ.get("FSTAT_PEAKS_TO_FIT", "").strip()
    if n_fit_env:
        peaks = peaks[: int(n_fit_env)]
        logger.info("[stageB] FSTAT_PEAKS_TO_FIT caps the fit to the top %d "
                    "peaks (of the full selection).", len(peaks))

    band_edges_mHz = np.asarray(band_edges_hz, dtype=float) * 1e3
    f0_los, f0_dxs, keep = [], [], []
    for row in peaks:
        f0p, bi = float(row[0]), int(row[3])
        lo = max(f0p - half_f0, band_edges_mHz[bi])
        hi = min(f0p + half_f0, band_edges_mHz[bi + 1])
        if not hi - lo > 0:
            logger.warning("[stageB] degenerate clamped box at f0=%.5f mHz "
                           "(band %d); skipped.", f0p, bi)
            continue
        f0_los.append(lo)
        f0_dxs.append((hi - lo) / (n_f0 - 1))
        keep.append(row)
    if not keep:
        logger.warning("[stageB] every peak box was degenerate; nothing to fit.")
        return None
    peaks = np.asarray(keep)
    f0_los = np.asarray(f0_los)
    f0_dxs = np.asarray(f0_dxs)
    # f0-SORT the boxes: the sweep runs them in array order, and a scorer
    # with f0-local state (the sig-het shared-reference cache) rebuilds on
    # every jump -- the F-ordered box list measured 4,600+ block rebuilds
    # (~350 s) on the first real-data fit. Every per-box output (grids,
    # weights, band ids, npz fields) is a parallel array, so a consistent
    # permutation is invisible to consumers. Applied AFTER the
    # FSTAT_PEAKS_TO_FIT cap, which must keep selecting the top-F peaks.
    _order = np.argsort(f0_los, kind="stable")
    peaks = peaks[_order]
    f0_los = f0_los[_order]
    f0_dxs = f0_dxs[_order]
    K = len(peaks)
    band_idx = peaks[:, 3].astype(int)

    mc_lims = list(mc_lims or [0.001, 1.0])
    mc_range = (max(float(mc_lims[0]), fstat_knob("FSTAT_MC_MIN", float)),
                float(mc_lims[-1]))
    # BANDED Mc density (user ruling 2026-08-26): the auto criterion is
    # evaluated PER BOX (the fdot span scales as f0^{11/3}), boxes are
    # grouped on the requirement ladder (contiguous in the f0-sorted
    # order since the requirement is monotone in f0), and each group
    # sweeps its own rectangular stack sized by ITS max raw requirement.
    # One group -- every confined probe, a pinned FSTAT_N_MC, or
    # FSTAT_MC_GROUPING=0 -- is bit-identical to the historical
    # single-stack path (n_Mc = the max-f0 requirement, legacy cache).
    # FSTAT_FDOT_AXIS: replace the Mc axis with a LINEAR fdot axis in Hz/s
    # (user ruling: fit in (f0, fdot_total), convert after). The Mc axis is
    # already an fdot axis -- rows are scored at fdot = fdot_gr(f0, Mc), the
    # r = 0 manifold -- so this changes what axis 2 MEANS, not the grid's
    # shape or the machinery around it. Three defects go at once: the axis
    # reaches NEGATIVE fdot (unreachable today, against 40% of low-f and 21%
    # of high-f v7 leaves), spacing becomes uniform in the coordinate
    # StackedFStatProposal4D assumes is uniform (fdot ~ Mc^(5/3) is not),
    # and coverage goes from 10% of the prior fdot range to all of it.
    # Measured node cost at 90 d: 6.5 mHz 3->3, 10.2 6->5, 15.6 27->20,
    # 20.38 71->53 -- FEWER nodes over 10x the range, because the aligned
    # coherence width is 13.4x coarser.
    _fdot_axis = fdot_axis_on()
    _rm_env = os.environ.get("FSTAT_FDOT_RATIO_MAX", "").strip()
    _ratio_max = (float(_rm_env) if _rm_env
                  else (None if ratio_max is None else float(ratio_max)))
    _c_t = 0.5 * float(Tobs)
    if _fdot_axis:
        if _ratio_max is None:
            raise ValueError(
                "FSTAT_FDOT_AXIS=1 needs the fdot_astro_ratio prior "
                "half-width: pass ratio_max= or set FSTAT_FDOT_RATIO_MAX. "
                "Guessing it would silently size the axis to the wrong "
                "physical range.")
        _fd_bounds = [fdot_axis_bounds(float(f0) * 1e-3, mc_range[1],
                                       _ratio_max) for f0 in peaks[:, 0]]
        n_req = np.array([n_fdot_nodes(lo, hi, float(Tobs), aligned=True,
                                       eta=fstat_knob("FSTAT_MC_ETA", float))
                          for lo, hi in _fd_bounds])
    else:
        n_req = np.array([
            fstat_n_mc(float(f0), mc_range[0], mc_range[1], float(Tobs))
            for f0 in peaks[:, 0]
        ])
    if os.environ.get("FSTAT_MC_GROUPING", "1") == "1":
        levels = mc_ladder_levels(n_req)
    else:
        levels = np.full(K, int(n_req.max()))

    # ---- ADAPTIVE SKY (2026-09-24) ------------------------------------
    # Until now the sky axes were ONE fixed 8x8 shared by every box, while
    # stage A already scaled its own sky grid as f0^2 -- so above ~6 mHz the
    # refinement was coarser than the scan that produced its input. The sky
    # requirement is now evaluated per box from the same law
    # (:func:`sky_nodes_required`) and enters the GROUPING key beside the Mc
    # ladder. Both requirements are monotone in f0 and the boxes are
    # f0-sorted, so the joint key still yields CONTIGUOUS groups -- the
    # property every downstream slice (g_edges, box weights, the parallel
    # fit's box ranges) depends on.
    #
    # The floor defaults to 64, the historical node count, so the change can
    # only ADD sky resolution, never remove it. FSTAT_STAGEB_NSKY_MAX=0 is
    # uncapped; the real protection is the per-group byte budget below, not
    # a node cap, because what actually breaks is memory and memory depends
    # on n_boxes * n_f0 * n_Mc * n_sky, not on n_sky alone.
    _sky_adapt = os.environ.get("FSTAT_STAGEB_SKY_ADAPT", "1") == "1"
    _sky_pinned = bool(os.environ.get("FSTAT_N_ALPHA", "").strip()
                       or os.environ.get("FSTAT_N_SINDELTA", "").strip()
                       or os.environ.get("FSTAT_N_PER_AXIS", "").strip())
    if _sky_adapt and not _sky_pinned:
        _nsky_min = int(os.environ.get("FSTAT_STAGEB_NSKY_MIN", "64"))
        _nsky_max = int(os.environ.get("FSTAT_STAGEB_NSKY_MAX", "0"))
        sky_lv = sky_nodes_required(peaks[:, 0], float(Tobs),
                                    nmin=_nsky_min, nmax=_nsky_max)
        sky_lv = np.maximum.accumulate(sky_lv)  # monotone => contiguous groups
    else:
        sky_lv = np.full(K, int(n_alpha) * int(n_sd), dtype=int)
        if _sky_pinned:
            logger.info(
                "[stageB] sky grid PINNED to %dx%d by FSTAT_N_ALPHA/"
                "FSTAT_N_SINDELTA; the f0-adaptive sky is off for this fit.",
                n_alpha, n_sd)

    _key = np.stack([np.asarray(levels, dtype=np.int64),
                     np.asarray(sky_lv, dtype=np.int64)], axis=1)
    _chg = np.flatnonzero(np.any(np.diff(_key, axis=0) != 0, axis=1)) + 1
    g_edges = np.concatenate([[0], _chg, [K]]).astype(int)
    n_groups = len(g_edges) - 1

    # Per-group sky axes, after a MEMORY step-down. The grid is materialized
    # dense and float64 -- xp.empty(node_shape) in run_stage_b_group -- and
    # StackedFStatProposal4D then holds the node grid, the cell weights AND
    # the flat CDF, so the resident cost at DRAW time is ~3x the grid and
    # FSTAT_GRID_MEM_MB does not bound it (it chunks construction; every
    # chunk is retained). A group whose grid exceeds the budget therefore
    # takes its sky down one power of two at a time, loudly. Without this,
    # uncapped adaptive sky asks for a single 27 GB allocation on the
    # 15-20 mHz group, where n_Mc is already 96.
    _grp_max_gb = float(os.environ.get("FSTAT_STAGEB_GROUP_MAX_GB", "8.0"))
    alpha_ax_g, sd_ax_g = [], []
    for gi in range(n_groups):
        a, b = int(g_edges[gi]), int(g_edges[gi + 1])
        if _sky_adapt and not _sky_pinned:
            want = int(sky_lv[a])
            n_Mc_g = int(n_req[a:b].max())
            while True:
                _a_ax, _s_ax = stage_b_sky_axes(want)
                gb = ((b - a) * n_f0 * n_Mc_g * len(_a_ax) * len(_s_ax)
                      * 8 / 1e9)
                # The step-down stops at 64 even if FSTAT_STAGEB_NSKY_MIN is
                # higher: 64 is the historical node count, so it is the one
                # size this pipeline is known to fit. Going below it to
                # satisfy a byte budget would make the grid COARSER than it
                # has ever been, which is never the trade we want; if 64
                # still does not fit, the budget is set wrong and the run
                # should fail loudly on the allocation rather than quietly
                # propose worse births.
                if _grp_max_gb <= 0 or gb <= _grp_max_gb or want <= 64:
                    break
                logger.warning(
                    "[stageB] group %d (%.2f-%.2f mHz, %d boxes, n_Mc %d) "
                    "wants %d sky nodes = %.3g GB > "
                    "FSTAT_STAGEB_GROUP_MAX_GB %.3g GB; stepping the sky "
                    "down to %d.",
                    gi, float(f0_los[a]), float(f0_los[b - 1]), b - a,
                    n_Mc_g, want, gb, _grp_max_gb, want // 2)
                want //= 2
            alpha_ax_g.append(_a_ax)
            sd_ax_g.append(_s_ax)
        else:
            alpha_ax_g.append(np.linspace(0.0, 2 * np.pi, n_alpha))
            sd_ax_g.append(np.linspace(-1.0, 1.0, n_sd))
    # The flat alpha_ax/sd_ax fields (spec fields, npz keys, the legacy
    # single-group proposal) can only carry ONE sky grid, so from here they
    # mean GROUP 0. Say so when the groups actually differ: several scripts
    # under scripts/fstat_proposal/ read the flat npz keys and would plot or
    # re-wrap a high-f0 group on the low-f0 group's axes.
    alpha_ax, sd_ax = alpha_ax_g[0], sd_ax_g[0]
    if not all(len(a_) == len(alpha_ax) and len(s_) == len(sd_ax)
               for a_, s_ in zip(alpha_ax_g, sd_ax_g)):
        logger.warning(
            "[stageB] the sky grid now VARIES BY GROUP (%s nodes); the flat "
            "alpha_ax/sin_delta_ax fields describe GROUP 0 only. Readers "
            "must prefer alpha_ax_g{gi}/sin_delta_ax_g{gi}.",
            [len(a_) * len(s_) for a_, s_ in zip(alpha_ax_g, sd_ax_g)])
    _parts = cache_path.replace(".npz", "_parts") if cache_path else None
    mem_mb = os.environ.get("FSTAT_GRID_MEM_MB", "").strip()
    mem_budget = float(mem_mb) if mem_mb else None
    # LIVE-path weighting. Use the SAME weighting the cache path applies,
    # so a run that fits in-process and one that reloads the epoch cache
    # draw from an identical proposal -- otherwise the first epoch after a
    # fit would silently differ from every resume.
    box_w = peak_box_weights(
        np.clip(peaks[:, 1], 0.0, None), peak_f0_mHz=peaks[:, 0],
        band_edges=band_edges_hz,
        alpha=peak_weight_alpha_env(epoch),
        cells=peak_weight_cells_env(),
        equal=(os.environ.get("FSTAT_PEAK_WEIGHTING", "fstat")
               .strip().lower() == "equal"))

    grids_g, mc_ax_g, n_mc_g = [], [], []
    for gi in range(n_groups):
        a, b = int(g_edges[gi]), int(g_edges[gi + 1])
        n_Mc = int(n_req[a:b].max())
        if _fdot_axis:
            # ONE reference f0 for the whole group, so fdot is independent
            # of f0 and the f_mid shear stays exactly unit-determinant.
            #
            # INTERSECTION, not union. The bound scales as f0^(11/3), so a
            # box at the group's LOW-f0 end cannot physically reach the fdot
            # a high-f0 box can: with the union, drawing that fdot makes the
            # feasible Mc interval EMPTY (mc_floor > mc_hi), and the birth
            # conversion then emits a row its own logpdf prices at -inf --
            # proposed forever, accepted never, and it reads as mere
            # inefficiency. Taking the min of the upper bounds (and the max
            # of the lower) sizes the axis at the group's lowest f0, so
            # every drawn fdot is reachable at every f0 in the group.
            # The ladder holds a group's f0 within 2^(3/11) = 1.21, so this
            # costs at most ~21% of span at a group edge -- efficiency.
            _lo = max(b_[0] for b_ in _fd_bounds[a:b])
            _hi = min(b_[1] for b_ in _fd_bounds[a:b])
            mc_ax = np.linspace(_lo, _hi, n_Mc)
        else:
            mc_ax = np.linspace(mc_range[0], mc_range[1], n_Mc)
        _al, _sd = alpha_ax_g[gi], sd_ax_g[gi]
        _na, _nsd = len(_al), len(_sd)
        node_shape = (b - a, n_f0, n_Mc, _na, _nsd)
        logger.info(
            "[stageB] group %d/%d: %d peak boxes x %dx%dx%dx%d nodes "
            "(f0 %.4f-%.4f mHz); axis2 = %s %s; sky %d nodes = %.2f GB; "
            "f0 boxes clamped to their sub-bands.", gi + 1, n_groups, b - a,
            n_f0, n_Mc, _na, _nsd, float(f0_los[a]), float(f0_los[b - 1]),
            "fdot[Hz/s]" if _fdot_axis else "Mc",
            (f"[{mc_ax[0]:.4e}, {mc_ax[-1]:.4e}]" if _fdot_axis
             else str(mc_range)),
            _na * _nsd, float(np.prod(node_shape)) * 8 / 1e9)
        # single group keeps the historical "stageb" checkpoint name so
        # in-flight fits resume across this code change.
        _ck = "stageb" if n_groups == 1 else f"stageb_g{gi}"
        spec = StageBGroupSpec(
            gi=gi, n_groups=n_groups, a=a, b=b,
            f0_los=f0_los[a:b], f0_dxs=f0_dxs[a:b], mc_ax=mc_ax,
            alpha_ax=_al, sd_ax=_sd, node_shape=node_shape,
            ckpt_name=_ck, parts_dir=_parts,
            fingerprint_extra=fingerprint_extra,
            fdot_axis=_fdot_axis, c_t=_c_t,
        )
        runner = sweep_runner if sweep_runner is not None else run_stage_b_group
        grids_g.append(runner(spec, call_fstat, xp=xp))  # beta = 1: logp = F
        mc_ax_g.append(mc_ax)
        n_mc_g.append(n_Mc)

    if n_groups == 1:
        stacked = StackedFStatProposal4D(
            grids_g[0], f0_los, f0_dxs, mc_ax_g[0], alpha_ax, sd_ax,
            weights=box_w, mem_budget_mb=mem_budget,
        )
        if cache_path:
            stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
            write_stacked_npz(
                stacked_path, grids_g=grids_g, mc_ax_g=mc_ax_g,
                f0_los=f0_los, f0_dxs=f0_dxs, alpha_ax=alpha_ax, sd_ax=sd_ax,
                grid_basis=("fdot" if _fdot_axis else "Mc"),
                grid_c_t=(_c_t if _fdot_axis else 0.0), peaks=peaks,
                band_idx=band_idx, band_edges_mHz=band_edges_mHz,
                band_edges_hz=band_edges_hz, group_sizes=None)
            logger.info("[cache] wrote %s", stacked_path)
            ckpt_clear(_parts, "stageb")
        return stacked

    comps = []
    for gi in range(n_groups):
        a, b = int(g_edges[gi]), int(g_edges[gi + 1])
        sub_w = None
        if box_w is not None:
            sub_w = np.asarray(box_w, dtype=float)[a:b]
            if not np.any(sub_w > 0):
                sub_w = None  # zero-mass group: equal inside, never drawn
        comps.append(StackedFStatProposal4D(
            grids_g[gi], f0_los[a:b], f0_dxs[a:b], mc_ax_g[gi],
            alpha_ax_g[gi], sd_ax_g[gi], weights=sub_w,
            mem_budget_mb=mem_budget))
    stacked = GroupedStackedFStatProposal(comps, box_weights=box_w)
    _sizes = np.diff(g_edges).astype(int)
    _tot_gb = sum(float(np.prod(g.shape)) for g in grids_g) * 8 / 1e9
    logger.info(
        "[stageB] banded Mc stacks: %d groups, sizes %s, n_Mc %s, sky %s "
        "(a single max-f0 stack would carry %d Mc nodes on every box); "
        "grids total %.1f GB, resident cost at draw time is ~3x that.",
        n_groups, _sizes.tolist(), n_mc_g,
        [len(a_) * len(s_) for a_, s_ in zip(alpha_ax_g, sd_ax_g)],
        int(n_req.max()), _tot_gb)

    if cache_path:
        stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
        write_stacked_npz(
            stacked_path, grids_g=grids_g, mc_ax_g=mc_ax_g,
            f0_los=f0_los, f0_dxs=f0_dxs, alpha_ax=alpha_ax, sd_ax=sd_ax,
            alpha_ax_g=alpha_ax_g, sd_ax_g=sd_ax_g,
            grid_basis=("fdot" if _fdot_axis else "Mc"),
            grid_c_t=(_c_t if _fdot_axis else 0.0), peaks=peaks,
            band_idx=band_idx, band_edges_mHz=band_edges_mHz,
            band_edges_hz=band_edges_hz, group_sizes=_sizes)
        logger.info("[cache] wrote %s (%d Mc groups)", stacked_path, n_groups)
        ckpt_clear(_parts, "stageb")
    return stacked


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def comb_cache_usable(comb_cache: str, fingerprint_extra: str, *,
                      band_edges_hz=None, band_skip=None) -> bool:
    """Is this FINISHED comb scan the one THIS fit would have run?

    ``fingerprint_extra`` salts :func:`ckpt_fingerprint`, i.e. the in-flight
    PROGRESS files only. A completed ``*_comb.npz`` is not a checkpoint, so
    until this existed it was reloaded on ``os.path.exists`` alone -- and a
    refit whose reference walker moved re-selected its peak BOXES from the
    previous walker's scan while stage B then scored inside those boxes at
    the NEW reference. Two residuals stitched into one fit, with nothing
    anywhere to say so.

    That is not hypothetical: 6mo epoch 1 hit exactly it (the global argmax
    moved from walker 0 to walker 2 across a restart). It was ruled a known
    limitation on 2026-09-17 because the effect is proposal quality, not
    correctness -- births are MH-corrected -- and because rescanning cost
    the full 47-minute serial comb. Stage A is split now, so the rescan is
    ~12 minutes and the limitation is not worth keeping.

    A cache with NO stamp is a legacy one, written before this landed. It is
    REUSED, with a warning: refusing it would throw away a finished comb on
    the first restart of every fit currently in flight, which is the exact
    cost this is meant to avoid -- and it is no worse than the behavior that
    wrote it. An UNREADABLE one is refused (there is nothing to reuse).

    A comb whose sweep SKIPPED bands (2026-10-10; it carries ``node_swept``)
    holds F_max = 0 at every node it never scored. It is reused only if it
    scored every node THIS fit's ``band_skip`` needs
    (:func:`comb_sweep_mask` over ``band_edges_hz``, default the cached
    edges): a band shut then and live now would otherwise re-select from
    zeros and lose its peaks without a word. In the GB move a changed skip
    set also changes the stamp (``fingerprint_extra`` carries
    ``|bandskip=``), so either test refuses it there; this one is what
    guards a caller whose fingerprint does not carry the skip set.
    """
    uncovered = 0
    try:
        with np.load(comb_cache, allow_pickle=False) as d:
            stamp = (str(d["fingerprint_extra"])
                     if "fingerprint_extra" in d else None)
            if "node_swept" in d:
                swept = np.asarray(d["node_swept"], dtype=bool)
                f0c = np.asarray(d["f0_nodes_mHz"], dtype=float)
                edges = (d["band_edges"] if band_edges_hz is None
                         else band_edges_hz)
                need = comb_sweep_mask(
                    f0c, edges, band_skip,
                    float(f0c[1] - f0c[0]) if len(f0c) > 1 else 1.0)
                if need is None:
                    need = np.ones(f0c.shape, dtype=bool)
                uncovered = (int(need.size) if swept.shape != need.shape
                             else int((need & ~swept).sum()))
    except (OSError, ValueError) as exc:
        logger.warning("[fit] comb cache %s could not be read (%s: %s); "
                       "re-running the comb scan.", comb_cache,
                       exc.__class__.__name__, exc)
        return False
    if uncovered:
        logger.info(
            "[fit] comb cache %s did not sweep %d f0 node(s) this fit needs "
            "(bands skipped when it was scanned are live now) -- re-running "
            "the comb scan rather than selecting peaks from zero-filled "
            "nodes.", comb_cache, uncovered)
        return False
    if stamp is None:
        logger.warning(
            "[fit] comb cache %s carries no reference-walker stamp (written "
            "before the salt landed), so it cannot be checked against this "
            "fit's %r -- reusing it. If this fit's reference walker moved, "
            "its peak boxes come from the PREVIOUS walker's scan; delete the "
            "file to force a clean rescan.", comb_cache, fingerprint_extra)
        return True
    if stamp != fingerprint_extra:
        logger.info(
            "[fit] comb cache %s was scored against a different reference "
            "(%r, this fit is %r) -- re-running the comb scan rather than "
            "selecting peaks from another walker's residual.",
            comb_cache, stamp, fingerprint_extra)
        return False
    return True


def stacked_grid_path(cache_dir: str) -> str:
    """The stage-B output npz ``run_fstat_grid_fit`` writes and reloads.

    Spelled exactly as every writer spells it -- ``os.path.join(cache_dir,
    GRID_BASENAME)`` with the suffix swapped -- so a caller's pre-check can
    never name a different file from the one the orchestrator tests.
    """
    return os.path.join(cache_dir, GRID_BASENAME).replace(
        ".npz", "_peaks_stacked.npz")


def stage_b_complete(cache_dir: str) -> bool:
    """True when ``cache_dir`` already holds a finished stage-B grid.

    The condition :func:`run_fstat_grid_fit` short-circuits on, exposed so a
    caller can test it BEFORE paying for anything the short circuit would
    throw away. Under the multi-rank fit that matters: the reference row is
    ~72 MB per rank on the wire and the sig-het scorer built beside it is
    ~GB on device, and a "fit" decision can legitimately land on a complete
    npz (a ``DONE.json`` lost beside one, an offline grid dropped in).

    Says nothing about a ZERO-PEAK epoch, for which no npz is ever written;
    that is the manifest's job (``_epoch_complete`` /
    ``_epoch_missing_for_ranks`` in the GB move).
    """
    return os.path.exists(stacked_grid_path(cache_dir))


#: The stage-B cache keys that are PER-BOX parallel arrays (one entry per
#: selected peak, in the cache's box order). Everything else is either a
#: group-level array (``logp_grids_g{i}`` / ``mc_ax_g{i}`` / the per-group sky
#: axes / ``group_sizes``) or run-level metadata copied through unchanged.
_STACKED_PER_BOX_KEYS = ("f0_los", "f0_dxs", "peak_f0_mHz", "peak_F",
                         "band_idx", "band_f0_lo", "band_f0_hi")


def stacked_cache_at_floor(d, min_F: float):
    """A stage-B cache restricted to the boxes whose peak F is >= ``min_F``.

    Returns ``(cache, n_kept, n_total)``. ``cache`` is ``d`` ITSELF when no
    box falls below the floor (nothing copied), ``None`` when every box does,
    and otherwise a lazy read-through view (:class:`_FloorCutStackedCache`)
    with the same key layout :func:`write_stacked_npz` produces (legacy single
    stack or grouped), which
    :func:`~lisatools.sampling.fstat_proposal.stacked_from_cache` and
    :func:`build_gb_birth_distribution` read exactly like the npz -- each box
    grid cut only when it is read, one group at a time.

    WHY THIS IS THE SAME CATALOG A RE-SELECTION AT ``min_F`` WOULD GIVE
    (user ruling 2026-10-12: the GB-free galaxy grid at SNR 6.5, while the
    running 9mo store holds a PE epoch selected at 6.25). Peak selection
    (:func:`select_comb_peaks`) takes a floor-INDEPENDENT candidate set
    (window champions | strict local maxima, interior bands), keeps those
    with ``F >= floor`` and caps each band at its ``FSTAT_PEAKS_PER_BAND``
    highest-F candidates. Raising the floor therefore keeps exactly the old
    selection's boxes with ``peak_F >= min_F``: in a band where the cap bound
    at the old floor the kept set is a top-F prefix either way, and where it
    did not, every candidate above the new floor was already selected. Each
    box's grid is scored independently of the others (stage B sweeps boxes,
    and box is the slowest axis), so the kept boxes' grids are exactly what
    they were. What does NOT re-derive: a group's SHARED axes (the Mc axis
    sized by the group's widest box, the sky axes after the per-group byte
    budget step-down) keep the values the larger group was fitted on -- a
    valid proposal, the same draw-and-logpdf object on both sides of every
    RJ factor, just not byte-identical to a fresh fit at the higher floor.

    Raising the floor only: a cache selected at a STRICTER floor than
    ``min_F`` cannot gain the peaks between the two, and is returned as is
    (the caller says so).
    """
    keys = list(getattr(d, "files", None) or list(d))
    if "peak_F" not in keys:
        # an unverifiable legacy cache: nothing to select on
        return d, None, None
    peak_F = np.asarray(d["peak_F"], dtype=float).reshape(-1)
    n_total = int(peak_F.size)
    keep = peak_F >= float(min_F)
    n_kept = int(keep.sum())
    if n_kept == n_total:
        return d, n_kept, n_total
    if n_kept == 0:
        return None, 0, n_total
    return _FloorCutStackedCache(d, keys, keep, float(min_F)), n_kept, n_total


class _FloorCutStackedCache:
    """Read-through view of a stage-B cache with the boxes below a floor cut
    (what :func:`stacked_cache_at_floor` returns when it drops something).

    LAZY ON PURPOSE: a box GRID is cut only when its key is read, one group
    at a time, exactly as the unmodified loader pulls one group at a time out
    of the npz. Materializing every cut group up front would hold the whole
    cut galaxy grid in each rank's host memory for the length of the device
    build (the 9mo PE epoch: 32,715 boxes, ~43 GB resident per card once on
    device). The small per-box arrays, the renumbered group table and the
    raised stamp are computed once here. Reads like the npz: ``files``,
    ``in``, iteration, ``[key]``.
    """

    _GROUP_PREFIXES = ("logp_grids_g", "mc_ax_g", "alpha_ax_g", "sin_delta_ax_g")

    def __init__(self, d, keys, keep, min_F):
        self._d = d
        self._keep = keep
        self._small = {}          # computed arrays, by NEW key
        self._alias = {}          # NEW key -> (source key, box mask or None)
        grouped = "group_sizes" in keys
        out_keys = []
        for k in keys:
            if k in _STACKED_PER_BOX_KEYS:
                self._small[k] = np.asarray(d[k])[keep]
            elif k == "logp_grids":
                self._alias[k] = (k, keep)
            elif grouped and (k == "group_sizes" or k.startswith(self._GROUP_PREFIXES)):
                continue          # renumbered below
            else:
                self._alias[k] = (k, None)
            out_keys.append(k)
        if grouped:
            sizes = np.asarray(d["group_sizes"], dtype=int).reshape(-1)
            bounds = np.concatenate([[0], np.cumsum(sizes)])
            if int(bounds[-1]) != int(keep.size):
                raise ValueError(
                    f"stage-B cache: group_sizes sum to {int(bounds[-1])} boxes "
                    f"but peak_F has {int(keep.size)}; refusing to subset a "
                    f"cache whose box axis disagrees with its groups.")
            new_sizes = []
            for gi, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
                m = keep[int(a):int(b)]
                if not m.any():
                    continue      # a group that lost every box
                gj = len(new_sizes)
                self._alias[f"logp_grids_g{gj}"] = (f"logp_grids_g{gi}", m)
                self._alias[f"mc_ax_g{gj}"] = (f"mc_ax_g{gi}", None)
                for ax in ("alpha_ax_g", "sin_delta_ax_g"):
                    if f"{ax}{gi}" in keys:
                        self._alias[f"{ax}{gj}"] = (f"{ax}{gi}", None)
                new_sizes.append(int(m.sum()))
            self._small["group_sizes"] = np.asarray(new_sizes, dtype=int)
            out_keys.append("group_sizes")
            out_keys += [k for k in self._alias if k.startswith(self._GROUP_PREFIXES)]
            # the flat sky keys describe GROUP 0 when the groups differ
            # (write_stacked_npz); keep that true when group 0 was dropped
            if "alpha_ax_g0" in self._alias:
                self._alias["alpha_ax"] = self._alias["alpha_ax_g0"]
                self._alias["sin_delta_ax"] = self._alias["sin_delta_ax_g0"]
        # the stamp now says what the cache holds: the higher floor
        have = float(d["peak_min_F"]) if "peak_min_F" in keys else -np.inf
        self._small["peak_min_F"] = np.float64(max(have, float(min_F)))
        if "peak_min_F" not in out_keys:
            out_keys.append("peak_min_F")
        for k in self._small:
            self._alias.pop(k, None)
        self._keys = list(dict.fromkeys(out_keys))

    @property
    def files(self):
        return list(self._keys)

    def __contains__(self, key):
        return key in self._small or key in self._alias

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, key):
        if key in self._small:
            return self._small[key]
        src, mask = self._alias[key]
        if mask is None:
            return self._d[src]
        return np.asarray(self._d[src])[mask]


def run_fstat_grid_fit(call_fstat: Callable, *, xp, Tobs: float,
                       band_edges_hz, f0_lims_hz, mc_lims, cache_dir: str,
                       fingerprint_extra: str = "", epoch=None,
                       ratio_max=None, sweep_runner=None, comb_runner=None,
                       band_min_F=None, band_skip=None):
    """Full fit with resume: comb scan -> peak select -> stage B.

    ``epoch`` selects the peak-box weighting tilt only (see
    :func:`peak_weight_alpha_env`); it does not change what is computed or
    cached, so the npz caches stay interchangeable across epochs.

    ``comb_runner`` is forwarded to :func:`run_comb_scan` and
    ``sweep_runner`` to :func:`run_stacked_stage_b`; ``None`` is the serial
    per-level / per-group kernel stream. The stacked-cache short circuit
    below it is unchanged -- a complete epoch never re-enters the kernel,
    parallel or not, and ``call_fstat`` is never called on that path (see
    :func:`stage_b_complete`, which a caller can test first to avoid
    BUILDING one).

    Cache reuse IS the mid-fit resume, and it is always on:

    * ``<cache_dir>/fstat_grid_peaks_stacked.npz`` present -> load and return
      (the fit is done; nothing recomputed).
    * ``<cache_dir>/fstat_grid_comb.npz`` present AND scored against this
      fit's reference (:func:`comb_cache_usable`) -> reload the comb,
      re-select peaks (cheap, deterministic), run only stage B.
    * otherwise -> run both stages.

    Returns ``(stacked_or_None, n_peaks)``.
    """
    from lisatools.sampling.fstat_proposal import stacked_from_cache

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, GRID_BASENAME)
    comb_cache = cache_path.replace(".npz", "_comb.npz")
    stacked_cache = stacked_grid_path(cache_dir)

    if os.path.exists(stacked_cache):
        d = np.load(stacked_cache, allow_pickle=False)
        logger.info("[fit] stage B already complete: %s", stacked_cache)
        _check_cached_peak_threshold(d, stacked_cache, comb_cache)
        mem_mb = os.environ.get("FSTAT_GRID_MEM_MB", "").strip()
        # Weight exactly as the other two paths do. This used to pass RAW
        # peak_F, i.e. alpha=1 with no cell equalisation -- and because the
        # caller forwards this object as ``stacked_live``, which
        # short-circuits build_gb_birth_distribution's npz reload, both
        # FSTAT_PEAK_WEIGHT_ALPHA and FSTAT_PEAK_WEIGHT_CELLS were silently
        # dropped whenever this branch was taken (a refit that finds stage B
        # already written -- a resume after a mid-fit death, and every epoch
        # whose grids were prebuilt offline).
        stacked = stacked_from_cache(
            d,
            weights=peak_box_weights(
                np.clip(np.asarray(d["peak_F"], dtype=float), 0.0, None),
                peak_f0_mHz=d["peak_f0_mHz"] if "peak_f0_mHz" in d else None,
                band_edges=d["band_edges"] if "band_edges" in d else None,
                alpha=peak_weight_alpha_env(epoch),
                cells=peak_weight_cells_env(),
                equal=(os.environ.get("FSTAT_PEAK_WEIGHTING", "fstat")
                       .strip().lower() == "equal")),
            mem_budget_mb=float(mem_mb) if mem_mb else None,
            use_cupy=(xp.__name__ != "numpy"),
        )
        return stacked, int(len(d["peak_f0_mHz"]))

    # Stage A's wall, on its own line -- it used to be the head-only half of
    # the fit and the only serial one left, so it has to stay readable
    # directly rather than inferred by subtracting stage B from the fit.
    # Measured at 6mo epoch 1 before the split: 2804 s on ONE GPU against
    # stage B's 3068 s on four, i.e. 48% of the epoch.
    _t_stage_a = time.time()
    if os.path.exists(comb_cache) and comb_cache_usable(
            comb_cache, fingerprint_extra, band_edges_hz=band_edges_hz,
            band_skip=band_skip):
        d = np.load(comb_cache, allow_pickle=False)
        logger.info("[fit] reusing comb cache %s; re-selecting peaks",
                    comb_cache)
        f0_nodes = np.asarray(d["f0_nodes_mHz"], dtype=float)
        spacing = float(f0_nodes[1] - f0_nodes[0]) if len(f0_nodes) > 1 else 1.0
        # RE-SELECT from the cached comb at the CURRENT floor. This is
        # the cheap half by design: the comb stores F_max for every
        # node, so a floor that moved between epochs costs a
        # re-selection and a stage-B refit, never a re-sweep.
        peaks = select_comb_peaks(f0_nodes, xp.asarray(d["F_max"]),
                                  band_edges_hz, spacing, xp,
                                  min_F=band_min_F, band_skip=band_skip)
        # What THAT sweep dropped (it covers this fit's needs, or
        # comb_cache_usable would have refused it).
        _n_nodes = int(len(f0_nodes))
        _n_dropped = (int((~np.asarray(d["node_swept"], dtype=bool)).sum())
                      if "node_swept" in d.files else 0)
    else:
        _f0, _F, peaks, _x = run_comb_scan(
            call_fstat, xp=xp, Tobs=Tobs, band_edges_hz=band_edges_hz,
            f0_lims_hz=f0_lims_hz, mc_lims=mc_lims, cache_path=cache_path,
            fingerprint_extra=fingerprint_extra, comb_runner=comb_runner,
            band_min_F=band_min_F, band_skip=band_skip,
        )
        _n_nodes = int(len(_f0))
        _n_dropped = int(_x.get("n_nodes_dropped", 0))
    _n_skip = (0 if band_skip is None
               else int(np.asarray(band_skip, dtype=bool).sum()))
    _n_band = int(len(band_edges_hz) - 1)
    # The skip set is the union of the per-walker valve (shut on EVERY
    # walker) and the per-band empty-band valve -- see
    # gbspecialstretch.fstat_band_skip_for -- so the line no longer says
    # which; the [comb] band_skip line above it carries the window split.
    logger.info("[stageA] comb + peak selection: %d peaks in %s (%s)%s",
                int(len(peaks)), _fmt_secs(time.time() - _t_stage_a),
                "split by node range over the compute ranks"
                if comb_runner is not None else "serial, this process",
                ("" if band_skip is None else
                 f"; {_n_skip} of {_n_band} bands skipped ({_n_dropped} of "
                 f"{_n_nodes} f0 nodes dropped from the sweep)"))

    stacked = run_stacked_stage_b(
        call_fstat, peaks, xp=xp, Tobs=Tobs, band_edges_hz=band_edges_hz,
        mc_lims=mc_lims, ratio_max=ratio_max, cache_path=cache_path,
        fingerprint_extra=fingerprint_extra, epoch=epoch,
        sweep_runner=sweep_runner,
    )
    return stacked, int(len(peaks))


# --------------------------------------------------------------------------
# birth container
# --------------------------------------------------------------------------

def _peak_min_F_now() -> float:
    """The peak-selection floor in force right now, in F units."""
    from lisatools.sampling.fstat_proposal import fstat_peak_min_F

    return float(fstat_peak_min_F())


def _check_cached_peak_threshold(d, stacked_cache: str, comb_cache: str) -> None:
    """Refuse a stage-B cache selected at a DIFFERENT peak threshold.

    The stage-B cache short-circuits everything: "load and return, nothing
    recomputed". Its peak LIST was selected at whatever
    ``FSTAT_PEAK_MIN_SNR`` was in force when it was fitted, and nothing else
    on this path consults the knob -- so lowering the threshold on a resume
    is a SILENT NO-OP, and the run keeps proposing from the old, stricter
    list while its log and its submit script both claim the new value. Same
    class of trap as the ``grid_basis`` and ``band_edges`` stamps above, and
    the same treatment: stamp it, refuse a mismatch.

    The migration is cheap and is named in the error. The COMB cache stores
    ``F_max`` for every node, so deleting only ``*_peaks_stacked.npz`` while
    KEEPING ``*_comb.npz`` re-selects peaks at the new threshold (cheap,
    deterministic) and reruns stage B alone -- no full refit.

    A cache with no stamp is legacy: warn rather than refuse, since every
    cache written before this stamp existed would otherwise become unusable.
    """
    from lisatools.sampling.fstat_proposal import fstat_peak_min_F

    want = float(fstat_peak_min_F())
    if "peak_min_F" not in d:
        logger.warning(
            "[fit] stage-B cache %s carries no peak_min_F stamp (written "
            "before the stamp existed); CANNOT verify it against the run's "
            "F >= %.3f (SNR %.2f). If you have changed FSTAT_PEAK_MIN_SNR, "
            "delete %s (KEEP the _comb.npz) to re-select peaks.",
            stacked_cache, want, np.sqrt(2.0 * want), stacked_cache)
        return
    have = float(d["peak_min_F"])
    if abs(have - want) > 1e-9 * max(1.0, abs(want)):
        raise ValueError(
            f"F-stat stage-B cache {stacked_cache!r} selected its peaks at "
            f"F >= {have:.4f} (SNR {np.sqrt(2.0 * have):.3f}), but this run "
            f"asks for F >= {want:.4f} (SNR {np.sqrt(2.0 * want):.3f}). "
            f"Loading it would silently keep the OLD peak list. Delete "
            f"{stacked_cache!r} and KEEP {comb_cache!r}: the comb stores "
            f"F_max for every node, so peaks re-select at the new threshold "
            f"and only stage B reruns (no full refit). Deleting the whole "
            f"epoch directory also works but pays for the comb again."
        )


def check_cached_band_grid(cache_dir: str, expected_band_edges) -> None:
    """Refuse a cached F-stat grid fitted against a DIFFERENT band grid.

    The stage-B cache stores per-peak ``band_idx`` (an index into the band
    grid the fit ran against) and the ``band_edges`` themselves; the comb
    cache stores ``band_edges`` too. If the run's band edges change (e.g.
    GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT / GB_SUBBAND_DIVISOR), every
    stored ``band_idx`` silently re-labels and the interior-band selection
    mask shifts -- the ONLY correct behavior is a refit. Raises
    :class:`ValueError` loudly when the cached edges differ from
    ``expected_band_edges`` (shape or values), or when a cache exists but
    carries no band metadata at all (unverifiable legacy cache). A missing
    cache passes (nothing to be stale).
    """
    expected = np.asarray(expected_band_edges, dtype=float)
    cache_path = os.path.join(cache_dir, GRID_BASENAME)
    for path in (cache_path.replace(".npz", "_peaks_stacked.npz"),
                 cache_path.replace(".npz", "_comb.npz")):
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=False)
        if "band_edges" not in d:
            raise ValueError(
                f"F-stat grid cache {path!r} carries no band_edges metadata; "
                f"cannot verify it against the run's "
                f"{len(expected) - 1}-band grid. Refusing the stale/legacy "
                f"cache -- delete the epoch directory (or point "
                f"GB_FSTAT_FIT_DIR elsewhere) to force a refit."
            )
        stored = np.asarray(d["band_edges"], dtype=float)
        if stored.shape != expected.shape or not np.allclose(stored, expected):
            raise ValueError(
                f"F-stat grid cache {path!r} was fitted against "
                f"{len(stored) - 1} sub-bands but the run now has "
                f"{len(expected) - 1} (band edges differ). Per-peak "
                f"band_idx labels are stale; a refit is required. Delete "
                f"the epoch directory or start a new epoch -- never reuse "
                f"a grid across a band-edge change (see scripts/"
                f"fstat_proposal/migrate_gb_band_edges.py)."
            )
    return None


def peak_box_weights(peak_F, peak_f0_mHz=None, band_edges=None,
                     alpha=1.0, cells=0, equal=False):
    """Mixture weights for the stacked F-stat peak boxes.

    Two schemes, both returned as FLAT per-box weights:

    * ``cells = K >= 1`` -- the HIERARCHICAL scheme and the DEFAULT: draw a
      SUB-BAND STRATUM uniformly, then draw within it with ``w ~ F**alpha``.
      ``K = 1`` (the default) is one stratum per sub-band; ``K > 1``
      sub-divides each sub-band into K parts of ITS OWN width. The
      composite weight of box ``j`` in stratum ``c`` is::

          w_j = (1 / N_occupied) * F_j**alpha / sum_{k in c} F_k**alpha

    * ``cells <= 0`` -- the historical GLOBAL mixture, ``w ~ F**alpha``
      over every box at once. Always reachable as a fallback.

    WHY SUB-BANDS AND NOT CAP CELLS (user ruling 2026-08-29). Until now the
    stratum count tracked ``GB_CAP_DIVISOR`` and the docstring claimed the
    cells were "the same construction as the leaf-cap grid". Both were
    wrong. Births are confined to sub-bands and scheduled
    serial-within-band, so the sub-band is the unit a birth competes in --
    "those are the effective rj limits as well". Cap cells are a cap
    overlay that deliberately STRADDLES sub-band seams (offset half a cell)
    so the two sides of a seam contend for one cap; stratifying a proposal
    on them mismatches the RJ limits. And the code never actually built cap
    cells here -- it built ``np.linspace(be[0], be[-1], nb*K + 1)``, which
    equals the sub-bands only on a uniform band grid and equals the
    staggered cap cells never.

    WHY THIS IS EXPRESSED AS FLAT WEIGHTS rather than a two-stage sampler.
    ``StackedFStatProposal4D`` drives BOTH ``rvs`` and ``logpdf`` from
    ``self.weights`` -- ``rvs`` builds its CDF as
    ``w_k * (cell weight / box total)`` and ``logpdf`` returns
    ``g + log(w_k) - log_norm_k``. A flat mixture carrying the composite
    weight above has exactly the distribution of the hierarchical draw, so
    the sampler and its density stay mutually exact BY CONSTRUCTION. Hand
    rolling a two-stage ``rvs`` with a matching ``logpdf`` would put the RJ
    acceptance ratio at the mercy of two implementations agreeing -- and a
    silent mismatch there biases the posterior rather than crashing.

    WHY IT IS WORTH DOING. The F-statistic goes like SNR^2, so a global
    ``w ~ F`` mixture concentrates birth draws on whichever peaks are
    loudest ANYWHERE in the band. Measured on the 3-month epoch-0 fit, the
    per-cell draw mass spans 21x between the median cell and the busiest
    one; equalising across cells removes that while keeping the local
    ``F**alpha`` preference inside each cell intact.

    A cell containing peaks that are ALL F == 0 cannot be normalised by
    ``F**alpha``; it falls back to uniform inside that cell so the cell
    still receives its share rather than silently vanishing.

    Returns ``None`` when the weights would be flat anyway, which is what
    ``StackedFStatProposal4D`` expects for "no weighting".
    """
    F = np.clip(np.asarray(peak_F, dtype=float).ravel(), 0.0, None)
    cells = int(cells or 0)
    if equal or (alpha == 0.0 and cells <= 0):
        return None
    w = F if alpha == 1.0 else np.power(F, alpha)
    if cells <= 0:
        return w if np.any(w > 0) else None
    if peak_f0_mHz is None or band_edges is None:
        logger.warning("[birth] per-cell peak weighting requested but the "
                       "cache carries no f0/band_edges; falling back to the "
                       "global w ~ F**alpha mixture.")
        return w if np.any(w > 0) else None

    # STRATA ARE SUB-BANDS (user ruling 2026-08-29): "the sub-bands should
    # define that grid since those are the effective rj limits as well."
    # A birth is confined to a sub-band (the draw is global, then candidates
    # are divided up and assigned by drawn f0) and scheduling is
    # serial-within-band, so the sub-band is the unit a birth competes in.
    #
    # THIS USED TO BE ``np.linspace(be[0], be[-1], ncell + 1)`` -- a uniform
    # grid over the whole range that ignored ``band_edges`` entirely. It
    # coincided with the sub-bands only when the band grid happened to be
    # uniform (it is today, GB_BAND_EDGES_MODE=uniform, but get_n grids are
    # frequency-dependent), and it never matched the leaf-cap cells the old
    # docstring claimed it mirrored -- those are STAGGERED, offset half a
    # cell so they straddle seams. Bucketing on band edges makes the strata
    # the RJ limits by construction, on any band grid.
    #
    # ``cells = K`` sub-divides each sub-band into K parts of ITS OWN width
    # (nested/aligned -- deliberately not staggered: these are proposal
    # limits, not caps). K=1 is one stratum per sub-band, the default.
    be = np.asarray(band_edges, dtype=float).ravel()
    nb = be.size - 1
    k = max(1, int(cells))
    ncell = nb * k
    f0 = np.asarray(peak_f0_mHz, dtype=float).ravel() * 1e-3      # mHz -> Hz
    b = np.clip(np.searchsorted(be, f0, side="right") - 1, 0, nb - 1)
    if k == 1:
        ci = b
    else:
        width = be[b + 1] - be[b]
        frac = np.divide(f0 - be[b], width, out=np.zeros_like(f0),
                         where=width > 0)
        ci = b * k + np.clip((frac * k).astype(np.int64), 0, k - 1)

    out = np.zeros_like(w)
    tot = np.bincount(ci, weights=w, minlength=ncell)
    cnt = np.bincount(ci, minlength=ncell)
    occ = np.nonzero(cnt > 0)[0]
    good = tot[ci] > 0
    out[good] = w[good] / tot[ci][good]
    if np.any(~good):
        out[~good] = 1.0 / cnt[ci][~good]
    # TODO(cell-weighting, 6-month or later): occupied cells are currently
    # given EQUAL total mass -- that is what this line does, and it is
    # deliberate (it replaced the global ``w ~ F**alpha`` mixture, whose
    # per-cell mass ran max/median ~10x and starved sparse cells). But equal
    # is almost certainly not optimal either: a cell holding 40 candidate
    # peaks and a cell holding 1 draw the same total mass, so a source in the
    # dense cell is proposed ~40x less often than one in the sparse cell.
    #
    # IDEA (user 2026-08-29): weight the CELL draw by roughly sqrt of the
    # number of sources/peaks it contains, i.e. ``out /= sum(sqrt(cnt[occ]))``
    # with a per-cell factor ``sqrt(cnt)`` instead of the flat 1. That sits
    # between today's flat-per-cell and the historical proportional mixture:
    # dense cells get more mass than sparse ones, but sub-linearly, so they
    # cannot dominate the way ``w ~ F**alpha`` did.
    #
    # WHY NOT NOW: this changes the RJ birth proposal distribution, so the
    # proposal density bookkeeping must be re-derived with it (the mixture is
    # exact today -- see GroupedStackedFStatProposal, group mass = summed box
    # weights). Worth doing when the source density per cell is high enough
    # for the difference to matter, i.e. the 6-month and longer runs, where
    # cells hold many more peaks than at 3 months. NOTE ``cnt`` counts PEAKS,
    # not confirmed sources -- decide deliberately which the weight should
    # follow before implementing.
    out /= max(occ.size, 1)
    if not np.any(out > 0):
        return None
    logger.info("[birth] per-SUB-BAND peak weighting: K=%d subdivision(s) "
                "-> %d strata over %d sub-bands, %d occupied (%.0f%%); "
                "alpha=%.3g; per-stratum mass equalised (was max/median "
                "%.1fx under the global mixture).",
                k, ncell, nb, occ.size,
                100.0 * occ.size / max(ncell, 1), alpha,
                float(np.max(tot[occ]) / max(np.median(tot[occ]), 1e-300))
                if occ.size else 1.0)
    return out


def peak_weight_cells_env():
    """Resolve ``FSTAT_PEAK_WEIGHT_CELLS`` -- sub-band subdivisions per draw
    stratum.

    DEFAULT 1 = ONE STRATUM PER SUB-BAND (user ruling 2026-08-29: "the
    sub-bands should define that grid since those are the effective rj
    limits as well").

    It used to default to ``GB_CAP_DIVISOR``, tying the birth draw's
    stratification to the LEAF-CAP grid. That was a 2026-08-16 choice which
    no longer holds: caps live on cap cells (staggered, straddling seams by
    design so the two sides of a seam compete for one cap) while births are
    confined to sub-bands and scheduled serial-within-band. Coupling them
    also meant a change to ``GB_CAP_DIVISOR`` -- e.g. the 2026-08-29 move
    to the midpoint-to-midpoint grid -- silently re-stratified the RJ birth
    proposal, which is not something a cap knob should do.

    ``0`` falls back to the historical global ``w ~ F**alpha`` mixture; an
    explicit integer ``K > 1`` sub-divides each sub-band into K strata of
    its own width.
    """
    raw = os.environ.get("FSTAT_PEAK_WEIGHT_CELLS", "").strip()
    if not raw:
        return 1
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        logger.warning("[birth] FSTAT_PEAK_WEIGHT_CELLS=%r is not an "
                       "integer; using one stratum per sub-band.", raw)
        return 1


def peak_weight_alpha_env(epoch=None):
    """The ``w ~ F**alpha`` exponent, flattened after the warm-up epoch.

    ``F = SNR**2 / 2``, so alpha reads directly as a power of SNR:
    ``alpha=1 -> w ~ SNR**2`` (historical), ``0.5 -> ~SNR``,
    ``0.25 -> ~sqrt(SNR)``, ``0 -> flat``.

    Why two values. Epoch 0 fits against a residual with NOTHING subtracted,
    so peak strength tracks true source strength and preferring the loud end
    is right -- those sources have to come out first for the confusion
    residual to drop at all. Every later epoch fits against a residual whose
    found sources are already GONE, so what survives is the faint tail plus
    whatever the updated foreground/PSD estimate has revealed. Holding the
    epoch-0 tilt there keeps spending birth attempts on the brightest
    survivors, which are the ones least in need of help.

    ``FSTAT_PEAK_WEIGHT_ALPHA_LATE`` applies from epoch
    ``FSTAT_PEAK_WEIGHT_ALPHA_AFTER_EPOCH`` (default 1, i.e. the first
    refit) onward. It is INERT for a run that never refits
    (``GB_FSTAT_REFIT_EVERY<=0`` fits epoch 0 and stops), and inert for any
    caller that does not pass an epoch.

    Not lower than 0.25 by default: the SNR-8 selection floor makes pure
    noise peaks rare but the peak list still runs to thousands against a few
    hundred detectable sources, so flat weighting hands a marginal peak the
    same share as a real one. The heavy lifting is done by the refit itself
    -- a found source is subtracted, so its peak collapses -- and this is a
    second-order tilt on top of that.
    """
    early = float(os.environ.get("FSTAT_PEAK_WEIGHT_ALPHA", "1.0"))
    if epoch is None:
        return early
    raw = os.environ.get("FSTAT_PEAK_WEIGHT_ALPHA_LATE", "0.25").strip()
    if not raw:
        return early
    after = int(float(os.environ.get(
        "FSTAT_PEAK_WEIGHT_ALPHA_AFTER_EPOCH", "1")))
    return float(raw) if int(epoch) >= after else early


def build_gb_birth_distribution(*, cache_dir: str, mc_lims, A_lims,
                                dist_lims=None, fdot_astro_ratio_max=None,
                                use_cupy: bool = False, gpu: Optional[int] = None,
                                floor_eps: Optional[float] = None,
                                comb_weight: Optional[float] = None,
                                stacked_live=None,
                                expected_band_edges=None, epoch=None,
                                ratio_tight=None, tobs=None,
                                seed: Optional[int] = None,
                                peak_min_F: Optional[float] = None):
    """Stacked grids (+ optional comb) -> floor -> RJ birth container.

    ``peak_min_F`` (optional, F units): the LOADING move's own peak floor.
    A cache selected at a LOWER floor is restricted to the boxes at or above
    it (:func:`stacked_cache_at_floor` -- the same catalog a re-selection
    from the comb at that floor gives, with no refit); a cache selected at a
    STRICTER floor is kept as is, loudly. ``None`` (the default, every move
    without a floor of its own) loads the cache unchanged. Never applied to
    ``stacked_live``: a grid fitted in this process was selected at the
    floor in force.

    ``stacked_live`` short-circuits the npz reload when the caller just built
    the grids in-process (the in-move fit). ``comb_weight > 0`` adds a
    :class:`CombIntrinsicProposal` with linear-in-F weighting -- proportional
    mass on EVERY comb peak, which is what successive births need after the
    loudest source is born and subtracted.

    ``seed`` (optional): ONE integer that re-derives the RNG stream of every
    node of the assembled container -- the cache-loaded / live stacks, the
    comb, the mixture, the uniform floor and the outer wrappers -- through
    :func:`~lisatools.sampling.fstat_proposal.reseed_birth_tree`. Every
    constructor above still builds with ``seed=None``; the single reseed at
    the end overrides them deterministically. ``seed=None`` changes NOTHING
    (every stream stays on OS entropy, exactly as before), which is the
    ``general.random_seed is None`` path.

    ``expected_band_edges`` (optional, Hz): when given and the grids come
    from the npz caches (not ``stacked_live``), the cached band grid is
    validated via :func:`check_cached_band_grid` -- a cache fitted against a
    different band count/edges raises instead of silently mis-labeling
    ``band_idx``.

    Returns ``None`` when no grid source is available.
    """
    from lisatools.sampling.fstat_proposal import (
        CombIntrinsicProposal,
        MixtureProposal,
        StackedFStatProposal4D,
        UniformFloorMixture,
        fstat_knob,
        make_gb_rj_birth_container,
        reseed_birth_tree,
        stacked_from_cache,
    )

    cache_path = os.path.join(cache_dir, GRID_BASENAME)
    comb_cache = cache_path.replace(".npz", "_comb.npz")
    stacked_cache = cache_path.replace(".npz", "_peaks_stacked.npz")

    if floor_eps is None:
        floor_eps = fstat_knob("FSTAT_FLOOR_EPS", float)
    if comb_weight is None:
        comb_weight = fstat_knob("FSTAT_COMB_WEIGHT", float)

    # Band-grid staleness gate: an npz reload against a changed band grid is
    # never valid (band_idx labels shift). The live in-process fit is exempt
    # (it was just run against the current edges).
    if expected_band_edges is not None and stacked_live is None:
        check_cached_band_grid(cache_dir, expected_band_edges)

    cache = None
    if os.path.exists(stacked_cache):
        cache = np.load(stacked_cache, allow_pickle=False)
    if peak_min_F is not None and stacked_live is None and cache is not None:
        _have = (float(cache["peak_min_F"]) if "peak_min_F" in cache
                 else None)
        cache, _n_kept, _n_all = stacked_cache_at_floor(cache, peak_min_F)
        _snr = float(np.sqrt(2.0 * float(peak_min_F)))
        if _n_all is None:
            logger.warning(
                "[birth] %s carries no peak_F; cannot apply this move's "
                "peak floor F >= %.4f (SNR %.3f) to it -- loaded unchanged.",
                stacked_cache, float(peak_min_F), _snr)
        elif _n_kept < _n_all:
            logger.info(
                "[birth] %s: selected at F >= %s; this move's floor is "
                "F >= %.4f (SNR %.3f) -> keeping %d of %d peak boxes (the "
                "catalog a re-selection from the comb at that floor gives; "
                "no refit).", stacked_cache,
                "unknown" if _have is None else
                f"{_have:.4f} (SNR {np.sqrt(2.0 * _have):.3f})",
                float(peak_min_F), _snr, _n_kept, _n_all)
        elif _have is not None and _have > float(peak_min_F) * (1 + 1e-9):
            logger.warning(
                "[birth] %s was selected at a STRICTER floor F >= %.4f (SNR "
                "%.3f) than this move's F >= %.4f (SNR %.3f): the peaks "
                "between the two are not in it and cannot be added without a "
                "refit (GB_FSTAT_FORCE_REFIT=1 on the launch line opens a new "
                "epoch). Loaded unchanged.", stacked_cache, _have,
                np.sqrt(2.0 * _have), float(peak_min_F), _snr)
        if cache is None:
            logger.warning(
                "[birth] %s: none of its %d peak boxes reaches this move's "
                "floor F >= %.4f (SNR %.3f); no peak component.",
                stacked_cache, _n_all, float(peak_min_F), _snr)
    if stacked_live is None and cache is None:
        return None

    mc_lims = list(mc_lims or [0.001, 1.0])
    if stacked_live is not None:
        peak_component = stacked_live
    else:
        # PEAK-BOX MIXTURE WEIGHT: w ~ F**alpha.
        #
        # WHY alpha EXISTS (measured on the 3-month run, 2026-08-16). The
        # F-statistic goes like SNR^2, so the historical w ~ F allocates
        # birth draws in proportion to SNR^2 -- an SNR-10 source gets 9x
        # FEWER attempts than an SNR-30 one, which is backwards: the faint
        # source is the one that needs more. On that run 80.3% of the birth
        # mass landed on cap cells whose source had ALREADY been found
        # (median peak F 125.5) against 7.8% on cells holding a detectable
        # source that had not (median peak F 26.6), and 146 of 332 cells
        # holding a detectable source were still empty at iteration 15.
        # Worse, the over-served cells are exactly the ones sitting at cap,
        # so much of that 80% was drawn only to be rejected by the cap gate.
        #
        # ``equal`` was the only escape hatch and it overcorrects: there are
        # ~5,000 peaks against a few hundred detectable sources, so flat
        # weighting hands noise peaks the same share as real ones. alpha
        # interpolates -- alpha=0.5 makes draws ~ SNR rather than SNR^2,
        # still preferring real signal without starving the faint tail by
        # the square.
        #
        # alpha=1 reproduces the historical ``fstat`` behaviour EXACTLY and
        # is the default, so this is inert until someone sets it; alpha=0
        # is ``equal`` (F**0 == 1 for every box, including F == 0).
        # ``FSTAT_PEAK_WEIGHTING=equal`` still short-circuits, and wins.
        weighting = os.environ.get("FSTAT_PEAK_WEIGHTING", "fstat").strip().lower()
        peak_F = np.clip(np.asarray(cache["peak_F"], dtype=float), 0.0, None)
        alpha = peak_weight_alpha_env(epoch)
        box_weights = peak_box_weights(
            peak_F,
            peak_f0_mHz=cache["peak_f0_mHz"] if "peak_f0_mHz" in cache else None,
            band_edges=cache["band_edges"] if "band_edges" in cache else None,
            alpha=alpha, cells=peak_weight_cells_env(),
            equal=(weighting == "equal"))
        if box_weights is not None and not np.any(box_weights > 0):
            # Every box would have zero mass -- StackedFStatProposal4D
            # normalizes by the sum, so this would be a divide-by-zero that
            # silently produces a NaN proposal. Fall back to flat.
            logger.warning("[birth] peak weights are all zero at alpha=%.3g; "
                           "falling back to equal weighting.", alpha)
            box_weights = None
        logger.info("[birth] peak-box weighting: %s (alpha=%.3g ~ SNR**%.3g, "
                    "epoch=%s)%s",
                    "equal" if box_weights is None else "w ~ F**alpha", alpha,
                    2.0 * alpha, epoch,
                    "" if box_weights is None else
                    f"; weight max/median = "
                    f"{box_weights.max() / max(np.median(box_weights), 1e-300):.1f}x")
        mem_mb = os.environ.get("FSTAT_GRID_MEM_MB", "").strip()
        peak_component = stacked_from_cache(
            cache, weights=box_weights,
            mem_budget_mb=float(mem_mb) if mem_mb else None,
            use_cupy=use_cupy,
        )

    if cache is not None and "band_idx" in cache and "band_edges" in cache:
        band_idx = np.asarray(cache["band_idx"], dtype=int)
        num_sub_bands = len(np.asarray(cache["band_edges"])) - 1
        counts = np.bincount(band_idx, minlength=num_sub_bands)
        logger.info("[birth] per-interior-band peak counts: %s",
                    dict(enumerate(counts.tolist())))
        zero = [k for k in range(1, num_sub_bands - 1) if counts[k] == 0]
        if zero:
            logger.warning("[birth] interior sub-band(s) %s have ZERO peak "
                           "grids -- births there rely on the comb/floor only.",
                           zero)

    comps_list, weights = [peak_component], [1.0]
    c = np.load(comb_cache, allow_pickle=False) if os.path.exists(comb_cache) else None
    if c is not None and comb_weight > 0:
        # AXIS 1 IS NOT ALWAYS Mc (same hazard as the floor box below).
        # Under the fdot basis the comb draws its column 1 as fdot
        # CONDITIONAL on the drawn f0 -- uniform over the feasible range at
        # that f0 -- because a global fdot range sized by the band's top is
        # unreachable almost everywhere below it, and the Mc box read as
        # Hz/s is garbage everywhere.
        _comb_kw = {}
        if fdot_axis_on():
            if fdot_astro_ratio_max is None:
                raise ValueError(
                    "FSTAT_FDOT_AXIS=1 with a comb component needs "
                    "fdot_astro_ratio_max to size the conditional fdot "
                    "draw; refusing rather than guessing.")
            _comb_kw = dict(fdot_conditional=True,
                            ratio_max=float(fdot_astro_ratio_max),
                            mc_hi=float(mc_lims[-1]))
        comps_list.append(CombIntrinsicProposal(
            c["f0_nodes_mHz"], c["F_max"], mc_lims=mc_lims, **_comb_kw,
        ))
        weights = [1.0 - comb_weight, float(comb_weight)]
        logger.info("[birth] comb component (linear-in-F) weight=%s",
                    comb_weight)
    base = comps_list[0] if len(comps_list) == 1 else MixtureProposal(
        comps_list, weights)

    # Floor box: f0 across the whole sampled band, everything else the full
    # prior ranges. UniformFloorMixture is MANDATORY -- BandSorter evaluates
    # logpdf for its factors and needs it finite everywhere reachable.
    if cache is not None and "band_edges" in cache:
        be = np.asarray(cache["band_edges"], dtype=float)
        f0_lo, f0_hi = float(be[1]) * 1e3, float(be[-2]) * 1e3
    elif c is not None:
        f0_lo, f0_hi = float(c["f0_nodes_mHz"][0]), float(c["f0_nodes_mHz"][-1])
    else:
        f0_los = np.asarray(peak_component.f0_los, dtype=float)
        f0_dxs = np.asarray(peak_component.f0_dxs, dtype=float)
        n0 = int(peak_component.logp_grids.shape[1])
        f0_lo = float(f0_los.min())
        f0_hi = float((f0_los + (n0 - 1) * f0_dxs).max())
    # AXIS 1 IS NOT ALWAYS Mc. Under FSTAT_FDOT_AXIS it is fdot in Hz/s, and
    # the floor must span the fdot axis, not the chirp-mass box. Getting this
    # wrong is spectacular rather than subtle -- a floor draw of "Mc = 1.0"
    # read as 1.0 Hz/s gives f0 = f_mid - (T/2)*1.0 = -3.9e9 mHz -- but it
    # surfaces only as a non-finite logpdf on the container's own draws,
    # which reads as a floor bug rather than an axis mix-up. Read the range
    # off the fitted axis itself, so it cannot drift from what stage B built.
    _ax1_lo, _ax1_hi = float(mc_lims[0]), float(mc_lims[-1])
    # ``in cache``, not ``in cache.files``: ``cache`` is the NpzFile or, cut to
    # a loading move's floor, the view stacked_cache_at_floor returns (which
    # has ``files`` too; a hand-built dict does not)
    if cache is not None and "grid_basis" in cache and \
            str(np.asarray(cache["grid_basis"]).item()) == "fdot":
        _axes = ([cache["mc_ax"]] if "mc_ax" in cache
                 else [cache[f"mc_ax_g{i}"]
                       for i in range(len(np.asarray(cache["group_sizes"])))])
        _ax1_lo = float(min(float(np.min(a)) for a in _axes))
        _ax1_hi = float(max(float(np.max(a)) for a in _axes))
    elif stacked_live is not None and fdot_axis_on():
        _comps = getattr(stacked_live, "components", [stacked_live])
        _ax1_lo = float(min(float(c._lo3[0]) for c in _comps))
        _ax1_hi = float(max(float(c._hi3[0]) for c in _comps))
    lo = [f0_lo, _ax1_lo, 0.0, -1.0]
    hi = [f0_hi, _ax1_hi, 2.0 * np.pi, 1.0]
    mix = UniformFloorMixture(base, lo, hi, eps=float(floor_eps))
    logger.info("[birth] floor box f0=[%.5f, %.5f] mHz  eps=%s  use_cupy=%s",
                f0_lo, f0_hi, floor_eps, use_cupy)
    if fdot_astro_ratio_max is not None:
        logger.info("[birth] appending fdot_astro_ratio ~ U[-%s, %s] "
                    "(9-column basis)", fdot_astro_ratio_max,
                    fdot_astro_ratio_max)
    if dist_lims is not None:
        logger.info("[birth] slot 0 = dist ~ U[%s, %s] kpc (distance basis)",
                    dist_lims[0], dist_lims[1])
    if ratio_tight is not None:
        logger.info("[birth] fdot_astro_ratio proposal TIGHTENED: "
                    "w = clip(%.3g/(pi*T^2*fdot_gr), %.3g, M), eps=%.3g "
                    "(prior unchanged)", ratio_tight.get("phase_rad", 6.283),
                    ratio_tight.get("w_min", 0.05),
                    ratio_tight.get("eps", 0.1))
    container = make_gb_rj_birth_container(
        mix, A_lims, use_cupy=use_cupy,
        fdot_astro_ratio_max=fdot_astro_ratio_max, dist_lims=dist_lims,
        ratio_tight=ratio_tight, tobs=tobs, mc_lims=mc_lims,
    )
    # ONE reseed for the whole tree (no-op when seed is None); see the
    # ``seed`` paragraph above and fstat_proposal's reseeding block.
    reseed_birth_tree(container, seed)
    return container


# --------------------------------------------------------------------------
# per-epoch F-stat CENTER table (the RJ distance-birth proposal's centers)
# --------------------------------------------------------------------------

def enumerate_center_nodes(cache_dir: str, *, mc_lims=None,
                           max_nodes: Optional[int] = None) -> dict:
    """The birth proposal's DRAWABLE f0 support, as a node list.

    The RJ birth container built by :func:`build_gb_birth_distribution` is
    ``UniformFloorMixture(MixtureProposal([stacked peaks, comb]), box)``, so
    the f0 a birth can carry comes from exactly three places:

    * **peak boxes** -- :class:`StackedFStatProposal4D` draws a cell of box
      ``k`` and jitters inside it, i.e. anywhere in ``[f0_los[k],
      f0_los[k] + (n_f0 - 1) * f0_dxs[k]]``. Its NODE grid is the finest
      structure the density has, so every node of every box is a table node.
      The other three axes are represented by the box's own F-stat argmax at
      that f0 node (``logp_grids[k, i].argmax()`` over Mc x alpha x
      sin_delta) -- the maximum the birth is aimed at.
    * **comb cells** -- :class:`CombIntrinsicProposal` draws a comb cell and
      jitters inside it, so its nodes (spacing ``~1/(2 Tobs)``) are table
      nodes too; sky comes from the comb scan's per-node best sky
      (``best_alpha`` / ``best_sin_delta``) and Mc from the comb's fixed
      ``FSTAT_COMB_MC``.
    * **uniform floor** -- f0 anywhere in the interior band span. No extra
      nodes: the comb covers that span at ``1/(2 Tobs)`` already, which is
      finer than anything a floor birth resolves.

    Returns a host dict with the sorted-by-f0 arrays ``f0_mHz``, ``mc``,
    ``alpha``, ``sin_delta`` (plus ``n_peak_nodes`` / ``n_comb_nodes``
    provenance counts). ``max_nodes`` (when the union exceeds it) thins the
    COMB nodes by a uniform stride and keeps every peak node -- peak nodes
    are where the birth mass actually is.
    """
    cache_path = os.path.join(cache_dir, GRID_BASENAME)
    stacked_cache = cache_path.replace(".npz", "_peaks_stacked.npz")
    comb_cache = cache_path.replace(".npz", "_comb.npz")

    f0_parts, mc_parts, al_parts, sd_parts = [], [], [], []
    n_peak = n_comb = 0

    if os.path.exists(stacked_cache):
        d = np.load(stacked_cache, allow_pickle=False)
        al_ax = np.asarray(d["alpha_ax"], dtype=float)
        sd_ax = np.asarray(d["sin_delta_ax"], dtype=float)
        f0_los_all = np.asarray(d["f0_los"], dtype=float)
        f0_dxs_all = np.asarray(d["f0_dxs"], dtype=float)
        # legacy single-stack npz vs banded-Mc groups (2026-08-26): one
        # (grids, f0 slice, mc_ax) block per group either way.
        # ⚠ THE SKY AXES ARE PER GROUP under the f0-adaptive sky
        # (2026-09-24). ``write_stacked_npz`` emits ``alpha_ax_g{gi}`` /
        # ``sin_delta_ax_g{gi}`` ONLY when the groups actually differ, and
        # keeps the flat ``alpha_ax`` / ``sin_delta_ax`` as the shared case
        # so a pinned-sky cache stays byte-identical. This reader used the
        # FLAT pair for every group, which is right only in that shared
        # case; with an adaptive grid it unravelled a group's argmax
        # against the wrong shape:
        #
        #   ValueError: index 358 is out of bounds for array with size 189
        #
        # -- and where the group's sky happened to be SMALLER than the flat
        # axes it would not have raised at all, just silently placed births
        # at sky angles the grid was never scored at. Resolve per group,
        # falling back to the flat axes when the suffixed keys are absent.
        def _sky_for(gi):
            ka, ks = f"alpha_ax_g{gi}", f"sin_delta_ax_g{gi}"
            if ka in d.files and ks in d.files:
                return (np.asarray(d[ka], dtype=float),
                        np.asarray(d[ks], dtype=float))
            return al_ax, sd_ax

        if "logp_grids" in d.files:
            blocks = [(np.asarray(d["logp_grids"], dtype=float),
                       f0_los_all, f0_dxs_all,
                       np.asarray(d["mc_ax"], dtype=float),
                       al_ax, sd_ax)]
        else:
            sizes = np.asarray(d["group_sizes"], dtype=int)
            bounds = np.concatenate([[0], np.cumsum(sizes)])
            blocks = [
                (np.asarray(d[f"logp_grids_g{gi}"], dtype=float),
                 f0_los_all[a:b], f0_dxs_all[a:b],
                 np.asarray(d[f"mc_ax_g{gi}"], dtype=float),
                 *_sky_for(gi))
                for gi, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))
            ]
        # Under the fdot basis axis 2 is fdot [Hz/s], not Mc, and axis 0
        # carries f_mid. The table is looked up BY f0, so a non-sheared
        # table puts every entry up to ~11 bins from its own maximum --
        # degrading birth quality with no error at all.
        _basis = (str(np.asarray(d["grid_basis"]).item())
                  if "grid_basis" in d.files else "Mc")
        _c_t = (float(np.asarray(d["grid_c_t"]).item())
                if "grid_c_t" in d.files else 0.0)
        for grids, f0_los, f0_dxs, mc_ax, g_al_ax, g_sd_ax in blocks:
            K, n_f0 = grids.shape[0], grids.shape[1]
            # argmax over the (axis2, alpha, sin_delta) block at each
            # (box, f0 node)
            flat = grids.reshape(K, n_f0, -1).argmax(axis=2)
            # The shape MUST be this group's own -- assert it against the
            # grid rather than trusting the axes, because a mismatch that
            # happens to be small enough not to overflow places births at
            # sky angles that were never scored.
            _want = tuple(int(s) for s in grids.shape[2:5])
            _have = (len(mc_ax), len(g_al_ax), len(g_sd_ax))
            if _want != _have:
                raise ValueError(
                    f"enumerate_center_nodes: grid block has "
                    f"(mc, alpha, sin_delta) shape {_want} but the axes "
                    f"resolved to {_have} -- the per-group sky axes and the "
                    f"grid disagree, so every center node from this group "
                    f"would be placed at the wrong sky angle.")
            i_mc, i_al, i_sd = np.unravel_index(flat, _have)
            f0 = f0_los[:, None] + np.arange(n_f0)[None, :] * f0_dxs[:, None]
            if _basis == "fdot":
                fd = mc_ax[i_mc]                       # Hz/s at the argmax
                f0 = f0 - _c_t * fd * 1e3              # f_mid -> f0 [mHz]
                # A REPRESENTATIVE chirp mass, since the contract is
                # (f0, mc, alpha, sd): the midpoint of the feasible
                # interval, i.e. the centre of the Mc range on which this
                # fdot is reachable with |r| <= M. mc_lims is the caller's
                # prior box; without it fall back to the axis' own range.
                _ml = list(mc_lims or [0.001, 1.0])
                _rm = float(os.environ.get("FSTAT_FDOT_RATIO_MAX", "5.0"))
                floor = mc_floor_for_fdot(fd, f0 * 1e-3, _rm, float(_ml[0]))
                # CLIP to the box: the floor is evaluated at the SHEARED f0
                # (up to ~1 bin below the node's f_mid), so at group-edge
                # nodes it can overshoot mc_hi by ~1e-4 -- the same
                # boundary-rounding class the birth block absorbs with an
                # epsilon. This is a REPRESENTATIVE centre for a smeared
                # lognormal table lookup, so clipping is exact enough by
                # construction.
                mc_parts.append(np.clip(
                    0.5 * (floor + float(_ml[-1])),
                    float(_ml[0]), float(_ml[-1])).ravel())
            else:
                mc_parts.append(mc_ax[i_mc].ravel())
            f0_parts.append(f0.ravel())
            al_parts.append(g_al_ax[i_al].ravel())
            sd_parts.append(g_sd_ax[i_sd].ravel())
            n_peak += int(f0.size)

    if os.path.exists(comb_cache):
        c = np.load(comb_cache, allow_pickle=False)
        f0c = np.asarray(c["f0_nodes_mHz"], dtype=float)
        mc_lims = list(mc_lims or [0.001, 1.0])
        mc_fix = float(os.environ.get(
            "FSTAT_COMB_MC", 0.5 * (float(mc_lims[0]) + float(mc_lims[-1]))))
        if "best_alpha" in c and "best_sin_delta" in c:
            alc = np.asarray(c["best_alpha"], dtype=float)
            sdc = np.asarray(c["best_sin_delta"], dtype=float)
        else:
            # Legacy comb cache with no per-node best sky: the F-stat maximum
            # would be evaluated at an arbitrary sky point, so say so loudly
            # rather than pretend the centers are on-peak.
            logger.warning(
                "[centers] comb cache %s has no best_alpha/best_sin_delta; "
                "comb nodes fall back to (alpha, sin_delta) = (0, 0).",
                comb_cache)
            alc = np.zeros_like(f0c)
            sdc = np.zeros_like(f0c)
        keep = np.ones(len(f0c), dtype=bool)
        if max_nodes and n_peak + len(f0c) > int(max_nodes):
            room = max(1, int(max_nodes) - n_peak)
            stride = int(np.ceil(len(f0c) / room))
            keep[:] = False
            keep[::stride] = True
            logger.warning(
                "[centers] %d peak + %d comb nodes exceed max_nodes=%d; "
                "comb thinned by stride %d (-> %d nodes).", n_peak, len(f0c),
                int(max_nodes), stride, int(keep.sum()))
        f0_parts.append(f0c[keep])
        mc_parts.append(np.full(int(keep.sum()), mc_fix))
        al_parts.append(alc[keep])
        sd_parts.append(sdc[keep])
        n_comb = int(keep.sum())

    if not f0_parts:
        return dict(f0_mHz=np.empty(0), mc=np.empty(0), alpha=np.empty(0),
                    sin_delta=np.empty(0), n_peak_nodes=0, n_comb_nodes=0)

    f0 = np.concatenate(f0_parts)
    order = np.argsort(f0, kind="stable")
    return dict(
        f0_mHz=f0[order],
        mc=np.concatenate(mc_parts)[order],
        alpha=np.concatenate(al_parts)[order],
        sin_delta=np.concatenate(sd_parts)[order],
        n_peak_nodes=n_peak,
        n_comb_nodes=n_comb,
    )


def run_center_sweep(call_fstat: Callable, nodes: dict, *, xp,
                     batch: Optional[int] = None) -> dict:
    """ONE batched F-stat sweep over the center nodes -> maximized extrinsics.

    Same row convention as :func:`run_comb_scan` / :func:`run_stacked_peak_sweep`
    (physical 9-column ``[A, f0, fdot, fddot, phi0, iota, psi, alpha, delta]``
    with a placeholder amplitude; the F-stat maximizes over amplitude/phase/
    iota/psi so those input columns are ignored), and the SAME
    Jaranowski-Krol inversion the per-row propose-time path uses
    (:func:`lisatools.sampling.fstat_proposal.fstat_maximized_extrinsics`).

    Returns host arrays ``ln_A_max``, ``phi0``, ``iota``, ``psi``,
    ``ln_snr`` and ``sigma_base`` (``= 1/SNR``, the UNSMEARED proposal
    width), one entry per node.
    """
    from gbgpu.utils.utility import get_fdot

    from lisatools.sampling.fstat_proposal import (
        fstat_knob,
        fstat_maximized_extrinsics,
    )

    f0 = np.asarray(nodes["f0_mHz"], dtype=float)
    n = int(f0.size)
    out = {k: np.zeros(n) for k in
           ("ln_A_max", "phi0", "iota", "psi", "ln_snr", "sigma_base")}
    if n == 0:
        return out
    batch = int(batch or fstat_knob("FSTAT_BATCH", int))
    mc = np.asarray(nodes["mc"], dtype=float)
    al = np.asarray(nodes["alpha"], dtype=float)
    sd = np.asarray(nodes["sin_delta"], dtype=float)
    t0 = time.time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        # Rows are assembled on the HOST -- the same convention
        # :func:`run_comb_scan` uses. Every input column is host data
        # already, so building the block on the device only forced the
        # scorer to read it back (the sig-het scorer downloads the f0
        # column; the shard-routed scorer downloads the whole block).
        pr = np.zeros((e - s, 9))
        pr[:, 0] = 1e-22
        pr[:, 1] = f0[s:e] * 1e-3
        pr[:, 2] = get_fdot(f=pr[:, 1], Mc=mc[s:e])
        pr[:, 5] = 0.5 * np.pi
        pr[:, 7] = al[s:e]
        pr[:, 8] = np.arcsin(np.clip(sd[s:e], -1.0, 1.0))
        N_arr, M_up = call_fstat(pr)
        A, phi0, iota, psi, F = fstat_maximized_extrinsics(
            xp.asarray(N_arr), xp.asarray(M_up))
        # ONE device->host transfer for the batch: five separate ``_host``
        # calls are five hard syncs, and the five outputs are all (m,)
        # float64 rows of the same block.
        blk = _to_host(xp.stack([A, phi0, iota, psi, F]))
        ln_snr = 0.5 * np.log(np.clip(2.0 * blk[4], 1.0, None))
        out["ln_A_max"][s:e] = np.log(np.clip(blk[0], 1e-300, None))
        out["phi0"][s:e] = blk[1]
        out["iota"][s:e] = blk[2]
        out["psi"][s:e] = blk[3]
        out["ln_snr"][s:e] = ln_snr
        out["sigma_base"][s:e] = np.exp(-ln_snr)
    logger.info("[centers] %d node F-stat evals in one batched sweep "
                "(batch=%d) in %s", n, batch, _fmt_secs(time.time() - t0))
    return out


def build_fstat_center_table(call_fstat: Optional[Callable], *, cache_dir: str,
                             xp, mc_lims=None, max_nodes: Optional[int] = None,
                             rebuild: bool = False) -> Optional[dict]:
    """Load (or build + persist) an epoch's F-stat center table.

    ``<cache_dir>/fstat_centers.npz`` is the epoch artifact, written next to
    the grids and reloaded the same way -- a restart that loads a complete
    epoch loads its centers too, in milliseconds. It is MISSING for any epoch
    fitted before this table existed (and for an offline grid dropped into
    ``epoch_0000``), so an absent file simply rebuilds it against
    ``call_fstat`` and writes it out.

    Returns the host table dict (keys ``f0_mHz``, ``mc``, ``alpha``,
    ``sin_delta``, ``ln_A_max``, ``phi0``, ``iota``, ``psi``, ``ln_snr``,
    ``sigma_base``), or ``None`` when the epoch has no drawable support at
    all (no grids and no comb -> the birth container fell back to the prior)
    or when a rebuild is needed but no ``call_fstat`` was supplied.
    """
    path = os.path.join(cache_dir, CENTER_TABLE_BASENAME)
    if os.path.exists(path) and not rebuild:
        d = np.load(path, allow_pickle=False)
        tbl = {k: np.asarray(d[k], dtype=float) for k in d.files}
        logger.info("[centers] loaded %d-node center table from %s",
                    len(tbl["f0_mHz"]), path)
        return tbl

    nodes = enumerate_center_nodes(cache_dir, mc_lims=mc_lims,
                                   max_nodes=max_nodes)
    if int(nodes["f0_mHz"].size) == 0:
        logger.warning("[centers] epoch %s has no drawable f0 support "
                       "(no peak grids, no comb); no center table.", cache_dir)
        return None
    if call_fstat is None:
        return None
    logger.info("[centers] building the center table for %s: %d nodes "
                "(%d peak-box + %d comb)", cache_dir,
                int(nodes["f0_mHz"].size), int(nodes["n_peak_nodes"]),
                int(nodes["n_comb_nodes"]))
    swept = run_center_sweep(call_fstat, nodes, xp=xp)
    tbl = dict(
        f0_mHz=nodes["f0_mHz"], mc=nodes["mc"], alpha=nodes["alpha"],
        sin_delta=nodes["sin_delta"], **swept,
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, **tbl)
        logger.info("[cache] wrote %s", path)
    except OSError as exc:  # the table is a cache, never fatal
        logger.warning("[centers] could not write %s (%r)", path, exc)
    return tbl
