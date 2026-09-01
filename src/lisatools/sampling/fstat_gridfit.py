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
    "select_comb_peaks",
    "run_comb_scan",
    "run_stacked_peak_sweep",
    "run_stacked_stage_b",
    "run_fstat_grid_fit",
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
            raise ValueError(
                f"F-stat checkpoint payload {dat!r} holds {max(have, 0)} "
                f"rows but the append starts at {s}; refusing to write a "
                f"payload with a hole.")
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


def select_comb_peaks(f0_nodes, F_max, band_edges_hz, spacing, xp):
    """Vectorized per-sub-band peak selection from the comb's ``F_max(f0)``.

    Two-tier candidate set -> absolute SNR floor + interior-band mask ->
    per-band cap:

    * **Tier 1** (window champion): a node survives iff it carries the max F
      within +-``min_sep`` nodes (the Doppler envelope ~3e-3 mHz).
    * **Tier 2** (local-max rescue): ALSO keep any strict 3-point local
      maximum, gated only by the shared SNR floor -- so a quiet real source
      inside a loud neighbor's Doppler window still gets its own grid.

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

    min_sep = max(1, int(round(3e-3 / spacing)))
    wmax = _windowed_max(F_max, min_sep, xp)
    tier1 = F_max >= wmax
    local3 = xp.zeros(F_max.shape, dtype=bool)
    local3[1:-1] = (F_max[1:-1] > F_max[:-2]) & (F_max[1:-1] > F_max[2:])

    min_F = fstat_peak_min_F()
    interior = (band_of_node >= 1) & (band_of_node <= num_sub_bands - 2)
    cand = (tier1 | local3) & (F_max >= min_F) & interior

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
    logger.info("[peaks] %d peaks (F >= %.1f ~ SNR %.1f, cap %d/band); "
                "per-interior-band counts: %s", len(peaks), min_F,
                np.sqrt(2 * min_F), cap, dict(enumerate(counts.tolist())))
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


def run_comb_scan(call_fstat: Callable, *, xp, Tobs: float, band_edges_hz,
                  f0_lims_hz, mc_lims, cache_path: Optional[str] = None,
                  fingerprint_extra: str = ""):
    """Dense-in-f0 F-stat comb scan across the sub-band.

    With months of data the F-stat f0 peaks are ~1/Tobs wide -- far too
    narrow for any feasible 4-D tensor grid to land on. But over the same
    stretch ``fdot*Tobs << 1/Tobs``, so Mc is nearly unmeasurable per band
    and is held FIXED, and sky only enters through the Doppler ridge. The
    right scan is therefore dense in f0 (spacing ~ 1/(2*Tobs)) x a small
    spread of sky points, maximized over sky per f0 node -- assembled into
    ONE parameter set per sky level and swept in a single chunked stream.

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
    # otherwise ADAPTIVE: the comb resolves a source through its Doppler
    # ridge, so the sky grid scales as ~(f0*Tobs*v/c)^2, floored, capped,
    # and snapped up to powers of two so the sweep runs as a few rectangular
    # batches instead of one giant fixed-n_sky design.
    vc = float(os.environ.get("FSTAT_COMB_SKY_VC", "1e-4"))
    nsky_min = int(os.environ.get("FSTAT_COMB_NSKY_MIN", "16"))
    nsky_max = int(os.environ.get("FSTAT_COMB_NSKY_MAX", "512"))
    _fixed = os.environ.get("FSTAT_COMB_NSKY", "").strip()
    if _fixed:
        nsky_per_node = np.full(n_nodes, int(_fixed), dtype=int)
    else:
        req = np.ceil((f0_nodes * 1e-3 * Tobs * vc) ** 2)
        lvl = (2 ** np.ceil(np.log2(np.clip(req, 1.0, None)))).astype(int)
        nsky_per_node = np.clip(lvl, nsky_min, nsky_max).astype(int)

    levels = np.unique(nsky_per_node)
    logger.info("[comb] %d f0 nodes; sky %s [%d..%d] over %d level(s); "
                "spacing %.3e mHz = %.2f/Tobs; Mc fixed %.3f", n_nodes,
                "FIXED=" + str(int(levels[0])) if len(levels) == 1
                else "ADAPTIVE",
                int(nsky_per_node.min()), int(nsky_per_node.max()),
                len(levels), spacing, spacing / (1e3 / Tobs), mc_fix)

    parts_dir = cache_path.replace(".npz", "_parts") if cache_path else None
    F_max_host = np.zeros(n_nodes)
    best_al_host = np.zeros(n_nodes)
    best_sd_host = np.zeros(n_nodes)
    # Per-level eval counts upfront so each level header carries the OVERALL
    # comb fraction (the per-sweep lines only know their own level).
    _lv_evals = {int(lv): int(lv) * int((nsky_per_node == lv).sum())
                 for lv in levels}
    _grand_total = sum(_lv_evals.values())
    total_evals = 0
    for _li, lv in enumerate(levels):
        idx = np.where(nsky_per_node == lv)[0]
        nodes = f0_nodes[idx]
        nn = len(nodes)
        logger.info(
            "[comb] level %d/%d (nsky=%d): %d nodes x %d sky = %d evals; "
            "comb overall %.1f%% done, %.1f%% after this level",
            _li + 1, len(levels), int(lv), nn, int(lv), _lv_evals[int(lv)],
            100.0 * total_evals / max(_grand_total, 1),
            100.0 * (total_evals + _lv_evals[int(lv)]) / max(_grand_total, 1))
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
        params = _CombRows(nodes, al, sd, mc_fix)
        Fd = chunked_fstat_sweep(
            call_fstat, params, xp=xp, label=f":comb.nsky{int(lv)}",
            ckpt=(os.path.join(parts_dir, f"comb_nsky{int(lv)}")
                  if parts_dir else None),
            fingerprint_extra=fingerprint_extra,
        ).reshape(nn, int(lv))
        _kb = _to_host(Fd.argmax(axis=1)).astype(int)
        F_max_host[idx] = _to_host(Fd.max(axis=1))
        best_al_host[idx] = al[_kb]
        best_sd_host[idx] = sd[_kb]
        total_evals += int(lv) * nn
    logger.info("[comb] total %d F-stat evals across %d sky level(s)",
                total_evals, len(levels))

    peaks = select_comb_peaks(f0_nodes, xp.asarray(F_max_host), band_edges_hz,
                              spacing, xp)
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
                 best_sin_delta=best_sd_host, nsky_per_node=nsky_per_node)
        logger.info("[cache] wrote %s", comb_cache)
        # The comb npz is now the durable artifact; drop the per-level
        # progress files so a later knob change can't resurrect stale rows.
        ckpt_clear(parts_dir, "comb_")

    extras = dict(sky_alpha=_al_top, sky_sin_delta=_sd_top,
                  F_all=F_max_host[None, :], best_alpha=best_al_host,
                  best_sin_delta=best_sd_host, mc_fix=mc_fix,
                  spacing=spacing, nsky_per_node=nsky_per_node)
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
                        fingerprint_extra: str = "", epoch=None):
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
    _fdot_axis = os.environ.get("FSTAT_FDOT_AXIS", "0").strip() == "1"
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
    g_edges = np.concatenate(
        [[0], np.flatnonzero(np.diff(levels)) + 1, [K]]).astype(int)
    n_groups = len(g_edges) - 1

    alpha_ax = np.linspace(0.0, 2 * np.pi, n_alpha)
    sd_ax = np.linspace(-1.0, 1.0, n_sd)
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
        node_shape = (b - a, n_f0, n_Mc, n_alpha, n_sd)
        logger.info(
            "[stageB] group %d/%d: %d peak boxes x %dx%dx%dx%d nodes "
            "(f0 %.4f-%.4f mHz); axis2 = %s %s; f0 boxes clamped to their "
            "sub-bands.", gi + 1, n_groups, b - a, n_f0, n_Mc, n_alpha,
            n_sd, float(f0_los[a]), float(f0_los[b - 1]),
            "fdot[Hz/s]" if _fdot_axis else "Mc",
            (f"[{mc_ax[0]:.4e}, {mc_ax[-1]:.4e}]" if _fdot_axis
             else str(mc_range)))
        # single group keeps the historical "stageb" checkpoint name so
        # in-flight fits resume across this code change.
        _ck = "stageb" if n_groups == 1 else f"stageb_g{gi}"
        grids_g.append(run_stacked_peak_sweep(
            call_fstat, f0_los[a:b], f0_dxs[a:b], mc_ax, alpha_ax, sd_ax,
            node_shape, xp=xp,
            ckpt=os.path.join(_parts, _ck) if _parts else None,
            fingerprint_extra=fingerprint_extra,
            fdot_axis=_fdot_axis, c_t=_c_t,
        ))  # beta = 1: logp = F
        mc_ax_g.append(mc_ax)
        n_mc_g.append(n_Mc)

    if n_groups == 1:
        stacked = StackedFStatProposal4D(
            grids_g[0], f0_los, f0_dxs, mc_ax_g[0], alpha_ax, sd_ax,
            weights=box_w, mem_budget_mb=mem_budget,
        )
        if cache_path:
            stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
            os.makedirs(os.path.dirname(stacked_path), exist_ok=True)
            np.savez(
                stacked_path,
                logp_grids=_to_host(grids_g[0]), f0_los=f0_los,
                f0_dxs=f0_dxs, mc_ax=mc_ax_g[0], alpha_ax=alpha_ax,
                sin_delta_ax=sd_ax,
                # THE BASIS IS PART OF THE CACHE. Axis 2's VALUES differ
                # between the two meanings, but a consumer that reads them
                # as chirp masses when they are Hz/s gets no error at all --
                # just births at absurd parameters. Stamp it, and refuse a
                # mismatch on load.
                grid_basis=("fdot" if _fdot_axis else "Mc"),
                grid_c_t=float(_c_t if _fdot_axis else 0.0),
                peak_f0_mHz=peaks[:, 0], peak_F=peaks[:, 1],
                band_idx=band_idx,
                band_f0_lo=band_edges_mHz[band_idx],
                band_f0_hi=band_edges_mHz[band_idx + 1],
                band_edges=np.asarray(band_edges_hz, dtype=float),
            )
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
            grids_g[gi], f0_los[a:b], f0_dxs[a:b], mc_ax_g[gi], alpha_ax,
            sd_ax, weights=sub_w, mem_budget_mb=mem_budget))
    stacked = GroupedStackedFStatProposal(comps, box_weights=box_w)
    _sizes = np.diff(g_edges).astype(int)
    logger.info(
        "[stageB] banded Mc stacks: %d groups, sizes %s, n_Mc %s "
        "(a single max-f0 stack would carry %d Mc nodes on every box).",
        n_groups, _sizes.tolist(), n_mc_g, int(n_req.max()))

    if cache_path:
        stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
        os.makedirs(os.path.dirname(stacked_path), exist_ok=True)
        group_arrays = {}
        for gi in range(n_groups):
            group_arrays[f"logp_grids_g{gi}"] = _to_host(grids_g[gi])
            group_arrays[f"mc_ax_g{gi}"] = mc_ax_g[gi]
        np.savez(
            stacked_path,
            group_sizes=_sizes, f0_los=f0_los, f0_dxs=f0_dxs,
            alpha_ax=alpha_ax, sin_delta_ax=sd_ax,
            grid_basis=("fdot" if _fdot_axis else "Mc"),
            grid_c_t=float(_c_t if _fdot_axis else 0.0),
            peak_f0_mHz=peaks[:, 0], peak_F=peaks[:, 1], band_idx=band_idx,
            band_f0_lo=band_edges_mHz[band_idx],
            band_f0_hi=band_edges_mHz[band_idx + 1],
            band_edges=np.asarray(band_edges_hz, dtype=float),
            **group_arrays,
        )
        logger.info("[cache] wrote %s (%d Mc groups)", stacked_path,
                    n_groups)
        ckpt_clear(_parts, "stageb")
    return stacked


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def run_fstat_grid_fit(call_fstat: Callable, *, xp, Tobs: float,
                       band_edges_hz, f0_lims_hz, mc_lims, cache_dir: str,
                       fingerprint_extra: str = "", epoch=None):
    """Full fit with resume: comb scan -> peak select -> stage B.

    ``epoch`` selects the peak-box weighting tilt only (see
    :func:`peak_weight_alpha_env`); it does not change what is computed or
    cached, so the npz caches stay interchangeable across epochs.

    Cache reuse IS the mid-fit resume, and it is always on:

    * ``<cache_dir>/fstat_grid_peaks_stacked.npz`` present -> load and return
      (the fit is done; nothing recomputed).
    * ``<cache_dir>/fstat_grid_comb.npz`` present -> reload the comb, re-select
      peaks (cheap, deterministic), run only stage B.
    * otherwise -> run both stages.

    Returns ``(stacked_or_None, n_peaks)``.
    """
    from lisatools.sampling.fstat_proposal import stacked_from_cache

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, GRID_BASENAME)
    comb_cache = cache_path.replace(".npz", "_comb.npz")
    stacked_cache = cache_path.replace(".npz", "_peaks_stacked.npz")

    if os.path.exists(stacked_cache):
        d = np.load(stacked_cache, allow_pickle=False)
        logger.info("[fit] stage B already complete: %s", stacked_cache)
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

    if os.path.exists(comb_cache):
        d = np.load(comb_cache, allow_pickle=False)
        logger.info("[fit] reusing comb cache %s; re-selecting peaks",
                    comb_cache)
        f0_nodes = np.asarray(d["f0_nodes_mHz"], dtype=float)
        spacing = float(f0_nodes[1] - f0_nodes[0]) if len(f0_nodes) > 1 else 1.0
        peaks = select_comb_peaks(f0_nodes, xp.asarray(d["F_max"]),
                                  band_edges_hz, spacing, xp)
    else:
        _f0, _F, peaks, _x = run_comb_scan(
            call_fstat, xp=xp, Tobs=Tobs, band_edges_hz=band_edges_hz,
            f0_lims_hz=f0_lims_hz, mc_lims=mc_lims, cache_path=cache_path,
            fingerprint_extra=fingerprint_extra,
        )

    stacked = run_stacked_stage_b(
        call_fstat, peaks, xp=xp, Tobs=Tobs, band_edges_hz=band_edges_hz,
        mc_lims=mc_lims, cache_path=cache_path,
        fingerprint_extra=fingerprint_extra, epoch=epoch,
    )
    return stacked, int(len(peaks))


# --------------------------------------------------------------------------
# birth container
# --------------------------------------------------------------------------

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
                                ratio_tight=None, tobs=None):
    """Stacked grids (+ optional comb) -> floor -> RJ birth container.

    ``stacked_live`` short-circuits the npz reload when the caller just built
    the grids in-process (the in-move fit). ``comb_weight > 0`` adds a
    :class:`CombIntrinsicProposal` with linear-in-F weighting -- proportional
    mass on EVERY comb peak, which is what successive births need after the
    loudest source is born and subtracted.

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
        comps_list.append(CombIntrinsicProposal(
            c["f0_nodes_mHz"], c["F_max"], mc_lims=mc_lims,
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
    lo = [f0_lo, mc_lims[0], 0.0, -1.0]
    hi = [f0_hi, mc_lims[-1], 2.0 * np.pi, 1.0]
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
    return make_gb_rj_birth_container(
        mix, A_lims, use_cupy=use_cupy,
        fdot_astro_ratio_max=fdot_astro_ratio_max, dist_lims=dist_lims,
        ratio_tight=ratio_tight, tobs=tobs, mc_lims=mc_lims,
    )


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
        if "logp_grids" in d.files:
            blocks = [(np.asarray(d["logp_grids"], dtype=float),
                       f0_los_all, f0_dxs_all,
                       np.asarray(d["mc_ax"], dtype=float))]
        else:
            sizes = np.asarray(d["group_sizes"], dtype=int)
            bounds = np.concatenate([[0], np.cumsum(sizes)])
            blocks = [
                (np.asarray(d[f"logp_grids_g{gi}"], dtype=float),
                 f0_los_all[a:b], f0_dxs_all[a:b],
                 np.asarray(d[f"mc_ax_g{gi}"], dtype=float))
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
        for grids, f0_los, f0_dxs, mc_ax in blocks:
            K, n_f0 = grids.shape[0], grids.shape[1]
            # argmax over the (axis2, alpha, sin_delta) block at each
            # (box, f0 node)
            flat = grids.reshape(K, n_f0, -1).argmax(axis=2)
            i_mc, i_al, i_sd = np.unravel_index(
                flat, (len(mc_ax), len(al_ax), len(sd_ax)))
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
                mc_parts.append((0.5 * (floor + float(_ml[-1]))).ravel())
            else:
                mc_parts.append(mc_ax[i_mc].ravel())
            f0_parts.append(f0.ravel())
            al_parts.append(al_ax[i_al].ravel())
            sd_parts.append(sd_ax[i_sd].ravel())
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
