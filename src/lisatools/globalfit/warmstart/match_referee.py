"""MATCH-STATISTIC REFEREE for the warm-start GB clustering.

Stage 2 of the warm-start build (:mod:`.fit_from_store` -> referee ->
:mod:`.referee_apply`), run in-process by
:func:`.build.ensure_warm_start_components` and by every self-refit epoch
(:func:`.self_refit.build_self_epoch`) with only ``--npz`` / ``--store``
given (every other option at its default). Judges the components npz
produced by :mod:`.fit_from_store` with actual waveform matches (existing
machinery only -- stock GB transform + gbgpu FD templates, per the
overlap-validation pattern of ``gb_mojito_match.py``):

  1. WITHIN-GROUP COHERENCE: for each refereed component, real member
     rows (re-extracted from the store -- the fit's own last
     ``meta["last_k"]`` stored rows -- by nearest-centroid assignment
     in each component's own sigma-whitened space, :func:`assign_members`)
     are matched against the member closest to the component mean
     ("anchor"). A good group has median member match ~1. Refereed:
     every component with ``p > --p-core``, a random ``--tail-n`` of the
     rest, and the island of ``--flagship-f0``.
  2. MERGE TEST: for every same-island component pair with ``|delta f0|``
     <= ``--pair-bins`` / Tobs (both with ``p > --pair-p-floor``, or in the
     flagship island), the centroid-vs-centroid match. High cross-match
     (> 0.9) with coherent groups = split artifact (should merge); low
     (< 0.5) = genuinely distinct.

For an observable-basis fit every check runs per MIXTURE COMPONENT (the
per-cluster ``p`` / ``mult`` / ``n_members`` / ``island_id`` repeated per
component), the geometry in the observable basis, the waveforms on the
components' sampling-basis means.

Match metric: normalized phase-maximized overlap summed over the X/Y/Z
FD channels with FLAT weighting over the narrow band,
``|<a|b>| / sqrt(<a|a><b|b>)`` (fine for a same-band relative match --
the PSD is constant to <~1% over a <~0.2 uHz-wide template support).
Templates: ``GBGPU.run_wave`` on the L1 orbits of the data release
(``MOJITO_DATA_PATH``; read as ICRS, the run frame), t0 = the hardcoded
mojito data start :data:`DATA_T0`, Tobs from the npz meta (``--tobs``),
``dt`` from ``--dt`` (default 10 s). GBGPU's FD response is an equal-arm
approximation: adequate for this same-band RELATIVE match, and not the
production (unequal-arm WDM) likelihood. The merge test is BATCHED (2026-10-13):
each component's template is generated once per template length and the
pair overlaps are computed in vectorized groups (``Matcher.pair_matches``
/ ``batched_overlaps``), equal to the per-pair ``Matcher.match`` it
replaced, which regenerated both templates for every pair (463,915 CPU
waveforms, 2,270 s = 38 of the 45 minutes of the 6mo -> 9mo build). On
that build's own components the batched CPU path takes 8 s on a laptop
and reproduces the cluster's 231,737 cross-matches to 3e-15.
``--backend`` (env ``GB_WARM_START_REFEREE_BACKEND``, default ``cpu``)
moves the orbits and templates to a GPU (``cuda``, or ``auto`` = CUDA
when cupy sees a device); the default stays on the CPU until the CUDA path
has run once on a GPU node. ``--chunk-gb`` (env
``GB_WARM_START_REFEREE_CHUNK_GB``, default 1.0) bounds the memory of one
chunk. Neither env knob is set by the 9mo / 12mo launchers (CPU referee).

Output npz (default ``<npz stem>_referee.npz``): per-component
``med_match``, ``min_match``, ``med_ratio`` (measured / sinc-predicted
match; NaN = not refereed), ``n_sampled``, ``assigned_counts``, the
refereed index list ``referee``, the merge-test ``pairs`` ``(P, 2)`` with
their ``cross_match``, ``flagship`` and a ``meta`` JSON.

Laptop rules: run with
``OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1``
(the module sets these three as defaults at import).
Peak RSS ~2.4 GB, transient, during the L1 orbit load (the ltt table
is materialized in full before trimming -- same peak the fit pays).

Example:
    The call the build makes (``--npz`` is the fitter output):

    .. code-block:: bash

        python -m lisatools.globalfit.warmstart.match_referee \\
            --npz gf_prod_6mo_v9_4gpu_refereed_fit.npz --store gf_prod_6mo_testing.h5

    All options: ``[--tobs S] [--dt 10] [--oversample 2]
    [--members-per-comp 8] [--tail-n 200] [--p-core 0.5]
    [--pair-p-floor 0.2] [--pair-bins 4.0] [--flagship-f0 20.380]
    [--max-wf 20000] [--seed 11] [--backend cpu|cuda|auto] [--chunk-gb 1.0]
    [--out PATH]``.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

def _orbit_source_file() -> str:
    """An L1 file to read ORBITS from, resolved the way the runs resolve data.

    ``MOJITO_DATA_PATH`` (the env every submit script exports --
    ``/shared/data/mojito_cache`` on the 9mo / 12mo launchers; default the
    local dev cache ``~/.mojito_cache/brickmarket/mojito_light_v1_0_0``) ->
    the GB brick ``data/GB/L1``, else the COMBINED stream file
    ``data/COMBINED/L1``. The 6mo campaign's launch died here on a
    hardcoded laptop dev-cache path (2026-09-14).

    Returns:
        str: The L1 file path.

    Raises:
        FileNotFoundError: When neither is found.
    """
    from lisatools.globalfit.preprocessing import (
        find_combined_file,
        find_file,
    )

    root = os.environ.get(
        "MOJITO_DATA_PATH",
        os.path.expanduser(
            "~/.mojito_cache/brickmarket/mojito_light_v1_0_0"),
    )
    tried = []
    gb = os.path.join(root, "data", "GB", "L1")
    try:
        return find_file(gb, "GB", 0)
    except FileNotFoundError:
        tried.append(gb)
    comb = os.path.join(root, "data", "COMBINED", "L1")
    try:
        return find_combined_file(comb)
    except (FileNotFoundError, ValueError):
        tried.append(comb)
    raise FileNotFoundError(
        "warmstart match_referee needs an L1 file to read orbits from; "
        f"tried {tried} under MOJITO_DATA_PATH={root!r}."
    )
DATA_T0 = 97729089.327664  # mojito data start (gb_mojito_match.py REF)
CIRC = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}  # sampled-basis periods
ASSIGN_RADIUS = 6.0        # whitened-sigma junk cut for member assignment
N_CAP = 16384
DEFAULT_CHUNK_GB = 1.0     # one template chunk + its overlap scratch
GEN_SCRATCH = 8            # run_wave working memory per template, x its size
REPORT_PAIRS = 10          # merge-test pairs printed per class


def rss_gb() -> float:
    """Peak RSS for the progress prints: GB on macOS (bytes); 1024x too small on Linux (KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3


def circ_diff(x, m, period):
    """Minimal-image difference ``x - m`` of a circular column, in ``[-period/2, period/2)``."""
    return (x - m + period / 2.0) % period - period / 2.0


def _to_host(a):
    """cupy-or-numpy -> numpy."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _array_module(a):
    """numpy, or cupy for a cupy array (without importing cupy otherwise)."""
    if isinstance(a, np.ndarray):
        return np
    import cupy as cp

    return cp.get_array_module(a)


def resolve_backend(spec) -> str:
    """``--backend`` string -> the backend name for the orbits and GBGPU.

    ``"auto"`` is ``"cuda"`` when cupy reports a device and ``"cpu"``
    otherwise, with the reason printed (login node, laptop, no driver) --
    the same rule as ``fit_from_store.resolve_gmm_gpu``: a referee that
    runs on the CPU is recoverable, one that does not run blocks a launch.
    Any other name passes through lower-cased (``cpu``, ``cuda``,
    ``cuda12x``, ...).

    Args:
        spec (str or None): The option value; ``None`` / empty = ``"auto"``.

    Returns:
        str: The backend name (a ``force_backend`` value).
    """
    s = str(spec or "auto").strip().lower()
    if s != "auto":
        return s
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            return "cuda"
        print("[warmstart] referee --backend auto: cupy sees no device; "
              "using CPU")
    except Exception as exc:  # no cupy, no driver, no visible device
        print(f"[warmstart] referee --backend auto: GPU unavailable "
              f"({exc}); using CPU")
    return "cpu"


# ------------------------------------------------------------------
# member re-extraction: nearest centroid in per-component sigma space
# ------------------------------------------------------------------
def load_leaf_rows(store: str, last_k: int):
    """Cold-chain GB leaf rows over the last K written STORED ROWS: the SAME rows the fit read.

    Delegates to :func:`.fit_from_store.load_leaf_table` (with its
    torn-tail and keep-window guards).

    This used to be a private copy of that loader that kept the old hard
    clamp at row 471, so on a store past row 471 the members were
    re-extracted from the 300 rows ENDING AT 471 instead of the fit's rows
    (the 6mo -> 9mo build fitted rows 741-1040 and refereed its members
    from rows 172-471).

    Args:
        store (str): The store h5 the fit read.
        last_k (int): The fit's ``meta["last_k"]``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``X (n, 9)`` sampling-basis rows and
        ``sample_id (n,)`` (``row * nwalkers + walker``).
    """
    from .fit_from_store import load_leaf_table

    X, sample_id, _info = load_leaf_table(store, int(last_k))
    return X, sample_id


def assign_members(X, means, covs, island_id, f0_window_edges):
    """Nearest-centroid member assignment, per island, in each component's whitened space.

    Each row is located in its island by the (disjoint) island windows and
    compared with that island's components in each component's own
    ``sqrt(diag(cov))``-whitened space (circular columns 3 / 5 / 6
    wrapped); a best distance above ``ASSIGN_RADIUS`` (6) leaves it
    unassigned. Also the member count of :func:`.self_refit.consensus_filter`.

    Args:
        X (np.ndarray): ``(n, 9)`` rows in the FIT's basis.
        means (np.ndarray): ``(ncomp, 9)`` component means (same basis).
        covs (np.ndarray): ``(ncomp, 9, 9)``.
        island_id (np.ndarray): ``(ncomp,)`` island of each component.
        f0_window_edges (np.ndarray): ``(n_islands, 2)`` island windows on
            column 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``assign (n,)`` int64 component index
        or -1, and ``dist (n,)`` the best whitened distance (``inf`` for a
        row in no island).
    """
    n = len(X)
    assign = np.full(n, -1, dtype=np.int64)
    adist = np.full(n, np.inf)
    sig = np.sqrt(np.maximum(
        np.einsum("kii->ki", covs), 1e-30))          # (ncomp, 9)
    # island of each row via the window edges (windows are disjoint)
    lo, hi = f0_window_edges[:, 0], f0_window_edges[:, 1]
    win = np.searchsorted(lo, X[:, 1], side="right") - 1
    valid = (win >= 0) & (X[:, 1] < hi[np.clip(win, 0, len(hi) - 1)])
    for isl in np.unique(island_id):
        comp_sel = np.flatnonzero(island_id == isl)
        row_sel = np.flatnonzero(valid & (win == isl))
        if not len(row_sel):
            continue
        x = X[row_sel]                               # (m, 9)
        d2 = np.zeros((len(row_sel), len(comp_sel)))
        for jj, k in enumerate(comp_sel):
            diff = x - means[k]
            for col, period in CIRC.items():
                diff[:, col] = circ_diff(x[:, col], means[k][col], period)
            d2[:, jj] = np.sum((diff / sig[k]) ** 2, axis=1)
        best = d2.argmin(axis=1)
        bestd = np.sqrt(d2[np.arange(len(row_sel)), best])
        keep = bestd <= ASSIGN_RADIUS
        assign[row_sel[keep]] = comp_sel[best[keep]]
        adist[row_sel] = bestd
    return assign, adist


# ------------------------------------------------------------------
# waveforms + matches
# ------------------------------------------------------------------
def build_wave_gen(tobs: float, backend: str = "cpu"):
    """GBGPU on the data release's L1 orbits, both on ``backend``.

    The orbit table is trimmed on the host to ``[DATA_T0 - 1e5 s, DATA_T0 +
    tobs + 1e5 s]`` before ``configure`` uploads it; the orbits are read
    with ``frame="icrs"`` and GBGPU's ``t0`` is :data:`DATA_T0`.

    Args:
        tobs (float): Template observation time [s].
        backend (str, optional): ``force_backend`` name. Default ``"cpu"``.

    Returns:
        gbgpu.gbgpu.GBGPU: The waveform generator.
    """
    from lisatools.detector import L1Orbits
    from gbgpu.gbgpu import GBGPU

    orb = L1Orbits(_orbit_source_file(),
                   force_backend=backend, frame="icrs")
    pad = 1.0e5
    lt = np.asarray(orb.ltt_t)
    mk = (lt >= max(DATA_T0 - pad, float(orb.sc_t0))) & \
         (lt <= DATA_T0 + tobs + pad)
    orb.ltt = np.asarray(orb.ltt)[mk].copy()
    orb.ltt_t = lt[mk].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True)
    return GBGPU(force_backend=backend, orbits=orb, t0=DATA_T0)


def batched_overlaps(T, kmin, pairs, max_pairs=None) -> np.ndarray:
    """:meth:`Matcher.match` for every row of ``pairs``, vectorized.

    Two templates overlap on bins [max(ka, kb), min(ka, kb) + N), so for a
    fixed bin shift s = kb - ka both slices are fixed: the pairs are
    grouped by s and each group is one gathered product + sum (in
    sub-batches of ``max_pairs`` to bound the scratch). ``|s| >= N`` shares no
    bin -> 0, a zero-norm template -> 0, as in the scalar reference.

    Args:
        T (``xp.ndarray``): ``(m, 3, N)`` complex XYZ templates sharing ONE
            length N (numpy or cupy).
        kmin (array-like): ``(m,)`` their first frequency bins.
        pairs (array-like): ``(P, 2)`` indices into ``T``.
        max_pairs (int, optional): Sub-batch size. Default ``None`` = all
            pairs of a shift at once.

    Returns:
        np.ndarray: ``(P,)`` float matches in ``[0, 1]``.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    out = np.zeros(len(pairs))
    if not len(pairs):
        return out
    xp = _array_module(T)
    N = int(T.shape[-1])
    k = _to_host(kmin).astype(np.int64)
    nrm = (xp.abs(T) ** 2).sum(axis=(1, 2))
    shift = k[pairs[:, 1]] - k[pairs[:, 0]]
    step = len(pairs) if max_pairs is None else max(1, int(max_pairs))
    for s in np.unique(shift):
        s = int(s)
        if abs(s) >= N:
            continue
        a_sl = slice(s, N) if s >= 0 else slice(0, N + s)
        b_sl = slice(0, N - s) if s >= 0 else slice(-s, N)
        sel = np.flatnonzero(shift == s)
        for lo in range(0, len(sel), step):
            sub = sel[lo:lo + step]
            ia = xp.asarray(pairs[sub, 0])
            ib = xp.asarray(pairs[sub, 1])
            num = (xp.conj(T[ia, :, a_sl]) * T[ib, :, b_sl]).sum(axis=(1, 2))
            den = xp.sqrt(nrm[ia] * nrm[ib])
            ok = den > 0
            out[sub] = _to_host(
                xp.where(ok, xp.abs(num) / xp.where(ok, den, 1.0), 0.0))
    return out


def merge_report_lines(pairs, cross_match, f0, p, mult, coher,
                       n_show: int = REPORT_PAIRS) -> list:
    """The MERGE TEST log block, as a list of lines.

    The class counts, then the ``n_show`` highest cross-matches of the
    merge (> 0.9) and middle (0.5-0.9] classes, each followed by how many
    were not shown. Every pair and its cross-match is in the output npz
    (``pairs``, ``cross_match``). Printing every candidate (with a scan of
    all pairs per printed line) put ~47k lines per self-refit build into
    the 9mo job log, 285k in job 763 by 2026-10-14.

    Args:
        pairs (array-like): ``(P, 2)`` component pairs.
        cross_match (array-like): ``(P,)`` their matches.
        f0, p, mult, coher (array-like): Per-component f0 [mHz], ``p``,
            ``mult`` and median member match, for the printed lines.
        n_show: Lines per class. Default ``REPORT_PAIRS`` (10).

    Returns:
        list of str: The lines.
    """
    pairs = np.asarray(pairs).reshape(-1, 2)
    cross_match = np.asarray(cross_match)
    hi_m = np.flatnonzero(cross_match > 0.9)
    mid = np.flatnonzero((cross_match > 0.5) & (cross_match <= 0.9))
    lines = ["\n=== MERGE TEST (centroid cross-matches) ===",
             f"  >0.9 (merge candidates): {len(hi_m)} | 0.5-0.9: {len(mid)} | "
             f"<0.5 (distinct): {int(np.sum(cross_match <= 0.5))}"]
    for tag, sel, what in (("MERGE", hi_m, "merge candidates"),
                           ("mid  ", mid, "0.5-0.9 pairs")):
        top = sel[np.argsort(-cross_match[sel], kind="stable")][:n_show]
        for n in top:
            i, j = (int(v) for v in pairs[n])
            ln = (f"   {tag} {i}+{j}: f0 {f0[i]:.5f}/{f0[j]:.5f} "
                  f"p {p[i]:.2f}/{p[j]:.2f} ")
            if tag == "MERGE":
                ln += (f"mult {mult[i]:.1f}/{mult[j]:.1f} "
                       f"cross {cross_match[n]:.4f} "
                       f"coher {coher[i]:.3f}/{coher[j]:.3f}")
            else:
                ln += f"cross {cross_match[n]:.4f}"
            lines.append(ln)
        if len(sel) > len(top):
            lines.append(f"   ... {len(sel) - len(top)} more {what} (all "
                         f"pairs + cross_match are in the referee npz)")
    return lines


def pair_chunks(pairs, max_templates) -> list:
    """Split ``pairs`` into consecutive blocks with at most ``max_templates`` unique templates.

    Greedy, in order; the cap is at least 2; a template shared across a
    block boundary is generated in both blocks.

    Args:
        pairs (array-like): ``(P, 2)`` template-index pairs.
        max_templates (int): Unique templates allowed per block.

    Returns:
        list of np.ndarray: int64 index arrays into ``pairs``.
    """
    cap = max(2, int(max_templates))
    chunks, cur, seen = [], [], set()
    for n, (i, j) in enumerate(np.asarray(pairs, dtype=np.int64).tolist()):
        new = (i not in seen) + (j not in seen and j != i)
        if cur and len(seen) + new > cap:
            chunks.append(np.asarray(cur, dtype=np.int64))
            cur, seen = [], set()
        cur.append(n)
        seen.update((i, j))
    if cur:
        chunks.append(np.asarray(cur, dtype=np.int64))
    return chunks


class Matcher:
    """Sampled-basis rows -> XYZ FD templates -> matches, on the backend of ``gb``.

    Rows go through the stock GB transform
    (``make_gb_transform_container`` with chirp mass, astrophysical fdot
    ratio and distance) to the 9 GBGPU parameters; template lengths follow
    gbgpu's ``get_N`` (capped at ``N_CAP``).

    Args:
        gb (gbgpu.gbgpu.GBGPU): The waveform generator
            (:func:`build_wave_gen`); its arrays are numpy or cupy.
        tobs (float): Observation time [s].
        dt (float): Sample cadence [s] handed to ``run_wave``.
        oversample (int): ``get_N`` oversampling.
        chunk_bytes (int, optional): Memory budget of one merge-test
            template chunk plus its overlap scratch. Default 1 GiB.

    Attributes:
        tc: The GB transform container.
        n_wf (int): Templates generated so far.
    """

    def __init__(self, gb, tobs, dt, oversample,
                 chunk_bytes=int(DEFAULT_CHUNK_GB * 2 ** 30)):
        from lisatools.globalfit.stock.erebor.transforms import (
            make_gb_transform_container,
        )

        self.gb, self.tobs, self.dt, self.oversample = gb, tobs, dt, oversample
        self.chunk_bytes = int(chunk_bytes)
        self.tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=True)
        self.n_wf = 0

    def row_n(self, p: np.ndarray) -> np.ndarray:
        """Template length per TRANSFORMED row: gbgpu ``get_N`` on (amp, f0), capped at ``N_CAP``.

        A ``run_wave`` call on several rows uses their max. Returns an
        ``(m,)`` int64 array.
        """
        from gbgpu.utils.utility import get_N

        n = get_N(p[:, 0], p[:, 1], self.tobs, oversample=self.oversample)
        return np.minimum(N_CAP, np.asarray(n)).astype(np.int64)

    def waves(self, rows_sampled: np.ndarray):
        """``(m, 9)`` sampled rows -> host ``(XYZf (m, 3, N) complex, kmin (m,))``, one length N."""
        p = self.tc.both_transforms(np.atleast_2d(rows_sampled))
        N = int(self.row_n(p).max())
        self.gb.run_wave(*[p[:, i] for i in range(9)],
                         N=N, T=self.tobs, dt=self.dt)
        self.n_wf += len(p)
        return _to_host(self.gb.XYZf), _to_host(self.gb.start_inds)

    def _templates(self, p: np.ndarray, N: int):
        """Templates of the transformed rows ``p`` at ONE length ``N``, batched to the budget.

        Returns ``(T (m, 3, N) on the backend, kmin (m,) host int)``.
        """
        m = len(p)
        per = max(1, self.chunk_bytes // (GEN_SCRATCH * 3 * N * 16))
        T = None
        kmin = np.empty(m, dtype=np.int64)
        for lo in range(0, m, per):
            q = p[lo:lo + per]
            self.gb.run_wave(*[q[:, i] for i in range(9)],
                             N=int(N), T=self.tobs, dt=self.dt)
            X = self.gb.XYZf
            if T is None:
                T = _array_module(X).empty((m,) + tuple(X.shape[1:]),
                                           dtype=X.dtype)
            T[lo:lo + len(q)] = X
            kmin[lo:lo + len(q)] = _to_host(self.gb.start_inds)
            self.n_wf += len(q)
        return T, kmin

    def pair_matches(self, rows_sampled: np.ndarray, pairs) -> np.ndarray:
        """:meth:`match` for every (i, j) row of ``pairs`` over the sampled ``rows_sampled``.

        Each pair is evaluated at the length ``waves`` gives the two rows
        together, N = max(N_i, N_j). Per length, the pairs are cut into
        chunks of at most ~half the budget of templates; each template of a
        chunk is generated once and the overlaps batched
        (:func:`batched_overlaps`, the other half).

        Args:
            rows_sampled (np.ndarray): ``(m, 9)`` sampling-basis rows.
            pairs (array-like): ``(P, 2)`` indices into ``rows_sampled``.

        Returns:
            np.ndarray: ``(P,)`` matches.
        """
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        out = np.zeros(len(pairs))
        if not len(pairs):
            return out
        p = self.tc.both_transforms(np.atleast_2d(rows_sampled))
        n_row = self.row_n(p)
        n_pair = np.maximum(n_row[pairs[:, 0]], n_row[pairs[:, 1]])
        for N in np.unique(n_pair):
            N = int(N)
            tmpl = 3 * N * 16                       # bytes per template
            cap = self.chunk_bytes // (2 * tmpl)
            max_pairs = max(1, self.chunk_bytes // (6 * tmpl))
            sel = np.flatnonzero(n_pair == N)
            for blk in pair_chunks(pairs[sel], cap):
                idx = sel[blk]
                u, inv = np.unique(pairs[idx].ravel(), return_inverse=True)
                T, kmin = self._templates(p[u], N)
                out[idx] = batched_overlaps(
                    T, kmin, inv.reshape(-1, 2), max_pairs=max_pairs)
                del T
        return out

    @staticmethod
    def match(a, ka, b, kb):
        """Phase-maximized normalized XYZ overlap, flat weighting.

        Args:
            a, b (np.ndarray): ``(3, N_a)`` / ``(3, N_b)`` complex templates.
            ka, kb (int): Their first frequency bins.

        Returns:
            float: ``|<a|b>| / sqrt(<a|a><b|b>)`` over the shared bins
            (0 with no shared bin or a zero norm).
        """
        lo = max(ka, kb)
        hi = min(ka + a.shape[-1], kb + b.shape[-1])
        if hi <= lo:
            return 0.0
        num = np.sum(np.conj(a[:, lo - ka:hi - ka]) * b[:, lo - kb:hi - kb])
        den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
        return float(abs(num) / den) if den > 0 else 0.0


# ------------------------------------------------------------------
def main(argv=None):
    """CLI: referee a fitter npz against its store and write the referee npz.

    The argparse description is the first line of the module docstring.
    Members are re-extracted over the fit's ``meta["last_k"]`` rows, so the
    fit must have been run with ``--last-k`` (every automatic build passes
    it). The default ``--backend`` is ``GB_WARM_START_REFEREE_BACKEND``
    (else ``cpu``), the default ``--chunk-gb``
    ``GB_WARM_START_REFEREE_CHUNK_GB`` (else 1.0). Prints the report;
    returns ``None``.

    Args:
        argv (list of str, optional): Arguments; ``None`` = ``sys.argv[1:]``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tobs", type=float, default=None,
                    help="default: npz meta tobs")
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--members-per-comp", type=int, default=8)
    ap.add_argument("--tail-n", type=int, default=200,
                    help="random sample size of the p < p-core tail")
    ap.add_argument("--p-core", type=float, default=0.5)
    ap.add_argument("--pair-p-floor", type=float, default=0.2)
    ap.add_argument("--pair-bins", type=float, default=4.0)
    ap.add_argument("--flagship-f0", type=float, default=20.380,
                    help="mHz; island refereed in detail regardless of p")
    ap.add_argument("--max-wf", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--backend",
                    default=os.environ.get("GB_WARM_START_REFEREE_BACKEND",
                                           "cpu"),
                    help="template backend: cpu (default), cuda, cuda12x, "
                         "... or auto (cuda when cupy sees a device, "
                         "else cpu)")
    ap.add_argument("--chunk-gb", type=float,
                    default=float(os.environ.get(
                        "GB_WARM_START_REFEREE_CHUNK_GB", DEFAULT_CHUNK_GB)),
                    help="memory budget of one merge-test template chunk "
                         "plus its overlap scratch [GB]")
    ap.add_argument("--out", default=None,
                    help="results npz (default <npz>_referee.npz)")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    z = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    basis = str(meta.get("basis", "sampling"))
    tobs = args.tobs or float(meta["tobs"])
    edges = z["f0_window_edges"]
    obs_map = None
    if basis == "observable":
        from . import basis as wb

        # The fit's geometry -- member assignment, whitening, f0 offsets,
        # island windows -- all lives in the OBSERVABLE basis, so the leaf
        # rows are mapped INTO it below. Only the waveform calls convert
        # back, and they read the sampling rows directly.
        obs_map = wb.build_map_from_stored_params(meta["map_params"])
        ncomp_per = np.asarray(z["gmm_ncomp"], dtype=int)
        means, covs = np.asarray(z["gmm_means"]), np.asarray(z["gmm_covs"])
        # per-CLUSTER -> per-COMPONENT so every check below is unchanged
        p, mult, n_members, island_id = (
            np.repeat(np.asarray(z[k]), ncomp_per, axis=0)
            for k in ("p", "mult", "n_members", "island_id"))
        means_sampling = np.asarray(obs_map.from_internal(means), dtype=float)
        df_seg = 1.0 / tobs                     # col 1 is f_mid [Hz]
        f_units = "Hz"
    else:
        means, covs, p, mult = z["means"], z["covs"], z["p"], z["mult"]
        n_members, island_id = z["n_members"], z["island_id"]
        means_sampling = np.asarray(means, dtype=float)
        df_seg = float(meta["df_mhz"])
        f_units = "mHz"
    ncomp = len(means)
    print(f"npz: {ncomp} comps ({basis} basis) | Tobs {tobs:.0f} s | "
          f"df {df_seg:.6g} {f_units}")

    # ---- member re-extraction ------------------------------------
    t0 = time.perf_counter()
    X, _sid = load_leaf_rows(args.store, int(meta["last_k"]))
    # X stays SAMPLING (the waveform generator needs it); G is the same
    # rows in the fit's basis, so X[i] and G[i] are the same leaf.
    if obs_map is not None:
        from .fit_from_store import to_observable

        G = to_observable(X, obs_map)
    else:
        G = X
    assign, _ad = assign_members(G, means, covs, island_id, edges)
    counts = np.bincount(assign[assign >= 0], minlength=ncomp)
    frac = counts / np.maximum(n_members, 1)
    print(f"leaf rows {len(X):,} | assigned {np.sum(assign >= 0):,} "
          f"({np.mean(assign >= 0):.1%}) | per-comp count vs npz "
          f"n_members: median ratio {np.median(frac):.2f} "
          f"[{time.perf_counter() - t0:.1f} s, RSS {rss_gb():.2f} GB]")

    # ---- referee sets --------------------------------------------
    core = np.flatnonzero(p > args.p_core)
    tail_all = np.flatnonzero((p <= args.p_core) & (counts >= 2))
    tail = rng.choice(tail_all, min(args.tail_n, len(tail_all)),
                      replace=False)
    # flagship island: highest-p comp within 6 bins of --flagship-f0
    near = np.flatnonzero(np.abs(means_sampling[:, 1] - args.flagship_f0)
                          < 6 * (1e3 / tobs))
    flag = int(near[np.argmax(p[near])]) if len(near) else -1
    flag_isl = island_id[flag] if flag >= 0 else -1
    referee = np.unique(np.concatenate(
        [core, tail, np.flatnonzero(island_id == flag_isl)]))
    referee = referee[counts[referee] >= 2]
    print(f"referee: {len(core)} core (p>{args.p_core}) + {len(tail)} tail "
          f"+ flagship island {flag_isl} (comp {flag}, "
          f"f0 {means_sampling[flag, 1]:.5f} mHz, p {p[flag]:.2f})"
          if flag >= 0 else
          f"referee: {len(core)} core + {len(tail)} tail (no flagship found)")

    # ---- same-island neighbor pairs ------------------------------
    pairs = []
    for isl in np.unique(island_id):
        sel = np.flatnonzero(island_id == isl)
        ok = (p[sel] > args.pair_p_floor) | (isl == flag_isl)
        sel = sel[ok]
        for a_i in range(len(sel)):
            for b_i in range(a_i + 1, len(sel)):
                i, j = sel[a_i], sel[b_i]
                if abs(means[i, 1] - means[j, 1]) <= args.pair_bins * df_seg:
                    pairs.append((i, j))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    print(f"pairs: {len(pairs)} same-island neighbors within "
          f"{args.pair_bins}/Tobs (p>{args.pair_p_floor} or flagship isl)")

    budget = len(referee) * (args.members_per_comp + 1) + 2 * len(pairs)
    if budget > args.max_wf:
        keep = (args.max_wf - 2 * len(pairs)) // (args.members_per_comp + 1)
        referee = rng.choice(referee, max(keep, 50), replace=False)
        print(f"BUDGET: subsampled referee comps to {len(referee)} "
              f"(est. {budget} > max {args.max_wf} waveforms)")

    # ---- waveform machinery --------------------------------------
    t0 = time.perf_counter()
    backend = resolve_backend(args.backend)
    try:
        gb = build_wave_gen(tobs, backend)
    except Exception as exc:
        if str(args.backend).strip().lower() != "auto" or backend == "cpu":
            raise
        print(f"[warmstart] referee --backend auto: the {backend} wave "
              f"generator failed ({exc!r}); using CPU")
        backend = "cpu"
        gb = build_wave_gen(tobs, backend)
    mm = Matcher(gb, tobs, args.dt, args.oversample,
                 chunk_bytes=int(args.chunk_gb * 2 ** 30))
    print(f"wave gen ready on {backend} [{time.perf_counter() - t0:.1f} s, "
          f"RSS {rss_gb():.2f} GB]")

    # ---- within-group coherence ----------------------------------
    t0 = time.perf_counter()
    K = args.members_per_comp
    med_match = np.full(ncomp, np.nan)
    min_match = np.full(ncomp, np.nan)
    med_ratio = np.full(ncomp, np.nan)   # measured / sinc-predicted
    n_sampled = np.zeros(ncomp, dtype=np.int64)
    sig = np.sqrt(np.maximum(np.einsum("kii->ki", covs), 1e-30))
    flag_detail = None
    for k in referee:
        ridx = np.flatnonzero(assign == k)
        # anchor: member closest to the component mean (whitened)
        diff = G[ridx] - means[k]
        for col, period in CIRC.items():
            diff[:, col] = circ_diff(G[ridx][:, col], means[k][col], period)
        d = np.sqrt(np.sum((diff / sig[k]) ** 2, axis=1))
        anchor = ridx[d.argmin()]
        others = ridx[ridx != anchor]
        n_take = min(K, len(others)) if k != flag else min(3 * K, len(others))
        take = rng.choice(others, n_take, replace=False)
        wf, kmin = mm.waves(X[np.concatenate(([anchor], take))])
        ms = np.array([mm.match(wf[0], kmin[0], wf[i], kmin[i])
                       for i in range(1, len(wf))])
        # sinc prediction: for a windowed sinusoid, an f0 offset of x
        # bins alone gives match ~ |sinc(x)| (np.sinc = sin(pi x)/(pi x)).
        # ratio ~1 -> the group's incoherence is fully explained by its
        # f0 spread (self-consistent posterior width / smooth blend);
        # ratio << 1 -> members also disagree in other params (junk).
        off_bins = (G[take, 1] - G[anchor, 1]) / df_seg
        pred = np.maximum(np.abs(np.sinc(off_bins)), 1e-3)
        med_match[k] = np.median(ms)
        min_match[k] = ms.min()
        med_ratio[k] = np.median(np.minimum(ms / pred, 1.2))
        n_sampled[k] = len(ms)
        if k == flag:
            off = (G[take, 1] - means[k, 1]) / df_seg  # 1/Tobs units
            # far-vs-near cross-check: match the two extreme-offset members
            lo_i, hi_i = int(off.argmin()), int(off.argmax())
            cross = mm.match(wf[1 + lo_i], kmin[1 + lo_i],
                             wf[1 + hi_i], kmin[1 + hi_i])
            flag_detail = dict(offsets=off, matches=ms, cross_extreme=cross,
                               off_lo=off[lo_i], off_hi=off[hi_i], pred=pred)
    print(f"within-group: {len(referee)} comps, {mm.n_wf} waveforms "
          f"[{time.perf_counter() - t0:.1f} s]")

    # ---- merge test ----------------------------------------------
    t0 = time.perf_counter()
    n_wf0 = mm.n_wf
    cross_match = mm.pair_matches(means_sampling, pairs)
    print(f"merge test: {len(pairs)} pairs [{time.perf_counter() - t0:.1f} s, "
          f"{mm.n_wf - n_wf0} templates on {backend}]")
    if backend != "cpu":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()

    # ---- report ---------------------------------------------------
    def bucket(vals):
        v = vals[np.isfinite(vals)]
        if not len(v):
            return "n/a"
        return (f"n={len(v)} med={np.median(v):.4f} | >0.99: "
                f"{np.mean(v > 0.99):.1%} | 0.9-0.99: "
                f"{np.mean((v > 0.9) & (v <= 0.99)):.1%} | <0.9: "
                f"{np.mean(v <= 0.9):.1%}")

    def consis(sel):
        v = med_ratio[sel]
        v = v[np.isfinite(v)]
        return f"{np.mean(v > 0.7):.1%}" if len(v) else "n/a"

    print("\n=== WITHIN-GROUP median member match (vs anchor) ===")
    print(f"  p>0.9      : {bucket(med_match[p > 0.9])} | "
          f"sinc-consistent: {consis(p > 0.9)}")
    print(f"  p 0.5-0.9  : {bucket(med_match[(p > 0.5) & (p <= 0.9)])} | "
          f"sinc-consistent: {consis((p > 0.5) & (p <= 0.9))}")
    print(f"  tail p<=0.5: {bucket(med_match[tail])} | "
          f"sinc-consistent: {consis(tail)}")
    print("  (sinc-consistent: median member match >= 0.7x the |sinc(df0)| "
          "prediction, i.e. the f0 spread explains the incoherence)")

    ref_ok = referee[np.isfinite(min_match[referee])]
    worst = ref_ok[np.argsort(med_ratio[ref_ok])][:10]
    print("\n  10 worst groups (by measured/sinc-predicted ratio):")
    print("   comp   f0[mHz]     p    mult  n_mem  sig_f0/df  med     min"
          "    ratio")
    for k in worst:
        print(f"   {k:5d} {means_sampling[k, 1]:10.5f} {p[k]:5.2f} "
              f"{mult[k]:6.2f} "
              f"{n_members[k]:6d} {sig[k, 1] / df_seg:8.2f} "
              f"{med_match[k]:7.4f} {min_match[k]:7.4f} {med_ratio[k]:7.3f}")

    print("\n".join(merge_report_lines(
        pairs, cross_match, f0=means_sampling[:, 1], p=p, mult=mult,
        coher=med_match)))

    if flag >= 0:
        print(f"\n=== FLAGSHIP comp {flag} "
              f"(f0 {means_sampling[flag, 1]:.5f} mHz, "
              f"p {p[flag]:.2f}, sig_f0 {sig[flag, 1] / df_seg:.2f}/Tobs) ===")
        if flag_detail is not None:
            fd = flag_detail
            o, m = fd["offsets"], fd["matches"]
            print(f"  {len(m)} members vs anchor: med {np.median(m):.4f} "
                  f"min {m.min():.4f}")
            pr = fd["pred"]
            for lo, hi in [(-99, -1.5), (-1.5, -0.3), (-0.3, 0.3),
                           (0.3, 1.5), (1.5, 99)]:
                s = (o >= lo) & (o < hi)
                if s.any():
                    print(f"   off [{lo:+5.1f},{hi:+5.1f})/Tobs: n={s.sum()}"
                          f" med match {np.median(m[s]):.4f} "
                          f"min {m[s].min():.4f} "
                          f"(sinc pred {np.median(pr[s]):.4f})")
            print(f"  extreme-offset members ({fd['off_lo']:+.2f} vs "
                  f"{fd['off_hi']:+.2f} /Tobs) direct match: "
                  f"{fd['cross_extreme']:.4f}")
        fp = np.flatnonzero(np.any(pairs == flag, axis=1))
        for n_i in fp:
            i, j = pairs[n_i]
            o = j if i == flag else i
            print(f"  pair w/ comp {o} (p {p[o]:.2f}, "
                  f"df0 {(means[o, 1] - means[flag, 1]) / df_seg:+.2f}"
                  f"/Tobs)"
                  f": cross-match {cross_match[n_i]:.4f}")

    out = args.out or os.path.splitext(args.npz)[0] + "_referee.npz"
    np.savez_compressed(
        out, referee=referee, med_match=med_match, min_match=min_match,
        med_ratio=med_ratio,
        n_sampled=n_sampled, pairs=pairs, cross_match=cross_match,
        assigned_counts=counts, flagship=flag,
        meta=json.dumps(dict(
            npz=args.npz, store=args.store, tobs=tobs, dt=args.dt,
            oversample=args.oversample, members_per_comp=K,
            tail_n=int(len(tail)), p_core=args.p_core,
            pair_p_floor=args.pair_p_floor, pair_bins=args.pair_bins,
            seed=args.seed, n_waveforms=mm.n_wf,
            backend=backend, chunk_gb=args.chunk_gb,
            weighting="flat-narrowband XYZ, phase-maximized")))
    print(f"\nDONE: {mm.n_wf} waveforms | wall "
          f"{time.perf_counter() - t_start:.1f} s | peak RSS "
          f"{rss_gb():.2f} GB | wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
