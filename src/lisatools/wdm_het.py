"""Host-side helpers for the chunked-heterodyne WDM kernels.

These are the Python primitives that
:class:`gbgpu.gbcomps.GBWDMComputations` /
:class:`bbhx.sobbhcomps.SOBBHWDMComputations` (through
:class:`lisatools.chunked_het.WDMComputationsBase`) hand off to the C++ / JAX
kernels. Moved here at Phase 3L.7i (2026-06-04) as part of the
lisa-on-gpu retirement work; previously lived at
``fastlisaresponse.utils.wdm_het``. ``gbgpu.wdm_het`` re-exports the chunked-het
names for older GB imports.

The chunked-het transform covers the ``Nt``-pixel WDM grid with overlapping
chunks of ``Nt_sub`` pixels each; ``n_pad`` pixels at each interior chunk edge are
discarded at the stitch.

Pieces here:

* :func:`compute_chunk_geometry` -- per-chunk ``starts / keep_lo /
  keep_hi / n_global_lo`` arrays the kernels consume. Handles the
  partial-slide tail when ``(Nt - Nt_sub) % step != 0``.
* :func:`compute_wdm_window` -- the Nt_sub-long phitilde sample the
  ``gb_chunk_fd_to_wdm`` kernel reads on the host.
* :func:`compute_layer_groups` / :func:`compute_swap_layer_groups` --
  cluster binaries by carrier WDM layer so each kernel launch reads
  data/PSD only over a narrow m-band per group.
* :func:`recommended_tukey_alpha` / :func:`resolve_tukey_alpha` --
  centralised Tukey alpha selection. ``USE_RECOMMENDED_TUKEY`` is the
  sentinel; ``recommended_tukey_alpha("heterodyne", N_sparse=...)``
  is the auto-pick (0.01 for N_sparse>=512, 0.05 otherwise).
* :func:`wdm_chunk_of_td`, :func:`chunk_start_for_pixels`,
  :func:`tail_chunk_plan`, :func:`splice_chunk` -- source-agnostic single-chunk
  primitives (TD -> WDM of one even-start chunk, its placement and splice). The
  first three are used by the EMRI direct-to-WDM plunge chunk
  (:mod:`lisatools.sources.emri.wdm_direct`); :func:`splice_chunk` is the matching
  raw splice (exercised by the tests; the EMRI path writes its chunks itself).

Algorithm reference for the chunk geometry / layer-grouping lives in
``scripts/diagnostics/check_shortened_wdm.py`` and the design notes
``scripts/notes/CHUNKED_HET_DESIGN_NOTES.md``.
"""
from __future__ import annotations

import numpy as np

from lisatools.domains import WDMSettings
from lisatools.utils.utility import get_array_module

# ----------------------------------------------------------------------
# Tukey alpha
# ----------------------------------------------------------------------

# Recommended alphas for the chunked-stitch variants. The TD path was
# validated at 0.02; the heterodyne path uses 0.01 at wide N_sparse
# (>=512) and 0.05 at narrow N_sparse (<512).
RECOMMENDED_TUKEY_ALPHA_TD = 0.02
RECOMMENDED_TUKEY_ALPHA_HET_WIDE = 0.01
RECOMMENDED_TUKEY_ALPHA_HET_NARROW = 0.05

# Sentinel: pass ``tukey_alpha=USE_RECOMMENDED_TUKEY`` to opt into the
# auto-pick. Float-typed so existing positional callers still work.
USE_RECOMMENDED_TUKEY = -1.0


def recommended_tukey_alpha(path, N_sparse=None):
    """Return the recommended Tukey alpha for a chunked-WDM workflow.

    Args:
        path (str): ``"td"`` (also ``"td_stitched"`` / ``"td-based"``) for the
            TD-based chunked stitch, or ``"heterodyne"`` / ``"het"`` / ``"fd"`` /
            ``"fd_heterodyne"`` for the FD-direct heterodyne stitch. Case-insensitive.
        N_sparse (int, optional): Heterodyne FFT length. ``None`` or ``>= 512``
            selects the wide value.

    Returns:
        float: ``0.02`` (TD), ``0.01`` (heterodyne, wide) or ``0.05``
        (heterodyne, ``N_sparse < 512``).

    Raises:
        ValueError: For an unknown ``path``.

    Example:
        >>> from lisatools.wdm_het import recommended_tukey_alpha
        >>> recommended_tukey_alpha("heterodyne", N_sparse=256)
        0.05
    """
    p = str(path).lower()
    if p in ("td", "td_stitched", "td-based"):
        return RECOMMENDED_TUKEY_ALPHA_TD
    if p in ("heterodyne", "het", "fd", "fd_heterodyne"):
        if N_sparse is None or N_sparse >= 512:
            return RECOMMENDED_TUKEY_ALPHA_HET_WIDE
        return RECOMMENDED_TUKEY_ALPHA_HET_NARROW
    raise ValueError(
        "path must be one of 'td', 'heterodyne', 'het', 'fd', 'fd_heterodyne'"
    )


def resolve_tukey_alpha(tukey_alpha, use_tukey, path, N_sparse=None):
    """Resolve the actual Tukey alpha to apply, per the ``use_tukey`` flag.

    ``use_tukey=False`` always gives ``0.0``. Otherwise ``USE_RECOMMENDED_TUKEY``
    (``-1.0``) triggers :func:`recommended_tukey_alpha` and any other value is
    returned as a float.

    Args:
        tukey_alpha (float): Requested alpha or ``USE_RECOMMENDED_TUKEY``.
        use_tukey (bool): Master switch.
        path (str): Workflow passed to :func:`recommended_tukey_alpha`.
        N_sparse (int, optional): Heterodyne FFT length for the auto-pick.

    Returns:
        float: The alpha to apply.
    """
    if tukey_alpha == USE_RECOMMENDED_TUKEY:
        if not use_tukey:
            return 0.0
        return recommended_tukey_alpha(path, N_sparse=N_sparse)
    if not use_tukey:
        return 0.0
    return float(tukey_alpha)


# ----------------------------------------------------------------------
# Chunk geometry + WDM window
# ----------------------------------------------------------------------

# TODO(CPU two-size chunks, 2026-08-03): allow the chunk transform to use
# TWO Nt_sub sizes instead of one, to stop paying ~1.9x the necessary work
# on CPU.
#
# THE PROBLEM. Nt_sub must be a power of two (radix-2 FFT; asserted in
# chunked_het.ChunkedHetBase) and Nt essentially never is, so a uniform
# Nt_sub leaves only two bad options:
#
#   ceil power of two  -> ONE chunk, but it overhangs the data by up to
#                         1.9x (7744 pixels past the end at Nt=8640).
#                         Reads past the buffer; segfaults on some builds.
#   floor power of two -> TWO chunks, no overhang, but the second chunk is
#                         full-size to cover a small remainder, so the
#                         interior overlap is enormous: 2 x 8192 = 16384
#                         pixels computed for the 8640 actually needed.
#
# Both land at 1.896x the required work at EVERY baseline (verified for
# Nt = 540 / 1080 / 2160 / 4320 / 8640).
#
# THE FIX. Size the last chunk to the remainder instead of to the first
# chunk: one big power-of-two covering the bulk, plus one SMALL power of
# two covering (Nt - big + n_pad) -- the leftover plus a single halo so
# the stitch still overlaps. Measured cost:
#
#     Nt     big   small   computed   vs Nt      uniform   vs Nt
#    540     512      64        576   1.067x        1024   1.896x
#   1080    1024     128       1152   1.067x        2048   1.896x
#   2160    2048     128       2176   1.007x        4096   1.896x
#   4320    4096     256       4352   1.007x        8192   1.896x
#   8640    8192     512       8704   1.007x       16384   1.896x
#
# i.e. ~1.9x less transform work, and the two-size split generalises to a
# per-chunk size array if a future geometry wants more than two.
#
# WHAT IT TOUCHES. Nt_sub is scalar today and the window is built once for
# it (compute_wdm_window(Nf, Nt_sub, dt) -> length tied to Nt_sub). Two
# sizes needs: a per-chunk Nt_sub array alongside starts/keep_lo/keep_hi,
# one window per DISTINCT size (two arrays, not n_chunks), and the kernel
# selecting window + transform length per chunk. keep_lo/keep_hi already
# express arbitrary per-chunk trimming, so the stitching logic itself does
# not change -- this is a sizing change, not a correctness change.
#
# WHY CPU SPECIFICALLY. On GPU the chunks ARE the parallel decomposition:
# they map to CUDA blocks (grid_dim defaults to n_chunks), so redundant
# span is hidden by concurrency and uniform sizing keeps the launch
# regular -- leave the GPU path alone. On CPU the chunks are a sequential
# loop, so every redundant pixel is wall-clock.
#
# INTERIM. A uniform Nt_sub sweep already recovers much of it, and the
# optimum is NOT the largest size: at Nt=8640, Nt_sub=512 computes 9216
# pixels (1.07x) across 18 chunks vs 16384 (1.90x) for two 8192 chunks.
# That trades transform waste for per-chunk fixed cost, which is why it
# should be measured (scripts/gb_chunked_het/gb_engine_benchmark.py,
# BENCH_NT_SUB) rather than modelled. The two-size split gets the same
# 1.007x with only TWO chunks, so it dominates the sweep once implemented.

def compute_chunk_geometry(Nt, Nt_sub, n_pad):
    """Pre-compute per-chunk stitching geometry for the het kernels.

    Returns a dict of numpy arrays:

    - ``starts``: ``(n_chunks,)`` int64 -- chunk start in WDM-pixel units
    - ``keep_lo``: ``(n_chunks,)`` int32 -- chunk-local lo to keep
    - ``keep_hi``: ``(n_chunks,)`` int32 -- chunk-local hi (exclusive)
    - ``n_global_lo``: ``(n_chunks,)`` int32 -- global WDM pixel where the
      first kept sample lands

    Chunks start every ``step = Nt_sub - 2 * n_pad`` pixels; the first chunk keeps
    from its local 0 and the last full chunk ending exactly at ``Nt`` keeps through
    its end, every other edge drops ``n_pad`` pixels. ``Nt_sub >= Nt`` gives a
    single chunk keeping ``[0, Nt)``.

    Handles the partial-slide case (``(Nt - Nt_sub) % step != 0``) by
    appending one extra chunk at ``Nt - Nt_sub`` whose kept range starts where the
    previous chunk's kept range ends, so the overlap region isn't double-counted.

    Args:
        Nt (int): WDM time pixels of the full grid.
        Nt_sub (int): Pixels per chunk.
        n_pad (int): Pixels discarded at each interior chunk edge.

    Returns:
        dict: The four arrays above.

    Raises:
        AssertionError: If ``step`` is not a positive even number.

    Example:
        >>> from lisatools.wdm_het import compute_chunk_geometry
        >>> g = compute_chunk_geometry(Nt=512, Nt_sub=256, n_pad=32)
        >>> g["starts"].tolist(), g["keep_lo"].tolist(), g["keep_hi"].tolist()
        ([0, 192, 256], [0, 32, 160], [224, 224, 256])
    """
    step = int(Nt_sub) - 2 * int(n_pad)
    assert step > 0 and step % 2 == 0, (Nt_sub, n_pad, step)

    # Single-chunk case: Nt_sub >= Nt. One chunk covers the full WDM grid;
    # the chunk's TD span (Nt_sub samples) is longer than Nt but we only
    # keep [0, Nt) WDM pixels. The waveform is evaluated on the fly at any
    # t, so extending the chunk past t_obs_end is harmless.
    if int(Nt_sub) >= int(Nt):
        return dict(
            starts      = np.asarray([0],          dtype=np.int64),
            keep_lo     = np.asarray([0],          dtype=np.int32),
            keep_hi     = np.asarray([int(Nt)],    dtype=np.int32),
            n_global_lo = np.asarray([0],          dtype=np.int32),
        )

    n_full = (int(Nt) - int(Nt_sub)) // step + 1
    starts = [j * step for j in range(n_full)]
    last_full_end = starts[-1] + Nt_sub

    keep_lo = []
    keep_hi = []
    n_global_lo = []
    for j, n0 in enumerate(starts):
        klo = 0 if j == 0 else n_pad
        if j == n_full - 1 and last_full_end == Nt:
            khi = Nt_sub
        else:
            khi = Nt_sub - n_pad
        keep_lo.append(klo)
        keep_hi.append(khi)
        n_global_lo.append(n0 + klo)

    if last_full_end < Nt:
        n0_partial = Nt - Nt_sub
        new_lo_global = starts[-1] + (Nt_sub - n_pad)
        new_lo_local = new_lo_global - n0_partial
        starts.append(n0_partial)
        keep_lo.append(new_lo_local)
        keep_hi.append(Nt_sub)
        n_global_lo.append(new_lo_global)

    return dict(
        starts      = np.asarray(starts,      dtype=np.int64),
        keep_lo     = np.asarray(keep_lo,     dtype=np.int32),
        keep_hi     = np.asarray(keep_hi,     dtype=np.int32),
        n_global_lo = np.asarray(n_global_lo, dtype=np.int32),
    )


def compute_wdm_window(Nf, Nt_sub, dt, backend="cpu"):
    """Return the Nt_sub-long phitilde sampled at the chunk's grid.

    Equivalent to ``WDMSettings(Nf, Nt_sub, dt).window`` -- exposed
    here as a free function so host code doesn't need to construct a
    full settings object. The window is a real-valued host array; we
    always build it through the ``cpu`` backend (lisatools' WDMSettings
    in-place initializer doesn't support jax arrays).

    Args:
        Nf (int): WDM frequency layers.
        Nt_sub (int): Pixels per chunk.
        dt (float): Time-domain sample step (s).
        backend (str): Ignored (the window is always built on the host).

    Returns:
        np.ndarray: Float array of length ``Nt_sub``.
    """
    del backend  # window is host-side; always CPU
    return np.asarray(
        WDMSettings(Nf=int(Nf), Nt=int(Nt_sub), dt=float(dt),
                    force_backend="cpu").window,
        dtype=float,
    )


# ----------------------------------------------------------------------
# Source-agnostic single-chunk primitives (EMRI plunge chunk)
# ----------------------------------------------------------------------

def wdm_chunk_of_td(td_arr, start_sample, Nf, Nt_sub, dt, backend="cpu"):
    """TD->WDM of ONE contiguous chunk of ``Nf * Nt_sub`` samples, no window.

    Lifted from scripts/diagnostics/check_shortened_wdm.py (``wdm_of_td_slice``)
    with the Tukey removed: the chunk EXCLUDES the global taper (the chunked-het
    edge rule), and the lookup side never sees a window. Start on an EVEN global
    pixel (``start_sample = n0 * Nf``, ``n0`` even) or the real WDM differs from
    the full transform; discard ``n_pad`` pixels at each edge.

    Args:
        td_arr: Time-domain array ``(..., N)`` (numpy or cupy).
        start_sample (int): First sample of the chunk.
        Nf (int): WDM frequency layers.
        Nt_sub (int): Pixels in the chunk (the chunk has ``Nf * Nt_sub`` samples).
        dt (float): Sample step (s).
        backend (str): ``force_backend`` of the chunk transform. Default ``"cpu"``.

    Returns:
        ``xp.ndarray``: The chunk's coefficients, ``td_arr.shape[:-1] + (Nf, Nt_sub)``
        (``(nch, Nf, Nt_sub)`` for a ``(nch, N)`` input; a 1-D input gets a channel
        axis of length 1). The chunk's WDM settings are full-band.

    Raises:
        ValueError: If ``td_arr`` has fewer than ``Nf * Nt_sub`` samples from
            ``start_sample``.
    """
    from .domains import TDSettings, TDSignal, WDMSettings  # domains imports this module

    n = int(Nf) * int(Nt_sub)
    sub = td_arr[..., int(start_sample):int(start_sample) + n]
    if sub.shape[-1] != n:
        raise ValueError(f"chunk needs {n} samples from {start_sample}, got {sub.shape[-1]}")
    sig = TDSignal(sub, TDSettings(n, dt, force_backend=backend)).transform(
        WDMSettings(Nf=int(Nf), Nt=int(Nt_sub), dt=dt, force_backend=backend))
    return sig.arr


def chunk_start_for_pixels(n_lo, n_hi, Nt, Nt_sub, n_pad):
    """EVEN global start pixel ``n0`` of a chunk whose interior holds ``[n_lo, n_hi)``.

    The interior is ``[n0 + n_pad, n0 + Nt_sub - n_pad)``; the grid-start chunk may
    keep pixels from 0, and a range reaching the last ``n_pad`` pixels of the grid
    uses the grid-end chunk ``n0 = Nt - Nt_sub`` (nothing follows it). Otherwise the
    lowest valid even start is returned.

    Args:
        n_lo (int): First pixel to cover.
        n_hi (int): One past the last pixel to cover.
        Nt (int): Pixels of the full grid.
        Nt_sub (int): Pixels per chunk.
        n_pad (int): Pixels discarded at each interior edge.

    Returns:
        int: The chunk start pixel ``n0``.

    Raises:
        ValueError: If the range exceeds the chunk interior or no even-start chunk
            covers it.
    """
    n_lo, n_hi, Nt, Nt_sub, n_pad = map(int, (n_lo, n_hi, Nt, Nt_sub, n_pad))
    # the last n_pad pixels of the grid can only come from the grid-end chunk
    right_pad = 0 if n_hi > Nt - n_pad else n_pad
    if n_hi - n_lo > Nt_sub - n_pad - right_pad:
        raise ValueError(
            f"pixels [{n_lo}, {n_hi}) exceed the chunk interior ({Nt_sub - n_pad - right_pad}); "
            "use a larger Nt_sub")
    # valid starts: the chunk must reach n_hi (n0 >= lower) and keep n_lo inside its left
    # pad (n0 <= upper; the grid-start chunk may keep pixels from 0); pick the lowest EVEN
    lower = max(0, n_hi - (Nt_sub - right_pad))
    upper = min(n_lo - n_pad if n_lo >= n_pad else 0, Nt - Nt_sub)
    # keeping the right edge is only valid for the chunk that ENDS at the grid end
    n0 = (Nt - Nt_sub) if right_pad == 0 else lower + (lower % 2)
    if n0 > upper or n0 < lower or n0 % 2:
        raise ValueError(f"no even-start chunk of {Nt_sub} covers [{n_lo}, {n_hi}) with n_pad={n_pad}")
    return n0


def tail_chunk_plan(n_lo, n_hi, Nt, Nt_sub):
    """Tile ``[n_lo, n_hi)`` with even-start chunks: list of ``(n0, keep_lo, keep_hi)``.

    Each chunk keeps its interior ``[n_pad, Nt_sub - n_pad)`` (``n_pad = Nt_sub // 4``;
    the grid-end chunk keeps through its end). The step is the interior minus 2 so the
    even-start constraint always fits.

    Args:
        n_lo (int): First pixel to cover.
        n_hi (int): One past the last pixel to cover.
        Nt (int): Pixels of the full grid.
        Nt_sub (int): Pixels per chunk.

    Returns:
        list[tuple[int, int, int]]: ``(n0, keep_lo, keep_hi)`` per chunk, with
        chunk-local keep bounds, tiling ``[n_lo, n_hi)`` without overlap.

    Raises:
        ValueError: From :func:`chunk_start_for_pixels`.
    """
    n_pad = Nt_sub // 4
    plan = []
    start = int(n_lo)
    while start < n_hi:
        stop = min(int(n_hi), start + Nt_sub - 2 * n_pad - 2)
        if stop > Nt - n_pad:                # the grid-end chunk takes the rest
            stop = int(n_hi)
        n0 = chunk_start_for_pixels(start, stop, Nt, Nt_sub, n_pad)
        plan.append((n0, start - n0, stop - n0))
        start = stop
    return plan


def splice_chunk(out, chunk, n0, keep_lo, keep_hi):
    """Raw-assignment splice ``out[..., n0+keep_lo : n0+keep_hi] = chunk[..., keep_lo:keep_hi]``.

    Not ``add_signal``: ``domains._apply_wdm_add`` has no partial-overlap path.
    Writes ``out`` in place along its last (time-pixel) axis.

    Args:
        out: Destination array, last axis = global pixels.
        chunk: Chunk coefficients, last axis = chunk-local pixels.
        n0 (int): Chunk start pixel.
        keep_lo (int): First chunk-local pixel to copy.
        keep_hi (int): One past the last chunk-local pixel to copy.
    """
    out[..., int(n0) + int(keep_lo):int(n0) + int(keep_hi)] = chunk[..., int(keep_lo):int(keep_hi)]


# ----------------------------------------------------------------------
# Layer-grouping (narrow-band GB inner product)
# ----------------------------------------------------------------------

def compute_layer_groups(params_array, layer_df, f0_param_index=1,
                          group_band_layers=5, margin_layers=0,
                          data_index_all=None, noise_index_all=None):
    """Group binaries by (m-band, data_index) for get_ll dispatch.

    Each group's kernel iteration is restricted to the m-band
    ``[m_lo - margin_layers, m_hi + margin_layers)`` -- cuts WDM-data
    global-mem reads by ~``Nf / group_band_layers``.

    The default ``group_band_layers=5`` gives a 5-wide band centered on
    each group's carrier WDM layer (``[m_floor - 2, m_floor + 3)`` for
    ``group_band_layers=5``), matching the validated ``mm5`` narrow-band
    convention for GB inner products. ``group_band_layers=3`` (``m_floor
    ± 1``) is also enough when a Tukey (or other) window is applied
    to the *injection* data -- the window suppresses the m_floor-1
    spillover so 2-3 layers around the carrier capture essentially all
    of the in-band power.

    Binaries are sorted by ``(data_index, m_floor)`` with ``m_floor =
    floor(f0 / layer_df)``; a group is a run of sorted binaries in one data slot
    whose carriers span fewer than ``group_band_layers`` layers. The scan runs on
    the host (one device-to-host copy on CuPy).

    Args:
        params_array: ``(num_bin, nparams)`` parameters (numpy or cupy; outputs
            follow).
        layer_df (float): WDM layer spacing (Hz).
        f0_param_index (int): Column of the carrier frequency. Default ``1`` (GB).
        group_band_layers (int): Band width in layers. Default ``5``.
        margin_layers (int): Extra layers on each side. Default ``0``.
        data_index_all: ``(num_bin,)`` data slot per binary; default all zeros.
        noise_index_all: ``(num_bin,)`` noise slot per binary; must equal
            ``data_index_all`` when not all zero.

    Returns:
        dict: The per-group arrays the kernels consume: ``binary_perm``
        ``(num_bin,)`` (sort order), ``group_starts`` / ``group_ends`` (ranges into
        the sorted order), ``group_m_lo`` / ``group_m_hi`` (layer band
        ``[m0 - half - margin, m_last + group_band_layers - half + margin)``, with
        ``half = group_band_layers // 2``), ``group_data_index`` (int32 arrays) and
        ``n_groups`` (int).

    Raises:
        AssertionError: For misshapen index arrays, or a non-trivial
            ``noise_index_all`` that differs from ``data_index_all``.
    """
    xp = get_array_module(params_array)
    params_array = xp.asarray(params_array)
    num_bin = params_array.shape[0]
    f0 = params_array[:, int(f0_param_index)]
    m_floor = xp.floor(f0 / float(layer_df)).astype(xp.int32)

    if data_index_all is None:
        data_index_all = xp.zeros(num_bin, dtype=xp.int32)
    else:
        data_index_all = xp.asarray(data_index_all, dtype=xp.int32)
        assert data_index_all.shape == (num_bin,)

    if noise_index_all is None:
        noise_index_all = xp.zeros(num_bin, dtype=xp.int32)
    else:
        noise_index_all = xp.asarray(noise_index_all, dtype=xp.int32)
        assert noise_index_all.shape == (num_bin,)
        if xp.any(noise_index_all != 0):
            assert xp.array_equal(noise_index_all, data_index_all), (
                "noise_index_all must equal data_index_all when noise_index "
                "is non-trivial; grouping path assumes data + PSD share a slab")
    if xp != np:
        # Manual lexsort equivalent to np.lexsort((m_floor, data_index_all)):
        # stable-sort by secondary key first, then stable-sort by primary.
        # Works under both numpy and cupy (cupy.lexsort requires a 2-D keys
        # array and rejects axis kwargs).
        _sec = xp.argsort(m_floor, kind='stable')
        order = _sec[xp.argsort(data_index_all[_sec], kind='stable')]
    else:
        order = xp.lexsort((m_floor, data_index_all))
    sorted_m = m_floor[order]
    sorted_data_idx = data_index_all[order]

    # Build the iterated m-band as a window of width ``group_band_layers``
    # *centred* on each group's carrier(s). Mirrors
    # :func:`compute_swap_layer_groups`'s ``pair_m_{lo,hi}_b`` convention so
    # a single-binary group still sees the full ``group_band_layers``-wide
    # window instead of just its carrier layer. ``half = group_band_layers
    # // 2`` puts more bins above the carrier than below (e.g. for
    # ``group_band_layers=5``, ``half=2`` -> band ``[m-2, m+3)``), matching
    # the WDM transform's slightly-asymmetric spectral spread.
    half = int(group_band_layers) // 2
    # Host-side boundary scan (perf, 2026-08): every condition below reads a
    # 0-d element, and on CuPy each ``bool()``/``int()`` of a 0-d device
    # array is one device synchronization — O(num_bin) syncs per call. Pull
    # the two small int arrays to HOST once (a single D2H copy) and run the
    # IDENTICAL loop on numpy (zero syncs). The group arrays are uploaded
    # via ``xp.asarray`` below exactly as before, so outputs are unchanged.
    if xp is np:
        sorted_m_host = sorted_m
        sorted_data_idx_host = sorted_data_idx
    else:
        sorted_m_host = np.asarray(sorted_m.get())
        sorted_data_idx_host = np.asarray(sorted_data_idx.get())
    starts, ends, m_los, m_his, di = [], [], [], [], []
    i = 0
    while i < num_bin:
        m0 = sorted_m_host[i]
        d0 = sorted_data_idx_host[i]
        j = i
        while j < num_bin and sorted_data_idx_host[j] == d0 and (
                sorted_m_host[j] - m0) < group_band_layers:
            j += 1
        m_lo = int(m0 - half - margin_layers)
        m_hi = int(sorted_m_host[j - 1] + (group_band_layers - half) + margin_layers)
        starts.append(int(i))
        ends.append(int(j))
        m_los.append(m_lo)
        m_his.append(m_hi)
        di.append(int(d0))
        i = j

    return dict(
        binary_perm      = xp.asarray(order,  dtype=xp.int32),
        group_starts     = xp.asarray(starts, dtype=xp.int32),
        group_ends       = xp.asarray(ends,   dtype=xp.int32),
        group_m_lo       = xp.asarray(m_los,  dtype=xp.int32),
        group_m_hi       = xp.asarray(m_his,  dtype=xp.int32),
        group_data_index = xp.asarray(di,     dtype=xp.int32),
        n_groups         = len(starts),
    )


def compute_swap_layer_groups(params_add, params_remove, layer_df,
                               f0_param_index=1,
                               group_band_layers=5, margin_layers=0,
                               data_index_all=None, noise_index_all=None):
    """Layer-grouping for swap_ll's two-pass cache strategy.

    Returns the same fields as :func:`compute_layer_groups` (driven by
    the ADD template's carrier WDM layer) plus per-pair remove-template
    bands (``pair_m_lo_b``, ``pair_m_hi_b``) the kernel uses for its
    second pass over the leftover <d|rem> / <rem|rem> pixels.

    Args:
        params_add: ``(num_bin, nparams)`` add-template parameters.
        params_remove: ``(num_bin, nparams)`` remove-template parameters.
        layer_df (float): WDM layer spacing (Hz).
        f0_param_index (int): Column of the carrier frequency. Default ``1``.
        group_band_layers (int): Band width in layers. Default ``5``.
        margin_layers (int): Extra layers on each side. Default ``0``.
        data_index_all: Per-binary data slot (see :func:`compute_layer_groups`).
        noise_index_all: Per-binary noise slot (see :func:`compute_layer_groups`).

    Returns:
        dict: :func:`compute_layer_groups` output plus ``pair_m_lo_b`` /
        ``pair_m_hi_b`` (``(num_bin,)`` int32, in ``binary_perm`` order).

    Raises:
        AssertionError: If the two parameter arrays differ in length.
    """
    xp = get_array_module(params_add)
    params_add    = xp.asarray(params_add)
    params_remove = xp.asarray(params_remove)
    num_bin = params_add.shape[0]
    assert params_remove.shape[0] == num_bin

    add_groups = compute_layer_groups(
        params_add, layer_df=layer_df, f0_param_index=f0_param_index,
        group_band_layers=group_band_layers, margin_layers=margin_layers,
        data_index_all=data_index_all, noise_index_all=noise_index_all)

    binary_perm = add_groups["binary_perm"]
    f0_b = params_remove[:, int(f0_param_index)]
    m_b  = xp.floor(f0_b / float(layer_df)).astype(xp.int32)
    half = group_band_layers // 2

    m_b_sorted     = m_b[binary_perm]
    pair_m_lo_b    = (m_b_sorted - half - margin_layers).astype(xp.int32)
    pair_m_hi_b    = (m_b_sorted + (group_band_layers - half)
                       + margin_layers).astype(xp.int32)

    add_groups["pair_m_lo_b"] = pair_m_lo_b
    add_groups["pair_m_hi_b"] = pair_m_hi_b
    return add_groups


__all__ = [
    "RECOMMENDED_TUKEY_ALPHA_TD",
    "RECOMMENDED_TUKEY_ALPHA_HET_WIDE",
    "RECOMMENDED_TUKEY_ALPHA_HET_NARROW",
    "USE_RECOMMENDED_TUKEY",
    "recommended_tukey_alpha",
    "resolve_tukey_alpha",
    "compute_chunk_geometry",
    "compute_wdm_window",
    "compute_layer_groups",
    "compute_swap_layer_groups",
]
