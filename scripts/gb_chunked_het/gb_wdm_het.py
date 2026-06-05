"""Python wrapper for the chunked-heterodyne WDM kernels (GB + SOBBH).

This module gives the user a single Python entry point for the
heterodyne WDM workflow we built and validated in
``check_shortened_wdm.py``:

* :func:`compute_chunk_geometry` -- pre-builds the per-chunk
  ``starts / keep_lo / keep_hi / n_global_lo`` arrays handed to the
  kernels (works for both aligned and partial-slide tilings).
* :func:`compute_wdm_window` -- phitilde samples expected by
  ``gb_chunk_fd_to_wdm`` on the host. Hands a Nt_sub-long array off to
  the C++ kernel.
* :func:`group_binaries_by_layer` -- clusters binaries by their carrier
  WDM layer so each kernel launch's shared PSD/data slab amortises over
  binaries in the same band.
* :class:`GBWDMHeterodyne` -- main user-facing class. Constructor takes
  the WDM grid + obs setup; methods ``fill_global``, ``get_ll``,
  ``swap_ll`` follow the same call signatures as the existing
  ``GBComputationGroup`` API.

Until the pybind binding for the new C++ kernels is wired
(``lisa-on-gpu/src/fastlisaresponse/cutils/TDIonthefly.cu``:
``gb_wdm_het_fill_global_kernel`` / ``gb_wdm_het_get_ll_kernel`` /
``gb_wdm_het_swap_ll_kernel``), the wrapper uses a pure-Python fallback
backed by ``check_shortened_wdm._stitched_wdm_from_heterodyne`` /
``chunked_get_ll_python_reference``. Switching to the native path
should be a single ``backend.gb_wdm_het_*_wrap(...)`` call replacement
inside each method.
"""
from __future__ import annotations

import numpy as np

# Reuse the validated chunked-heterodyne primitives from the test script.
from check_shortened_wdm import (
    CachedHeterodyneGenerator,
    USE_RECOMMENDED_TUKEY,
    chunked_get_ll_python_reference,
    group_binaries_by_layer,
    recommended_tukey_alpha,
    _resolve_tukey_alpha,
)
from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import FDSettings, FDSignal, WDMSettings
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.utils.parallelbase import FastLISAResponseParallelModule


# ---------------------------------------------------------------------------
# Host helpers (mirrored on the C++ side at kernel-launch time)
# ---------------------------------------------------------------------------

def compute_chunk_geometry(Nt, Nt_sub, n_pad):
    """Pre-compute per-chunk stitching geometry for the heterodyne kernels.

    Returns a dict of numpy arrays:

      ``starts``           (n_chunks,) int   -- chunk start in WDM-pixel units
      ``keep_lo``          (n_chunks,) int   -- chunk-local lo to keep
      ``keep_hi``          (n_chunks,) int   -- chunk-local hi (exclusive)
      ``n_global_lo``      (n_chunks,) int   -- global WDM pixel where the
                                                first kept sample lands

    Handles the partial-slide case (when ``(Nt - Nt_sub) % step != 0``):
    appends one extra chunk at ``Nt - Nt_sub`` and adjusts the previous
    chunk's ``keep_hi`` so the overlap region isn't double-counted.

    Mirrors the Python implementation in
    ``check_shortened_wdm._stitched_wdm_partial_slide``.
    """
    step = int(Nt_sub) - 2 * int(n_pad)
    assert step > 0 and step % 2 == 0, (Nt_sub, n_pad, step)
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


def compute_layer_groups(params_array, layer_df, f0_param_index=1,
                          group_band_layers=5, margin_layers=0,
                          data_index_all=None, noise_index_all=None):
    """Group binaries by (m-band, data_index) for get_ll / swap_ll dispatch.

    The chunked-het get_ll kernel accumulates ``<d|h>``, ``<h|h>`` over a
    range of m-layers per binary. A binary's slow signal has support only
    in a narrow band (~5 layers) around its carrier WDM layer
    ``m0 = floor(f0 / layer_df)``. Without grouping, the kernel reads
    data_d / invC over ALL Nf layers; with grouping, it iterates only the
    band for each group of binaries that share an m-window. That cuts
    WDM-data global-mem reads by roughly ``Nf / group_band_layers``.

    Binaries are sorted by their m_layer (within a fixed data_index
    bucket), and contiguous runs spanning at most ``group_band_layers``
    are bundled into one group. The kernel iterates the m-band
    ``[m_lo - margin_layers, m_hi + margin_layers)`` per group (margin
    for Doppler / wavelet-tail flexibility).

    Args:
        params_array: ``(num_bin, nparams)`` float array.
        layer_df: WDM layer spacing (Hz).
        f0_param_index: column of ``params_array`` holding the carrier
            frequency (1 for GB's ``f0``, 5 for SOBBH's ``f_low``).
        group_band_layers: maximum m-layer span per group (default 5).
        margin_layers: extra layers added on each side of each group's
            band (default 0; set 1 or 2 for Doppler flexibility).
        data_index_all: optional ``(num_bin,)`` int array. If supplied,
            grouping respects data-index buckets (binaries with different
            data slabs land in separate groups). Default = all zeros
            (single shared data slab).
        noise_index_all: optional ``(num_bin,)`` int array. Asserted to
            match ``data_index_all`` (we only support the case where
            data and PSD live on the same data slab). If both are all
            zeros, the assertion is trivially satisfied.

    Returns:
        dict with keys
            ``binary_perm``: ``(num_bin,) int32`` permutation -- kernel
                                indexes binaries via this array.
            ``group_starts``: ``(n_groups,) int32`` start index into
                                ``binary_perm`` (inclusive).
            ``group_ends``  : ``(n_groups,) int32`` end index (exclusive).
            ``group_m_lo``  : ``(n_groups,) int32`` lower m-layer (inclusive).
            ``group_m_hi``  : ``(n_groups,) int32`` upper m-layer (exclusive).
            ``group_data_index`` : ``(n_groups,) int32`` data-slab index
                                per group.
            ``n_groups``    : int.
    """
    params_array = np.asarray(params_array)
    num_bin = params_array.shape[0]
    f0 = params_array[:, int(f0_param_index)]
    m_floor = np.floor(f0 / float(layer_df)).astype(np.int32)

    if data_index_all is None:
        data_index_all = np.zeros(num_bin, dtype=np.int32)
    else:
        data_index_all = np.asarray(data_index_all, dtype=np.int32)
        assert data_index_all.shape == (num_bin,), data_index_all.shape

    if noise_index_all is None:
        noise_index_all = np.zeros(num_bin, dtype=np.int32)
    else:
        noise_index_all = np.asarray(noise_index_all, dtype=np.int32)
        assert noise_index_all.shape == (num_bin,)
        # Only assert if noise_index has anything other than zeros.
        if np.any(noise_index_all != 0):
            assert np.array_equal(noise_index_all, data_index_all), (
                "noise_index_all must equal data_index_all when noise_index "
                "is non-trivial; grouping path assumes data + PSD share a slab")

    # Sort by (data_index, m_floor) so consecutive binaries share the band.
    order = np.lexsort((m_floor, data_index_all))
    sorted_m = m_floor[order]
    sorted_data_idx = data_index_all[order]

    starts, ends, m_los, m_his, di = [], [], [], [], []
    i = 0
    while i < num_bin:
        m0 = sorted_m[i]
        d0 = sorted_data_idx[i]
        # Walk forward as long as we stay in the same data_index AND within
        # group_band_layers of m0.
        j = i
        while j < num_bin and sorted_data_idx[j] == d0 and (
                sorted_m[j] - m0) < group_band_layers:
            j += 1
        m_lo = int(m0 - margin_layers)
        m_hi = int(sorted_m[j - 1] + 1 + margin_layers)  # exclusive
        starts.append(int(i))
        ends.append(int(j))
        m_los.append(m_lo)
        m_his.append(m_hi)
        di.append(int(d0))
        i = j

    return dict(
        binary_perm     = np.asarray(order,  dtype=np.int32),
        group_starts    = np.asarray(starts, dtype=np.int32),
        group_ends      = np.asarray(ends,   dtype=np.int32),
        group_m_lo      = np.asarray(m_los,  dtype=np.int32),
        group_m_hi      = np.asarray(m_his,  dtype=np.int32),
        group_data_index= np.asarray(di,     dtype=np.int32),
        n_groups        = len(starts),
    )


def compute_swap_layer_groups(params_add, params_remove, layer_df,
                              f0_param_index=1,
                              group_band_layers=5, margin_layers=0,
                              data_index_all=None, noise_index_all=None):
    """Layer-grouping for swap_ll's two-pass cache strategy.

    Returns a group structure identical to :func:`compute_layer_groups`
    (driven by the ADD template's carrier WDM layer m_a = floor(f0_add /
    layer_df)) PLUS a per-pair remove-template band (m_lo_b, m_hi_b)
    derived from m_b = floor(f0_rem / layer_df). The kernel uses the
    group's add band [group_m_lo, group_m_hi) for pass 1 (accumulates all
    five inner products, since w_add is zero outside) and the pair's
    remove band [pair_m_lo_b, pair_m_hi_b) for pass 2 (picks up the
    leftover <d|rem> and <rem|rem> pixels outside the add band).

    Per-pair output arrays are indexed by SORTED bin_iter (i.e. in
    binary_perm order), matching the kernel's iteration.
    """
    params_add    = np.asarray(params_add)
    params_remove = np.asarray(params_remove)
    num_bin = params_add.shape[0]
    assert params_remove.shape[0] == num_bin

    add_groups = compute_layer_groups(
        params_add, layer_df=layer_df, f0_param_index=f0_param_index,
        group_band_layers=group_band_layers, margin_layers=margin_layers,
        data_index_all=data_index_all, noise_index_all=noise_index_all)

    binary_perm = add_groups["binary_perm"]
    f0_b = params_remove[:, int(f0_param_index)]
    m_b  = np.floor(f0_b / float(layer_df)).astype(np.int32)
    half = group_band_layers // 2

    # Per-pair remove band, ordered by sorted bin_iter (binary_perm).
    m_b_sorted     = m_b[binary_perm]
    pair_m_lo_b    = (m_b_sorted - half - margin_layers).astype(np.int32)
    pair_m_hi_b    = (m_b_sorted + (group_band_layers - half)
                       + margin_layers).astype(np.int32)

    add_groups["pair_m_lo_b"] = pair_m_lo_b
    add_groups["pair_m_hi_b"] = pair_m_hi_b
    return add_groups


def compute_wdm_window(Nf, Nt_sub, dt, backend="cpu"):
    """Return the Nt_sub-long phitilde sampled at the chunk's grid.

    Equivalent to ``WDMSettings(Nf, Nt_sub, dt).window`` -- exposed here
    as a free function so the kernel host code can call it without
    constructing a full settings object.
    """
    return np.asarray(
        WDMSettings(Nf=int(Nf), Nt=int(Nt_sub), dt=float(dt),
                    force_backend=backend).window,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# GPU grid-sizing for the chunked-heterodyne kernels
# ---------------------------------------------------------------------------

# Per-arch resource limits relevant for occupancy on the chunked-het kernels.
#   max_shared_per_sm_bytes : usable shared-mem budget per SM (the "shared"
#                             carveout of the unified L1+shared block; opt-in
#                             via cudaFuncSetAttribute when > 48 KB).
#   max_threads_per_sm      : max concurrent threads per SM.
#   max_blocks_per_sm       : hard cap on concurrent blocks per SM.
#   num_sms                 : SM count on a representative variant of the
#                             card (SXM5 for A100/H100; PCIe variants have
#                             fewer SMs).
#   max_shared_per_block    : opt-in upper bound for static-or-dynamic shared
#                             per block (used to raise an error if the kernel
#                             genuinely won't fit at all).
_GPU_ARCH_LIMITS = {
    "A100": dict(
        max_shared_per_sm_bytes=164 * 1024,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        num_sms=108,
        max_shared_per_block=163 * 1024,
    ),
    "H100": dict(
        max_shared_per_sm_bytes=228 * 1024,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        num_sms=132,
        max_shared_per_block=227 * 1024,
    ),
}


def chunked_het_grid_dim(
    gpu_arch,
    n_chunks,
    shared_per_block_bytes,
    threads_per_block,
    latency_hide_factor=2,
):
    """Pick (grid_dim_x, resident_blocks_per_sm) for the chunked-het kernels.

    Maximizes per-SM occupancy under the joint shared-mem + threads-per-SM
    constraints, then sizes the grid so the runtime always has at least
    ``latency_hide_factor`` blocks queued ahead of each SM. Caps the grid at
    ``n_chunks`` so different blocks process different chunks (no double
    work on fill_global's per-pixel write).

    Args:
        gpu_arch: ``"A100"`` or ``"H100"``. Raises ``ValueError`` otherwise.
        n_chunks: number of chunks the kernel will iterate over.
        shared_per_block_bytes: static + dynamic shared-mem per block, in bytes.
        threads_per_block: NUM_THREADS used by the kernel launch.
        latency_hide_factor: launch this many times the resident block count
            per SM so the scheduler always has a ready block to start when one
            retires. Default 2 -- conservative; 1 leaves no slack, 4+ wastes
            per-block launch overhead.

    Returns:
        dict with keys
          ``grid_dim``                 (int) -- the launch grid size,
          ``resident_blocks_per_sm``   (int) -- occupancy-limited blocks/SM,
          ``max_blocks_per_sm_shared`` (int) -- shared-mem-limited blocks/SM,
          ``max_blocks_per_sm_threads``(int) -- threads-limited blocks/SM,
          ``num_sms``                  (int),
          ``threads_per_sm_resident``  (int) -- threads * resident blocks,
          ``theoretical_occupancy``    (float in [0,1]) -- threads_per_sm_resident
                                                          / max_threads_per_sm,
          ``shared_headroom_bytes``    (int) -- spare shared-mem at the chosen
                                                resident count; negative if the
                                                kernel doesn't fit on this arch.

    Raises:
        ValueError: unknown ``gpu_arch`` or kernel doesn't fit at all.
    """
    arch = str(gpu_arch).upper()
    if arch not in _GPU_ARCH_LIMITS:
        raise ValueError(
            f"Unsupported gpu_arch={gpu_arch!r}; supported: "
            f"{sorted(_GPU_ARCH_LIMITS)}"
        )
    L = _GPU_ARCH_LIMITS[arch]

    s_per_block = int(shared_per_block_bytes)
    t_per_block = int(threads_per_block)
    if s_per_block > L["max_shared_per_block"]:
        raise ValueError(
            f"Kernel needs {s_per_block / 1024:.1f} KB shared per block; "
            f"{arch} max per block is {L['max_shared_per_block'] / 1024:.1f} KB. "
            f"Reduce N_sparse or move more buffers to heap."
        )
    if t_per_block > 1024:
        raise ValueError(
            f"threads_per_block={t_per_block} exceeds CUDA hard cap of 1024."
        )

    max_blocks_shared = (
        L["max_shared_per_sm_bytes"] // s_per_block if s_per_block > 0 else L["max_blocks_per_sm"]
    )
    max_blocks_threads = L["max_threads_per_sm"] // t_per_block

    resident = min(max_blocks_shared, max_blocks_threads, L["max_blocks_per_sm"])
    if resident < 1:
        raise ValueError(
            f"Kernel does not fit one resident block on {arch} "
            f"(shared={s_per_block/1024:.1f} KB, threads={t_per_block})."
        )

    target_grid = resident * L["num_sms"] * int(latency_hide_factor)
    grid_dim = min(int(n_chunks), target_grid)
    if grid_dim < 1:
        grid_dim = 1

    threads_per_sm_resident = resident * t_per_block
    headroom = L["max_shared_per_sm_bytes"] - resident * s_per_block

    return dict(
        grid_dim=int(grid_dim),
        resident_blocks_per_sm=int(resident),
        max_blocks_per_sm_shared=int(max_blocks_shared),
        max_blocks_per_sm_threads=int(max_blocks_threads),
        num_sms=int(L["num_sms"]),
        threads_per_sm_resident=int(threads_per_sm_resident),
        theoretical_occupancy=float(threads_per_sm_resident) / float(L["max_threads_per_sm"]),
        shared_headroom_bytes=int(headroom),
    )


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class GBWDMHeterodyne(FastLISAResponseParallelModule):
    """High-level Python entry point for the chunked-heterodyne WDM GB kernels.

    Inherits :class:`FastLISAResponseParallelModule`, so the compute
    backend (``"cpu"`` / ``"cuda11x"`` / ``"cuda12x"`` / ``"cuda13x"`` /
    ``"jax"``) is chosen at instantiation via ``force_backend`` and
    all subsequent dispatch reads ``self.backend`` / ``self.backend.xp``
    / ``self.backend.name``. Methods MUST NOT accept a ``backend=`` or
    ``use_cpp=`` runtime kwarg (per the sprint-wide rule in
    ``CLAUDE.md``); construct a fresh instance to switch backends.

    Constructor sets up the WDM grid, chunk geometry, phitilde window,
    and a cached :class:`CachedHeterodyneGenerator`. Methods
    ``fill_global``, ``get_ll``, ``swap_ll`` mirror the existing
    ``GBComputationGroup`` API but route through the heterodyne path.

    Args:
        Nf, Nt: WDM grid dimensions of the global template.
        dt: TD sample step (seconds).
        T_full, t_ref_full: full observation duration + source-phase
            reference time.
        Nt_sub: per-chunk WDM time pixels.
        n_pad: WDM pixels discarded at each chunk edge during stitching.
        N_sparse: heterodyne FFT length per chunk; must be <= 256
            (FAST_WDM_N_SPARSE_MAX in the C++ kernel).
        tukey_alpha: explicit Tukey alpha, or ``USE_RECOMMENDED_TUKEY``
            (default) to auto-pick.
        use_tukey: if False, force alpha = 0.
        nchannels: 3 for XYZ.
        force_backend: forwarded to :class:`FastLISAResponseParallelModule`.
            Accepts ``"cpu"`` / ``"cuda11x"`` / ``"cuda12x"`` /
            ``"cuda13x"`` / ``"jax"`` (or a full backend tuple). The
            resulting ``self.backend`` drives both the C++/CUDA
            template-kernel path (CPU/CUDA backends) and the JAX
            autograd path (``"jax"`` backend).
        tdi_gen: TDI generation string passed through to the GB source.
        orbits: lisatools Orbits subclass; auto-defaults to
            EqualArmlengthOrbits on the same backend if None.
        t_obs_start: absolute time at which the WDM grid starts.
        N_cp_sig, N_cp_orbit: source-signal / orbit spline cache
            density (see kernel docstrings for the validated values).
        jax_chunk: default leaf-axis chunk size for JAX dispatch
            (memory escape hatch; ``None`` resolves via env at call
            time).
    """

    # Subclass-overridable config: which C++ Wrap class on the backend
    # to instantiate, and which prefix the chunked-het methods use.
    # GBWDMHeterodyne -> GBComputationGroupWrap.gb_wdm_het_*
    # SOBBHWDMHeterodyne -> SOBBHComputationGroupWrap.sobbh_wdm_het_*
    _CPP_WRAP_ATTR = "GBComputationGroupWrap"
    _CPP_METHOD_PREFIX = "gb_wdm_het"
    _CPP_NPARAMS = 9
    _CPP_F0_PARAM_INDEX = 1   # GBTDIonTheFly: params[1] = f0

    def __init__(self, Nf, Nt, dt, T_full, t_ref_full,
                 Nt_sub=256, n_pad=32, N_sparse=256,
                 tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
                 nchannels=3, force_backend=None,
                 tdi_gen="2nd generation",
                 orbits=None,
                 t_obs_start=0.0,
                 N_cp_sig=0,
                 N_cp_orbit=0,
                 d_d=0.0,
                 jax_chunk=None):
        # Resolve the backend object up front so all nested
        # constructors share the same backend string.
        super().__init__(force_backend=force_backend)
        # ``backend_short`` is the trailing tag of the resolved backend
        # name (``cpu``, ``cuda12x``, ``jax``) -- the form lisatools /
        # gbgpu / fastlisaresponse constructors accept as
        # ``force_backend=...``.
        self._backend_short = self.backend.name.split("_")[-1]
        # jax_chunk:
        #   Default leaf-axis chunk size for the JAX paths
        #   (get_ll_jax / get_ll_grad_jax / hessian and their *_wdm
        #   ACA wrappers). ``None`` -> resolve at call time via
        #   _resolve_jax_chunk (env GBHET_JAX_CHUNK / JAX_GRAD_CHUNK,
        #   else 0 = single big batched call). C++ paths are
        #   unchunked -- the kernel's per-source loop limits CPU
        #   throughput regardless of batch size.
        self.jax_chunk = jax_chunk if jax_chunk is None else int(jax_chunk)
        # N_cp_sig:
        #   0  -> direct path (call get_tdi at all N_sparse points).
        #         Bit-precision match to the Python reference; default.
        #   >0 -> source-signal spline cache. Calls get_tdi_heterodyned at
        #         N_cp_sig control points, fits cubic splines through
        #         (amp, tdi_phase, dphi_ref_het), evaluates at the N_sparse
        #         grid. ~N_sparse/N_cp_sig fewer get_tdi calls per
        #         (chunk, binary). Typical value: 48 (half-day baseline,
        #         clears mm < 1e-9 for GB; ~4e-9 for SOBBH).
        # N_cp_orbit:
        #   0  -> raw global-mem orbit lookups inside get_tdi.
        #   >0 -> orbit spline cache. Per chunk, samples orbits at N_cp_orbit
        #         uniform times within the chunk and PCR-fits cubic splines
        #         for the 6 link LTTs and 9 spacecraft-xyz positions. Reused
        #         across all binaries in the chunk -- replaces ~num_bin x
        #         N_sparse x (32-64) global-mem orbit lookups per chunk with
        #         one fit + cheap shared-mem cubic evals. Typical value: 32
        #         (30-day chunks: 2e-12 s L err, 350 m X err). See the
        #         density study in dev_orbit_spline_density.py and
        #         CHUNKED_HET_DESIGN_NOTES.md.
        self.N_cp_sig = int(N_cp_sig)
        self.N_cp_orbit = int(N_cp_orbit)
        self.Nf       = int(Nf)
        self.Nt       = int(Nt)
        self.dt       = float(dt)
        self.T_full   = float(T_full)
        self.t_ref_full = float(t_ref_full)
        # Absolute time at which the observation begins. WDM pixel 0
        # corresponds to absolute time ``t_obs_start``; chunk j starts at
        # ``t_obs_start + n0 * layer_dt``. The CachedHeterodyneGenerator
        # evaluates GBTDIonTheFly at these absolute times, so this MUST
        # match the t_arr range the injection was built on (otherwise
        # the orbits / source phase land at the wrong absolute time and
        # the template comes out uncorrelated with the injection).
        self.t_obs_start = float(t_obs_start)
        self.Nt_sub   = int(Nt_sub)
        self.n_pad    = int(n_pad)
        self.N_sparse = int(N_sparse)
        self.tukey_alpha = float(tukey_alpha)
        self.use_tukey   = bool(use_tukey)
        self.nchannels = int(nchannels)
        # ``d_d`` constant the source-side ``get_ll`` adds to
        # ``h_h - 2 d_h``. Defaults to 0 (constructor kwarg) so the
        # returned log-likelihood is the source-only piece; callers
        # add ``-0.5 <d|d>`` themselves if they want the full
        # ``log p(d|h)``. Mirrors the same convention as
        # :class:`fastlisaresponse.gbcomps.GBWDMComputations`.
        self.d_d = float(d_d)

        self.T_chunk    = self.Nf * self.Nt_sub * self.dt
        self.chunk_df   = 1.0 / self.T_chunk
        self.layer_df   = 1.0 / (2.0 * self.Nf * self.dt)
        self.n_rfft_chunk = self.Nf * self.Nt_sub // 2 + 1
        self.log2_N_sparse = int(np.log2(self.N_sparse))
        self.log2_Nt_sub   = int(np.log2(self.Nt_sub))
        assert 2 ** self.log2_N_sparse == self.N_sparse
        assert 2 ** self.log2_Nt_sub   == self.Nt_sub

        self.geometry = compute_chunk_geometry(self.Nt, self.Nt_sub, self.n_pad)
        self.wdm_window = compute_wdm_window(self.Nf, self.Nt_sub, self.dt,
                                             backend=self._backend_short)

        # Cached heterodyne generator, reused across all binaries and chunks.
        # If the caller didn't supply orbits, default to EqualArmlength
        # (cheap and good enough for most tests); for prior-draw scripts
        # the injection's orbit model (e.g. ESAOrbits) MUST be passed
        # here too or the template won't match.
        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=self._backend_short)
        self._gb_kwargs = dict(
            tdi_config=TDIConfig(tdi_gen), orbits=orbits,
            tdi_chan="XYZ", force_backend=self._backend_short,
        )
        self._gen = CachedHeterodyneGenerator(
            T_window=self.T_chunk, t_ref_source=self.t_ref_full,
            N_sparse=self.N_sparse, dt=self.dt, nchannels=self.nchannels,
            gb_kwargs=self._gb_kwargs,
        )

        # FD/WDM settings reused per chunk.
        self._chunk_fd_set  = FDSettings(self.n_rfft_chunk, self.chunk_df,
                                         force_backend=self._backend_short)
        self._chunk_wdm_set = WDMSettings(Nf=self.Nf, Nt=self.Nt_sub,
                                          dt=self.dt,
                                          force_backend=self._backend_short)

        # Native-kernel routing setup. Deferred until the first
        # fill_global / get_ll / swap_ll call (so callers can construct
        # GBWDMHeterodyne before orbits have been auto-configured by
        # GBTDIonTheFly). The chosen backend (``self.backend``) drives
        # which path runs -- CPU/CUDA backends use the C++ kernel,
        # the ``"jax"`` backend uses the JAX kernel.
        self._cpp_setup_done = False
        self._cpp_orbits_obj = orbits
        self._cpp_tdi_gen = tdi_gen

    @property
    def _uses_cpp_kernel(self) -> bool:
        """True when ``self.backend`` is a C++ backend (cpu/cuda*)
        rather than the JAX backend. Used internally to dispatch
        between the C++ and JAX implementations -- callers never need
        to pass a ``backend=`` kwarg.
        """
        return self.backend.name != "fastlisaresponse_jax"

    # ------------------------------------------------------------------
    # C++ chunked-het routing
    # ------------------------------------------------------------------

    def _ensure_cpp_setup(self):
        if self._cpp_setup_done:
            return
        self._setup_cpp(self._cpp_orbits_obj, self._cpp_tdi_gen)
        self._cpp_setup_done = True

    def _setup_cpp(self, orbits, tdi_gen):
        """Construct the C++ Wrap objects and precompute geometry arrays.

        Called lazily on the first ``fill_global`` / ``get_ll`` /
        ``swap_ll`` call when ``self.backend`` is one of the C++
        backends (``cpu`` / ``cuda*``). Uses the inherited
        :class:`FastLISAResponseParallelModule` ``self.backend`` --
        the ``GBComputationGroupWrap`` / ``SOBBHComputationGroupWrap``
        classes live on it. Orbits must be configured by this point
        (typically done implicitly by GBTDIonTheFly's first use).
        """
        # ``self.backend`` is the resolved FastLISAResponseBackend
        # object (CPU / CUDA variant); kept as ``self._be`` as an alias
        # so the rest of the class keeps the short name.
        self._be = self.backend
        wrap_cls = getattr(self._be, self._CPP_WRAP_ATTR)
        self._cpp_group = wrap_cls()

        # Ensure orbits are configured for linear-interp C++ access.
        # ESAOrbits etc. need ``configure(..., linear_interp_setup=True)``
        # before their ``pycppdetector_args`` are populated; if a caller
        # hands us an orbits object that hasn't been so configured, we
        # do it now using the full obs span.
        if getattr(orbits, "_pycppdetector_args", None) is None:
            t_arr = np.arange(0.0, self.T_full + self.dt, self.dt) + self.t_obs_start
            try:
                orbits.configure(t_arr=t_arr, dt=self.dt, linear_interp_setup=True)
            except TypeError:
                orbits.configure(t_arr=t_arr)

        # Build the C++ Orbits / TDIConfig wrapper objects (mirrors
        # gbcomps.GBLikelihood.__init__). We MUST keep the Python
        # TDIConfig alive as a class member -- TDIConfigWrap stores raw
        # pointers into its numpy arrays, and if the Python object is
        # GC'd the data is freed and the kernel reads garbage. (Got
        # bitten by this -- manifested as ``Bad link ind`` from
        # ``Orbits::get_link_ind`` with bogus combination_link values.)
        self._cpp_orbits = self._be.OrbitsWrap(*orbits.pycppdetector_args)
        from fastlisaresponse.tdiconfig import TDIConfig as _TDIConfig
        self._tdi_cfg_py = _TDIConfig(tdi_gen)
        self._cpp_tdi_config = self._be.TDIConfigWrap(
            *self._tdi_cfg_py.pytdiconfig_args
        )
        # Same anti-GC pinning for orbits' pycppdetector_args arrays
        # (the OrbitsWrap stores raw pointers into them).
        self._orbits_py = orbits

        # WDMSettings wrap (recent C++-binding change: every
        # ``gb_wdm_het_*`` entry point now takes a ``WDMSettingsWrap *``
        # immediately after ``tdi_config_wrap``, reading Nf / Nt /
        # ind_min_f / ind_max_f / ind_min_t / ind_max_t off it instead
        # of taking ``Nf`` / ``Nt`` ints in the int-arg block). For
        # GBWDMHeterodyne the active band IS the full grid -- this
        # class doesn't carry an ind_min/max concept like
        # GBWDMComputations does -- so we point ind_min_* at 0 and
        # ind_max_* at the corresponding N-1, making Nf_active=Nf and
        # Nt_active=Nt for the kernel's data_d / invC length checks.
        layer_df = 1.0 / (2.0 * self.Nf * self.dt)
        layer_dt_grid = self.Nf * self.dt
        self._cpp_wdm_settings = self._be.WDMSettingsWrap(
            float(layer_df),
            float(layer_dt_grid),
            int(self.Nf), int(self.Nt),
            int(self.nchannels),
            int(0), int(self.Nt - 1),   # ind_min_t, ind_max_t (full Nt active)
            int(0), int(self.Nf - 1),   # ind_min_f, ind_max_f (full Nf active)
        )
        # ``tdi_type`` int the kernel uses to switch the invC layout
        # (XYZ cross-channel vs AET/AE diagonal). GBWDMHeterodyne is
        # XYZ-only today; the constant lives on the backend dict.
        self._tdi_type_int = int(self._be.TDITypeDict["XYZ"])

        # Geometry arrays passed straight to the kernel. chunk_t_starts
        # is ABSOLUTE seconds (= t_obs_start + n0 * layer_dt).
        layer_dt = self.Nf * self.dt
        starts = np.asarray(self.geometry["starts"], dtype=float)
        self._cpp_chunk_t_starts        = (self.t_obs_start + starts * layer_dt).copy()
        self._cpp_chunk_keep_lo         = np.asarray(self.geometry["keep_lo"], dtype=np.int32)
        self._cpp_chunk_keep_hi         = np.asarray(self.geometry["keep_hi"], dtype=np.int32)
        self._cpp_chunk_n_global_offset = np.asarray(self.geometry["n_global_lo"], dtype=np.int32)
        self._cpp_wdm_window            = np.asarray(self.wdm_window, dtype=float).copy()
        self.n_chunks                   = int(starts.size)

        # Resolve tukey_alpha once -- the kernel takes a single double.
        self._cpp_tukey_alpha = float(_resolve_tukey_alpha(
            self.tukey_alpha, self.use_tukey, path="heterodyne",
            N_sparse=self.N_sparse,
        ))

    def _flatten_params(self, params_list):
        """Pack a list of length-nparams param vectors into one (num_bin*nparams,) array."""
        arr = np.asarray(params_list, dtype=float).reshape(-1)
        num_bin = arr.size // self._CPP_NPARAMS
        assert num_bin * self._CPP_NPARAMS == arr.size, (
            f"params_list inconsistent with nparams={self._CPP_NPARAMS}: "
            f"got {arr.size} elements, not divisible by {self._CPP_NPARAMS}.")
        return arr.copy(), num_bin

    def _call_cpp_fill_global(self, template_out, params_flat, factors, num_bin, grid_dim):
        # Binding signature (recent): (template, orbits, tdi_config,
        # wdm_settings, params, factors, chunk_t_starts, keep_lo, keep_hi,
        # n_global_offset, wdm_window, n_chunks, num_bin, nparams,
        # Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse, nchannels,
        # n_rfft_chunk, T_chunk, dt, T, t_ref, tukey_alpha, grid_dim,
        # N_cp_sig, N_cp_orbit, m_band_half_width). Nf / Nt are read off
        # ``wdm_settings`` by the kernel and no longer passed as ints.
        getattr(self._cpp_group, f"{self._CPP_METHOD_PREFIX}_fill_global")(
            np.asarray(template_out).reshape(-1),
            self._cpp_orbits, self._cpp_tdi_config,
            self._cpp_wdm_settings,
            params_flat, factors,
            self._cpp_chunk_t_starts,
            self._cpp_chunk_keep_lo, self._cpp_chunk_keep_hi,
            self._cpp_chunk_n_global_offset,
            self._cpp_wdm_window,
            self.n_chunks, int(num_bin), int(self._CPP_NPARAMS),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T_full), float(self.t_ref_full),
            float(self._cpp_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            int(1),  # m_band_half_width
        )

    def _call_cpp_get_ll(self, d_h_out, h_h_out, data_d, invC,
                         params_flat, num_bin, grid_dim, groups=None):
        # data_index_all / noise_index_all are unused by the chunked-het
        # kernel today (data_d / invC are caller-pre-sliced), but the
        # signature still requires them.
        data_idx = np.zeros(num_bin, dtype=np.int32)
        noise_idx = np.zeros(num_bin, dtype=np.int32)
        # Layer-group arrays: when groups is None, pass length-1 zero
        # arrays + n_groups=0 -- the kernel takes the un-grouped path.
        if groups is None:
            binary_perm    = np.zeros(num_bin, dtype=np.int32)
            group_starts   = np.zeros(1,       dtype=np.int32)
            group_ends     = np.zeros(1,       dtype=np.int32)
            group_m_lo     = np.zeros(1,       dtype=np.int32)
            group_m_hi     = np.zeros(1,       dtype=np.int32)
            n_groups       = 0
        else:
            binary_perm    = np.asarray(groups["binary_perm"],  dtype=np.int32)
            group_starts   = np.asarray(groups["group_starts"], dtype=np.int32)
            group_ends     = np.asarray(groups["group_ends"],   dtype=np.int32)
            group_m_lo     = np.asarray(groups["group_m_lo"],   dtype=np.int32)
            group_m_hi     = np.asarray(groups["group_m_hi"],   dtype=np.int32)
            n_groups       = int(groups["n_groups"])
        # Binding signature (recent): inserts ``wdm_settings`` after
        # ``tdi_config`` and ``tdi_type`` after ``t_ref``. Nf / Nt are
        # now read off ``wdm_settings`` by the kernel and dropped from
        # the int-arg block.
        getattr(self._cpp_group, f"{self._CPP_METHOD_PREFIX}_get_ll")(
            d_h_out, h_h_out,
            self._cpp_orbits, self._cpp_tdi_config,
            self._cpp_wdm_settings,
            params_flat, data_idx, noise_idx,
            self._cpp_chunk_t_starts,
            self._cpp_chunk_keep_lo, self._cpp_chunk_keep_hi,
            self._cpp_chunk_n_global_offset,
            self._cpp_wdm_window,
            np.asarray(data_d).reshape(-1),
            np.asarray(invC).reshape(-1),
            self.n_chunks, int(num_bin), int(self._CPP_NPARAMS),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T_full), float(self.t_ref_full),
            int(self._tdi_type_int),
            float(self._cpp_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            binary_perm, group_starts, group_ends,
            group_m_lo, group_m_hi, int(n_groups),
            int(1),  # m_band_half_width (default narrow-band)
        )

    def _call_cpp_swap_ll(self, out_5tuple, data_d, invC,
                          params_add_flat, params_rem_flat,
                          num_bin, grid_dim, groups=None):
        d_h_add, d_h_rem, aa, rr, ar = out_5tuple
        data_idx = np.zeros(num_bin, dtype=np.int32)
        noise_idx = np.zeros(num_bin, dtype=np.int32)
        if groups is None:
            binary_perm    = np.zeros(num_bin, dtype=np.int32)
            group_starts   = np.zeros(1,       dtype=np.int32)
            group_ends     = np.zeros(1,       dtype=np.int32)
            group_m_lo     = np.zeros(1,       dtype=np.int32)
            group_m_hi     = np.zeros(1,       dtype=np.int32)
            pair_m_lo_b    = np.zeros(num_bin, dtype=np.int32)
            pair_m_hi_b    = np.zeros(num_bin, dtype=np.int32)
            n_groups       = 0
        else:
            binary_perm    = np.asarray(groups["binary_perm"],  dtype=np.int32)
            group_starts   = np.asarray(groups["group_starts"], dtype=np.int32)
            group_ends     = np.asarray(groups["group_ends"],   dtype=np.int32)
            group_m_lo     = np.asarray(groups["group_m_lo"],   dtype=np.int32)
            group_m_hi     = np.asarray(groups["group_m_hi"],   dtype=np.int32)
            pair_m_lo_b    = np.asarray(groups["pair_m_lo_b"],  dtype=np.int32)
            pair_m_hi_b    = np.asarray(groups["pair_m_hi_b"],  dtype=np.int32)
            n_groups       = int(groups["n_groups"])
        # Binding signature (recent): same wdm_settings + tdi_type
        # insertions as get_ll. Nf / Nt dropped from the int args.
        getattr(self._cpp_group, f"{self._CPP_METHOD_PREFIX}_swap_ll")(
            d_h_add, d_h_rem, aa, rr, ar,
            self._cpp_orbits, self._cpp_tdi_config,
            self._cpp_wdm_settings,
            params_add_flat, params_rem_flat,
            data_idx, noise_idx,
            self._cpp_chunk_t_starts,
            self._cpp_chunk_keep_lo, self._cpp_chunk_keep_hi,
            self._cpp_chunk_n_global_offset,
            self._cpp_wdm_window,
            np.asarray(data_d).reshape(-1),
            np.asarray(invC).reshape(-1),
            self.n_chunks, int(num_bin), int(self._CPP_NPARAMS),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T_full), float(self.t_ref_full),
            int(self._tdi_type_int),
            float(self._cpp_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            binary_perm, group_starts, group_ends,
            group_m_lo, group_m_hi, int(n_groups),
            pair_m_lo_b, pair_m_hi_b,
            int(1),  # m_band_half_width
        )

    # ------------------------------------------------------------------
    # Pure-Python fallback implementations
    # ------------------------------------------------------------------

    def _stitched_wdm(self, source_params):
        """Build the stitched WDM template for one binary (host fallback)."""
        Nf, Nt = self.Nf, self.Nt
        layer_dt = Nf * self.dt
        stitched = np.zeros((self.nchannels, Nf, Nt), dtype=float)
        g = self.geometry
        for j, n0 in enumerate(g["starts"]):
            chunk_t_start = self.t_obs_start + float(n0) * layer_dt
            chunk_fd, _ = self._gen.chunk_fd(
                source_params, chunk_t_start, Nf * self.Nt_sub,
                tukey_alpha=self.tukey_alpha, use_tukey=self.use_tukey,
            )
            chunk_wdm = FDSignal(chunk_fd, self._chunk_fd_set).transform(
                self._chunk_wdm_set
            )
            w_chunk = np.asarray(chunk_wdm.arr)
            klo, khi = int(g["keep_lo"][j]), int(g["keep_hi"][j])
            n_lo = int(g["n_global_lo"][j])
            n_hi = n_lo + (khi - klo)
            stitched[:, :, n_lo:n_hi] = w_chunk[:, :, klo:khi]
        return stitched

    def fill_global(self, template_out, params_list, factors=None,
                    grid_dim=0):
        """Accumulate per-binary stitched WDM templates into ``template_out``.

        Args:
            template_out: ``(nchannels, Nf, Nt)`` float array, the
                accumulator (caller pre-zeros).
            params_list: iterable of length-nparams source-parameter
                vectors (one per binary).
            factors: per-binary multiplicative factors (default = +1).
            grid_dim: CUDA launch grid size (use 0 to default to
                ``n_chunks``). Use :func:`chunked_het_grid_dim` to pick
                an optimal value for A100 / H100.
        """
        num_bin = len(params_list)
        if factors is None:
            factors = np.ones(num_bin, dtype=float)
        factors = np.asarray(factors, dtype=float).copy()

        if self._uses_cpp_kernel:
            self._ensure_cpp_setup()
            params_flat, n_check = self._flatten_params(params_list)
            assert n_check == num_bin
            self._call_cpp_fill_global(template_out, params_flat, factors,
                                       num_bin, grid_dim)
            return template_out

        for src, f in zip(params_list, factors):
            template_out += float(f) * self._stitched_wdm(src)
        return template_out

    def get_ll(self, data_d, invC, params_list, grid_dim=0,
                use_layer_groups=True, margin_layers=0,
                group_band_layers=5, phase_maximize=True, return_cupy: bool = False):
        """Per-binary log-likelihood via chunked heterodyne.

        Computes ``-0.5 * (self.d_d + h_h - 2 d_h)`` after running the
        chunked-het kernel (or its Python fallback). With ``self.d_d
        = 0`` (the default), the return is the source-only
        log-likelihood piece; the caller is responsible for adding
        the global ``-0.5 <d|d>`` constant if they want the full
        ``log p(d|h)``. The raw per-source inner products are stashed
        on ``self.d_h`` / ``self.h_h`` for later access.

        ``phase_maximize=True`` (default) marginalises analytically
        over the source phase by replacing ``d_h`` with ``|d_h|``
        before forming the likelihood; the pre-marg complex/signed
        ``d_h`` is preserved on ``self.non_marg_d_h``.

        The default ``use_layer_groups=True`` restricts both template
        construction and the inner-product accumulator to a narrow
        ``group_band_layers``-wide m-band around each source's carrier
        layer. This is the canonical **narrow-band GB inner product**
        validated against mm5 / mm2 (median mm5 ~1e-9; see
        ``gb_chunked_prior_draws.py`` and the sprint root CLAUDE.md).
        Setting ``use_layer_groups=False`` enables the full-Nf path,
        which picks up spectral-tail contributions outside the GB band
        and is *not* the right physical model for narrow-band GBs --
        edge effects create large divergences from mm5.

        Args:
            data_d: ``(nchannels, Nf, Nt)`` WDM data.
            invC: ``(nchannels, Nf, Nt)`` inverse-PSD weighting on the
                same grid.
            params_list: iterable of length-nparams source-parameter vectors.
            grid_dim: CUDA launch grid size (use 0 for n_chunks).
            use_layer_groups: if True (default), sort binaries by
                carrier WDM layer and iterate only the per-group
                m-band. ~Nf/group_band_layers reduction in data_d /
                invC traffic.
            margin_layers: extend each group's iterated m-band by this
                many layers on each side (Doppler / wavelet-tail margin).
            group_band_layers: max m-layer span per group. Default 5
                matches the mm5 narrow band ``[f0-3*df, f0+2*df]``.
            phase_maximize: if True (default), analytically marginalise
                the source phase by taking ``|d_h|`` before forming
                the likelihood.
            return_cupy: if True, leave the return on the GPU (cupy
                array); default False = move to host (numpy).

        Returns:
            ``like_out`` -- length ``len(params_list)`` array of the
            per-source log-likelihood ``-0.5 * (d_d + h_h - 2 d_h)``.
        """
        num_bin = len(params_list)

        if self._uses_cpp_kernel:
            self._ensure_cpp_setup()
            d_h = np.zeros(num_bin, dtype=float)
            h_h = np.zeros(num_bin, dtype=float)
            params_flat, _ = self._flatten_params(params_list)
            groups = None
            if use_layer_groups:
                groups = compute_layer_groups(
                    np.asarray(params_list).reshape(num_bin, self._CPP_NPARAMS),
                    layer_df=self.layer_df,
                    f0_param_index=self._CPP_F0_PARAM_INDEX,
                    group_band_layers=int(group_band_layers),
                    margin_layers=int(margin_layers),
                )
            self._call_cpp_get_ll(d_h, h_h,
                                  np.asarray(data_d, dtype=float),
                                  np.asarray(invC, dtype=float),
                                  params_flat, num_bin, grid_dim,
                                  groups=groups)

        else:
            # Pure-Python fallback: build the stitched template per binary
            # and form the inner products directly. Matches the C++ kernel's
            # accumulator loop (sum over keep regions only).
            d_h = np.zeros(num_bin, dtype=float)
            h_h = np.zeros(num_bin, dtype=float)
            data_d = np.asarray(data_d)
            invC   = np.asarray(invC)
            for i, p in enumerate(params_list):
                w = self._stitched_wdm(p)
                d_h[i] = float(np.sum(data_d * w * invC))
                h_h[i] = float(np.sum(w * w * invC))
        
        if phase_maximize:
            self.non_marg_d_h = d_h.copy()
            try:
                self.non_marg_d_h = self.non_marg_d_h.get()
            except AttributeError:
                pass

            d_h = self.xp.abs(d_h)

        # store these likelihood terms for later if needed
        self.h_h = h_h
        self.d_h = d_h

        # compute Likelihood
        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2 * d_h).real

        if return_cupy:
            return like_out

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out


    def swap_ll(self, data_d, invC, params_add_list, params_remove_list,
                grid_dim=0,
                use_layer_groups=True, margin_layers=0,
                group_band_layers=5):
        """5-way swap-ll accumulator.

        Returns ``(d_h_add, d_h_remove, add_add, remove_remove,
        add_remove)`` -- each length ``num_bin``.

        Default ``use_layer_groups=True`` matches the narrow-band GB
        inner product validated against mm5 / mm2; see
        :meth:`get_ll` for the rationale.

        Layer-grouping (``use_layer_groups=True``): the (add, remove) pair
        index ``bin_i`` is grouped by the carrier WDM layer derived from
        ``params_add_list[bin_i]`` -- the add parameters drive the group
        assignment because the data has been built around the inj/add
        carriers. The remove template uses the same (m_lo, m_hi) band as
        the add for that pair; if their f0s differ by more than the
        margin the remove contribution is truncated (caller's
        responsibility to pick adequate ``margin_layers``).
        """
        num_bin = len(params_add_list)
        d_h_add = np.zeros(num_bin); d_h_rem = np.zeros(num_bin)
        aa = np.zeros(num_bin); rr = np.zeros(num_bin); ar = np.zeros(num_bin)

        if self._uses_cpp_kernel:
            self._ensure_cpp_setup()
            pa, _ = self._flatten_params(params_add_list)
            pr, _ = self._flatten_params(params_remove_list)
            groups = None
            if use_layer_groups:
                groups = compute_swap_layer_groups(
                    np.asarray(params_add_list).reshape(num_bin, self._CPP_NPARAMS),
                    np.asarray(params_remove_list).reshape(num_bin, self._CPP_NPARAMS),
                    layer_df=self.layer_df,
                    f0_param_index=self._CPP_F0_PARAM_INDEX,
                    group_band_layers=int(group_band_layers),
                    margin_layers=int(margin_layers),
                )
            self._call_cpp_swap_ll(
                (d_h_add, d_h_rem, aa, rr, ar),
                np.asarray(data_d, dtype=float),
                np.asarray(invC, dtype=float),
                pa, pr, num_bin, grid_dim,
                groups=groups,
            )
            return d_h_add, d_h_rem, aa, rr, ar

        for i in range(num_bin):
            w_add = self._stitched_wdm(params_add_list[i])
            w_rem = self._stitched_wdm(params_remove_list[i])
            d_h_add[i] = float(np.sum(data_d * w_add * invC))
            d_h_rem[i] = float(np.sum(data_d * w_rem * invC))
            aa[i]      = float(np.sum(w_add  * w_add  * invC))
            rr[i]      = float(np.sum(w_rem  * w_rem  * invC))
            ar[i]      = float(np.sum(w_add  * w_rem  * invC))
        return d_h_add, d_h_rem, aa, rr, ar

    # ------------------------------------------------------------------
    # Likelihood gradients via central finite differences (lightest wrap).
    #
    # Both methods loop k = 0..nparams-1 and call the existing C++
    # get_ll / swap_ll wrappers at theta +/- eps[k] per binary, then form
    # the central FD. No new C++ code; the gradient is a thin Python
    # layer on top of the kernels we already validated.
    #
    # `param_eps` must be a length-nparams array. eps[k] <= 0 freezes
    # parameter k (its gradient stays 0). Recommended default for GB:
    #   eps = [A*1e-4, 2e-14, 1e-22, 1e-26, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    # i.e. ~1e-4 fractional for log-quantities, 2e-14 Hz for f0, small
    # absolute steps for fdot/fddot, ~1e-3 rad for angles.
    # ------------------------------------------------------------------
    def get_ll_grad(self, data_d, invC, params_list, param_eps,
                     grid_dim=0):
        """Central-FD gradient of L = <d|h> - 0.5 <h|h> w.r.t. params.

        Args:
            data_d, invC: same as :meth:`get_ll`.
            params_list: iterable of length-nparams source-parameter
                vectors.
            param_eps: length-nparams array; eps[k] is the FD step for
                parameter k (eps[k] <= 0 freezes that parameter).
            grid_dim: CUDA launch grid size.

        Returns:
            grad: ``(num_bin, nparams)`` array, ``grad[b, k] = dL_b/dtheta_k``.
        """
        num_bin = len(params_list)
        nparams = int(self._CPP_NPARAMS)
        param_eps = np.asarray(param_eps, dtype=float).reshape(-1)
        assert param_eps.shape[0] == nparams, (
            f"param_eps length {param_eps.shape[0]} != nparams {nparams}"
        )
        params_arr = np.asarray(params_list, dtype=float).reshape(num_bin, nparams)

        grad = np.zeros((num_bin, nparams), dtype=float)
        for k in range(nparams):
            eps_k = float(param_eps[k])
            if eps_k <= 0.0:
                continue
            # +eps
            p_plus = params_arr.copy()
            p_plus[:, k] += eps_k
            dh_p, hh_p = self.get_ll(data_d, invC, [p for p in p_plus],
                                       grid_dim=grid_dim)
            # -eps
            p_minus = params_arr.copy()
            p_minus[:, k] -= eps_k
            dh_m, hh_m = self.get_ll(data_d, invC, [p for p in p_minus],
                                       grid_dim=grid_dim)
            inv2eps = 1.0 / (2.0 * eps_k)
            grad[:, k] = inv2eps * (
                (np.asarray(dh_p) - np.asarray(dh_m))
                - 0.5 * (np.asarray(hh_p) - np.asarray(hh_m))
            )
        return grad

    def swap_ll_grad(self, data_d, invC,
                      params_add_list, params_remove_list,
                      param_eps_add, grid_dim=0):
        """Central-FD gradients of all 5 swap_ll terms w.r.t. theta_add,
        plus the combined likelihood gradient for "swap residual" model.

        Likelihood model: the **remove** template is fully extracted from
        the data (not in residual, not in template), and we evaluate L
        for the add template against that clean data:

            L_add = <d - h_rem | h_add> - 0.5 <h_add|h_add>
                  = dh_add - ar - 0.5 * aa

        So dL_add/dtheta_add = d(dh_add)/dtheta_add - d(ar)/dtheta_add
                                                    - 0.5 * d(aa)/dtheta_add.

        Args:
            data_d, invC: same as :meth:`swap_ll`.
            params_add_list, params_remove_list: per-binary param vectors.
            param_eps_add: length-nparams FD step for the add template
                (eps[k] <= 0 freezes parameter k).
            grid_dim: CUDA launch grid size.

        Returns:
            dict with keys:
                ``grad_dh_add``  : (num_bin, nparams) d(<d|h_add>)/dtheta_add
                ``grad_dh_rem``  : (num_bin, nparams) d(<d|h_rem>)/dtheta_add
                                    (mathematically 0; FD noise floor)
                ``grad_aa``      : (num_bin, nparams) d(<h_add|h_add>)/dtheta_add
                ``grad_rr``      : (num_bin, nparams) d(<h_rem|h_rem>)/dtheta_add
                                    (mathematically 0; FD noise floor)
                ``grad_ar``      : (num_bin, nparams) d(<h_add|h_rem>)/dtheta_add
                ``grad_L_add``   : (num_bin, nparams) combined
                                    = grad_dh_add - grad_ar - 0.5 * grad_aa
        """
        num_bin = len(params_add_list)
        nparams = int(self._CPP_NPARAMS)
        param_eps_add = np.asarray(param_eps_add, dtype=float).reshape(-1)
        assert param_eps_add.shape[0] == nparams, (
            f"param_eps_add length {param_eps_add.shape[0]} != nparams {nparams}"
        )
        p_add  = np.asarray(params_add_list,    dtype=float).reshape(num_bin, nparams)
        p_rem  = np.asarray(params_remove_list, dtype=float).reshape(num_bin, nparams)
        rem_as_list = [p for p in p_rem]

        g_dh_add = np.zeros((num_bin, nparams))
        g_dh_rem = np.zeros((num_bin, nparams))
        g_aa     = np.zeros((num_bin, nparams))
        g_rr     = np.zeros((num_bin, nparams))
        g_ar     = np.zeros((num_bin, nparams))

        for k in range(nparams):
            eps_k = float(param_eps_add[k])
            if eps_k <= 0.0:
                continue
            # +eps on theta_add[k]
            p_add_p = p_add.copy()
            p_add_p[:, k] += eps_k
            dha_p, dhr_p, aa_p, rr_p, ar_p = self.swap_ll(
                data_d, invC, [p for p in p_add_p], rem_as_list,
                grid_dim=grid_dim,
            )
            # -eps on theta_add[k]
            p_add_m = p_add.copy()
            p_add_m[:, k] -= eps_k
            dha_m, dhr_m, aa_m, rr_m, ar_m = self.swap_ll(
                data_d, invC, [p for p in p_add_m], rem_as_list,
                grid_dim=grid_dim,
            )
            inv2eps = 1.0 / (2.0 * eps_k)
            g_dh_add[:, k] = inv2eps * (np.asarray(dha_p) - np.asarray(dha_m))
            g_dh_rem[:, k] = inv2eps * (np.asarray(dhr_p) - np.asarray(dhr_m))
            g_aa    [:, k] = inv2eps * (np.asarray(aa_p)  - np.asarray(aa_m))
            g_rr    [:, k] = inv2eps * (np.asarray(rr_p)  - np.asarray(rr_m))
            g_ar    [:, k] = inv2eps * (np.asarray(ar_p)  - np.asarray(ar_m))

        grad_L_add = g_dh_add - g_ar - 0.5 * g_aa
        return dict(
            grad_dh_add=g_dh_add, grad_dh_rem=g_dh_rem,
            grad_aa=g_aa, grad_rr=g_rr, grad_ar=g_ar,
            grad_L_add=grad_L_add,
        )

    # ------------------------------------------------------------------
    # JAX backend (autograd-friendly likelihood / gradient / Hessian)
    #
    # The C++ get_ll / get_ll_grad above remain the canonical path; the
    # JAX methods below mirror them by calling the same chunked-het
    # primitives in fastlisaresponse.jax.wdm.heterodyne_kernels. They
    # share the C++-built chunk geometry / WDM window arrays via
    # _ensure_cpp_setup so the two backends are guaranteed to consume
    # the same geometry (any drift would surface as a JAX-vs-C++
    # reldiff in validate_jax_vs_cpp_chunked_het.py).
    # ------------------------------------------------------------------

    # Subclass-overridable JAX source factory: () -> JaxAmpPhaseSource
    # built from this instance. GB uses JaxUCBSource; SOBBH would point
    # to its own JAX source class via this hook.
    _JAX_SOURCE_CLASS_PATH = ("fastlisaresponse.jax.sources.ucb", "JaxUCBSource")

    def _ensure_jax_setup(self):
        if getattr(self, "_jax_setup_done", False):
            return
        self._ensure_cpp_setup()  # need cpp-side chunk arrays
        # Resolve the JAX backend's ``xp`` module through fastlisaresponse's
        # backend manager rather than importing ``jax.numpy`` directly --
        # keeps the chunked-het JAX path consistent with the rest of the
        # sprint's backend-dispatch convention (see CLAUDE.md). Cached as
        # ``self.jax_xp`` so callers (e.g. gb_chunked_test_script.py) can
        # build JAX-side wrappers without a top-level ``import jax.numpy``.
        try:
            from fastlisaresponse import get_backend as _get_be
            self._jax_backend = _get_be("fastlisaresponse_jax")
            jnp = self._jax_backend.xp
        except Exception as e:
            # Fallback to direct jax.numpy import (still functional --
            # the backend lookup only fails if fastlisaresponse_jax isn't
            # registered, which means jax isn't installed either).
            try:
                import jax.numpy as jnp
                self._jax_backend = None
            except ImportError as e2:
                raise RuntimeError(
                    "JAX is not installed; install jax to use the *_jax methods."
                ) from e2
        self._jax_xp = jnp
        from importlib import import_module
        from fastlisaresponse.jax.orbits import OrbitsWrapJAX
        from fastlisaresponse.jax.tdi_config import TDIConfigWrapJAX
        src_mod = import_module(self._JAX_SOURCE_CLASS_PATH[0])
        SrcCls   = getattr(src_mod, self._JAX_SOURCE_CLASS_PATH[1])
        self._jax_orbits = OrbitsWrapJAX(*self._orbits_py.pycppdetector_args)
        self._jax_tdi    = TDIConfigWrapJAX(*self._tdi_cfg_py.pytdiconfig_args)
        self._jax_source = SrcCls(t_ref=self.t_ref_full)
        self._jax_chunk_t_starts = jnp.asarray(self._cpp_chunk_t_starts)
        self._jax_chunk_keep_lo  = jnp.asarray(self._cpp_chunk_keep_lo)
        self._jax_chunk_keep_hi  = jnp.asarray(self._cpp_chunk_keep_hi)
        self._jax_chunk_n_lo     = jnp.asarray(self._cpp_chunk_n_global_offset)
        self._jax_wdm_window     = jnp.asarray(self._cpp_wdm_window)
        self._jax_setup_done = True

    @property
    def jax_xp(self):
        """JAX numpy module (``jax.numpy``), resolved via the
        ``fastlisaresponse_jax`` backend. Lazy: triggers ``_ensure_jax_setup``
        on first access.
        """
        self._ensure_jax_setup()
        return self._jax_xp

    def _jax_native_group(self):
        """Return the JAX backend's ``GBComputationGroupWrap()`` --
        the per-source-class chunked-het kernel wrap that exposes
        ``gb_wdm_het_get_ll`` / ``gb_wdm_het_swap_ll`` /
        ``gb_wdm_het_fill_global`` / ``gb_wdm_het_get_ll_grad`` /
        ``gb_wdm_het_hessian`` as methods. Lazy-resolves through
        ``self._jax_backend`` (the ``fastlisaresponse_jax`` backend
        object) -- no direct imports of the standalone JAX functions.
        Cached as ``self._jax_group``.
        """
        if getattr(self, "_jax_group", None) is not None:
            return self._jax_group
        self._ensure_jax_setup()
        if self._jax_backend is None:
            # The backend lookup failed at setup time; fall back to
            # constructing the wrap directly from the JAX submodule
            # so the methods still work even without the backend
            # registry. This path is exercised only when
            # ``fastlisaresponse_jax`` is unavailable in the registry
            # but ``jax`` is importable.
            from fastlisaresponse.jax.wdm import GBComputationGroupWrapJAX
            self._jax_group = GBComputationGroupWrapJAX()
        else:
            wrap_cls = getattr(self._jax_backend, self._CPP_WRAP_ATTR)
            self._jax_group = wrap_cls()
        return self._jax_group

    def _jax_kernel_args(self, data_d, invC):
        """Return the (positional, keyword) tuple consumed by every
        gb_wdm_het_*_jax kernel. ``data_d`` / ``invC`` are converted to
        jnp arrays here so callers don't have to.
        """
        import jax.numpy as jnp
        self._ensure_jax_setup()
        pos = (
            jnp.asarray(data_d), jnp.asarray(invC),
            self._jax_chunk_t_starts,
            self._jax_chunk_keep_lo, self._jax_chunk_keep_hi,
            self._jax_chunk_n_lo,
            self._jax_source, self._jax_orbits, self._jax_tdi,
            self._jax_wdm_window,
        )
        kw = dict(
            Nf=self.Nf, Nt=self.Nt, Nt_sub=self.Nt_sub, N_sparse=self.N_sparse,
            dt=self.dt, T_chunk=self.T_chunk,
            tukey_alpha=self._cpp_tukey_alpha,
        )
        return pos, kw

    def _resolve_jax_chunk(self, chunk):
        """Pick the effective leaf-axis chunk size for a JAX call.

        Priority: explicit ``chunk`` arg > instance default
        ``self.jax_chunk`` > env ``GBHET_JAX_CHUNK`` /
        ``JAX_GRAD_CHUNK`` (legacy). Returns ``0`` if no chunking is
        requested (single big batched call).

        Rationale: the C++ kernel's per-source loop limits per-call
        throughput on CPU regardless of batch size, so we only
        plumbed batch chunking through the JAX paths -- where the
        knob is mostly a memory escape hatch (jax.grad / jax.hessian
        autograd tapes blow up with the binary axis).
        """
        import os
        if chunk is not None:
            return int(chunk)
        # instance-level default, set at construction
        inst = getattr(self, "jax_chunk", 0)
        if inst:
            return int(inst)
        # env fallbacks (GBHET_JAX_CHUNK preferred; JAX_GRAD_CHUNK kept
        # for back-compat with earlier scripts)
        env = os.environ.get("GBHET_JAX_CHUNK") or os.environ.get("JAX_GRAD_CHUNK")
        return int(env) if env else 0

    def _chunked_jax_apply(self, fn, params_batch, chunk):
        """Apply ``fn(p_chunk)`` over ``params_batch`` split into
        leaf-axis chunks of size ``chunk`` (None / 0 / negative
        disables chunking). ``fn`` must return either a single ndarray
        ``(K, ...)`` or a tuple/list of such ndarrays. Outputs are
        concatenated along axis 0.

        Recompiles JAX once per distinct chunk shape -- callers should
        pick a chunk size that divides ``num_bin`` exactly (or accept a
        recompile on the partial tail chunk).
        """
        import numpy as _np
        num_bin = params_batch.shape[0]
        if not chunk or chunk <= 0 or chunk >= num_bin:
            return fn(params_batch)
        outs = []
        for start in range(0, num_bin, chunk):
            end = min(start + chunk, num_bin)
            out = fn(params_batch[start:end])
            outs.append(out)
        if isinstance(outs[0], tuple) or isinstance(outs[0], list):
            return tuple(
                _np.concatenate([_np.asarray(o[k]) for o in outs], axis=0)
                for k in range(len(outs[0]))
            )
        return _np.concatenate([_np.asarray(o) for o in outs], axis=0)

    def get_ll_jax(self, params_batch, data_d, invC, chunk=None):
        """JAX-backed ``<d|h>``, ``<h|h>`` per binary.

        Returns numpy arrays of length ``num_bin`` (so the call shape
        matches :meth:`get_ll`).

        ``chunk`` (or env ``GBHET_JAX_CHUNK`` / ``JAX_GRAD_CHUNK``):
        split the leaf axis into groups of this size before each JAX
        call -- mirrors the same knob on :meth:`get_ll_grad_jax` and
        :meth:`hessian`. Useful when the kernel's per-call memory
        scales linearly with ``num_bin`` and the full batch doesn't
        fit. ``None`` -> resolved per :meth:`_resolve_jax_chunk`.
        """
        import jax.numpy as jnp
        # All JAX chunked-het kernel functions are accessed through the
        # JAX backend's ``GBComputationGroupWrap`` (gb_wdm_het_*
        # methods registered on :class:`GBComputationGroupWrapJAX`).
        jax_group = self._jax_native_group()
        pos, kw = self._jax_kernel_args(data_d, invC)
        p_full = jnp.asarray(np.asarray(params_batch).reshape(-1, self._CPP_NPARAMS))
        chunk_eff = self._resolve_jax_chunk(chunk)
        def _fn(p_sub):
            d_h, h_h = jax_group.gb_wdm_het_get_ll(p_sub, *pos, **kw)
            return (np.asarray(d_h), np.asarray(h_h))
        return self._chunked_jax_apply(_fn, p_full, chunk_eff)

    def get_ll_grad_jax(self, params_batch, data_d, invC, chunk=None):
        """JAX-autograd gradient of L = sum_i (<d|h_i> - 0.5 <h_i|h_i>)
        w.r.t. ``params_batch``. Returns ``(num_bin, nparams)`` numpy.

        ``chunk`` (or env ``GBHET_JAX_CHUNK`` / ``JAX_GRAD_CHUNK``):
        split the leaf axis into groups of this size before calling
        jax.grad (memory escape hatch). ``None`` -> resolved per
        :meth:`_resolve_jax_chunk`.
        """
        import jax.numpy as jnp
        jax_group = self._jax_native_group()
        pos, kw = self._jax_kernel_args(data_d, invC)
        p_full = jnp.asarray(np.asarray(params_batch).reshape(-1, self._CPP_NPARAMS))
        chunk_eff = self._resolve_jax_chunk(chunk)
        def _fn(p_sub):
            g = jax_group.gb_wdm_het_get_ll_grad(p_sub, *pos, **kw)
            return np.asarray(g)
        return self._chunked_jax_apply(_fn, p_full, chunk_eff)

    def hessian(self, params_batch, data_d, invC,
                chunk=None, psd_fix=False, psd_floor_rel=1e-30):
        """Per-binary Hessian of L = <d|h> - 0.5 <h|h> w.r.t. params.

        Returns ``(num_bin, nparams, nparams)``.

        JAX-autograd only -- requires a JAX-backend instance. The
        instance's backend is fixed at construction (no ``backend=``
        kwarg); construct ``GBWDMHeterodyne(force_backend="jax", ...)``
        to use this method, or accept the NotImplementedError on a
        C++ backend until the Stage-2 native Hessian kernel lands.

        Args:
            params_batch: ``(num_bin, nparams)`` ndarray.
            data_d, invC: same shape contract as :meth:`get_ll`.
            chunk: leaf-axis chunking for memory; falls back to the env
                ``JAX_GRAD_CHUNK`` (same knob as get_ll_grad_jax).
            psd_fix: if True, apply eigendecompose + ``|lambda|`` clip
                (floor ``psd_floor_rel * lambda_max``) to ``-H`` so the
                returned matrix is positive-definite. This is the
                M = -H = inverse-covariance form -- ready to feed into
                ``NUTSSampler(metric=...)`` as a mass matrix.
        """
        if self._uses_cpp_kernel:
            raise NotImplementedError(
                "hessian requires a JAX backend "
                "(GBWDMHeterodyne(force_backend='jax')); the C++/CUDA "
                "Hessian kernel is the Stage-2 follow-up."
            )
        import os
        import jax.numpy as jnp
        jax_group = self._jax_native_group()
        pos, kw = self._jax_kernel_args(data_d, invC)
        p_full = jnp.asarray(np.asarray(params_batch).reshape(-1, self._CPP_NPARAMS))
        if chunk is None:
            chunk = int(os.environ.get("JAX_GRAD_CHUNK", "0"))
        def _fn(p_sub):
            H = jax_group.gb_wdm_het_hessian(p_sub, *pos, **kw)
            return np.asarray(H)
        H = self._chunked_jax_apply(_fn, p_full, chunk)
        if psd_fix:
            H = psd_fix_eigabs(-H, floor_rel=psd_floor_rel)
        return H

    # ------------------------------------------------------------------
    # ACA-aware wrappers (mirror GBWDMComputations.*_wdm signatures)
    #
    # Drop-in surface so this class can replace the legacy
    # GBWDMComputations in the global-fit Buffer / WDMBandLikelihoodEngine
    # path. Each ``*_wdm`` accepts ``source`` as one of:
    #
    #   - tuple ``(data_d, invC)`` of global-grid ``(nchannels, Nf, Nt)``
    #     ndarrays. Used directly.
    #   - :class:`AnalysisContainer` (single source). ``arr`` is taken
    #     verbatim; PSD inverted with the sanitiser below.
    #   - :class:`AnalysisContainerArray` (multi-band buffer, legacy
    #     ``wdm_holder``). The per-band slabs are zero-padded into the
    #     global ``(nchannels, Nf, Nt)`` grid via ``ind_min_f`` /
    #     ``ind_min_t`` offsets read off the basis settings; one global
    #     buffer is built per unique ``data_index`` and reused for all
    #     sources binding to that band.
    #
    # Inner products / gradients / Hessians are then routed through the
    # existing array-based methods (``get_ll`` / ``get_ll_grad_jax`` /
    # ``hessian``) per unique data_index group, and outputs are
    # reassembled into the original source ordering.
    # ------------------------------------------------------------------

    # Default per-parameter FD step (used by get_ll_grad_wdm when caller
    # doesn't pass param_eps). Sized for GB params at the natural
    # chunked-het scales -- NOT the legacy ``GBWDMComputations`` table,
    # which assumed the lookup-table kernel's normalisation.
    #
    # Heuristic: pick eps small enough that the kernel-side FD truncation
    # error is small, but large enough to stay well above the FP noise
    # floor of the get_ll inner-product accumulator (~1e-12 of |L|).
    #
    #   amp     ~ 1e-22  -> 1e-26      (amp * 1e-4; overridden in grad
    #                                    by caller's amp-scaling if any)
    #   f0      ~ 5e-3 Hz -> 1e-8 Hz   (fractional ~ 2e-6; big enough
    #                                    that the carrier truly shifts a
    #                                    measurable amount, small enough
    #                                    that the heterodyne stays valid)
    #   fdot    ~ 1e-17 - 1e-13 Hz/s -> 1e-14
    #   fddot   frozen (0.0)
    #   angles  ~ rad   -> 1e-3
    _DEFAULT_PARAM_EPS = (
        1.0e-26,   # amp     (absolute; ~ amp * 1e-4 for amp ~ 1e-22)
        1.0e-8,    # f0      (Hz)
        1.0e-14,   # fdot    (Hz/s)
        0.0,       # fddot   (frozen)
        1.0e-3,    # phi0
        1.0e-3,    # inc
        1.0e-3,    # psi
        1.0e-3,    # lam
        1.0e-3,    # beta
    )

    @staticmethod
    def _invert_psd(psd):
        """Sanitised 1/psd: non-finite / nonpositive entries -> 0
        (so the corresponding inner-product cell drops out cleanly).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(
                np.isfinite(psd) & (psd > 0), psd, np.inf,
            )
        return np.where(np.isfinite(inv), inv, 0.0)

    @staticmethod
    def _maybe_aca_basis(source):
        """Return basis_settings (with ind_min_f / ind_max_f / Nf / Nt
        attributes) on ``source`` if it's an AC/ACA, else None.
        """
        bs = getattr(source, "_basis_settings", None)
        if bs is not None:
            return bs
        dr = getattr(source, "data_res_arr", None)
        if dr is not None:
            return getattr(dr, "settings", None)
        return None

    def _zero_pad_band_to_global(self, band_arr, basis_settings, is_psd):
        """Zero-pad a per-band slab into a full ``(nchannels, Nf, Nt)``
        ndarray sized to the chunked-het instance.

        Handles two band-slab shapes seen in the wild:

          - ``(nch, Nf_active, Nt_active)`` -- WDM domain proper. The
            slab is placed at ``[ind_min_f : ind_max_f+1, slice_t]`` of
            a fresh global grid; PSD inversion is applied here so the
            returned ``invC`` slab is ``1/PSD``.
          - ``(nch, nch, Nf_active, Nt_active)`` -- cross-channel PSD.
            Diagonalised into per-channel ``(nch, Nf_active, Nt_active)``
            before padding.
        """
        nch = self.nchannels
        arr = np.asarray(band_arr)
        if arr.ndim == 4:
            arr = np.stack([arr[c, c] for c in range(nch)], axis=0)
        assert arr.ndim == 3 and arr.shape[0] == nch, (
            f"unexpected band slab shape {arr.shape}; expected "
            f"({nch}, Nf_band, Nt_band) or ({nch}, {nch}, Nf_band, Nt_band)"
        )
        ilo = int(getattr(basis_settings, "ind_min_f", 0))
        ihi = int(getattr(basis_settings, "ind_max_f", self.Nf - 1))
        if hasattr(basis_settings, "active_slice_t") and getattr(
            basis_settings, "Nt_active", self.Nt
        ) != self.Nt:
            tslice = basis_settings.active_slice_t
        else:
            tslice = slice(None)
        out = np.zeros((nch, self.Nf, self.Nt), dtype=float)
        out[:, ilo: ihi + 1, tslice] = self._invert_psd(arr) if is_psd else arr
        return out

    def _resolve_source_to_arrays(self, source, data_index=None, noise_index=None,
                                   num_bin=None):
        """Resolve ``source`` to ``{data_idx: (data_d_global, invC_global)}``
        keyed by the unique ``data_index`` values present in this call.

        For the array-tuple and AC cases there is exactly one entry (key
        ``0``); for ACAs there is one entry per unique ``data_index``.
        """
        # 1) Raw tuple (data_d, invC) -- pass through.
        if isinstance(source, (tuple, list)) and len(source) == 2 and not hasattr(source, "linear_data_arr"):
            d, c = source
            return {0: (np.asarray(d, dtype=float), np.asarray(c, dtype=float))}

        # 2) AnalysisContainer -- single source.
        if hasattr(source, "data_res_arr") and hasattr(source, "sens_mat") and not hasattr(source, "linear_data_arr"):
            bs = self._maybe_aca_basis(source)
            d = np.asarray(source.data_res_arr.arr, dtype=float)
            psd = np.asarray(source.sens_mat.sens_mat, dtype=float)
            if d.shape == (self.nchannels, self.Nf, self.Nt):
                return {0: (d, self._invert_psd(psd if psd.ndim == 3 else np.stack(
                    [psd[c, c] for c in range(self.nchannels)], axis=0
                )))}
            # Active-band slab -> zero-pad.
            assert bs is not None, (
                "AnalysisContainer with non-global data shape requires "
                "basis_settings exposing ind_min_f / ind_max_f"
            )
            d_g = self._zero_pad_band_to_global(d, bs, is_psd=False)
            c_g = self._zero_pad_band_to_global(psd, bs, is_psd=True)
            return {0: (d_g, c_g)}

        # 3) AnalysisContainerArray (legacy ``wdm_holder``).
        if hasattr(source, "linear_data_arr") or hasattr(source, "data_shaped"):
            if data_index is None:
                if num_bin is None:
                    raise ValueError("ACA source requires data_index (or num_bin)")
                data_index = np.zeros(num_bin, dtype=np.int32)
            data_index = np.asarray(data_index)
            if noise_index is None:
                noise_index = data_index
            noise_index = np.asarray(noise_index)
            data_shaped = np.asarray(source.data_shaped[0])
            psd_shaped  = np.asarray(source.psd_shaped[0])
            bs = self._maybe_aca_basis(source)
            unique_data = np.unique(data_index)
            out = {}
            for di in unique_data:
                ni = int(noise_index[np.argmax(data_index == di)])
                d_band = data_shaped[int(di)]
                c_band = psd_shaped[int(ni)]
                if d_band.shape == (self.nchannels, self.Nf, self.Nt):
                    d_g = d_band.astype(float, copy=False)
                    c_g = self._invert_psd(
                        c_band if c_band.ndim == 3 else
                        np.stack([c_band[c, c] for c in range(self.nchannels)], axis=0)
                    )
                else:
                    if bs is None:
                        raise ValueError(
                            "ACA source with per-band slab requires basis_settings "
                            "exposing ind_min_f / ind_max_f"
                        )
                    d_g = self._zero_pad_band_to_global(d_band, bs, is_psd=False)
                    c_g = self._zero_pad_band_to_global(c_band, bs, is_psd=True)
                out[int(di)] = (d_g, c_g)
            return out

        raise TypeError(
            f"unsupported source type {type(source).__name__}; expected "
            "tuple/list (data_d, invC), AnalysisContainer, or "
            "AnalysisContainerArray"
        )

    @staticmethod
    def _maybe_ecl_to_icrs(params, convert_to_ra_dec):
        if not convert_to_ra_dec:
            return params
        from fastlisaresponse.response import ecliptic_to_icrs
        params = np.asarray(params, dtype=float).copy()
        if params.ndim == 1:
            params = params.reshape(1, -1)
        lam = params[:, -2].copy()
        beta = params[:, -1].copy()
        lam, beta = ecliptic_to_icrs(lam, beta)
        params[:, -2] = lam
        params[:, -1] = beta
        return params

    def _dispatch_per_band(self, params, source, data_index, noise_index,
                            convert_to_ra_dec, _fn):
        """Common per-band loop for the ``*_wdm`` methods. ``_fn`` is a
        callable ``(data_d, invC, params_band) -> per_source_output`` --
        either an ndarray of shape ``(K, ...)`` or a tuple of such
        ndarrays. The first axis is reassembled in the original source
        ordering and returned.
        """
        params = np.asarray(params, dtype=float).reshape(-1, self._CPP_NPARAMS)
        num_bin = params.shape[0]
        params_used = self._maybe_ecl_to_icrs(params, convert_to_ra_dec)
        resolved = self._resolve_source_to_arrays(
            source, data_index=data_index, noise_index=noise_index,
            num_bin=num_bin,
        )
        if data_index is None:
            data_index = np.zeros(num_bin, dtype=np.int32)
        data_index = np.asarray(data_index)

        # Collect per-band outputs, then reassemble in original ordering.
        gathered = None
        for di, (d_g, c_g) in resolved.items():
            mask = (data_index == di)
            if not mask.any():
                continue
            params_band = params_used[mask]
            out_band = _fn(d_g, c_g, params_band)
            if isinstance(out_band, (tuple, list)):
                if gathered is None:
                    gathered = tuple(
                        np.zeros((num_bin,) + np.asarray(o).shape[1:], dtype=float)
                        for o in out_band
                    )
                for slot, o in zip(gathered, out_band):
                    slot[mask] = np.asarray(o)
            else:
                o = np.asarray(out_band)
                if gathered is None:
                    gathered = np.zeros((num_bin,) + o.shape[1:], dtype=float)
                gathered[mask] = o
        if gathered is None:
            raise RuntimeError("no sources dispatched (empty data_index?)")
        return gathered

    def get_ll_wdm(self, params, source, *,
                   data_index=None, noise_index=None,
                   convert_to_ra_dec=False, return_inner_products=False,
                   chunk=None):
        """ACA-aware ``<d|h>`` / ``<h|h>``. Drop-in for
        :meth:`GBWDMComputations.get_ll_wdm`.

        Dispatches via ``self.backend`` -- C++ kernel for cpu/cuda
        backends, JAX kernel for the jax backend. No ``backend=`` kwarg
        (per the sprint-wide rule); construct a different instance to
        switch backends. ``chunk`` applies only to the JAX path; the
        C++ path is unchunked.
        """
        if self._uses_cpp_kernel:
            def _fn(d_g, c_g, params_band):
                d_h, h_h = self.get_ll(
                    d_g, c_g,
                    [params_band[i] for i in range(params_band.shape[0])],
                )
                return (np.asarray(d_h), np.asarray(h_h))
        else:
            def _fn(d_g, c_g, params_band):
                d_h, h_h = self.get_ll_jax(params_band, d_g, c_g, chunk=chunk)
                return (np.asarray(d_h), np.asarray(h_h))
        d_h, h_h = self._dispatch_per_band(
            params, source, data_index, noise_index, convert_to_ra_dec, _fn,
        )
        self.d_h_out = d_h
        self.h_h_out = h_h
        like_out = -0.5 * (h_h - 2.0 * d_h)
        if return_inner_products:
            return like_out, d_h, h_h
        return like_out

    def get_ll_grad_wdm(self, params, source, *,
                        param_eps=None,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec=False,
                        chunk=None):
        """ACA-aware gradient of L = <d|h> - 0.5<h|h>.

        Dispatches via ``self.backend``: cpu/cuda backends take the
        central-FD path on :meth:`get_ll_grad` (FD over the C++
        kernel); the jax backend uses :meth:`get_ll_grad_jax`
        (autograd). No ``backend=`` kwarg (per the sprint-wide rule);
        construct a different instance to switch backends.
        Returns ``(num_bin, nparams)`` numpy.
        """
        if self._uses_cpp_kernel:
            eps = (np.asarray(param_eps, dtype=float)
                   if param_eps is not None else
                   np.asarray(self._DEFAULT_PARAM_EPS, dtype=float))
            def _fn(d_g, c_g, params_band):
                return self.get_ll_grad(
                    d_g, c_g, [params_band[i] for i in range(params_band.shape[0])],
                    eps,
                )
        else:
            def _fn(d_g, c_g, params_band):
                return self.get_ll_grad_jax(params_band, d_g, c_g, chunk=chunk)
        return self._dispatch_per_band(
            params, source, data_index, noise_index, convert_to_ra_dec, _fn,
        )

    def hessian_wdm(self, params, source, *,
                    data_index=None, noise_index=None,
                    convert_to_ra_dec=False,
                    chunk=None, psd_fix=False, psd_floor_rel=1e-30):
        """ACA-aware per-binary Hessian of L = <d|h> - 0.5<h|h>.

        Requires a JAX backend (autograd-only for now). Returns
        ``(num_bin, nparams, nparams)`` numpy. When ``psd_fix=True``
        returns ``M = |-H|`` (eigendecompose-then-abs) which is ready
        to be fed to ``NUTSSampler(metric=M)`` as a per-leaf mass
        matrix.
        """
        def _fn(d_g, c_g, params_band):
            return self.hessian(
                params_band, d_g, c_g,
                chunk=chunk, psd_fix=psd_fix, psd_floor_rel=psd_floor_rel,
            )
        return self._dispatch_per_band(
            params, source, data_index, noise_index, convert_to_ra_dec, _fn,
        )

    def get_swap_ll_wdm(self, params_add, params_remove, source, *,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec=False, phase_marginalize=False):
        """ACA-aware swap-LL pieces. Drop-in for
        :meth:`GBWDMComputations.get_swap_ll_wdm`.

        Stashes ``self.d_h_add_out`` / ``self.d_h_remove_out`` /
        ``self.add_add_out`` / ``self.remove_remove_out`` /
        ``self.add_remove_out`` for back-compat, and returns
        ``(like_add, like_remove, d_h_add, d_h_remove, aa, rr, ar)`` in
        the same shape contract as the legacy generator.

        ``phase_marginalize=True`` is currently not supported by the
        chunked-het kernel; we surface this loudly rather than silently
        returning the un-maximised result.
        """
        if phase_marginalize:
            raise NotImplementedError(
                "GBWDMHeterodyne.get_swap_ll_wdm does not yet support "
                "phase_marginalize=True. Use the gradient path "
                "(get_ll_grad_wdm / hessian_wdm) which does not "
                "phase-maximise, or fall back to the FD legacy "
                "generator for RJ moves."
            )
        params_add    = np.asarray(params_add,    dtype=float).reshape(-1, self._CPP_NPARAMS)
        params_remove = np.asarray(params_remove, dtype=float).reshape(-1, self._CPP_NPARAMS)
        assert params_add.shape == params_remove.shape
        num_bin = params_add.shape[0]
        params_add_used    = self._maybe_ecl_to_icrs(params_add,    convert_to_ra_dec)
        params_remove_used = self._maybe_ecl_to_icrs(params_remove, convert_to_ra_dec)

        resolved = self._resolve_source_to_arrays(
            source, data_index=data_index, noise_index=noise_index,
            num_bin=num_bin,
        )
        if data_index is None:
            data_index = np.zeros(num_bin, dtype=np.int32)
        data_index = np.asarray(data_index)

        d_h_add_out    = np.zeros(num_bin)
        d_h_remove_out = np.zeros(num_bin)
        aa_out         = np.zeros(num_bin)
        rr_out         = np.zeros(num_bin)
        ar_out         = np.zeros(num_bin)
        for di, (d_g, c_g) in resolved.items():
            mask = (data_index == di)
            if not mask.any():
                continue
            pa = [params_add_used[i]    for i in np.flatnonzero(mask)]
            pr = [params_remove_used[i] for i in np.flatnonzero(mask)]
            d_h_a, d_h_r, aa, rr, ar = self.swap_ll(d_g, c_g, pa, pr)
            d_h_add_out[mask]    = d_h_a
            d_h_remove_out[mask] = d_h_r
            aa_out[mask]         = aa
            rr_out[mask]         = rr
            ar_out[mask]         = ar

        self.d_h_add_out      = d_h_add_out
        self.d_h_remove_out   = d_h_remove_out
        self.add_add_out      = aa_out
        self.remove_remove_out = rr_out
        self.add_remove_out   = ar_out
        like_add    = -0.5 * (aa_out    - 2.0 * d_h_add_out)
        like_remove = -0.5 * (rr_out    - 2.0 * d_h_remove_out)
        return like_add, like_remove, d_h_add_out, d_h_remove_out, aa_out, rr_out, ar_out

    def fill_global_wdm(self, params, templates, source, *,
                        convert_to_ra_dec=False, data_index=None,
                        factors=None):
        """ACA-aware fill_global. Drop-in for
        :meth:`GBWDMComputations.fill_global_wdm` (params first per the
        sprint convention; 1D / 2D ``params`` are auto-promoted via
        ``atleast_2d`` internally).

        ``templates`` may be either a single global ``(nchannels, Nf,
        Nt)`` slab (data_index=None or all-zero) or a multi-slab
        ``(num_templates, nchannels, Nf, Nt)`` buffer with
        ``data_index[b] in [0, num_templates)`` selecting which slab
        each binary writes into. Flat (1D) buffers are also accepted
        and reshaped to ``(num_templates, nchannels, Nf, Nt)``.

        Internally ``fill_global`` doesn't take ``data_index`` -- it
        writes every binary into a single slab. To honour
        ``data_index``, we loop over the unique slab indices and call
        ``fill_global`` once per slab on the corresponding 3D view,
        with only the binaries assigned to that slab.

        ``source`` is unused (the chunked-het kernel writes templates
        directly); kept in the signature for drop-in compatibility
        with :meth:`GBWDMComputations.fill_global_wdm`.
        """
        params = np.atleast_2d(np.asarray(params, dtype=float)).reshape(-1, self._CPP_NPARAMS)
        params_used = self._maybe_ecl_to_icrs(params, convert_to_ra_dec)
        num_bin = params_used.shape[0]
        del source

        nch, Nf, Nt = self.nchannels, self.Nf, self.Nt
        per_template = nch * Nf * Nt
        # Reshape templates to (num_templates, nch, Nf, Nt) -- supports
        # 1D flat, 3D single-slab, and 4D multi-slab inputs the same way
        # GBWDMComputations.fill_global_wdm does. Aliasing the original
        # buffer (no .copy()) so writes propagate back to the caller.
        if templates.ndim == 1:
            assert templates.shape[-1] % per_template == 0, (
                f"templates flat size {templates.shape[-1]} not divisible "
                f"by nchannels*Nf*Nt = {per_template}"
            )
            num_templates = int(templates.shape[-1] // per_template)
            templates_view = templates.reshape(num_templates, nch, Nf, Nt)
        elif templates.ndim == 3:
            assert templates.shape == (nch, Nf, Nt), (
                f"3D templates must be ({nch}, {Nf}, {Nt}); got {templates.shape}"
            )
            num_templates = 1
            templates_view = templates[None, :, :, :]
        elif templates.ndim == 4:
            assert templates.shape[1:] == (nch, Nf, Nt), (
                f"4D templates must be (num_templates, {nch}, {Nf}, {Nt}); "
                f"got {templates.shape}"
            )
            num_templates = int(templates.shape[0])
            templates_view = templates
        else:
            raise ValueError(
                "templates must be 1D (flat), 3D (nchannels, Nf, Nt), or "
                "4D (num_templates, nchannels, Nf, Nt)."
            )

        if data_index is None:
            data_index_arr = np.zeros(num_bin, dtype=np.int32)
        else:
            data_index_arr = np.asarray(data_index, dtype=np.int32).reshape(-1)
            assert data_index_arr.shape == (num_bin,), (
                f"data_index must have shape ({num_bin},); "
                f"got {data_index_arr.shape}"
            )
            if data_index_arr.size and int(data_index_arr.max()) >= num_templates:
                raise ValueError(
                    f"data_index.max()={int(data_index_arr.max())} >= "
                    f"num_templates={num_templates}: templates buffer is "
                    "too small for the requested data_index range."
                )

        if factors is None:
            factors_arr = np.ones(num_bin, dtype=float)
        else:
            factors_arr = np.asarray(factors, dtype=float).reshape(-1)
            assert factors_arr.shape == (num_bin,), (
                f"factors must have shape ({num_bin},); got {factors_arr.shape}"
            )

        # Loop over slabs. Inside fill_global, the kernel accepts
        # template_out = (nch, Nf, Nt) and writes accumulator-style;
        # each slab gets only the binaries whose data_index hits it.
        for slab in np.unique(data_index_arr):
            mask = (data_index_arr == int(slab))
            params_slab = [params_used[i] for i in np.flatnonzero(mask)]
            factors_slab = factors_arr[mask]
            self.fill_global(templates_view[int(slab)], params_slab,
                             factors=factors_slab)


def psd_fix_eigabs(M, floor_rel=1e-30):
    """Project each ``(d, d)`` slice of ``M`` onto the PSD cone via
    symmetric eigendecomposition with ``|lambda|`` clipping.

    Args:
        M: ``(..., d, d)`` ndarray. Must be (approximately) symmetric
            along the last two axes; the routine symmetrises explicitly
            (``0.5 * (M + M^T)``) before decomposing.
        floor_rel: floor on eigenvalues as a fraction of the per-slice
            max |lambda| (avoids singular metrics where the curvature
            collapsed along one direction).

    Returns:
        ``(..., d, d)`` PSD reconstruction with eigenvalues
        ``max(|lambda_i|, floor_rel * max_j |lambda_j|)``.
    """
    Msym = 0.5 * (M + np.swapaxes(M, -1, -2))
    w, V = np.linalg.eigh(Msym)
    w_abs = np.abs(w)
    w_max = np.max(w_abs, axis=-1, keepdims=True)
    w_floor = floor_rel * np.maximum(w_max, np.finfo(w.dtype).tiny)
    w_fixed = np.maximum(w_abs, w_floor)
    return np.einsum("...ij,...j,...kj->...ik", V, w_fixed, V)


class SOBBHWDMHeterodyne(GBWDMHeterodyne):
    """Stellar-origin BBH variant of :class:`GBWDMHeterodyne`.

    Mirrors the GB pipeline but plugs ``SOBBHTDIonTheFly`` in as the
    source class and uses ``f_low`` (param index 5) as the heterodyne
    carrier. All other knobs (Nt_sub, N_sparse, n_pad, Tukey, chunk
    geometry, partial-slide handling) are unchanged -- the C++ kernel
    bodies in ``TDIonthefly.cu`` take ``GBTDIonTheFly *gb`` and treat
    ``gb->f0_index`` as opaque, so the SOBBH C++ specialisation only
    needs the analogous ``SOBBHTDIonTheFly`` pointer and ``f0_index =
    5`` instead.

    Param order on ``params_list`` (matches ``SOBBHTDIonTheFly``):
        (m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)
    """

    # Routes ``fill_global`` / ``get_ll`` / ``swap_ll`` through the
    # ``SOBBHComputationGroupWrap.sobbh_wdm_het_*`` C++ methods.
    _CPP_WRAP_ATTR = "SOBBHComputationGroupWrap"
    _CPP_METHOD_PREFIX = "sobbh_wdm_het"
    _CPP_NPARAMS = 11
    _CPP_F0_PARAM_INDEX = 5   # SOBBHTDIonTheFly: params[5] = f_low

    def __init__(self, *args, **kwargs):
        # Override the source class + param index before the parent's
        # CachedHeterodyneGenerator is constructed. We do this by
        # rebuilding self._gen at the end of __init__.
        super().__init__(*args, **kwargs)
        from fastlisaresponse.tdionfly import SOBBHTDIonTheFly
        from check_shortened_wdm import CachedHeterodyneGenerator
        self._gen = CachedHeterodyneGenerator(
            T_window=self.T_chunk, t_ref_source=self.t_ref_full,
            N_sparse=self.N_sparse, dt=self.dt, nchannels=self.nchannels,
            gb_kwargs=self._gb_kwargs,
            source_class=SOBBHTDIonTheFly, n_params=11,
            f0_param_index=5,                                  # f_low
        )


__all__ = [
    "compute_chunk_geometry",
    "compute_wdm_window",
    "group_binaries_by_layer",
    "recommended_tukey_alpha",
    "GBWDMHeterodyne",
    "SOBBHWDMHeterodyne",
]
