"""Shared chunked-heterodyne Python wrapper for source-class likelihoods.

Provides :class:`WDMComputationsBase`, the source-agnostic chunked-heterodyne
Python frontend that GB (``gbgpu.gbcomps.GBWDMComputations``) and SOBBH
(``bbhx.sobbhcomps.SOBBHWDMComputations``) inherit from. Sub-classes pick
their own:

* ``_BACKEND_PREFIX``  -- which backend family to dispatch through
  (``"gbgpu"`` for GBGPU, ``"bbhx"`` for BBHx).
* ``_WRAP_ATTR``       -- which ``*ComputationGroupWrap`` class to look for
  on ``self.backend`` (``GBComputationGroupWrap`` /
  ``SOBBHComputationGroupWrap``).
* ``_METHOD_PREFIX``   -- which kernel-name family to dispatch through
  (``"gb_wdm_het"`` / ``"sobbh_wdm_het"``).
* ``_NPARAMS``         -- per-source parameter count (9 for GB / 11 for SOBBH).
* ``_F0_PARAM_INDEX``  -- column index of the carrier frequency in the
  parameter array (1 for GB / 5 for SOBBH).

All chunked-het mechanics -- chunk geometry, WDM window precomputation,
layer-grouping, ``fill_global_wdm`` / ``get_ll_wdm`` / ``get_swap_ll_wdm``
/ ``get_ll_grad_wdm`` -- live on the base class and work identically for
both source classes.

This file was extracted from ``gbgpu.gbcomps.GBWDMComputations`` at the
2026-06-06 architectural cleanup that moved SOBBH out of GBGPU and into
BBHx. See the ``GBWDMComputations`` and ``SOBBHWDMComputations`` thin
sub-classes in their respective packages for the per-source-class
constants. ``_BACKEND_PREFIX`` is not defined on the base class: a subclass
(or its parallel-module mixin) must provide it.

Pipeline: in the stock WDM global fits the GB branch builds a
``GBWDMComputations`` (``gb_info.gb_wdm_comp``, on the run's orbits with
``t_obs_start = data_t0``); with sig-het in-model scoring on, the move receives a
``GBSignalHetComputations`` wrapping it (``for_band_engine``) while RJ, fills and
swaps keep this chunked-het path, as does the F-statistic
(:meth:`WDMComputationsBase.get_fstat_ll_wdm`).

Environment knobs read once at import: ``GB_INDEX_ASSERTS`` (``1`` enables
per-call index-bound asserts, each a device sync; default off),
``GB_FSTAT_FOLD`` (F-stat basis-filter fold, default ``1`` = on) and
``GB_FSTAT_ORBIT_CACHE`` (F-stat orbit spline cache, default ``1`` = on). Read per
call: ``SOBBH_GEOM_CACHE`` (``1`` default; anything else disables the per-device
geometry cache of :meth:`WDMComputationsBase._geometry_kernel_args`).
"""
import os
import time
from copy import deepcopy
from typing import Optional

import numpy as np

from .detector import EqualArmlengthOrbits, Orbits
from .domains import WDMSettings
from .response.directresponse import ecliptic_to_icrs, warn_deprecated_frame_conversion
from .response.tdiconfig import TDIConfig
from .utils.parallelbase import LISAToolsParallelModule
from .wdm_het import (
    USE_RECOMMENDED_TUKEY,
    compute_chunk_geometry,
    compute_layer_groups,
    compute_swap_layer_groups,
    compute_wdm_window,
    resolve_tukey_alpha,
)

# Index-bound sanity asserts (debugging aids): each ``int(arr.max())`` is a
# full device sync per engine call on CuPy. Default OFF in production;
# opt back in with GB_INDEX_ASSERTS=1. Checked ONCE at import.
_GB_INDEX_ASSERTS = os.environ.get("GB_INDEX_ASSERTS", "0") == "1"

# F-stat basis-filter fold. The 4 Cornish & Crowder basis filters are 2 psi
# stages x 2 phase quadratures, and the analytic TDI is an EXACT phasor in
# phi0, so the quadrature partners are a constant unit-modulus rotation of the
# two generated stages -- halving the waveform generations inside
# ``wdm_het_get_fstat_ll_kernel`` at zero physics cost (the fold is exact;
# only floating-point reassociation separates the two paths). See the FOLD
# note in lat_chunked_het_kernels.hh for the rotation and its sign.
#
# DEFAULT ON (user ruling 2026-08-28, promoted from opt-in after the parity
# suite passed in full ON GPU as well as CPU, with the rotation sign pinned
# by a deliberately conjugated build that failed the gate). The fold is
# EXACT -- two generations span the identical 4-space, rank 4, condition
# 1.000 -- so there is no physics reason to keep the slower path as the
# default. ``GB_FSTAT_FOLD=0`` restores the unfolded 4-generation path
# bit-for-bit if it is ever needed as a reference.
#
# ⚠ CONSEQUENCE OF THE PROMOTION: a pull + GBGPU rebuild is NO LONGER a
# no-op -- this is now the production path. It REQUIRES the rebuilt kernel:
# with a stale GBGPU .so the extra kernel argument has nowhere to go and the
# call raises (loudly, which is the good failure mode) rather than silently
# running the old code.
#
# Checked ONCE at import; the per-instance ``fstat_fold`` attribute (and the
# ``get_fstat_ll_wdm`` kwarg of the same name) override it, which is how the
# parity gate runs both paths in one process.
_GB_FSTAT_FOLD = int(os.environ.get("GB_FSTAT_FOLD", "1") != "0")

# Orbit spline cache for the F-stat kernel (2026-09-01). The get_ll kernel has
# routed its TD-builds through a per-chunk shared-mem cubic-spline cache of the
# orbit tables whenever the comp's ``N_cp_orbit`` > 0; the F-stat kernel never
# did, so every basis generation paid direct global-mem orbit lookups. With the
# port, the F-stat follows the SAME per-comp ``N_cp_orbit`` -- i.e. it scores
# candidates with exactly the orbit approximation the likelihood uses, which is
# the consistent choice. ``GB_FSTAT_ORBIT_CACHE=0`` forces the F-stat back to
# direct lookups (get_ll unaffected) as the reference / safety valve.
_GB_FSTAT_ORBIT_CACHE = int(os.environ.get("GB_FSTAT_ORBIT_CACHE", "1") != "0")

# Fused phase-max sign contract: the chunked kernels also accumulate the
# quadrature of each candidate-linear inner product (Im of the parity-map
# coefficient W with Re(W) = w, whose rotation tracks a carrier phase shift
# exactly). These constants normalize the raw kernel outputs to EXACTLY
# "the same inner product at phi0 + pi/2", the convention
# ``TwoQuadraturePhaseMaxMixin`` consumes. Pinned EMPIRICALLY by the signed
# stash-parity asserts in GBGPU tests/test_phase_max_fused.py (|D| is
# sign-blind; only the signed compare can catch a wrong constant) -- never
# derive them analytically, flip them only if that test says so. The swap
# cross term <h_add|h_rem> gets its own constant: its add coefficient sits
# in a different (real-product) slot than <d|h_add>'s.
_QUAD_SIGN_CHUNKED = +1.0
_QUAD_SIGN_CHUNKED_SWAP_ADD = +1.0
_QUAD_SIGN_CHUNKED_SWAP_CROSS = +1.0


class WDMComputationsBase(LISAToolsParallelModule):
    """Source-class-agnostic WDM-domain chunked-heterodyne likelihood base.

    Routes through the chunked-heterodyne kernel set
    ``gb_wdm_het_{fill_global, get_ll, swap_ll}`` on the backend (C++
    on CPU / CUDA, JAX on the ``jax`` backend). The constructor
    pre-computes the chunk geometry, the Nt_sub-long WDM window, and
    wraps orbits / TDI config so methods are a thin packing of
    per-call kernel arguments.

    Returns ``-0.5 * (d_d + h_h - 2 d_h)``. ``d_d`` defaults to 0; the
    return is then the source-only piece, and the caller adds
    ``-0.5 <d|d>`` for the full ``log p(d|h)``. There is **no**
    separate ``source_only`` flag.

    Subclass (e.g. ``GBWDMComputations`` in gbgpu, ``SOBBHWDMComputations``
    in bbhx)
    supported through the routing constants below: override
    ``_WRAP_ATTR`` to pick a different ``*ComputationGroupWrap`` on the
    backend, ``_METHOD_PREFIX`` to pick the kernel-name family
    (``gb_wdm_het`` -> ``sobbh_wdm_het``), ``_NPARAMS`` to change the
    per-source parameter count, and ``_F0_PARAM_INDEX`` to point the
    layer-grouping logic at the carrier-frequency column of
    ``params``.

    Methods: :meth:`get_ll_wdm` (per-template log-likelihood),
    :meth:`get_swap_ll_wdm` (add/remove pair), :meth:`fill_global_wdm` (scatter
    templates into a buffer), :meth:`get_fstat_ll_wdm` (F-statistic basis
    products), :meth:`information_matrix`, and the gradient hooks
    :meth:`get_ll_grad_wdm` / :meth:`get_swap_ll_grad_wdm`. Every ``*_wdm``
    method takes a single-shard holder (an AnalysisContainerArray with one
    ``linear_data_arr`` entry, or one AnalysisContainer).

    Args:
        wdm_settings: :class:`lisatools.domains.WDMSettings` instance
            exposing the WDM grid (``Nf``, ``Nt``, ``data_dt``,
            ``Tobs``, ``t0``, ``layer_df``, ``layer_dt``) and the active box that
            the global template buffer is built on.
        t_ref: Source-phase reference time in seconds (PHYSICAL epoch
            where ``f0``/``phi_c`` are defined). Forwarded to the
            kernel as ``t_ref_full``.
        Nt_sub: Per-chunk WDM time pixels (power of two). Default ``256``.
        n_pad: WDM pixels discarded at each chunk edge during the stitch.
            Default ``32``.
        N_sparse: Heterodyne FFT length per chunk; must be a power of two
            ``<= FAST_WDM_N_SPARSE_MAX`` in the C kernel. Default ``256``.
        tukey_alpha: Tukey alpha, or the default ``USE_RECOMMENDED_TUKEY``
            (auto-pick per :func:`~lisatools.wdm_het.recommended_tukey_alpha`,
            "heterodyne" path).
        use_tukey: Tukey master switch (``False`` -> alpha 0). Default ``True``.
        N_cp_sig: Source-signal spline-cache density per chunk
            (0 = direct path; >0 = cache). Default ``0``.
        N_cp_orbit: Orbit spline-cache density per chunk
            (0 = global-mem lookups; >0 = cache). Default ``0``.
        orbits: :class:`~lisatools.detector.Orbits` instance or class (deep-copied;
            configured on ``[t_obs_start, t_obs_start + Tobs]`` if not yet
            configured). ``None`` falls back to
            :class:`~lisatools.detector.EqualArmlengthOrbits`; the global-fit
            builders pass the run's orbits.
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig`, a TDI
            generation string, or ``None`` (``"1st generation"``).
        force_backend: Chooses the dispatch backend at construction;
            all subsequent methods read ``self.backend`` /
            ``self.backend.xp``. Per the sprint-wide rule, no
            method on this class takes a backend kwarg.
        d_d: Constant added to ``h_h - 2 d_h`` inside the returned
            likelihood (default 0 = source-only return).
        tdi_type: ``"XYZ"`` (default), ``"AET"`` or ``"AE"``.
        t_obs_start: PHYSICAL start time of the first grid sample.
            Defaults to ``wdm_settings.t0`` — override it when the
            run's WDM settings carry an array-space ``t0`` (e.g. the
            stock global-fit domains are 0-based) while the data
            actually starts elsewhere, so source phase/orbits are
            evaluated at true absolute times and ``t_ref`` keeps its
            physical meaning.

    Raises:
        TypeError: If ``wdm_settings`` is not a :class:`~lisatools.domains.WDMSettings`.
        ValueError: For an unknown ``tdi_type`` or ``tdi_config``.
        AssertionError: If ``Nt_sub`` or ``N_sparse`` is not a power of two, or the
            chunk step is not positive and even.

    Attributes:
        n_chunks (int): Number of stitched chunks.
        chunk_t_starts (``xp.ndarray``): Physical start time of each chunk (s).
        resolved_tukey_alpha (float): The alpha passed to the kernels.
        cpp_orbits, cpp_tdi_config, cpp_wdm_settings: Backend wraps of the orbits,
            TDI configuration and WDM grid.
        d_h_out, h_h_out: Raw per-template inner products of the last
            :meth:`get_ll_wdm` call.
    """

    #: Wall breakdown of the most recent :meth:`get_ll_wdm` call --
    #: ``{stage, geom, wrap, launch, total, num_bin, n_groups}`` (times in s; ``None``
    #: before the first call). Read by scoring-cost telemetry; no method
    #: reads it back, so nothing depends on it being present.
    last_call_spans = None

    # Routes ``fill_global_wdm`` / ``get_ll_wdm`` / ``swap_ll_wdm``
    # through ``GBComputationGroupWrap.gb_wdm_het_*`` on the backend.
    # Subclasses (e.g. :class:`SOBBHWDMComputations`) override these.
    _WRAP_ATTR = "GBComputationGroupWrap"
    _METHOD_PREFIX = "gb_wdm_het"
    _NPARAMS = 9
    _F0_PARAM_INDEX = 1   # GBTDIonTheFly: params[1] = f0
    #: Whether this comp's C++ get_ll/swap_ll BINDINGS accept the trailing
    #: fused phase-max quadrature array(s). The LAT kernels/impls default
    #: them off (nullptr), so a source class opts in only after its own
    #: binding TU grows the parameter -- GBGPU has (GBWDMComputations sets
    #: True); bbhx/SOBBH has not (stays False; its calls append nothing and
    #: ``d_h_im_out`` stashes None).
    _FUSED_QUAD_KERNELS = False
    #: Whether this comp's C++ ``get_fstat_ll`` BINDING accepts the trailing
    #: ``fstat_fold`` argument (basis-filter fold). Exactly the
    #: ``_FUSED_QUAD_KERNELS`` contract: the LAT kernel/impl default the
    #: parameter OFF, so a source class opts in only after its own binding TU
    #: grows it -- GBGPU has (``GBWDMComputations`` sets True); bbhx/SOBBH has
    #: not (stays False, nothing is appended, and the fold is simply
    #: unreachable there). Appending unconditionally would be a TypeError in
    #: every downstream wheel that has not rebuilt.
    _FSTAT_FOLD_KERNELS = False
    #: Basis-filter fold for :meth:`get_fstat_ll_wdm`: 0 = OFF (the unfolded
    #: 4-generation path, bit-for-bit the pre-fold kernel), 1 = ON (2 generations
    #: + the exact phi0 rotation; the default since 2026-08-28). Seeded from
    #: ``GB_FSTAT_FOLD`` (default "1") at import; override per instance or per call.
    fstat_fold = _GB_FSTAT_FOLD
    #: Whether this comp's C++ ``get_fstat_ll`` BINDING accepts the trailing
    #: ``N_cp_orbit`` argument (orbit spline cache). Same contract as
    #: ``_FSTAT_FOLD_KERNELS``: opt-in per source class after its binding TU
    #: grows the parameter -- GBGPU has (``GBWDMComputations`` sets True);
    #: bbhx/SOBBH has not (stays False, nothing appended).
    _FSTAT_ORBIT_CACHE_KERNELS = False
    #: Orbit spline cache for :meth:`get_fstat_ll_wdm`: 1 (default) = follow
    #: the comp's own :attr:`N_cp_orbit` (so the F-stat and get_ll share one
    #: orbit approximation); 0 = force direct orbit lookups in the F-stat
    #: kernel only. Seeded from ``GB_FSTAT_ORBIT_CACHE`` at import; override
    #: per instance.
    fstat_orbit_cache = _GB_FSTAT_ORBIT_CACHE
    #: Whether this comp's C++ ``get_ll`` / ``swap_ll`` / ``get_fstat_ll``
    #: BINDINGS accept the trailing shared-psd MIRROR args ``(invC_Nf,
    #: invC_row)`` (2026-09-09). Same opt-in contract as
    #: ``_FUSED_QUAD_KERNELS``: GBGPU sets True after its binding TU grew the
    #: params; bbhx/SOBBH stays False and appends nothing. A mirror-layout
    #: holder (``psd_row_index`` set: ``linear_psd_arr[0]`` is the parent's
    #: per-walker full-band invC plane) meeting a False class is a HARD
    #: error, never a silent per-slot fallback (plan risk R2).
    _PSD_MIRROR_KERNELS = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The mirror args are the LAST positional pair of the kernel
        # signatures, after the fused-quadrature / fstat_fold / N_cp_orbit
        # params. A class that appends them while skipping any of the earlier
        # opt-ins would splat them into the wrong slots (plan risk R6).
        if cls.__dict__.get("_PSD_MIRROR_KERNELS", False) and not (
            cls._FUSED_QUAD_KERNELS
            and cls._FSTAT_FOLD_KERNELS
            and cls._FSTAT_ORBIT_CACHE_KERNELS
        ):
            raise TypeError(
                f"{cls.__name__}: _PSD_MIRROR_KERNELS=True requires "
                "_FUSED_QUAD_KERNELS, _FSTAT_FOLD_KERNELS and "
                "_FSTAT_ORBIT_CACHE_KERNELS to be True as well (the mirror "
                "args are positional AFTER those)."
            )

    def __init__(self, wdm_settings, t_ref,
                 Nt_sub=256, n_pad=32, N_sparse=256,
                 tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
                 N_cp_sig=0, N_cp_orbit=0,
                 orbits=None, tdi_config=None, force_backend=None,
                 d_d=0.0, tdi_type="XYZ", t_obs_start=None):
        """Precompute chunk geometry, the chunk window and the backend wraps (see the class)."""
        super().__init__(force_backend=force_backend)

        # ---- multi-GPU replica metadata (see ``_device_local_gb_comp`` in
        # ``globalfit/stock/erebor/source_runtime.py``) --------------------
        # Everything this constructor allocates -- the chunk geometry, the
        # WDM window, ``cpp_orbits`` / ``cpp_tdi_config`` -- lands on
        # whatever CUDA device is current RIGHT NOW and is never migrated.
        # Record (a) that device, so a routed launch can assert its shard's
        # kernels are reading device-local pointers, and (b) the constructor
        # arguments, so a non-primary shard can rebuild an equivalent comp
        # inside its own device context. Same "remember your constructor
        # arguments so a replica can be rebuilt" contract ``Orbits.args`` /
        # ``DomainSettingsBase.args`` already carry; the recorded objects are
        # the ones this instance already holds, so nothing new reaches the
        # settings tree (pickle-safety rule).
        from .utils.device import current_device as _current_device

        self._build_device = _current_device(self.xp)
        self._ctor_args = (wdm_settings, t_ref)
        self._ctor_kwargs = dict(
            Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
            tukey_alpha=tukey_alpha, use_tukey=use_tukey,
            N_cp_sig=N_cp_sig, N_cp_orbit=N_cp_orbit,
            orbits=orbits, tdi_config=tdi_config,
            force_backend=force_backend, d_d=d_d, tdi_type=tdi_type,
            t_obs_start=t_obs_start,
        )

        if not isinstance(wdm_settings, WDMSettings):
            raise TypeError(
                "wdm_settings must be a lisatools.domains.WDMSettings "
                f"instance; got {type(wdm_settings).__name__}.")
        self.wdm_settings = wdm_settings

        # WDM grid + obs setup pulled from the WDMSettings instance.
        # Stored first so the orbits setter can configure on the full
        # obs span before kernel-arg packing.
        self.Nf       = int(wdm_settings.Nf)
        self.Nt       = int(wdm_settings.Nt)
        self.dt       = float(wdm_settings.data_dt)
        self.T        = float(wdm_settings.Tobs)
        self.t_ref    = float(t_ref)
        self.t_obs_start = float(
            wdm_settings.t0 if t_obs_start is None else t_obs_start
        )
        self.Nt_sub   = int(Nt_sub)
        self.n_pad    = int(n_pad)
        self.N_sparse = int(N_sparse)
        self.tukey_alpha = float(tukey_alpha)
        self.use_tukey   = bool(use_tukey)
        self.N_cp_sig    = int(N_cp_sig)
        self.N_cp_orbit  = int(N_cp_orbit)

        # Derived quantities the kernel consumes directly. ``layer_df``
        # is read straight from the WDMSettings so the two definitions
        # cannot drift.
        self.T_chunk      = self.Nf * self.Nt_sub * self.dt
        self.layer_df     = float(wdm_settings.layer_df)
        self.n_rfft_chunk = self.Nf * self.Nt_sub // 2 + 1
        self.log2_N_sparse = int(np.log2(self.N_sparse))
        self.log2_Nt_sub   = int(np.log2(self.Nt_sub))
        assert 2 ** self.log2_N_sparse == self.N_sparse, (
            f"N_sparse={self.N_sparse} must be a power of two.")
        assert 2 ** self.log2_Nt_sub == self.Nt_sub, (
            f"Nt_sub={self.Nt_sub} must be a power of two.")
        # NOTE: this power-of-two requirement, combined with Nt never being
        # one, is what forces every uniform chunking to compute ~1.9x the
        # pixels actually needed on CPU. See the "CPU two-size chunks" TODO
        # above compute_chunk_geometry in wdm_het.py for the fix (one big +
        # one remainder-sized chunk -> 1.007x) and for why the GPU path
        # should keep uniform sizing.

        # ``d_d`` constant the source-side likelihood adds to
        # ``h_h - 2 d_h``. Defaults to 0 -> source-only.
        self.d_d = float(d_d)

        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError(
                f"tdi_type must be one of 'XYZ', 'AET', 'AE'; got {tdi_type!r}.")
        self.tdi_type = tdi_type

        # Chunk geometry + WDM phitilde -- precomputed once on the host
        # (compute_chunk_geometry / compute_wdm_window are host-side by
        # design), then uploaded ONCE to this instance's backend: every
        # kernel call passes these as-is and the CUDA bindings require
        # device arrays. On CPU xp is numpy and asarray is a no-op.
        backend_short = self.backend.name.split("_")[-1]
        geom = compute_chunk_geometry(self.Nt, self.Nt_sub, self.n_pad)
        layer_dt = self.Nf * self.dt
        starts_f = np.asarray(geom["starts"], dtype=float)
        self.chunk_t_starts = self.xp.asarray(
            self.t_obs_start + starts_f * layer_dt
        )
        self.chunk_keep_lo  = self.xp.asarray(geom["keep_lo"],     dtype=self.xp.int32)
        self.chunk_keep_hi  = self.xp.asarray(geom["keep_hi"],     dtype=self.xp.int32)
        self.chunk_n_global_offset = self.xp.asarray(geom["n_global_lo"],
                                                     dtype=self.xp.int32)
        self.wdm_window = self.xp.asarray(
            compute_wdm_window(self.Nf, self.Nt_sub, self.dt,
                               backend=backend_short)
        )
        self.n_chunks = int(starts_f.size)

        # Resolved Tukey alpha (one double passed to the kernel).
        self.resolved_tukey_alpha = float(resolve_tukey_alpha(
            self.tukey_alpha, self.use_tukey,
            path="heterodyne", N_sparse=self.N_sparse,
        ))

        # Orbits / TDI config setters wrap the C++ / JAX objects on the
        # chosen backend.
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.nchannels = self.tdi_config.nchannels

        # WDMSettings wrap -- single object the het kernels read for all
        # of (Nf, Nt, ind_min_f, ind_min_t, Nf_active, Nt_active, layer_df,
        # layer_dt, num_channel). No per-call duplicate ints; mirrors the
        # cpp_orbits / cpp_tdi_config pattern.
        self.cpp_wdm_settings = self.backend.WDMSettingsWrap(
            float(self.wdm_settings.layer_df),
            float(self.wdm_settings.layer_dt),
            int(self.wdm_settings.Nf),
            int(self.wdm_settings.Nt),
            int(self.nchannels),
            int(self.wdm_settings.ind_min_t),
            int(self.wdm_settings.ind_max_t),
            int(self.wdm_settings.ind_min_f),
            int(self.wdm_settings.ind_max_f),
        )

    @property
    def tdi_config(self) -> TDIConfig:
        """TDI configuration; setting it (str, :class:`TDIConfig`, ``None``) rebuilds the wrap."""
        return self._tdi_config

    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig):
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation", force_backend=self.backend)
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=self.backend)
        elif not isinstance(tdi_config, TDIConfig):
            raise ValueError("TDI Config needs to be a string or an instnace of TDIConfig.")
        self._tdi_config = tdi_config

        self.cpp_tdi_config = self.backend.TDIConfigWrap(*self._tdi_config.pytdiconfig_args)

    @property
    def xp(self) -> object:
        """Array module of the dispatch backend (numpy or cupy)."""
        return self.backend.xp

    @property
    def args(self) -> tuple:
        """Positional arguments for recreating this comp (``(wdm_settings, t_ref)``).

        Mirrors :attr:`lisatools.detector.Orbits.args` /
        :attr:`lisatools.domains.DomainSettingsBase.args`: the per-device
        replica helper rebuilds ``type(comp)(*comp.args, **comp.kwargs)``
        inside the owning device context so every buffer and every ``*Wrap``
        pointer field is device-local.
        """
        return self._ctor_args

    @property
    def kwargs(self) -> dict:
        """Keyword arguments for recreating this comp (see :attr:`args`)."""
        return dict(self._ctor_kwargs)

    @property
    def orbits(self) -> object:
        """Response orbits (a private deep copy); setting them rebuilds ``cpp_orbits``."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        """Set response orbits and (re)build the backend orbits wrap."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)

        if not self._orbits.configured:
            # Configure on the full observation span at the kernel's dt
            # so pycppdetector_args populate for the C++ / JAX wrap.
            # (_configure directly: this is an explicit custom grid, not the
            # lazy default configuration.)
            t_arr = np.arange(0.0, self.T + self.dt, self.dt) + self.t_obs_start
            try:
                self._orbits._configure(t_arr=t_arr, dt=self.dt,
                                        linear_interp_setup=True)
            except TypeError:
                self._orbits._configure(linear_interp_setup=True)
        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)

    @classmethod
    def supported_backends(cls):
        """``_BACKEND_PREFIX + "_" + name`` for each GPU-recommended backend, plus ``jax``."""
        # ``_BACKEND_PREFIX`` lets subclasses (e.g. ``SOBBHWDMComputations``)
        # re-route to their own backend family without overriding this method.
        # GPU_RECOMMENDED_WITH_JAX appends 'jax' so the JAX backend is also
        # listed as a candidate.
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    # ------------------------------------------------------------------
    # Backend dispatch helpers
    # ------------------------------------------------------------------

    def _layer_groups(self, params_2d, group_band_layers, margin_layers,
                       data_index=None, noise_index=None):
        """Cluster binaries by carrier WDM layer for narrow-band dispatch."""
        return compute_layer_groups(
            self.xp.asarray(params_2d), layer_df=self.layer_df,
            f0_param_index=self._F0_PARAM_INDEX,
            group_band_layers=int(group_band_layers),
            margin_layers=int(margin_layers),
            data_index_all=data_index, noise_index_all=noise_index,
        )

    def _swap_layer_groups(self, params_add_2d, params_remove_2d,
                            group_band_layers, margin_layers,
                            data_index=None, noise_index=None):
        """Layer groups of an add/remove batch (``wdm_het.compute_swap_layer_groups``)."""
        # compute_swap_layer_groups is xp-agnostic (dispatches on the array
        # module of its inputs) -- keep params on this instance's backend
        # like _layer_groups does; np.asarray here raised CuPy's
        # implicit-conversion TypeError on GPU.
        return compute_swap_layer_groups(
            self.xp.asarray(params_add_2d), self.xp.asarray(params_remove_2d),
            layer_df=self.layer_df,
            f0_param_index=self._F0_PARAM_INDEX,
            group_band_layers=int(group_band_layers),
            margin_layers=int(margin_layers),
            data_index_all=data_index, noise_index_all=noise_index,
        )

    def _comp_group(self):
        """The ``*ComputationGroupWrap`` instance for this source class."""
        return getattr(self.backend, self._WRAP_ATTR)()

    def _kernel(self, name):
        """Resolve ``self._METHOD_PREFIX + '_' + name`` on the comp group."""
        return getattr(self._comp_group(), f"{self._METHOD_PREFIX}_{name}")

    def _geometry_kernel_args(self):
        """The five STATIC geometry kernel args, device-local and cached.

        ``(chunk_t_starts, chunk_keep_lo, chunk_keep_hi,
        chunk_n_global_offset, wdm_window)`` in kernel-signature order.

        All five are built once in ``__init__`` and never mutated -- on the
        comp's HOME device, because that is where ``__init__`` ran. This is
        the single point every kernel path goes through to get them onto
        whatever device is current NOW, and it exists because the SOBBH
        multi-GPU walker-shard route runs ONE SHARED comp under both
        shards' device contexts (``sobbhspecialmove.py`` enters
        ``device_context(view.device)`` around ``get_ll_wdm`` for scoring
        and around ``fill_global_wdm`` for the residual fill). Anything
        sourced from ``self`` is therefore a home-device pointer until
        :meth:`_to_device_local` moves it.

        2026-09-16: this used to be ``get_ll_wdm``'s private helper while
        ``fill_global_wdm`` / ``get_swap_ll_wdm`` / both grad methods passed
        ``self.chunk_*`` RAW and only ``get_fstat_ll_wdm`` re-asserted. A
        device-1 launch of ``wdm_het_fill_global_kernel`` fed device-0
        geometry is the null run's ``GPUassert: an illegal memory access``.
        ALL SIX kernel paths now share this helper -- do not reintroduce a
        raw ``self.chunk_*`` in a kernel argument list.

        Cached keyed on the CUDA device current at call time (``None`` on
        the CPU path). NEVER invalidated: this is fixed geometry, so there
        is nothing that could go stale. Anything that changed it would have
        to rebuild the comp anyway. On the home device the cached entry is
        the comp's own arrays (``asarray`` of an already-resident array is
        the same object), so single-GPU runs are pointer-identical to
        before.
        """
        from .utils.device import current_device

        # ESCAPE (2026-09-16): SOBBH_GEOM_CACHE=0 restores the pre-cache
        # per-call re-assert so the cluster can bisect the cache without a
        # revert. Still device-local -- the escape drops the CACHE, not the
        # relocation.
        if os.environ.get("SOBBH_GEOM_CACHE", "1").strip() != "1":
            return self._geometry_args_fresh()

        cache = getattr(self, "_geom_args_by_device", None)
        if cache is None:
            cache = self._geom_args_by_device = {}
        dev = current_device(self.xp)
        args = cache.get(dev)
        if args is None:
            args = self._geometry_args_fresh()
            cache[dev] = args
        return args

    def _geometry_args_fresh(self):
        """The five geometry args, freshly relocated onto the current device."""
        return (
            self._to_device_local(self.chunk_t_starts),
            self._to_device_local(self.chunk_keep_lo),
            self._to_device_local(self.chunk_keep_hi),
            self._to_device_local(self.chunk_n_global_offset),
            self._to_device_local(self.wdm_window),
        )

    def _to_device_local(self, arr):
        """``arr`` resident on the CUDA device that is current RIGHT NOW.

        ``xp.asarray`` uploads HOST data to the current device, but handed
        an array that ALREADY lives on another device it is not a
        relocation. That is the whole trap on the multi-GPU walker-shard
        path: the five geometry arrays are built once in ``__init__`` on
        the comp's home device, and the SOBBH move routes ONE SHARED comp
        through both shards' device contexts, so every kernel arg sourced
        from ``self`` is a home-device pointer unless something moves it.
        A device-1 kernel dereferencing a device-0 pointer is
        ``GPUassert: an illegal memory access was encountered``.

        The cross-device leg goes VIA HOST, matching
        :func:`lisatools.utils.device.to_current_device`: peer access is
        unavailable between GPUs on some nodes, so a device-to-device copy
        is not safe to assume.

        CPU / NumPy ``xp`` (no ``cuda`` attribute) and arrays already on
        the current device take the plain ``asarray`` path, bit-identical
        to the behaviour before this helper existed.
        """
        xp = self.xp
        if hasattr(xp, "cuda"):
            dev = getattr(arr, "device", None)
            dev_id = getattr(dev, "id", None)
            if (dev_id is not None
                    and int(dev_id) != int(xp.cuda.runtime.getDevice())):
                return xp.asarray(xp.asnumpy(arr))
        return xp.asarray(arr)

    def _slab_kernel_args(self, holder):
        """Task-b per-band slab args to splat at the end of the C++ buffer
        kernels (get_ll / swap_ll / fill_global / get_fstat_ll).

        Narrow per-band slabs (each buffer slot spans ``band_slab_Nf`` WDM
        layers starting at ``slab_min_f[slot]`` instead of the full active
        band) are a **C++-only** memory optimization. Returns:

        * ``()`` on the JAX backend -- the JAX kernels keep their original
          signature (narrow mode there is a documented follow-up), so nothing
          is appended.
        * ``(0, empty_int32)`` on the C++ backends when the holder carries no
          slab metadata -- the required kernel args in their OFF state
          (bit-identical to the pre-task-b full-active-band layout).
        * ``(int(band_slab_Nf), slab_min_f_int32)`` when the holder (a GB
          ``SubBandBuffer`` built with ``wdm_band_slab_layers``) exposes the
          per-slot origins.
        """
        return self._slab_args_from(
            getattr(holder, "band_slab_Nf", None),
            getattr(holder, "slab_min_f", None),
        )

    def _fstat_fold_kernel_args(self, fstat_fold=None):
        """Trailing ``fstat_fold`` arg for the C++ ``get_fstat_ll`` kernel.

        Splatted AFTER :meth:`_slab_kernel_args` (the kernel signature order
        is ``..., Nf_slab, slab_min_f, fstat_fold``). Returns:

        * ``()`` on the JAX backend (its ``get_fstat_ll`` is unimplemented)
          and on any source class whose binding TU has not grown the
          parameter (``_FSTAT_FOLD_KERNELS`` False) -- appending there would
          be a TypeError in a downstream wheel that has not rebuilt.
        * ``(int(fold),)`` otherwise, resolving ``None`` to the instance's
          :attr:`fstat_fold` (seeded from ``GB_FSTAT_FOLD``, default 1/ON).
        """
        if not self._FSTAT_FOLD_KERNELS:
            return ()
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            return ()
        if fstat_fold is None:
            fstat_fold = self.fstat_fold
        return (int(bool(int(fstat_fold))),)

    def _fstat_orbit_cache_kernel_args(self):
        """Trailing ``N_cp_orbit`` arg for the C++ ``get_fstat_ll`` kernel.

        Splatted AFTER :meth:`_fstat_fold_kernel_args` (the kernel signature
        order is ``..., Nf_slab, slab_min_f, fstat_fold, N_cp_orbit``).
        Returns ``()`` on JAX / on a source class whose binding TU has not
        grown the parameter (``_FSTAT_ORBIT_CACHE_KERNELS`` False), else
        ``(int(self.N_cp_orbit),)`` -- or ``(0,)`` when
        :attr:`fstat_orbit_cache` is disarmed (``GB_FSTAT_ORBIT_CACHE=0``),
        which forces the F-stat back to direct orbit lookups without
        touching get_ll.
        """
        if not self._FSTAT_ORBIT_CACHE_KERNELS:
            return ()
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            return ()
        if not self.fstat_orbit_cache:
            return (0,)
        return (int(self.N_cp_orbit),)

    def _psd_kernel_args(self, holder):
        """Trailing shared-psd MIRROR args ``(invC_Nf, invC_row)`` for the C++
        ``get_ll`` / ``swap_ll`` / ``get_fstat_ll`` kernels (2026-09-09).

        Splatted LAST (after the fused quadratures on get_ll/swap_ll, after
        ``N_cp_orbit`` on get_fstat_ll). Returns:

        * ``()`` on the JAX backend and on any source class whose binding TU
          has not grown the params (``_PSD_MIRROR_KERNELS`` False) -- and a
          HARD error there if the holder is in mirror layout (a mirror holder
          scored through per-slot addressing would read the wrong bytes).
        * ``(0, empty_int32)`` -- the OFF state, bit-identical to the
          pre-mirror per-slot slab addressing -- when the holder carries no
          ``psd_row_index``.
        * ``(int(Nf_active), psd_row_index_int32)`` when the holder exposes
          ``psd_row_index`` (a GB ``SubBandBuffer`` bound to the parent's
          psd mirror): ``linear_psd_arr[0]`` is then the parent ACA's
          per-WALKER full-active-band invC plane and slot ``s`` reads row
          ``psd_row_index[s]`` at its own absolute layers.

        Under ``GB_INDEX_ASSERTS`` the row map and the slab origins are
        bounds-checked against the plane (the buffer's ``bind_psd_mirror``
        does the same check unconditionally at bind time).
        """
        rows = getattr(holder, "psd_row_index", None)
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            if rows is not None:
                raise NotImplementedError(
                    "shared-psd mirror holders (psd_row_index) are C++-only; "
                    "the JAX chunked-het kernels keep the per-slot invC layout.")
            return ()
        if not self._PSD_MIRROR_KERNELS:
            if rows is not None:
                raise RuntimeError(
                    f"{type(self).__name__} does not carry the shared-psd "
                    "mirror kernel args (_PSD_MIRROR_KERNELS is False) but the "
                    "holder is in mirror layout (psd_row_index set). Refusing "
                    "to score a mirror plane through per-slot addressing.")
            return ()
        if rows is None:
            return (0, self.xp.zeros(0, dtype=np.int32))
        ws = self.wdm_settings
        Nf_active = int(getattr(ws, "Nf_active", self.Nf))
        rows = self.xp.ascontiguousarray(self.xp.asarray(rows, dtype=np.int32))
        if _GB_INDEX_ASSERTS:
            Nt_active = int(getattr(ws, "Nt_active", self.Nt))
            nch = int(self.nchannels)
            per_row = (nch * nch if self.tdi_type == "XYZ" else nch) * Nf_active * Nt_active
            n_rows = int(holder.linear_psd_arr[0].size) // per_row
            assert n_rows * per_row == int(holder.linear_psd_arr[0].size), (
                "mirror plane size is not a whole number of per-walker rows")
            assert int(rows.max()) < n_rows, (
                f"psd_row_index max {int(rows.max())} >= mirror rows {n_rows}")
            slab_min_f = getattr(holder, "slab_min_f", None)
            band_slab_Nf = getattr(holder, "band_slab_Nf", None)
            if slab_min_f is not None and band_slab_Nf is not None:
                ind_min_f = int(getattr(ws, "ind_min_f", 0))
                smf = self.xp.asarray(slab_min_f)
                assert int(smf.min()) >= ind_min_f and (
                    int(smf.max()) + int(band_slab_Nf) <= ind_min_f + Nf_active), (
                    "a slab origin runs outside the parent active band")
        return (Nf_active, rows)

    def _slab_args_from(self, band_slab_Nf, slab_min_f):
        """Build the trailing C++ slab kernel args from explicit values.

        ``()`` on JAX (narrow mode is C++-only); ``(0, empty_int32)`` when
        off (the required args in their bit-identical OFF state);
        ``(int(band_slab_Nf), slab_min_f_int32)`` when narrow.
        """
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            return ()
        if band_slab_Nf is None or slab_min_f is None:
            return (0, self.xp.zeros(0, dtype=np.int32))
        return (
            int(band_slab_Nf),
            self.xp.ascontiguousarray(self.xp.asarray(slab_min_f, dtype=np.int32)),
        )

    def _empty_groups(self, num_bin):
        """Zero-length grouping arrays for the un-grouped C-kernel path."""
        return dict(
            binary_perm  = np.zeros(num_bin, dtype=np.int32),
            group_starts = np.zeros(1,       dtype=np.int32),
            group_ends   = np.zeros(1,       dtype=np.int32),
            group_m_lo   = np.zeros(1,       dtype=np.int32),
            group_m_hi   = np.zeros(1,       dtype=np.int32),
            n_groups     = 0,
        )

    def _empty_swap_groups(self, num_bin):
        """:meth:`_empty_groups` plus zero ``pair_m_lo_b`` / ``pair_m_hi_b``."""
        g = self._empty_groups(num_bin)
        g["pair_m_lo_b"] = np.zeros(num_bin, dtype=np.int32)
        g["pair_m_hi_b"] = np.zeros(num_bin, dtype=np.int32)
        return g

    def _prep_indices(self, num_bin, num_data, num_noise,
                       data_index, noise_index):
        """``int32`` device slot indices (``None`` -> zeros), checked under ``GB_INDEX_ASSERTS``."""
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)
        if _GB_INDEX_ASSERTS:
            assert int(data_index.max()) < num_data
            assert int(noise_index.max()) < num_noise
        return data_index, noise_index

    def _as_wdm_holder(self, wdm_holder):
        """Normalise ``wdm_holder`` to an :class:`AnalysisContainerArray`.

        The ``*_wdm`` kernels consume an ACA's flat ``linear_data_arr[0]`` /
        ``linear_psd_arr[0]`` and index it per-binary via ``data_index``.
        Callers may pass either a full (multi-walker) ACA or a single
        :class:`AnalysisContainer` (e.g. one walker slab for a data-independent
        ``<h|h>``).  A lone container is wrapped into a 1-element ACA under the
        hood so the kernel interior is identical either way (one AC == one
        walker slab; ``data_index`` must then be all-zero).  ``gpus`` is
        inferred from where the container's residual lives so the repack stays
        on-device.
        """
        # Already an ACA (or anything exposing the flat buffers) -> pass through.
        if hasattr(wdm_holder, "linear_data_arr"):
            if len(wdm_holder.linear_data_arr) != 1:
                raise NotImplementedError(
                    "the *_wdm kernels are single-shard by contract (they "
                    "consume linear_data_arr[0]). Multi-shard holders must "
                    "go through the LAT shard router (lisatools.globalfit."
                    "moves.gbbands._RoutedBandEngine), which presents one "
                    "per-split view per call (engine ops via the instance "
                    "router; raw-comp F-stat via the route_fstat_ll "
                    "classmethod)."
                )
            return wdm_holder
        from .analysiscontainer import AnalysisContainer, AnalysisContainerArray
        if isinstance(wdm_holder, AnalysisContainer):
            arr = getattr(getattr(wdm_holder, "data_res_arr", None), "arr", None)
            # cupy arrays expose ``.device.id``; NumPy has no such attr -> None.
            dev = getattr(getattr(arr, "device", None), "id", None)
            gpus = None if dev is None else [int(dev)]
            # Wrap COPIES of the domain/sens holders, NOT the caller's
            # container: AnalysisContainerArray construction REBINDS its
            # containers' arrays into the wrap's private linear buffers
            # (reset_linear_data_arr / reset_linear_psd_arr set
            # ``._arr`` / ``.invC`` to views of the new buffers). Wrapping a
            # parent-ACA-owned container directly would silently disconnect
            # that walker from its parent's ``linear_data_arr`` — the GB
            # kernels keep writing the parent buffer while ``acs.likelihood``
            # reads the stale private copy (walker-0 frozen-readout bug,
            # 2026-07-10). Shallow copies share the ndarray content at
            # repack time, so the numerics are identical; only the rebind
            # side effect is confined to the throwaway wrap.
            import copy as _copy

            holder_ac = AnalysisContainer(
                _copy.copy(wdm_holder.data),
                _copy.copy(wdm_holder.sens_mat),
            )
            return AnalysisContainerArray(holder_ac, gpus=gpus)
        raise TypeError(
            f"wdm_holder must be an AnalysisContainerArray or AnalysisContainer; "
            f"got {type(wdm_holder).__name__}"
        )

    # ---- per-source in-model likelihood setup hooks -----------------------
    #
    # The GB special move calls setup_in_model() once per picked source
    # batch -- at the same stage the proposal cholesky / friend table are
    # built, AFTER the sources are taken out of their cell residuals --
    # and clear_in_model() after the repeat block. Chunked-het scores every
    # proposal directly against the residual, so it needs no per-source
    # preparation: both hooks just return. Heterodyne-reference
    # computations (gbgpu's GBSignalHetComputations engine mode) override
    # them to build the reference against the source-free residual and
    # hold it CONSTANT for the duration of the in-model repeats.

    def setup_in_model(self, buffer_aca, params_ref_phys, data_index,
                       N_vals=None) -> None:
        """No-op in-model setup hook (see class comment above)."""
        return None

    def clear_in_model(self) -> None:
        """No-op in-model teardown hook."""
        return None

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None,
                   convert_to_ra_dec: Optional[bool] = None,
                   grid_dim: int = 0,
                   use_layer_groups: bool = True,
                   group_band_layers: int = 5,
                   margin_layers: int = 0,
                   m_band_half_width: int = 1):
        """Per-binary chunked-heterodyne ``<d|h>`` / ``<h|h>``.

        Returns ``-0.5 * (d_d + h_h - 2 d_h)`` per binary. ``d_d``
        defaults to 0 (set in ``__init__``), making the return the
        source-only piece; pass ``d_d`` at construction time to fold
        the global ``<d|d>`` in.

        The raw per-binary inner products are stashed on
        ``self.d_h_out`` / ``self.h_h_out`` for callers that need them
        (e.g. analytic phase maximisation), and the fused phase-max quadrature
        of ``<d|h>`` on ``self.d_h_im_out`` (``None`` unless the subclass sets
        ``_FUSED_QUAD_KERNELS`` and the backend is C++).

        The narrow band: the get_ll kernel restricts each binary, per chunk, to
        the layers ``m_floor +/- m_band_half_width`` around its carrier (the
        narrow-band GB inner product validated by mm5 / mm2). The layer groups
        (``use_layer_groups`` / ``group_band_layers`` / ``margin_layers``) are
        still computed and passed, but this kernel ignores them (its grid is one
        block per binary); their count is recorded in :attr:`last_call_spans`. Of
        the other paths, the swap and gradient kernels receive group arrays;
        :meth:`fill_global_wdm` and :meth:`get_fstat_ll_wdm` do not take them.

        Records a per-call wall breakdown on :attr:`last_call_spans`
        (``stage`` = param staging + indices + layer grouping, ``geom`` =
        the static-geometry re-assert, ``wrap`` = comp-group wrap
        construction, ``launch`` = the kernel invocation, ``total``,
        ``num_bin``, ``n_groups``). Four ``perf_counter`` reads per call; the caller
        decides whether to sync the device around ``launch`` and what to do
        with the numbers (the SOBBH move accumulates them per leaf window
        and logs one [SOBBH_LL_TIMING] companion line).

        Args:
            params: ``(num_bin, nparams)`` array (1D auto-promoted via
                ``atleast_2d``). Last two columns are the sky longitude / latitude
                ``(lam, beta)`` in the orbits frame (ICRS on the stock runs;
                ``convert_to_ra_dec=True`` is the deprecated legacy ecliptic ->
                ICRS conversion).
            wdm_holder: ``AnalysisContainerArray`` (single shard) or one
                ``AnalysisContainer``, providing ``linear_data_arr[0]`` (WDM data)
                and ``linear_psd_arr[0]`` (invC) -- both flat.
            data_index, noise_index: per-binary slab indices into
                ``wdm_holder``. Default = all zeros.
            convert_to_ra_dec: Deprecated ecliptic -> ICRS conversion of the last
                two columns. Default ``None`` (off).
            grid_dim: CUDA launch grid size (use 0 for ``n_chunks``).
            use_layer_groups, group_band_layers, margin_layers: layer-grouping
                controls (see above). Defaults ``True``, ``5``, ``0``.
            m_band_half_width: Layers on each side of the carrier. Default ``1``.

        Returns:
            ``xp.ndarray``: ``(num_bin,)`` values of ``-0.5 * (d_d + h_h - 2 d_h)``.

        Raises:
            NotImplementedError: For a multi-shard holder.
            RuntimeError: For a shared-psd mirror holder on a class without the
                mirror kernels.
        """
        _perf = time.perf_counter
        _t_entry = _perf()
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]
        nparams = int(self._NPARAMS)

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        # JAX backend's wrap mutates host (numpy) buffers since jnp
        # arrays are immutable. The JAX kernels also keep their original
        # signature (no fused quadrature output yet -- documented follow-up
        # under the backend-hierarchy rule), so the quadrature array is
        # neither allocated nor appended there and ``d_h_im_out`` stashes
        # ``None`` (the phase-max mixin then falls back to its two-call body).
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            d_h_out = np.zeros(num_bin)
            h_h_out = np.zeros(num_bin)
            d_h_im = None
            quad_args = ()
        else:
            d_h_out = self.xp.zeros(num_bin)
            h_h_out = self.xp.zeros(num_bin)
            if self._FUSED_QUAD_KERNELS:
                d_h_im = self.xp.zeros(num_bin)
                quad_args = (d_h_im,)
            else:
                d_h_im = None
                quad_args = ()

        wdm_holder = self._as_wdm_holder(wdm_holder)
        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_in = params_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._layer_groups(
                params_tmp, group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_groups(num_bin)
        _t_stage = _perf() - _t_entry

        _t_mark = _perf()
        _geom_args = self._geometry_kernel_args()
        _t_geom = _perf() - _t_mark

        _t_mark = _perf()
        _kern = self._kernel("get_ll")
        _t_wrap = _perf() - _t_mark

        _t_mark = _perf()
        _kern(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config,
            self.cpp_wdm_settings,
            params_in, data_index, noise_index,
            *_geom_args,
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            int(self.backend.TDITypeDict[self.tdi_type]),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            self.xp.asarray(groups["binary_perm"],  dtype=np.int32),
            self.xp.asarray(groups["group_starts"], dtype=np.int32),
            self.xp.asarray(groups["group_ends"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_lo"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
            int(m_band_half_width),
            *self._slab_kernel_args(wdm_holder),
            *quad_args,
            *self._psd_kernel_args(wdm_holder),
        )
        # NB ``launch`` is only the true kernel time if the binding (or the
        # caller) synchronizes; an un-synced CUDA launch returns at once and
        # the wall lands on the caller's first blocking D2H instead. The
        # SOBBH move syncs explicitly and reports the tail separately.
        self.last_call_spans = {
            "stage": _t_stage, "geom": _t_geom, "wrap": _t_wrap,
            "launch": _perf() - _t_mark, "total": _perf() - _t_entry,
            "num_bin": int(num_bin),
            # Groups are keyed by (m-band, data_index). RECORDED ONLY AS A
            # DIAGNOSTIC for this path: ``wdm_het_get_ll_impl`` explicitly
            # ``(void)``s binary_perm / group_starts / group_ends /
            # group_m_lo / group_m_hi / n_groups -- the get_ll kernel takes
            # its narrow band from ``m_band_half_width`` per binary per
            # chunk instead, and its grid is ONE BLOCK PER BINARY
            # (``gd_x = num_bin``). Measured on the 6-mo production run
            # 2026-09-18 over 174 leaf windows: the call wall is flat at
            # 1.71-1.73 s across groups/call 1.0-2.8 AND rows/call 6.7-24.2,
            # i.e. it scales with NEITHER. An earlier note here claimed
            # "kernel work scales with the GROUP count"; that is wrong for
            # get_ll and it sent the 22x-per-row hunt down the wrong path.
            # The wall is one block's serial sweep over all n_chunks, so
            # rows are ~free until the device runs out of blocks.
            # Of the OTHER kernel paths, swap_ll and the *_grad kernels are
            # passed these arrays; fill_global and get_fstat_ll do not take them.
            "n_groups": int(groups["n_groups"]),
        }

        self.d_h_out = d_h_out
        self.h_h_out = h_h_out
        self.d_h_im_out = (None if d_h_im is None
                           else _QUAD_SIGN_CHUNKED * d_h_im)
        return -0.5 * (self.d_d + h_h_out - 2.0 * d_h_out)

    def get_swap_ll_wdm(self, params_add, params_remove, wdm_holder,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec: Optional[bool] = None,
                        grid_dim: int = 0,
                        use_layer_groups: bool = True,
                        group_band_layers: int = 5,
                        margin_layers: int = 0,
                        m_band_half_width: int = 1):
        """Swap-proposal 5-way accumulator via chunked-heterodyne.

        Mirrors :meth:`get_ll_wdm` but evaluates the five inner products
        ``<d|h_add>``, ``<d|h_remove>``, ``<h_add|h_add>``,
        ``<h_remove|h_remove>``, ``<h_add|h_remove>`` per binary. The swap kernel
        is passed the swap layer groups (driven by the add template, plus per-pair
        remove-template bands). The raw products are also stashed on
        ``self.d_h_add_out`` ... ``self.add_remove_out``.

        Args:
            params_add: ``(num_bin, nparams)`` add-template parameters.
            params_remove: ``(num_bin, nparams)`` remove-template parameters (same
                shape).
            wdm_holder, data_index, noise_index, convert_to_ra_dec, grid_dim: See
                :meth:`get_ll_wdm`.
            use_layer_groups, group_band_layers, margin_layers, m_band_half_width: See
                :meth:`get_ll_wdm`.

        Returns:
            tuple: ``(like_add, like_remove, d_h_add, d_h_remove, add_add,
            remove_remove, add_remove)``, each ``(num_bin,)``: the two
            ``-0.5 * (d_d + <hh> - 2 <dh>)`` values and the five raw inner products
            (e.g. for cross-term Hastings ratios).

        Raises:
            AssertionError: If the two parameter arrays differ in shape.
        """
        params_add_tmp = self.xp.asarray(self.xp.atleast_2d(params_add)).copy()
        params_remove_tmp = self.xp.asarray(self.xp.atleast_2d(params_remove)).copy()
        assert params_add_tmp.shape == params_remove_tmp.shape, (
            "params_add and params_remove must have the same shape; got "
            f"{params_add_tmp.shape} vs {params_remove_tmp.shape}"
        )
        num_bin = params_add_tmp.shape[0]
        nparams = int(self._NPARAMS)

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            for p_tmp in (params_add_tmp, params_remove_tmp):
                lam = p_tmp[:, -2].copy()
                beta = p_tmp[:, -1].copy()
                lam, beta = ecliptic_to_icrs(lam, beta)
                p_tmp[:, -2] = lam
                p_tmp[:, -1] = beta

        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            d_h_a = np.zeros(num_bin); d_h_r = np.zeros(num_bin)
            aa    = np.zeros(num_bin); rr    = np.zeros(num_bin)
            ar    = np.zeros(num_bin)
            d_h_a_im = None; ar_im = None
            quad_args = ()   # JAX kernels keep their original signature
        else:
            d_h_a = self.xp.zeros(num_bin); d_h_r = self.xp.zeros(num_bin)
            aa    = self.xp.zeros(num_bin); rr    = self.xp.zeros(num_bin)
            ar    = self.xp.zeros(num_bin)
            if self._FUSED_QUAD_KERNELS:
                d_h_a_im = self.xp.zeros(num_bin)
                ar_im    = self.xp.zeros(num_bin)
                quad_args = (d_h_a_im, ar_im)
            else:
                d_h_a_im = None; ar_im = None
                quad_args = ()

        wdm_holder = self._as_wdm_holder(wdm_holder)
        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_add_in    = params_add_tmp.flatten().copy()
        params_remove_in = params_remove_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._swap_layer_groups(
                params_add_tmp, params_remove_tmp,
                group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_swap_groups(num_bin)

        self._kernel("swap_ll")(
            d_h_a, d_h_r, aa, rr, ar,
            self.cpp_orbits, self.cpp_tdi_config,
            self.cpp_wdm_settings,
            params_add_in, params_remove_in,
            data_index, noise_index,
            *self._geometry_kernel_args(),
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            int(self.backend.TDITypeDict[self.tdi_type]),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            self.xp.asarray(groups["binary_perm"],  dtype=np.int32),
            self.xp.asarray(groups["group_starts"], dtype=np.int32),
            self.xp.asarray(groups["group_ends"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_lo"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
            self.xp.asarray(groups["pair_m_lo_b"],  dtype=np.int32),
            self.xp.asarray(groups["pair_m_hi_b"],  dtype=np.int32),
            int(m_band_half_width),
            *self._slab_kernel_args(wdm_holder),
            *quad_args,
            *self._psd_kernel_args(wdm_holder),
        )

        self.d_h_add_out       = d_h_a
        self.d_h_remove_out    = d_h_r
        self.add_add_out       = aa
        self.remove_remove_out = rr
        self.add_remove_out    = ar
        # Fused phase-max quadratures of the ADD-linear terms (None on JAX
        # -> the mixin's two-call fallback), comp-normalized to "value at
        # add-phi0 + pi/2".
        self.d_h_add_im_out = (None if d_h_a_im is None
                               else _QUAD_SIGN_CHUNKED_SWAP_ADD * d_h_a_im)
        self.add_remove_im_out = (None if ar_im is None
                                  else _QUAD_SIGN_CHUNKED_SWAP_CROSS * ar_im)

        like_add    = -0.5 * (self.d_d + aa - 2.0 * d_h_a)
        like_remove = -0.5 * (self.d_d + rr - 2.0 * d_h_r)
        return like_add, like_remove, d_h_a, d_h_r, aa, rr, ar

    # ------------------------------------------------------------------
    # Chain-rule gradients of get_ll_wdm / get_swap_ll_wdm
    #
    # Dispatch through ``self.backend.GBComputationGroupWrap()`` exactly
    # like the likelihood methods above. On the JAX backend the
    # gradient is built with ``jax.grad`` over the standalone JAX
    # chunked-het kernel. On the C++ backends the corresponding
    # ``gb_wdm_het_*_grad`` symbol is the canonical hook for a future
    # C++ implementation; until it lands the backend wrap will raise.
    # ------------------------------------------------------------------

    def get_ll_grad_wdm(self, params, wdm_holder,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec: Optional[bool] = None,
                        grid_dim: int = 0,
                        use_layer_groups: bool = True,
                        group_band_layers: int = 5,
                        margin_layers: int = 0):
        """Chain-rule gradient of :meth:`get_ll_wdm`.

        Dispatches to ``<_METHOD_PREFIX>_get_ll_grad`` on the backend with
        the same C++-style argument list as :meth:`get_ll_wdm`. The JAX
        backend evaluates the gradient via autograd over the
        chunked-het kernel; the C++ backends raise until the matching
        kernel is implemented.

        Args:
            params: ``(num_bin, nparams)`` array.
            wdm_holder: ``AnalysisContainerArray``.
            data_index, noise_index, convert_to_ra_dec, grid_dim: See :meth:`get_ll_wdm`.
            use_layer_groups, group_band_layers, margin_layers: See :meth:`get_ll_wdm`.

        Returns:
            ``xp.ndarray``: ``(num_bin, nparams)``, ``grad[i, k] = dL/dtheta_k``.
        """
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]
        nparams = int(self._NPARAMS)
        assert params_tmp.shape[1] == nparams, (
            f"params has {params_tmp.shape[1]} columns, expected {nparams}")

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        wdm_holder = self._as_wdm_holder(wdm_holder)
        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_in = params_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._layer_groups(
                params_tmp, group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_groups(num_bin)

        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            grad_out = np.zeros(num_bin * nparams, dtype=np.float64)
        else:
            grad_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)

        self._kernel("get_ll_grad")(
            grad_out,
            self.cpp_orbits, self.cpp_tdi_config,
            params_in, data_index, noise_index,
            *self._geometry_kernel_args(),
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            self.xp.asarray(groups["binary_perm"],  dtype=np.int32),
            self.xp.asarray(groups["group_starts"], dtype=np.int32),
            self.xp.asarray(groups["group_ends"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_lo"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
        )

        return self.xp.asarray(grad_out).reshape(num_bin, nparams)

    # TODO check: keep/remove/document -- no run callers; only gb_lookup_table_test_script.py
    def get_swap_ll_grad_wdm(self, params_add, params_remove, wdm_holder,
                             data_index=None, noise_index=None,
                             convert_to_ra_dec: Optional[bool] = None,
                             grid_dim: int = 0,
                             use_layer_groups: bool = True,
                             group_band_layers: int = 5,
                             margin_layers: int = 0):
        """Chain-rule gradient of :meth:`get_swap_ll_wdm`.

        Dispatches to ``<_METHOD_PREFIX>_swap_ll_grad`` on the backend (JAX autograd;
        the C++ backends raise until the kernel exists).

        Args:
            params_add, params_remove: ``(num_bin, nparams)`` arrays.
            wdm_holder, data_index, noise_index, convert_to_ra_dec, grid_dim,
            use_layer_groups, group_band_layers, margin_layers: See
                :meth:`get_ll_wdm`.

        Returns:
            tuple: ``(grad_add, grad_remove)``, each ``(num_bin, nparams)``.
        """
        params_add_tmp = self.xp.asarray(self.xp.atleast_2d(params_add)).copy()
        params_remove_tmp = self.xp.asarray(self.xp.atleast_2d(params_remove)).copy()
        assert params_add_tmp.shape == params_remove_tmp.shape, (
            f"params_add {params_add_tmp.shape} != "
            f"params_remove {params_remove_tmp.shape}"
        )
        num_bin = params_add_tmp.shape[0]
        nparams = int(self._NPARAMS)
        assert params_add_tmp.shape[1] == nparams, (
            f"params has {params_add_tmp.shape[1]} columns, expected {nparams}")

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            for p_tmp in (params_add_tmp, params_remove_tmp):
                lam = p_tmp[:, -2].copy()
                beta = p_tmp[:, -1].copy()
                lam, beta = ecliptic_to_icrs(lam, beta)
                p_tmp[:, -2] = lam
                p_tmp[:, -1] = beta

        wdm_holder = self._as_wdm_holder(wdm_holder)
        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_add_in    = params_add_tmp.flatten().copy()
        params_remove_in = params_remove_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._swap_layer_groups(
                params_add_tmp, params_remove_tmp,
                group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_swap_groups(num_bin)

        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            grad_add_out    = np.zeros(num_bin * nparams, dtype=np.float64)
            grad_remove_out = np.zeros(num_bin * nparams, dtype=np.float64)
        else:
            grad_add_out    = self.xp.zeros(num_bin * nparams,
                                            dtype=self.xp.float64)
            grad_remove_out = self.xp.zeros(num_bin * nparams,
                                            dtype=self.xp.float64)

        self._kernel("swap_ll_grad")(
            grad_add_out, grad_remove_out,
            self.cpp_orbits, self.cpp_tdi_config,
            params_add_in, params_remove_in,
            data_index, noise_index,
            *self._geometry_kernel_args(),
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            self.xp.asarray(groups["binary_perm"],  dtype=np.int32),
            self.xp.asarray(groups["group_starts"], dtype=np.int32),
            self.xp.asarray(groups["group_ends"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_lo"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
            self.xp.asarray(groups["pair_m_lo_b"],  dtype=np.int32),
            self.xp.asarray(groups["pair_m_hi_b"],  dtype=np.int32),
        )

        grad_add    = self.xp.asarray(grad_add_out).reshape(num_bin, nparams)
        grad_remove = self.xp.asarray(grad_remove_out).reshape(num_bin, nparams)
        return grad_add, grad_remove

    def fill_global_wdm(self, params, templates,
                        convert_to_ra_dec: Optional[bool] = None,
                        data_index=None, factors=None,
                        grid_dim: int = 0,
                        m_band_half_width: int = 1,
                        band_slab_Nf: Optional[int] = None,
                        slab_min_f=None):
        """Scatter per-source chunked-heterodyne WDM templates into a global buffer.

        Argument order matches the other ``*_wdm`` methods on this
        class: ``params`` (1D or 2D ndarray; lists auto-promoted via
        ``atleast_2d`` internally) is the first positional argument,
        followed by the output ``templates`` buffer.

        ``templates`` may be supplied as ``(nchannels, Nf, Nt)``,
        ``(num_templates, nchannels, Nf, Nt)``, or already flattened to
        ``(num_templates * nchannels * Nf * Nt,)``. With
        ``force_backend='jax'`` it must be a *numpy* array (not jnp),
        since JAX arrays are immutable -- the kernel writes back into
        the numpy buffer to honour the C++ in-place contract.

        The buffer may also be an AnalysisContainerArray (single shard; writes
        ``linear_data_arr[0]``), one AnalysisContainer, or a
        :class:`~lisatools.domains.DomainBase` (its ``arr``). The layout is the full
        grid ``(nchannels, Nf, Nt)`` (absolute ``(m, n)``) or, when the settings
        carry a restricted band, the active box ``(nchannels, Nf_active,
        Nt_active)``; it is picked from the buffer size. An empty ``params`` batch
        returns immediately.

        Args:
            params: ``(num_bin, nparams)`` array (1D auto-promoted).
            templates: output buffer (see shapes above). Pre-zero
                before calling -- the kernel accumulates.
            convert_to_ra_dec: **Deprecated** legacy ecliptic-to-ICRS
                conversion for the last two parameter columns; default
                ``None`` consumes them in the orbits frame directly.
            data_index: per-binary slab index into ``templates``.
                Default = all zeros (single shared template slab).
            factors: per-binary multiplicative factor at the
                accumulation step. Default ``+1``; pass ``-1`` to remove.
            grid_dim: CUDA launch grid size (use 0 to default to
                ``n_chunks``).
            m_band_half_width: Layers written on each side of the carrier.
                Default ``1``.
            band_slab_Nf: Narrow per-band slab width in layers (C++ only); the
                active layout then spans ``band_slab_Nf`` layers. Default ``None``.
            slab_min_f: Per-slot first layer of the narrow slabs. Default ``None``.

        Raises:
            NotImplementedError: For a multi-shard holder.
            AssertionError: For a buffer size matching neither layout, a wrong
                parameter count, or a buffer on the wrong array module.
            ValueError: For a buffer of unsupported dimensionality.
        """
        # Call-time holder acceptance (sprint rule, 2026-07): the fill
        # target may be an AnalysisContainerArray (incl. sub-band buffers),
        # a lone AnalysisContainer, or a DomainBase child (e.g. WDMSignal)
        # -- resolved here to the flat/shaped array the kernel accumulates
        # into. Raw ndarrays keep the legacy path below.
        from .domains import DomainBase as _DomainBase
        if hasattr(templates, "linear_data_arr"):
            if len(templates.linear_data_arr) != 1:
                raise NotImplementedError(
                    "fill_global_wdm writes into linear_data_arr[0] and is "
                    "single-shard by contract. For a multi-shard ACA either "
                    "go through the LAT shard router (_RoutedBandEngine "
                    "fill_template) or gather -> fill -> "
                    "scatter_linear_data_arr (init-time pattern)."
                )
            templates = templates.linear_data_arr[0]
        elif isinstance(templates, _DomainBase):
            templates = templates.arr
        else:
            from .analysiscontainer import AnalysisContainer as _AC
            if isinstance(templates, _AC):
                templates = self._as_wdm_holder(templates).linear_data_arr[0]

        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            assert isinstance(templates, np.ndarray), (
                "On the JAX backend, ``templates`` must be a numpy "
                "ndarray (not jnp.ndarray). The kernel writes back to "
                "it in place to match the C++ contract.")
        else:
            assert isinstance(templates, self.xp.ndarray)

        # Choose the output layout from the buffer: full parent grid (dense,
        # absolute (m, n)) or the restricted active band (Nf_active, Nt_active,
        # written at (m - ind_min_f, n - ind_min_t) via active_band=True). The
        # cpp_wdm_settings already carries the real band, so active mode only
        # flips the write addressing. Dense is preferred when the buffer is
        # full-grid-sized; active when this comp carries a restricted band AND
        # the buffer is the smaller active size (the global-fit settings path).
        ws = self.wdm_settings
        # Task-b: a narrow per-band slab makes the active layout span band_slab_Nf
        # layers instead of the full Nf_active. The per-template flat size + the
        # (nchannels, Nfa, Nta) shape check below key off this reduced extent.
        Nfa = int(band_slab_Nf) if band_slab_Nf else int(getattr(ws, "Nf_active", self.Nf))
        Nta = int(getattr(ws, "Nt_active", self.Nt))
        restricted = (Nfa < self.Nf) or (Nta < self.Nt)
        dense_per, active_per = self.nchannels * self.Nf * self.Nt, self.nchannels * Nfa * Nta
        active_band = False
        if templates.ndim == 1:
            flat = int(templates.shape[-1])
            if flat % dense_per == 0:
                per_template = dense_per
            elif restricted and flat % active_per == 0:
                active_band, per_template = True, active_per
            else:
                raise AssertionError(
                    f"templates flat size {flat} matches neither dense "
                    f"(nchannels*Nf*Nt={dense_per}) nor active "
                    f"(nchannels*Nf_active*Nt_active={active_per}).")
            num_templates = int(flat // per_template)
        elif templates.ndim in (3, 4):
            shp = tuple(templates.shape[-3:])
            if shp == (self.nchannels, self.Nf, self.Nt):
                active_band = False
            elif restricted and shp == (self.nchannels, Nfa, Nta):
                active_band = True
            else:
                raise AssertionError(
                    f"templates trailing shape {shp} matches neither dense "
                    f"({self.nchannels}, {self.Nf}, {self.Nt}) nor active "
                    f"({self.nchannels}, {Nfa}, {Nta}).")
            num_templates = 1 if templates.ndim == 3 else int(templates.shape[0])
        else:
            raise ValueError(
                "templates must be 3D (nchannels, Nf, Nt), 4D "
                "(num_templates, nchannels, Nf, Nt), or flat 1D.")

        params_tmp = self.xp.atleast_2d(self.xp.asarray(params)).copy()
        num_bin = params_tmp.shape[0]
        # Empty batch (e.g. a sub-band cell with no live sources to
        # inject): nothing to accumulate -- and the data_index bounds
        # check below cannot reduce over an empty array.
        if num_bin == 0 or params_tmp.size == 0:
            return
        nparams = int(self._NPARAMS)
        assert params_tmp.shape[1] == nparams, (
            f"params has {params_tmp.shape[1]} columns, expected {nparams}")

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        params_in = params_tmp.flatten().copy()

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if _GB_INDEX_ASSERTS:
            assert int(data_index.max()) < num_templates

        # factor +1 (add) / -1 (remove). Mirrors
        # gbgpu.generate_global_template's factors API.
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=self.xp.float64)
        else:
            factors = self.xp.ascontiguousarray(
                self.xp.asarray(factors, dtype=self.xp.float64))
            assert factors.shape == (num_bin,), (
                f"factors must have shape ({num_bin},), got {factors.shape}")

        self._kernel("fill_global")(
            templates,
            self.cpp_orbits, self.cpp_tdi_config,
            self.cpp_wdm_settings,
            params_in, factors,
            data_index,
            # DEVICE-LOCAL (2026-09-16). These were passed RAW -- the comp's
            # home-device arrays -- while the SOBBH move routes ONE SHARED
            # comp through both shards' device contexts
            # (sobbhspecialmove.py::_apply_cold_chain_sources). A device-1
            # launch of wdm_het_fill_global_kernel then dereferenced
            # device-0 pointers: the null run's illegal memory access.
            *self._geometry_kernel_args(),
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            int(m_band_half_width),
            bool(active_band),
            *self._slab_args_from(band_slab_Nf, slab_min_f),
        )

    def get_fstat_ll_wdm(self, params, wdm_holder,
                          data_index=None, noise_index=None,
                          convert_to_ra_dec: Optional[bool] = None,
                          grid_dim: int = 0,
                          m_band_half_width: int = 1,
                          fstat_fold: Optional[int] = None):
        """Chunked-heterodyne F-stat over the WDM domain.

        Builds the 4 Cornish & Crowder '05 basis filters per binary at the
        binary's (f0, fdot, lam, beta) with fixed ``A = 2``, ``iota = pi/2``,
        ``psi = {0, pi/4, 0, pi/4}`` and ``phi0 = {0, pi, 3*pi/2, pi/2}``.
        The narrow band is ``m_floor +/- m_band_half_width`` per binary (no layer
        groups).

        Compute ``F = N^T M^{-1} N / 2`` from these on the Python side
        (numpy.linalg.solve / inv).

        WDM coefficients are real -> N and M are real-valued; the imag
        parts returned by the C++ kernel are always 0 and dropped here. Both are
        also stored on ``self.N_arr`` / ``self.M_mat``.

        Args:
            params: ``(num_bin, nparams)`` array of GB parameters.
            wdm_holder: ``AnalysisContainerArray`` providing
                ``linear_data_arr[0]`` (WDM data) and
                ``linear_psd_arr[0]`` (invC).
            data_index, noise_index: per-binary slab indices into
                ``wdm_holder``.
            convert_to_ra_dec: Deprecated ecliptic -> ICRS conversion of the last
                two columns. Default ``None`` (off).
            grid_dim: CUDA launch grid size (0 -> use num_bin).
            m_band_half_width: layer-band half-width around f0
                (m_floor +/- m_band_half_width, 2*m_band_half_width+1
                layers total per binary). Default 1.
            fstat_fold: basis-filter fold. ``None`` (default) uses
                :attr:`fstat_fold`, itself seeded from ``GB_FSTAT_FOLD``
                (default 1 = ON; set on the 9-month / 12-month runs). 0 generates
                all 4 filters independently;
                1 generates only the 2 psi stages and recovers the
                quadrature partners by an exact constant rotation -- same
                (N, M) to floating-point reassociation, half the waveform
                work. Ignored unless ``_FSTAT_FOLD_KERNELS`` is set (i.e.
                unless this source class's binding TU accepts the argument).

        Returns:
            tuple: ``(N_arr, M_mat)``: ``N_arr`` of shape ``(num_bin, 4)``, the
            per-binary ``<d | A_i>`` for i in 0..3; ``M_mat`` of shape
            ``(num_bin, 10)``, the per-binary upper-triangle ``<A_i | A_j>`` for
            i <= j, flattened in row-major order ``[M00, M01, M02, M03, M11, M12,
            M13, M22, M23, M33]``.
        """
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]
        nparams = int(self._NPARAMS)

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        # Outputs (kernel writes both re + im; we drop im since WDM is real).
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            N_re = np.zeros((num_bin, 4))
            N_im = np.zeros((num_bin, 4))
            M_re = np.zeros((num_bin, 10))
            M_im = np.zeros((num_bin, 10))
        else:
            N_re = self.xp.zeros((num_bin, 4))
            N_im = self.xp.zeros((num_bin, 4))
            M_re = self.xp.zeros((num_bin, 10))
            M_im = self.xp.zeros((num_bin, 10))

        wdm_holder = self._as_wdm_holder(wdm_holder)
        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_in = params_tmp.flatten().copy()

        # F-stat has its own narrow-band path inside the kernel; the
        # layer-grouping interface is not used here. Route through the
        # standard ``_kernel(name)`` so SOBBHWDMComputations picks up the
        # ``sobbh_`` prefix automatically.
        self._kernel("get_fstat_ll")(
            N_re.reshape(-1), N_im.reshape(-1),
            M_re.reshape(-1), M_im.reshape(-1),
            self.cpp_orbits, self.cpp_tdi_config,
            self.cpp_wdm_settings,
            params_in, data_index, noise_index,
            *self._geometry_kernel_args(),
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            int(self.backend.TDITypeDict[self.tdi_type]),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(m_band_half_width),
            *self._slab_kernel_args(wdm_holder),
            *self._fstat_fold_kernel_args(fstat_fold),
            *self._fstat_orbit_cache_kernel_args(),
            *self._psd_kernel_args(wdm_holder),
        )

        # WDM coefs are real -> imag is identically 0.
        self.N_arr = N_re
        self.M_mat = M_re
        return N_re, M_re

    # GB-oriented default finite-difference steps (per physical parameter),
    # matching GBFDComputations._DEFAULT_PARAM_EPS. SOBBH (11 params) callers
    # should pass ``param_eps`` explicitly; extra columns fall back to 1e-6.
    _DEFAULT_PARAM_EPS = (
        1.0e-25,   # amp
        2.0e-14,   # f0    (Hz)
        1.0e-21,   # fdot  (Hz/s)
        1.0e-28,   # fddot (Hz/s^2)
        1.0e-6,    # phi0
        1.0e-6,    # iota
        1.0e-6,    # psi
        1.0e-6,    # lam
        1.0e-6,    # beta
    )

    def _info_matrix_param_eps(self, nparams, param_eps):
        """Finite-difference steps: ``param_eps``, else the padded ``_DEFAULT_PARAM_EPS``."""
        if param_eps is not None:
            eps = self.xp.asarray(param_eps, dtype=self.xp.float64)
            assert eps.shape[0] == nparams, (
                f"param_eps length {eps.shape[0]} != nparams {nparams}")
            return eps
        base = list(self._DEFAULT_PARAM_EPS)
        if nparams > len(base):
            base = base + [base[-1]] * (nparams - len(base))
        return self.xp.asarray(base[:nparams], dtype=self.xp.float64)

    def information_matrix(self, params, wdm_holder,
                           inds=None, param_eps=None,
                           noise_index=None,
                           convert_to_ra_dec: Optional[bool] = None,
                           **swap_kwargs):
        """Per-source information matrix over the WDM domain.

        Single-template parity with ``GBGPU.information_matrix``. The information matrix
        ``Gamma_ij = <dh/dtheta_i | dh/dtheta_j>`` is assembled from
        central-difference cross inner products ``<h_a | h_b>``, which
        :meth:`get_swap_ll_wdm` returns as its ``ar`` term with the canonical
        WDM inner-product normalization -- so the matrix matches the WDM
        likelihood convention exactly, with no factor re-derivation. With
        ``theta_i^+/- = theta +/- eps_i e_i`` and 2nd-order central
        differences::

            Gamma_ij = [ar(i+,j+) - ar(i+,j-) - ar(i-,j+) + ar(i-,j-)]
                       / (4 eps_i eps_j).

        Cost: 4 ``get_swap_ll_wdm`` launches per unique (i, j) pair; heavier
        than the FD path but correct-by-construction. (A direct dh-buffer
        variant via :meth:`fill_global_wdm` can replace this once its inner
        product is factor-calibrated against ``<h|h>``.)

        Args:
            params: ``(num_bin, nparams)`` GB parameters (1D auto-promoted).
            wdm_holder: ``AnalysisContainerArray`` (or a single
                ``AnalysisContainer``, wrapped under the hood) providing invC.
            inds: parameter indices to include (default all ``nparams``).
            param_eps: per-parameter finite-difference step (default the
                GB-oriented table; SOBBH callers should pass this).
            noise_index: per-binary index into the invC slabs (also used as the
                data index, which the layer grouping requires).
            convert_to_ra_dec: Deprecated ecliptic -> ICRS conversion of the last
                two columns. Default ``None`` (off).
            **swap_kwargs: forwarded to :meth:`get_swap_ll_wdm`
                (``grid_dim``/``use_layer_groups``/``group_band_layers``/
                ``margin_layers``/``m_band_half_width``).

        Returns:
            ``xp.ndarray``: ``(num_bin, num_derivs, num_derivs)`` real float64
            information matrices, ``num_derivs = len(inds)``.

        Raises:
            AssertionError: If ``params`` or ``param_eps`` has the wrong length.
        """
        p = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin, nparams = p.shape[0], int(self._NPARAMS)
        assert p.shape[1] == nparams, (
            f"params has {p.shape[1]} columns, expected {nparams}")
        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam = p[:, -2].copy(); beta = p[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            p[:, -2] = lam; p[:, -1] = beta

        if inds is None:
            inds = list(range(nparams))
        else:
            inds = [int(i) for i in self.xp.asarray(inds).tolist()]
        num_derivs = len(inds)
        eps = self._info_matrix_param_eps(nparams, param_eps)

        plus, minus = [], []
        for ind in inds:
            ei = float(eps[ind])
            up = p.copy(); up[:, ind] += ei
            dn = p.copy(); dn[:, ind] -= ei
            plus.append(up); minus.append(dn)

        def _ar(pa, pb):
            # <h(pa) | h(pb)> per binary (index 6 of the get_swap_ll_wdm tuple).
            # The information matrix only consumes template cross terms, so the data slab
            # is irrelevant -- but the layer-grouping path requires
            # data_index == noise_index, so align them.
            out = self.get_swap_ll_wdm(
                pa, pb, wdm_holder,
                data_index=noise_index, noise_index=noise_index,
                **swap_kwargs)
            return self.xp.asarray(out[6]).real

        info = self.xp.zeros((num_bin, num_derivs, num_derivs),
                             dtype=self.xp.float64)
        for i in range(num_derivs):
            ei = float(eps[inds[i]])
            for j in range(i, num_derivs):
                ej = float(eps[inds[j]])
                g = (_ar(plus[i],  plus[j])
                     - _ar(plus[i],  minus[j])
                     - _ar(minus[i], plus[j])
                     + _ar(minus[i], minus[j])) / (4.0 * ei * ej)
                info[:, i, j] = g
                info[:, j, i] = g
        return info


