"""Stock variant ``gb_no_fg``: GB-only global fit, no foreground fitting.

The installed version of ``global_fit_input/gb_no_foreground_global_fit_settings.py``:

* **No galactic-foreground branch** — the foreground is not fit.
* **Fixed PSD** — no ``psd`` branch; the sensitivity is a fixed
  :class:`InstrumentNoise` from ``general.fixed_psd_params`` via the
  engine's no-psd-branch path. Default levels are read off the mojito
  NOISE brick's tabulated estimates when the run uses mojito data
  (``psd_from_noise_file`` auto; stock analytic ``[15e-12, 3e-15]``
  otherwise).
* **Frequency band restricted to f > 6 mHz** so the unresolved galactic
  confusion is out of band — no foreground model needed anywhere.
* Data: the mojito L1 GB galaxy (``catalogue["GB"]`` populated so leaves
  can start at true catalogue points, SNR > 3 subset).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(nwalkers=4, file_base_name="my_run")
    fit.gb.min_freq = 9.8e-3           # direct GB band bounds (primary
    fit.gb.max_freq = 10.4e-3          #   interface; snapped to WDM layers)
    fit.gb.center_freq = 8.0e-3        # or: center/n_layers (secondary)
    fit.add_move("rj_fstat_mcmc", branch="gb", stage="gb_pe")
    fit.build()                        # heavy: loads + pours the data
    fit.run()                          # or hand it to run_global.py / GlobalFit

Every env var the legacy file honored (``TOBS_TARGET``, ``NWALKERS``,
``GB_MODE``, ``GB_DEBUG``, ``CHUNKED_*``, ...) still works: it feeds the
matching knob's default at construction (explicit kwargs win).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import typing
from copy import deepcopy

import numpy as np

from gbgpu.utils.utility import get_fdot

from lisatools.domains import FDSettings, WDMSettings

from ....engine import GeneralSetup, Settings
from ....moves import Move, MoveBuildContext
from ....recipe import (
    MOJITO_REFERENCE_TIME,
    Recipe,
    Stage,
    build_gb_moves,
    gb_catalogue_to_sampling_basis,
    select_gb_injection_subset_by_snr,
    setup_state_for_injection,
    subtract_gb_neighbors_from_data,
)
from ...base import env_default, env_resolve
from ..common import tdi_generation_info
from ..fit import EreborFit, EreborGeneralSettings
from ..gb import GBSettings, GBSetup

logger = logging.getLogger(__name__)

_DELTA_SAFE = 1e-5

# GB phase/frequency reference epoch = the mojito catalogue epoch
# (``MOJITO_REFERENCE_TIME`` = TimeReferenceSSBFrame). Validated convention
# from scripts/gb/gb_mojito_match.py (params at REF, phys phi0 = +TrueAnomaly).
GB_MOJITO_T_REF = 97729089.327664

# GB_DEBUG=1 preset: the fast smoke configuration. Seeded through
# ``os.environ.setdefault`` so every knob keeps its usual explicit env
# override — identical semantics to the legacy settings file.
_DEBUG_ENV_PRESET = {
    "TOBS_TARGET": str(3 * 86400.0),  # 3 days
    "NWALKERS": "3",
    "GB_NTEMPS": "2",
    "CHUNKED_NT_SUB": "64",
    "CHUNKED_N_PAD": "8",
    "CHUNKED_N_SPARSE": "64",
    "CHUNKED_N_CP_SIG": "16",
    "CHUNKED_N_CP_ORBIT": "16",
    "NUM_ITERATIONS": "4",
}


def _default_phi0_lims():
    return [0.0, 2 * np.pi]


def _default_iota_lims():
    return [0.0 + _DELTA_SAFE, np.pi - _DELTA_SAFE]


def _default_psi_lims():
    return [0.0, np.pi]


def _default_alpha_lims():
    return [0.0, 2 * np.pi]


def _default_delta_lims():
    return [-np.pi / 2.0 + _DELTA_SAFE, np.pi / 2.0 - _DELTA_SAFE]


def _default_gb_periodic():
    # Sampling-basis periodic parameters, keyed by plain names (run.py
    # translates names -> indices through the branch transform).
    return {"gb": {"phi0": 2 * np.pi, "psi": np.pi, "alpha": 2 * np.pi}}


@dataclasses.dataclass
class GBNoFgGBSettings(GBSettings):
    """GB branch block for ``gb_no_fg``: stock GBSettings + this variant's knobs.

    Band-derivation fields (``center_freq``/``n_layers``) and chunked-het
    kernel sizes are resolved into the underlying :class:`GBSettings`
    fields / the runtime likelihood object at build time. Env-backed
    defaults mirror the legacy settings file's env vars.
    """

    # -- parameter limits (prior ranges) --
    A_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [7e-26, 1e-19]
    )
    m_chirp_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.001, 1.0]
    )
    phi0_lims: typing.List[float] = dataclasses.field(default_factory=_default_phi0_lims)
    iota_lims: typing.List[float] = dataclasses.field(default_factory=_default_iota_lims)
    psi_lims: typing.List[float] = dataclasses.field(default_factory=_default_psi_lims)
    alpha_lims: typing.List[float] = dataclasses.field(default_factory=_default_alpha_lims)
    delta_lims: typing.List[float] = dataclasses.field(default_factory=_default_delta_lims)
    # f0_lims / fdot_lims: derived at build from the data band + m_chirp_lims
    # unless set explicitly.

    # -- shape / sampling --
    # None (default) -> resolved at build in ``prepare_gb_branch`` to 8, or
    # 9 when ``use_fdot_astro`` (use_chirp_mass and
    # use_astrophysical_f0_mc_prior) appends the ``fdot_astro_ratio``
    # column. Explicit int wins (e.g. a downstream override).
    ndim: typing.Optional[int] = None
    nleaves_min: int = 0
    # None (default) -> resolved at build to 2x the number of catalogue
    # sources inside the SAMPLED central band (the f0 prior span). The RJ
    # move visits every leaf slot once per proposal (one pick round each:
    # alive slots get death proposals, dead slots get births), so
    # nleaves_max directly sets the proposal's sequential depth; sizing it
    # to the local catalogue keeps RJ birth headroom without paying for a
    # tail of ~always-empty slots. GB_NLEAVES_MAX / explicit kwarg wins.
    nleaves_max: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GB_NLEAVES_MAX", None, int)
    )
    num_repeat_proposals: int = dataclasses.field(
        default_factory=env_default("GB_NUM_REPEAT_PROPOSALS", 100, int)
    )
    periodic: dict = dataclasses.field(default_factory=_default_gb_periodic)
    # Phase/frequency reference epoch. None -> resolved at build from the
    # data mode: the mojito catalogue epoch in mojito mode, the synthetic
    # stream start in synthetic mode.
    t0: typing.Optional[float] = None
    extra_buffer: int = 5
    start_freq_ind: int = 0
    # ``oversample=None`` -> auto at build: 2 on FD (halves per-band N at
    # short Tobs), 4 on WDM.
    oversample: typing.Optional[int] = None

    # -- narrow GB band, decoupled from the (wide) data band --
    # PRIMARY interface: direct band bounds (Hz). The GB band becomes
    # [min_freq, max_freq] snapped INWARD to WDM layer boundaries, with one
    # sub-band separator on every layer boundary in between (>= 3 snapped
    # layers required so an interior sampled span exists). Must lie inside
    # the data band (general min_freq/max_freq). The defaults reproduce the
    # legacy 3-layer band around 7.5 mHz on typical grids. On the WDM path
    # the bounds resolve into equivalent canonical (center_freq, n_layers)
    # at build so every downstream consumer (band edges, f0 prior,
    # data-band clipping, dynamic nleaves_max, debug plots) stays
    # consistent.
    min_freq: float = dataclasses.field(
        default_factory=env_default("GB_MIN_FREQ", 7.36e-3, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("GB_MAX_FREQ", 7.78e-3, float)
    )
    # SECONDARY interface: center + layer count, in integer WDM-layer units
    # around ``center_freq`` (the samplable interior layer CONTAINS
    # center_freq; floor, not round). Used only when EITHER field is set
    # explicitly (kwarg or GB_CENTER_FREQ / GB_N_LAYERS env) — the unset one
    # fills from the legacy defaults (7.5e-3 / 3) — and then it overrides
    # min_freq/max_freq.
    center_freq: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_CENTER_FREQ", None, float)
    )
    n_layers: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GB_N_LAYERS", None, int)
    )
    # GB_HIGHEST_FREQUENCIES (N, 1-based): center the GB band on the N-th
    # HIGHEST-frequency catalogue source (1 = highest, 2 = 2nd highest, ...),
    # ``n_layers`` wide. None (default) leaves the band at min_freq/max_freq
    # (or center_freq/n_layers). When set it OVERRIDES those, resolving the
    # source f0 from the loaded catalogue at build. General replacement for
    # the debug runner's hardcoded per-source dicts.
    highest_frequencies: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GB_HIGHEST_FREQUENCIES", None, int)
    )
    # DATA_BAND_LAYERS (memory knob): clip the DATA band — and every
    # per-walker ACA slab — to +-N layers around the GB band. None = full band.
    data_band_layers: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("DATA_BAND_LAYERS", None, int)
    )
    # FD basis only: GB band width in Hz centered on center_freq (None ->
    # the full data band so several bands survive the band-walker trim).
    fd_bandwidth: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_FD_BANDWIDTH", None, float)
    )

    # -- run-mode knobs --
    # mode="pe": leaves start AT the true catalogue points (SNR > 3 subset).
    # mode="search": zero-knowledge start; per-band progressive leaf caps.
    mode: str = dataclasses.field(default_factory=env_default("GB_MODE", "pe", str))
    # start="prior": PE dimensionality without truth seeding.
    start: str = dataclasses.field(default_factory=env_default("GB_START", "truth", str))
    snr_threshold: float = 3.0
    # Focused-central-band mode: subtract neighboring-layer catalogue
    # sources from the data as KNOWN signals.
    subtract_neighbors: bool = dataclasses.field(
        default_factory=env_default("GB_SUBTRACT_NEIGHBORS", False, bool)
    )
    neighbor_window_layers: int = 4
    # Opt-in generalization of the above: subtract EVERY catalogue source
    # OUTSIDE the sampled band (``f0_lims``) as a known signal, over the
    # whole remaining data band. Default OFF -- out-of-band sources stay in
    # the data. Supersedes ``subtract_neighbors`` when on.
    subtract_out_of_band: bool = dataclasses.field(
        default_factory=env_default("GB_SUBTRACT_OUT_OF_BAND", False, bool)
    )

    # -- chunked-het kernel sizes (WDM likelihood; validated defaults) --
    nt_sub: int = dataclasses.field(default_factory=env_default("CHUNKED_NT_SUB", 256, int))
    n_pad: int = dataclasses.field(default_factory=env_default("CHUNKED_N_PAD", 32, int))
    n_sparse: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_SPARSE", 256, int)
    )
    n_cp_sig: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_SIG", 48, int)
    )
    n_cp_orbit: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_ORBIT", 32, int)
    )
    # In-move F-stat grid fit (GBSpecialRJFStatGridMove). Opt-in: when set,
    # the RJ birth move fits its own comb/peak grids inside setup() against
    # the live residual on its first hit, so a run needs no offline
    # plot_fstat_proposal_mojito.py prep. Scalars only (deepcopy/pickle-safe).
    # Pure in-model move inserted between the fstat-birth and removal moves
    # (GB_MODE=search only). Off by default -- opt in with GB_SEARCH_IN_MODEL=1.
    search_in_model: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_IN_MODEL", False, bool)
    )
    fstat_fit_in_move: bool = dataclasses.field(
        # Default ON (2026-08-12 user ruling): the in-move fit against the
        # live residual is the production birth-grid path.
        default_factory=env_default("GB_FSTAT_FIT_IN_MOVE", True, bool)
    )
    fstat_refit_every: int = dataclasses.field(
        default_factory=env_default("GB_FSTAT_REFIT_EVERY", 0, int)
    )
    fstat_fit_dir: str = dataclasses.field(
        default_factory=env_default("GB_FSTAT_FIT_DIR", "", str)
    )
    # Sig-het in-model scoring (chunked-het delegate for RJ/fills/swaps).
    # TODO(sighet cleanup): these three knobs are the temporary switch for the
    # signal-heterodyne in-model likelihood. Sig-het is CPU-only right now
    # (GBSignalHetComputations.supported_backends()==["cpu"]; the CUDA kernels
    # in gb_tdi_on_the_fly.cu still `#ifdef __CUDACC__ throw` -- see
    # gb_signal_het_get_ll_wrap ~L1878). Once the sig-het GPU kernels land,
    # revisit: promote sighet_inmodel to the default in-model path (the
    # compression is the point) and fold nt_layer / n_sparse_fd into the
    # validated chunked-het size block above rather than a separate knob group.
    sighet_inmodel: bool = dataclasses.field(
        # Default ON (2026-08-12 user ruling): sig-het v5 is the production
        # in-model likelihood; the WDMSettings isinstance guard keeps FD
        # runs on their own path.
        default_factory=env_default("GB_SIGHET_INMODEL", True, bool)
    )
    sighet_nt_layer: int = dataclasses.field(
        default_factory=env_default("SIGHET_NT_LAYER", 64, int)
    )
    sighet_n_sparse_fd: int = dataclasses.field(
        default_factory=env_default("SIGHET_N_SPARSE_FD", 1024, int)
    )
    # Heterodyne-ratio magnitude clip; 0 = disabled (the kernel's FLOOR_EPS
    # guards the divide-by-small). A positive value silently saturates any
    # candidate whose amplitude ratio vs the fixed reference exceeds it —
    # diagnostic only, never for sampling runs.
    sighet_max_r: float = dataclasses.field(
        default_factory=env_default("SIGHET_MAX_R", 0.0, float)
    )
    # Control-point count for the sig-het spline waveform build (shared
    # machinery with chunked-het's N_cp_sig splines). -1 = AUTO (one node
    # per ~4 days of Tobs, clamped [32, 256]); 0 = direct per-point
    # evaluation (legacy); >1 = explicit count.
    sighet_n_cp: int = dataclasses.field(
        default_factory=env_default("SIGHET_N_CP", -1, int)
    )
    # Signal-het V3: ratio-spline candidate build (r modeled directly from
    # n raw evals per candidate; no FFT/polyphase/division). 0 = off (v2
    # exact build); >0 = fixed node count; -1 = ADAPTIVE from the batch's
    # predicted displacement (prototype-calibrated policy, clip [8, 64]).
    sighet_v3_nodes: int = dataclasses.field(
        # 64 = the full-frontier-validated node count (32 blows up at
        # 6mo/1yr; 128 at 4yr). Default ON with the v5 stack (2026-08-12).
        default_factory=env_default("SIGHET_V3_NODES", 64, int)
    )
    # Signal-het V4: the fitted ratio is resampled onto ``sighet_v4_knots``
    # FIXED, candidate-independent knots as linear complex values before the
    # pixel-time evaluation -- the representation that makes the fold a fixed
    # linear operator.  0 = off (v3/v2 path).  128 is the measured-converged
    # value; 64 is lossy, 256 buys nothing.  Requires sighet_v3_nodes > 0
    # (v4 reuses the v3 node fit).
    sighet_v4_knots: int = dataclasses.field(
        # 128 = measured-converged (64 lossy REJECTED, 256 buys nothing).
        default_factory=env_default("SIGHET_V4_KNOTS", 128, int)
    )
    # Sig-het REFERENCE-BUILD Tukey alpha, pinned EXPLICITLY (user ruling
    # 2026-08-19: equivalent tukey alphas across chunked-het / sig-het /
    # TD->FD; never inherit a knob-resolved value). The engine kernels apply
    # this as a fraction of the WHOLE OBSERVATION, unlike the chunked
    # delegate's per-chunk stitching alpha -- inheriting the delegate's 0.05
    # tapered 54 layers against the 20-layer time crop and suppressed the
    # reference inside the active region (the flat ~1% h_h bias = the
    # dissect's high-f 0.984). 0.01 = the recommended HET_WIDE value: taper
    # 11 layers, fully subsumed by the crop (EC >= taper); measured EXACT
    # (hh ratio 1.0000) in gb_sighet_invitro_probe.py.
    sighet_tukey_alpha: float = dataclasses.field(
        default_factory=env_default("SIGHET_TUKEY_ALPHA", 0.01, float)
    )
    # V4 evaluation mode: 0 = cooperative fixed-knot spline solve (PCR on
    # GPU); >0 = precomputed cardinal weights with this half-band -- no
    # solve, no block syncs, ~18 KB less shared memory.  Banded and PCR agree
    # to 1e-11 relative at half-band 16 and banded is never slower.
    # Signal-het V5: v4-banded with the per-candidate fold scratch relocated
    # out of shared memory. IDENTICAL arithmetic and ordering to v4-banded
    # (bit-identical likelihoods, verified in the engine bench-off); it only
    # changes where scratch lives, buying occupancy. 0 = off (v4 path),
    # 1 = phase-aliased shared arena, 2 = flat carve (the occupancy control).
    # Requires sighet_v4_knots > 0. Env: SIGHET_V5.
    sighet_v5: int = dataclasses.field(
        # v5 mode 1 = bit-identical to v4-banded, occupancy win.
        default_factory=env_default("SIGHET_V5", 1, int)
    )
    sighet_v4_band: int = dataclasses.field(
        # 16 = banded-gate default (1e-11 vs PCR, never slower).
        default_factory=env_default("SIGHET_V4_BAND", 16, int)
    )
    # NOTE: no __post_init__ here — Setup.__init__ re-runs this dataclass's
    # __init__ on the (non-dataclass) GBSetup instance, which cannot resolve
    # dataclass hooks. Value validation happens in prepare_branch_settings.


@dataclasses.dataclass
class GBNoFgGeneralSettings(EreborGeneralSettings):
    """General block for ``gb_no_fg`` (defaults mirror the legacy file)."""

    # foreground-free band restriction; env-backed so MIN_FREQ/MAX_FREQ
    # widen the data band (they were plain defaults before, which silently
    # ignored the env vars while the GB band guard told users to set them)
    min_freq: float = dataclasses.field(
        default_factory=env_default("MIN_FREQ", 6e-3, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("MAX_FREQ", 2.5e-2, float)
    )
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_gb_no_fg/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "gb_no_fg_test_2")
    )
    source_types: typing.Tuple[str, ...] = ("GB",)
    # None -> auto at build: [Soms_d, Sa_a] fit to the mojito NOISE brick's
    # tabulated estimates when available (mojito data mode; see
    # ``psd_from_noise_file`` / ``noise_file``), else the stock analytic
    # levels [15e-12, 3e-15]. An explicit list always wins.
    fixed_psd_params: typing.Optional[typing.List[float]] = None
    # Fixed PSD (no psd branch) -> report source-only log L = -1/2 <r|r>
    # (drop the constant -sum(log|detC|) noise normalization term).
    likelihood_source_only: bool = True
    # data_mode="synthetic" injects these GB rows in-process instead of
    # loading the mojito galaxy: ``(num_sources, 9)`` in the GBGPU basis
    # ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``. None -> the
    # stock two-source table (injections.GB_INJECTION_PARAMS).
    gb_injection_params: typing.Optional[typing.Any] = None


# ============================================================
# Shared gb_no_fg-style GB branch setup (module-level so all_sources reuses it)
# ============================================================


def _band_klohi(gb, layer_df: float) -> typing.Tuple[int, int]:
    """Resolved (k_lo, k_hi) GB-band layer bounds (pure; no mutation)."""
    if gb.center_freq is not None or gb.n_layers is not None:
        center = gb.center_freq if gb.center_freq is not None else 7.5e-3
        n_layers = int(gb.n_layers) if gb.n_layers is not None else 3
        k_center = int(math.floor(center / layer_df))
        k_lo = k_center - n_layers // 2
        return k_lo, k_lo + n_layers
    k_lo = int(math.ceil(float(gb.min_freq) / layer_df))
    k_hi = int(math.floor(float(gb.max_freq) / layer_df))
    if k_hi - k_lo < 3:
        raise ValueError(
            f"GB band [{gb.min_freq:.6e}, {gb.max_freq:.6e}] Hz spans only "
            f"{max(k_hi - k_lo, 0)} whole WDM layer(s) after snapping "
            f"inward (layer_df={layer_df:.4e} Hz); need >= 3 layers so an "
            "interior sampled span exists. Widen min_freq/max_freq."
        )
    return k_lo, k_hi


def _resolve_nleaves_max(gb, general_setup, is_fd, layer_df) -> int:
    """Dynamic RJ leaf budget: 2x the catalogue sources in the sampled band."""
    fallback = 100
    catalogue = (getattr(general_setup, "catalogue", None) or {}).get("GB", {})
    if is_fd or not catalogue:
        logger.info(
            "GB nleaves_max: dynamic sizing unavailable (%s); using the "
            "legacy default %d.",
            "FD basis" if is_fd else "no GB catalogue",
            fallback,
        )
        return fallback

    trim_duration = general_setup.data_t0 - MOJITO_REFERENCE_TIME
    sampling = np.array(
        [
            gb_catalogue_to_sampling_basis(catalogue[k], trim_duration=trim_duration)
            for k in sorted(catalogue.keys())
        ]
    )
    if sampling.ndim == 3:
        sampling = sampling.reshape(-1, sampling.shape[-1])
    f0_hz = np.asarray(sampling[:, 1], dtype=float) * 1e-3
    lo, hi = gb.start_freq + layer_df, gb.end_freq - layer_df
    n_in = int(((f0_hz >= lo) & (f0_hz <= hi)).sum())
    out = max(2 * n_in, 4)
    cap = int(getattr(gb, "nleaves_max_cap", 20000) or 20000)
    if out > cap:
        # Uncapped, the fresh-start prior draw at (ntemps, nwalkers,
        # nleaves_max, ndim) float64 x several copies OOMs the host on a
        # full-band catalogue (83.8 GB peak observed on the 3-month
        # combined run). GB_NLEAVES_MAX_CAP raises the ceiling; setting
        # gb.nleaves_max explicitly bypasses the resolver entirely.
        logger.warning(
            "GB nleaves_max: dynamic sizing 2 x %d = %d exceeds the cap "
            "%d (GB_NLEAVES_MAX_CAP); clamping. Leaf slots cover %.0f%% of "
            "the in-band catalogue.",
            n_in, out, cap, 100.0 * cap / max(n_in, 1),
        )
        out = cap
    logger.info(
        "GB nleaves_max: %d catalogue sources in the sampled band "
        "[%.6e, %.6e] Hz -> nleaves_max = %d (cap %d).",
        n_in, lo, hi, out, cap,
    )
    return out


def _nth_highest_catalogue_f0(mojito_data_path: str, n_th: int) -> float:
    """f0 (Hz) of the N-th HIGHEST-frequency GB in the mojito catalogue file.

    Reads only the frequency column of the GB catalogue HDF5 (in the
    ``catalogues/`` folder, which is separate from the L1 data), so it can run
    during general-settings adjustment -- before the data processor loads the
    galaxy -- to let the DATA band track the chosen source. 1-based N.
    """
    import h5py

    cat_path = os.path.join(
        mojito_data_path, "catalogues", "wdwd_cat_mojito_lite_processed.hdf5"
    )
    if not os.path.exists(cat_path):
        raise ValueError(
            f"GB_HIGHEST_FREQUENCIES needs the GB catalogue file, not found at "
            f"{cat_path!r}."
        )
    with h5py.File(cat_path, "r") as f:
        f0 = np.asarray(
            f["Binaries"]["GW22FrequencySSBFrame"][:], dtype=float
        ).ravel()
    if n_th < 1 or n_th > f0.size:
        raise ValueError(
            f"GB_HIGHEST_FREQUENCIES={n_th} out of range (1..{f0.size})."
        )
    # N-th highest = smallest of the top-N.
    return float(np.sort(f0[np.argpartition(f0, -n_th)[-n_th:]])[0])


def prepare_gb_branch(gb, general_setup, *, data_mode, synthetic_t_start):
    """gb_no_fg-style GB branch prep: band derivation + f0/fdot/nleaves/betas.

    Shared by ``gb_no_fg`` and ``all_sources`` so their GB setup is identical
    (all_sources just passes a GB band down to 0.1 mHz via the branch's
    ``min_freq``/``max_freq``).
    """
    if gb.t0 is None:
        gb.t0 = GB_MOJITO_T_REF if data_mode == "mojito" else float(synthetic_t_start)
    if gb.mode not in ("pe", "search"):
        raise ValueError(f"GB mode must be 'pe' or 'search', got {gb.mode!r}.")
    if gb.start not in ("truth", "prior"):
        raise ValueError(f"GB start must be 'truth' or 'prior', got {gb.start!r}.")
    domain_settings = general_setup.domain_settings

    if not gb.f0_lims:
        gb.f0_lims = [general_setup.min_freq, general_setup.max_freq]
    if not gb.fdot_lims:
        fdot_max_val = get_fdot(gb.f0_lims[-1], Mc=gb.m_chirp_lims[-1])
        gb.fdot_lims = [-fdot_max_val, fdot_max_val]
    if gb.ndim is None:
        # 9-parameter sampling basis adds the fdot_astro_ratio column iff
        # the astrophysical-prior package is on (single source of truth:
        # GBSettings.use_fdot_astro). See gb.py init_sampling_info.
        gb.ndim = 8 + int(getattr(gb, "use_fdot_astro", False))

    is_fd = isinstance(domain_settings, FDSettings)
    layer_df = 1.0 / (2 * general_setup.domain_settings.Nf * gb.dt) if not is_fd else None

    # GB_HIGHEST_FREQUENCIES=N: center the band on the N-th highest-frequency
    # catalogue GB (1-based). Fallback path for when the center was NOT already
    # resolved in adjust_general (e.g. non-mojito variants) -- reads the loaded
    # general_setup.catalogue. For mojito gb_no_fg, adjust_general has already
    # set center_freq (so the DATA band tracks it) and this is skipped.
    _highest = getattr(gb, "highest_frequencies", None)
    if _highest and gb.center_freq is None:
        n_th = int(_highest)
        cat = (getattr(general_setup, "catalogue", None) or {}).get("GB", {})
        f0_all = np.concatenate(
            [np.asarray(cat[k]["GW22FrequencySSBFrame"], dtype=float).ravel()
             for k in sorted(cat.keys())]
        ) if cat else np.array([])
        if f0_all.size == 0:
            raise ValueError(
                "GB_HIGHEST_FREQUENCIES set but no GB catalogue is available "
                "to resolve source frequencies."
            )
        if n_th < 1 or n_th > f0_all.size:
            raise ValueError(
                f"GB_HIGHEST_FREQUENCIES={n_th} out of range "
                f"(1..{f0_all.size} catalogue sources)."
            )
        # N-th highest = smallest of the top-N.
        nth_f0 = float(np.sort(f0_all[np.argpartition(f0_all, -n_th)[-n_th:]])[0])
        gb.center_freq = nth_f0
        if gb.n_layers is None:
            gb.n_layers = 3
        logger.info(
            "GB_HIGHEST_FREQUENCIES=%d -> centering GB band on catalogue "
            "source f0=%.6f mHz (%d layers).", n_th, nth_f0 * 1e3, gb.n_layers,
        )

    _center_mode = gb.center_freq is not None or gb.n_layers is not None
    if is_fd:
        # FD basis has no WDM layers: the narrow ``gb.min_freq/max_freq``
        # defaults are WDM layer-snapping semantics and are FAR too narrow
        # for the FD band walker (a single FD sub-band is already ~2*N*df
        # wide, which exceeds a few-layer band at short Tobs and collapses
        # ``band_edges`` to empty). So on FD:
        #   * ``fd_bandwidth is None`` -> the FULL data band (the documented
        #     default; several sub-bands survive the band-walker trim).
        #   * ``fd_bandwidth`` set     -> a band of that width centered on the
        #     requested center (``center_freq`` in center mode, else the
        #     midpoint of ``gb.min_freq/max_freq``), clipped to the data band.
        if _center_mode:
            _center = gb.center_freq if gb.center_freq is not None else 7.5e-3
        else:
            _center = 0.5 * (float(gb.min_freq) + float(gb.max_freq))
        if gb.fd_bandwidth is None:
            gb.start_freq = general_setup.min_freq
            gb.end_freq = general_setup.max_freq
        else:
            half_bw = 0.5 * float(gb.fd_bandwidth)
            gb.start_freq = max(general_setup.min_freq, _center - half_bw)
            gb.end_freq = min(general_setup.max_freq, _center + half_bw)
        # FD has no WDM-layer count; expose sane, non-None values so the
        # shared WDM-oriented consumers (debug-band selection, plots) that
        # read ``n_layers``/``center_freq`` don't trip over ``None``.
        gb.center_freq = _center
        if gb.n_layers is None:
            gb.n_layers = 3
        logger.info(
            "FD basis: GB band set to [%.6e, %.6e] Hz for the FD band walker.",
            gb.start_freq, gb.end_freq,
        )
    else:
        k_lo, k_hi = _band_klohi(gb, layer_df)
        gb.start_freq, gb.end_freq = k_lo * layer_df, k_hi * layer_df
        if (gb.start_freq < general_setup.min_freq - 1e-12
                or gb.end_freq > general_setup.max_freq + 1e-12):
            raise ValueError(
                f"GB band [{gb.start_freq:.6e}, {gb.end_freq:.6e}] Hz "
                f"extends outside the data band "
                f"[{general_setup.min_freq:.6e}, {general_setup.max_freq:.6e}] Hz; "
                "adjust gb.min_freq/max_freq (or center_freq/n_layers), or "
                "widen the general band."
            )
        gb.min_freq, gb.max_freq = gb.start_freq, gb.end_freq
        gb.n_layers = k_hi - k_lo
        gb.center_freq = ((k_lo + k_hi) // 2 + 0.5) * layer_df
        logger.info(
            "GB band: [%.6e, %.6e] Hz (%d WDM layers, layer_df=%.4e Hz)",
            gb.start_freq, gb.end_freq, gb.n_layers, layer_df,
        )

    if gb.oversample is None:
        gb.oversample = 2 if is_fd else 4
    if gb.nleaves_max is None:
        gb.nleaves_max = _resolve_nleaves_max(gb, general_setup, is_fd, layer_df)

    gb.tdi_setup = general_setup.tdi_chan
    gb.use_tdi2 = tdi_generation_info(general_setup.tdi_chan)[0] == 2
    gb.initialize_kwargs = dict(force_backend=general_setup.force_backend)
    if gb.betas is None:
        # GB's OWN ladder, sized by the per-branch knob (the engine runs
        # cold-chain only)
        betas = 1.0 / 1.2 ** np.arange(gb.ntemps)
        betas[-1] = 1e-4
        gb.betas = betas
    gb.gb_wdm_comp = None
    return gb


class GBNoForegroundGlobalFit(EreborFit):
    """GB-only global fit without foreground fitting (fixed PSD, f > 6 mHz)."""

    option_name = "gb_no_fg"
    description = (
        "GB-only fit on the mojito L1 GB galaxy: fixed PSD, no foreground "
        "branch, f in [6, 25] mHz, WDM chunked-het likelihood."
    )
    general_settings_class = GBNoFgGeneralSettings
    setup_classes = {"gb": GBSetup}

    def __init__(self, **knobs):
        # GB_DEBUG preset FIRST (before any env-backed default resolves), so
        # the debug values seed the environment exactly like the legacy file.
        debug = knobs.pop("debug", None)
        if debug is None:
            debug = env_resolve("GB_DEBUG", False, bool)
        if debug:
            for _knob, _debug_val in _DEBUG_ENV_PRESET.items():
                os.environ.setdefault(_knob, _debug_val)
            # Arm the GB special-move debug instrumentation too (band
            # residual round-trip checks + begin/middle/end band plots under
            # GB_DEBUG_DIR) — consumed inside build_gb_moves from the env.
            os.environ.setdefault("GB_DEBUG", "1")
        super().__init__(**knobs)
        self.debug = bool(debug)

    def apply_debug_preset(self) -> None:
        """Apply the GB_DEBUG smoke preset to the current fields.

        Field-level version of the env preset used at construction (clones
        never re-run ``__init__``, so ``fit(debug=True)`` routes here).
        Explicit env values still win, mirroring the setdefault semantics.
        """
        os.environ.setdefault("GB_DEBUG", "1")
        self.general.tobs_target = env_resolve("TOBS_TARGET", 3 * 86400.0, float)
        self.general.nwalkers = env_resolve("NWALKERS", 3, int)
        self.gb.ntemps = env_resolve("GB_NTEMPS", 2, int)
        self.general.num_iterations = env_resolve("NUM_ITERATIONS", 4, int)
        self.gb.nt_sub = env_resolve("CHUNKED_NT_SUB", 64, int)
        self.gb.n_pad = env_resolve("CHUNKED_N_PAD", 8, int)
        self.gb.n_sparse = env_resolve("CHUNKED_N_SPARSE", 64, int)
        self.gb.n_cp_sig = env_resolve("CHUNKED_N_CP_SIG", 16, int)
        self.gb.n_cp_orbit = env_resolve("CHUNKED_N_CP_ORBIT", 16, int)
        self.debug = True

    def apply_overrides(self, overrides: dict) -> None:
        debug = overrides.pop("debug", None)
        if debug:
            self.apply_debug_preset()
        elif debug is not None:
            self.debug = False
        super().apply_overrides(overrides)

    # -- default blocks -------------------------------------------------------

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {"gb": GBNoFgGBSettings()}

    def default_recipe(self) -> Recipe:
        # Smoke-validated default: ONLY the prior-based RJ proposal (which
        # carries the in-model repeat blocks internally). Add
        # Move("rj_fstat_mcmc"/"rj_refit", branch="gb") for the full PE
        # stack, or *_search names in a kind="search" stage.
        return Recipe(
            [
                Stage(
                    name="gb_pe",
                    kind="pe",
                    moves=[Move("rj_fstat_pe", branch="gb")],
                    combine_kwargs=dict(share_temperature_control=False),
                )
            ]
        )

    # -- general resolution -----------------------------------------------------

    def adjust_general(self, gs: GBNoFgGeneralSettings) -> None:
        # Fixed PSD levels: explicit list > mojito NOISE brick fit > stock.
        if gs.fixed_psd_params is None:
            file_params = self.resolve_noise_file_psd_params(gs)
            gs.fixed_psd_params = (
                file_params if file_params is not None else [15e-12, 3e-15]
            )

        gb = self.gb
        # GB_HIGHEST_FREQUENCIES: resolve the band center from the catalogue
        # HERE (before the data-band derivation below) so the DATA band tracks
        # the chosen source. Reads the catalogue file directly since the data
        # processor has not populated general_setup.catalogue yet. mojito only;
        # prepare_gb_branch handles the (already-loaded) catalogue fallback.
        _highest = getattr(gb, "highest_frequencies", None)
        if _highest and gb.center_freq is None and gs.data_mode == "mojito":
            gb.center_freq = _nth_highest_catalogue_f0(
                gs.mojito_data_path, int(_highest)
            )
            if gb.n_layers is None:
                gb.n_layers = 3
            logger.info(
                "GB_HIGHEST_FREQUENCIES=%d -> band center f0=%.6f mHz "
                "(from catalogue; data band will track it).",
                int(_highest), gb.center_freq * 1e3,
            )

        # ACA frequency clipping (memory knob): narrow the DATA band to
        # +-data_band_layers WDM layers around the GB band.
        if gb.data_band_layers is not None:
            layer_df = self.layer_df
            L = int(gb.data_band_layers)
            # Clip is measured from the BAND EDGES (works identically for the
            # min/max and center/n_layers interfaces); >= 5 layers of margin
            # are required for the 5-layer chunked-het gating.
            assert L >= 5, (
                f"data_band_layers={L} too narrow: need >= 5 layers beyond "
                "the GB band edges to cover the chunked-het gating."
            )
            k_lo, k_hi = _band_klohi(gb, layer_df)
            gs.min_freq = max(gs.min_freq, (k_lo - L) * layer_df)
            gs.max_freq = min(gs.max_freq, (k_hi + L) * layer_df)
            logger.info(
                "ACA data band CLIPPED to [%.6e, %.6e] Hz (+-%d layers around "
                "the GB band).", gs.min_freq, gs.max_freq, L,
            )

    def set_default_processor(self, gs: GBNoFgGeneralSettings) -> None:
        if gs.data_mode == "mojito":
            # Mojito L1 GB galaxy (populates catalogue["GB"] for the
            # true-point start).
            super().set_default_processor(gs)
            return
        if gs.data_mode == "synthetic":
            # In-process GB injection — no external data needed. With no
            # catalogue the true-point seeding in setup_recipe is a no-op
            # and the sampler starts from prior draws / RJ births.
            from ..injections import GB_INJECTION_PARAMS, SyntheticGBProcessingStep

            params = (
                gs.gb_injection_params
                if gs.gb_injection_params is not None
                else GB_INJECTION_PARAMS
            )
            gs.data_processor_class = SyntheticGBProcessingStep
            gs.processor_init_kwargs = dict(
                Tobs=self.wdm_grid[3],
                dt=gs.dt,
                t_start=gs.synthetic_t_start,
                injection_params=np.atleast_2d(np.asarray(params, dtype=float)),
                tdi_chan=gs.tdi_chan,
                nchannels=gs.nchannels,
                verbose=gs.verbose,
                use_tdi2=tdi_generation_info(gs.tdi_chan)[0] == 2,
                # Follow the run's compute setup. This used to be pinned to
                # "cpu", which was harmless when the injection was a legacy
                # per-source FD kernel but defeats the batched chunked-het
                # generator: a whole-galaxy injection wants the GPU.
                force_backend=(
                    gs.gpu_backend if gs.gpus is not None else "cpu"
                ),
            )
            return
        raise ValueError(
            f"gb_no_fg data_mode={gs.data_mode!r} not recognised; use "
            "'mojito' or 'synthetic' (or swap data_processor_class wholesale)."
        )

    def default_preprocess_kwargs(self) -> dict:
        if self.general.data_mode == "synthetic":
            # The synthetic stream covers exactly Tobs = Nf*Nt*dt; skip the
            # engine's default highpass + edge-trim + Tobs trim so the WDM
            # shape stays exact.
            return dict(
                highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False
            )
        return super().default_preprocess_kwargs()

    def make_domain_settings(self, gs, Nf, Nt, wavelet_duration, edge_crop):
        # GB_DOMAIN=fd switches the run basis to the frequency domain (same
        # band restriction); the object-level swap (fit.general.domain_settings
        # = FDSettings.make_factory(...)) is the primary interface.
        if env_resolve("GB_DOMAIN", "wdm", str).lower() == "fd":
            return FDSettings.make_factory(min_freq=gs.min_freq, max_freq=gs.max_freq)
        return super().make_domain_settings(gs, Nf, Nt, wavelet_duration, edge_crop)

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        if name != "gb":
            return settings
        return prepare_gb_branch(
            settings,
            general_setup,
            data_mode=self.general.data_mode,
            synthetic_t_start=self.general.synthetic_t_start,
        )

    def attach_runtime_objects(self) -> None:
        """Register the GB engine ``signal_gen`` and resolve the pe seed EARLY.

        With the generator registered, run.py's ``load_info`` seeds the GB
        leaves (named gb seeder) and ``setup_acs(rebuild_residuals=True)``
        subtracts them through the engine together with every other branch;
        the recipe-side late seeding + manual subtraction auto-skip. The
        SNR cut runs HERE against a NOISE-ONLY ``AnalysisContainer`` (a
        zeroed copy of the input data + the run's fixed sensitivity --
        existing pieces only, no acs needed): ``<h|h>`` reads only invC.
        WDM path only; FD keeps the legacy late path.
        """
        from copy import deepcopy

        from lisatools.analysiscontainer import AnalysisContainer

        from ....recipe import (
            MOJITO_REFERENCE_TIME,
            GBFillGlobalSignalGen,
            gb_catalogue_to_sampling_basis,
            select_gb_injection_subset_by_snr,
        )

        info = self.source_info["gb"]
        gi = self.general_info
        if not isinstance(gi.domain_settings, WDMSettings):
            return  # FD: legacy late seeding/subtraction path
        info.signal_gen = GBFillGlobalSignalGen(
            "gb", info.transform, gi, info
        )

        if getattr(info, "mode", "pe") == "search":
            return  # zero-leaf start: nothing to seed
        catalogue = (getattr(gi, "catalogue", None) or {}).get("GB", {})
        if not catalogue:
            return  # no catalogue (e.g. synthetic): legacy behavior (no seed)

        # Noise-only AC: zeroed data copy + the run's fixed sensitivity.
        data0 = deepcopy(gi.input_data_residual_array)
        data0.arr[:] = 0.0
        sens0 = gi.sensitivity_backend("gb_seed_snr", **gi.fixed_psd_kwargs)
        ac0 = AnalysisContainer(data0, sens0)

        # Disjointness with the neighbor-subtraction window (legacy pairing):
        # when subtract_neighbors is on, only the CENTRAL layer's sources
        # are modeled; the neighbors get subtracted as known signals later.
        _f0_lims = None
        if getattr(info, "subtract_neighbors", False) and not getattr(
            info, "subtract_out_of_band", False
        ):
            layer_df = self.layer_df
            k_center = int(math.floor(
                getattr(info, "center_freq", 7.5e-3) / layer_df
            ))
            _f0_lims = (k_center * layer_df, (k_center + 1) * layer_df)

        subset_inds = select_gb_injection_subset_by_snr(
            self, [ac0], info, info.signal_gen._comp(),
            snr_threshold=getattr(info, "snr_threshold", 3.0),
            f0_lims=_f0_lims,
        )
        if not len(subset_inds):
            return

        if getattr(info, "start", "truth") == "prior":
            # PE dimensionality WITHOUT truth: same leaf count, prior rows.
            _prior = list(info.priors.values())[0]
            rows = _prior.rvs(size=int(len(subset_inds)))
            rows = rows.get() if hasattr(rows, "get") else np.asarray(rows)
            logger.info(
                "GB start='prior': %d leaves seeded from PRIOR draws "
                "(load_info path).", len(subset_inds),
            )
        else:
            trim_duration = gi.data_t0 - MOJITO_REFERENCE_TIME
            sampling = np.array([
                gb_catalogue_to_sampling_basis(
                    catalogue[k], trim_duration=trim_duration
                )
                for k in sorted(catalogue.keys())
            ])
            if sampling.ndim == 3:
                sampling = sampling.reshape(-1, sampling.shape[-1])
            rows = sampling[subset_inds].copy()
            # gb_catalogue_to_sampling_basis returns FDOT-basis rows by
            # contract; map onto the run's sampling basis (Mc + optional
            # fdot_astro_ratio) with the transform's own conventions. The
            # 9-col ratio basis represents fdot<=0 sources exactly; the
            # 8-col Mc basis still floor-clamps them (mis-modeled).
            _use_fa = getattr(info, "use_fdot_astro", False)
            _use_cm = getattr(info, "use_chirp_mass", False)
            if _use_cm:
                from ....recipe import gb_fdot_rows_to_run_basis

                _nneg = int((rows[:, 2] <= 0).sum())
                rows = gb_fdot_rows_to_run_basis(
                    rows, use_chirp_mass=_use_cm, use_fdot_astro=_use_fa,
                    use_distance=getattr(info, "use_distance", False),
                    m_chirp_lims=info.m_chirp_lims,
                )
                if _nneg and _use_fa:
                    logger.info(
                        "GB seeding: %d fdot<=0 source(s) represented via "
                        "fdot_astro_ratio (~ -2).", _nneg,
                    )
                elif _nneg:
                    logger.warning(
                        "GB seeding: %d fdot<=0 source(s) clamped to the Mc "
                        "floor (mis-modeled; enable the astrophysical prior "
                        "for the fdot_astro_ratio basis).", _nneg,
                    )
        info.injection = rows
        info.injection_subset_inds = np.asarray(subset_inds)

def setup_gb_moves(engine_info, curr, acs, priors, state) -> dict:
    """Build the ``gb_no_fg``-style GB move stack (shared helper).

    Variant-specific GB pre-work — chunked-het likelihood build, true-point /
    prior seeding, neighbor subtraction — then ``build_gb_moves``. Returns the
    ``{name: move}`` dict; the caller materializes the recipe. Imported by both
    ``gb_no_fg`` and ``all_sources`` so their GB setup is identical.
    """
    general_info = curr.general_info
    gpus = general_info.gpus
    if gpus is not None:
        import cupy as cp

        cp.cuda.runtime.setDevice(gpus[0])

    gb_info = curr.source_info["gb"]
    tdi_gen = 2 if getattr(gb_info, "use_tdi2", True) else 1
    tdi_gen_str = f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation"

    # Search-mode arming + debug-plot cell selection: consumed inside
    # ``build_gb_moves`` (and the GB moves) from the environment; seeded via
    # setdefault so explicit env overrides still win.
    if getattr(gb_info, "mode", "pe") == "search":
        os.environ.setdefault("GB_LEAF_CAP_START", "1")
        os.environ.setdefault("GB_RJ_PHASE_MAXIMIZE", "1")
    os.environ.setdefault("GB_DEBUG_PLOT_WALKER", "0")
    # ``n_layers`` is a WDM concept and is ``None`` on the FD basis; fall back
    # to 3 so the debug-band index is well defined for either domain.
    _n_layers_for_plot = getattr(gb_info, "n_layers", None) or 3
    os.environ.setdefault("GB_DEBUG_PLOT_BAND", str(_n_layers_for_plot // 2))

    # Build the WDM-domain GB likelihood here (after the deepcopy in
    # ``CurrentInfoGlobalFit.__init__``) — the underlying C++ orbits wrap
    # is not picklable, so it must live outside the settings dataclass.
    # Chunked-heterodyne ``GBWDMComputations`` is the only WDM backend.
    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and gb_info.gb_wdm_comp is None
    ):
        from gbgpu.gbcomps import GBWDMComputations

        _wdm = general_info.domain_settings
        # Anchor the WDM domain t0 on the absolute data start so the het
        # samples the orbits inside their span (WDM coefficient addressing
        # is t0-independent; only absolute orbit-evaluation times depend on
        # this).
        _t_obs_start = float(getattr(general_info, "data_t0", 0.0))
        _orig_wdm_t0 = float(_wdm.t0)
        _wdm.t0 = _t_obs_start
        gb_info.gb_wdm_comp = GBWDMComputations(
            _wdm,
            t_ref=gb_info.t0,
            Nt_sub=int(gb_info.nt_sub),
            n_pad=int(gb_info.n_pad),
            N_sparse=int(gb_info.n_sparse),
            N_cp_sig=int(gb_info.n_cp_sig),
            N_cp_orbit=int(gb_info.n_cp_orbit),
            orbits=general_info.gpu_orbits,
            tdi_config=tdi_gen_str,
            force_backend=general_info.force_backend,
            tdi_type="XYZ",
        )
        logger.info(
            "Chunked-het GB likelihood: Nf=%d Nt=%d Nt_sub=%d N_sparse=%d "
            "N_cp_sig=%d N_cp_orbit=%d (domain t0 %.6e -> het t_obs_start=%.6e, "
            "t_ref=%.6e, chunk_t_starts=[%.6e, %.6e])",
            _wdm.Nf, _wdm.Nt,
            gb_info.gb_wdm_comp.Nt_sub,
            gb_info.gb_wdm_comp.N_sparse,
            gb_info.gb_wdm_comp.N_cp_sig,
            gb_info.gb_wdm_comp.N_cp_orbit,
            _orig_wdm_t0, gb_info.gb_wdm_comp.t_obs_start, gb_info.t0,
            float(gb_info.gb_wdm_comp.chunk_t_starts.min()),
            float(gb_info.gb_wdm_comp.chunk_t_starts.max()),
        )

        # Sig-het in-model scoring: the move receives a
        # GBSignalHetComputations wrapping the chunked comp; RJ / fills /
        # swaps keep the chunked-het path (pure type dispatch).
        if getattr(gb_info, "sighet_inmodel", False):
            from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

            # v5 is gated INSIDE the kernel selector on v4_knots, and its
            # phase-aliased arena additionally needs a non-zero band
            # (band_len == 0 falls back to the flat carve, i.e. v5=2). Both
            # degrade SILENTLY to a slower path, so a run that asked for v5
            # would look fine and quietly not be v5. Fail loudly instead.
            _v5 = int(getattr(gb_info, "sighet_v5", 0))
            if _v5:
                _knots = int(getattr(gb_info, "sighet_v4_knots", 0))
                _band = int(getattr(gb_info, "sighet_v4_band", 0))
                if _knots <= 0:
                    raise ValueError(
                        f"SIGHET_V5={_v5} requires SIGHET_V4_KNOTS > 0 (got "
                        f"{_knots}): the v5 kernel is selected only when the "
                        "v4 fixed-knot resample is active, so v5 would be "
                        "silently ignored. Benchmarked config: "
                        "SIGHET_V3_NODES=64 SIGHET_V4_KNOTS=128 "
                        "SIGHET_V4_BAND=16."
                    )
                if _v5 == 1 and _band <= 0:
                    raise ValueError(
                        f"SIGHET_V5=1 requires SIGHET_V4_BAND > 0 (got "
                        f"{_band}): with an empty band the kernel takes the "
                        "flat carve, which is the v5=2 control arm, not the "
                        "phase-aliased arena. Set SIGHET_V4_BAND=16, or ask "
                        "for the control explicitly with SIGHET_V5=2."
                    )

            # THE EDGE-EXCLUSION INVARIANT (user ruling 2026-08-19): any
            # region where taper error is created must be REMOVED by the WDM
            # [min_time, max_time] crop -- the crop serves the taper, never
            # the taper shrunk to fit the crop. Enforced HERE, at build time,
            # where both knobs are on the table: either raise
            # EDGE_CROP_WAVELETS or lower SIGHET_TUKEY_ALPHA. (The engine
            # also carries a last-resort clamp for non-erebor callers, but
            # this refusal is the real guard.)
            _alpha_sh = float(getattr(gb_info, "sighet_tukey_alpha", 0.01))
            _wdm_dom = getattr(gb_info.gb_wdm_comp, "wdm_settings", None)
            _Nt_dom = int(getattr(_wdm_dom, "Nt", 0) or 0)
            _crop = int(getattr(_wdm_dom, "ind_min_t", 0) or 0)
            if _Nt_dom:
                _taper = int(np.ceil(0.5 * _alpha_sh * _Nt_dom))
                if _taper + 8 > _crop:
                    raise ValueError(
                        f"sig-het reference taper (SIGHET_TUKEY_ALPHA="
                        f"{_alpha_sh} -> {_taper} WDM layers/side + 8 margin)"
                        f" is not excluded by the time crop (ind_min_t="
                        f"{_crop}). Error-created edges must be REMOVED by "
                        f"[min_time, max_time]: raise EDGE_CROP_WAVELETS to "
                        f">= {_taper + 8} or lower SIGHET_TUKEY_ALPHA to <= "
                        f"{max(0.0, 2.0 * (_crop - 8) / _Nt_dom):.4f}.")
            gb_info.gb_wdm_comp = GBSignalHetComputations.for_band_engine(
                gb_info.gb_wdm_comp,
                tukey_alpha=float(getattr(gb_info, "sighet_tukey_alpha",
                                          0.01)),
                nt_layer=int(gb_info.sighet_nt_layer),
                n_sparse_fd=int(gb_info.sighet_n_sparse_fd),
                max_r=float(getattr(gb_info, "sighet_max_r", 0.0)),
                n_cp_build=int(getattr(gb_info, "sighet_n_cp", -1)),
                v3_n_nodes=int(getattr(gb_info, "sighet_v3_nodes", 0)),
                v4_knots=int(getattr(gb_info, "sighet_v4_knots", 0)),
                v4_band=int(getattr(gb_info, "sighet_v4_band", 0)),
                **({"v5": int(getattr(gb_info, "sighet_v5", 0))}
                   if int(getattr(gb_info, "sighet_v5", 0)) else {}),
            )
            # RESOLVED-CONFIG ECHO -- emitted HERE, not from GBGPU. The
            # equivalent gbgpu-side logger.info (GBGPU b412089) is INVISIBLE:
            # this pipeline attaches handlers to the ``lisatools`` logger tree
            # only, and a 74k-line production log contains ZERO ``gbgpu.*``
            # records. Every GBGPU log message is silently discarded --
            # including _resolve_nt_layer's "nt_layer does not divide Nt;
            # snapping to N" WARNING, whose absence we spent four runs
            # misreading as evidence the knob had applied.
            #
            # Why it has to exist at all: nothing else reports what the
            # sig-het engine actually resolved to. Every field below is the
            # END of a chain of env reads, dataclass defaults, divisor
            # snapping and device clamps, and run_settings.log records none
            # of them -- so an accuracy A/B on SIGHET_NT_LAYER /
            # SIGHET_V3_NODES / SIGHET_V4_KNOTS had no way to confirm the
            # knob took. The obvious proxy fails too: kernel wall time is
            # launch-bound at production batch sizes (24-48 sources, against
            # the ~256 where the node stage starts to show), so a 32-vs-128
            # node A/B moved inmodel_repeats by 2%.
            #
            # Sparse spacing is reported in HOURS because the accuracy
            # prescription is stated that way (~35 h constant density) while
            # the knob is a layer count; the mapping runs through Nf, dt and
            # Nt and is not something to do in your head at 3am.
            _g = getattr(gb_info.gb_wdm_comp, "_g", {}) or {}
            _nf, _dt = _g.get("Nf", 0), _g.get("dt", 0.0)
            _stride = _g.get("stride", 0)
            logger.info(
                "GB in-model likelihood: SIGNAL-HET "
                "(chunked-het delegate for RJ / fills / swaps). "
                "RESOLVED: v3_n_nodes=%s v4_knots=%s v4_band=%s v5=%s | "
                "nt_layer=%s (stride %s) N_sparse_t=%s n_sparse_fd=%s "
                "n_cp_build=%s m_half=%s max_r=%s | Nf=%s Nt=%s "
                "Nt_active=%s Tobs=%.6gs -> sparse spacing %.1f h",
                _g.get("v3_n_nodes"), _g.get("v4_knots"), _g.get("v4_band"),
                _g.get("v5"), _g.get("nt_layer"), _stride,
                _g.get("N_sparse_t"), _g.get("n_sparse_fd"),
                _g.get("n_cp_build"), _g.get("m_half"), _g.get("max_r"),
                _nf, _g.get("Nt"), _g.get("Nt_active"), _g.get("Tobs", 0.0),
                (_stride * _nf * _dt / 3600.0) if _nf and _dt else float("nan"),
            )

    # FD-domain mirror: the GB move auto-builds a config-only
    # ``GBFDComputations`` from the ACA's FDSettings; it only needs the
    # orbits + TDI-config handles (wired here after the settings deepcopy,
    # exactly like ``gb_wdm_comp`` on the WDM path).
    if isinstance(general_info.domain_settings, FDSettings):
        if getattr(gb_info, "orbits", None) is None:
            gb_info.orbits = general_info.gpu_orbits
        if getattr(gb_info, "tdi_config", None) is None:
            gb_info.tdi_config = tdi_gen_str

    # ============== START GBs AT TRUE CATALOG POINTS (SNR cut) ==============
    if gb_info.gb_wdm_comp is not None:
        layer_df = 1.0 / (2 * general_info.domain_settings.Nf * gb_info.dt)
        k_center = int(math.floor(getattr(gb_info, "center_freq", 7.5e-3) / layer_df))
        _injection_f0_lims = None
        if getattr(gb_info, "subtract_out_of_band", False):
            # Remove ALL catalogue sources OUTSIDE the sampled band
            # (f0_lims) as known signals, over the whole remaining data
            # band. exclude_f0_lims == the band select_gb_injection_subset_
            # by_snr samples, so subtracted sources and injected/modeled
            # leaves stay disjoint.
            _band_lims = (float(gb_info.f0_lims[0]), float(gb_info.f0_lims[1]))
            _injection_f0_lims = _band_lims
            # Extend the subtraction window a BUFFER of WDM layers PAST the
            # data-band edges: a source whose carrier sits just outside the
            # band still has WDM support (wavelet ~a few layers wide) leaking
            # INTO the edge layers, so it must be subtracted too or it leaves
            # residual pinned at the first/last active layer. GB_SUBTRACT_
            # BUFFER_LAYERS (default 8) sets the buffer in layers.
            _buffer_hz = int(os.environ.get("GB_SUBTRACT_BUFFER_LAYERS", "8")) * layer_df
            _window_hz = max(
                _band_lims[0] - float(general_info.min_freq),
                float(general_info.max_freq) - _band_lims[1],
                0.0,
            ) + _buffer_hz
            _n_oob = subtract_gb_neighbors_from_data(
                curr, acs, gb_info, gb_info.gb_wdm_comp,
                exclude_f0_lims=_band_lims,
                window_hz=_window_hz,
            )
            logger.info(
                "GB_SUBTRACT_OUT_OF_BAND=1: subtracted %d out-of-band "
                "catalogue sources (outside [%.6e, %.6e] Hz, within the "
                "data band).", _n_oob, _band_lims[0], _band_lims[1],
            )
        elif getattr(gb_info, "subtract_neighbors", False):
            _central_lims = (k_center * layer_df, (k_center + 1) * layer_df)
            _injection_f0_lims = _central_lims
            subtract_gb_neighbors_from_data(
                curr, acs, gb_info, gb_info.gb_wdm_comp,
                exclude_f0_lims=_central_lims,
                window_hz=getattr(gb_info, "neighbor_window_layers", 4) * layer_df,
            )
        # mode="search": NO true-point seeding — the gb branch keeps the
        # engine's zero-leaf state and the sampler must FIND the sources
        # through RJ under the per-band progressive leaf cap.
        # Early-seeded runs (attach_runtime_objects resolved gb.injection +
        # registered the signal_gen): load_info seeded and the engine
        # rebuild subtracted -- the late path must not run again.
        _early_seeded = (
            getattr(gb_info, "signal_gen", None) is not None
            and getattr(gb_info, "injection", None) is not None
        )
        if _early_seeded:
            logger.info(
                "GB seeding handled in load_info (engine signal_gen path); "
                "late seeding skipped."
            )
        if getattr(gb_info, "mode", "pe") != "search" and not _early_seeded:
            gb_snr_subset_inds = select_gb_injection_subset_by_snr(
                curr, acs, gb_info, gb_info.gb_wdm_comp,
                snr_threshold=getattr(gb_info, "snr_threshold", 3.0),
                f0_lims=_injection_f0_lims,
            )
            if getattr(gb_info, "start", "truth") == "prior":
                # PE dimensionality WITHOUT truth seeding: same leaf count as
                # the SNR-selected subset, coordinates drawn from the PRIOR.
                _n_true = int(len(gb_snr_subset_inds))
                _gb_inds = state.branches["gb"].inds
                _gb_coords = state.branches["gb"].coords
                _nt, _nw, _nl, _ndim = _gb_coords.shape
                _draws = priors["gb"].rvs(size=(_nt, _nw, _n_true))
                _draws = (
                    _draws.get() if hasattr(_draws, "get") else np.asarray(_draws)
                )
                _gb_inds[:] = False
                _gb_inds[:, :, :_n_true] = True
                _gb_coords[:, :, :_n_true, :] = _draws
                logger.info(
                    "GB start: %d leaves per walker at PRIOR draws "
                    "(start='prior'; truth seeding skipped).", _n_true,
                )
            else:
                # GB_START_FACTOR: MULTIPLICATIVE ``x * (1 + factor * randn)``
                # seed scatter around the injection (the sprint-wide
                # START_FACTOR convention; magnitude-robust, so fdot ~1e-16
                # scatters sensibly without a prior-width scale). 0 seeds
                # walkers EXACTLY at truth (residual -> 0, per-source
                # convention check). With N sources the start logL offset
                # accumulates over ~8*N dims, so it is only ~0 when this is
                # 0/near-0.
                _start_factor = float(os.environ.get("GB_START_FACTOR", "1e-4"))
                logger.info(
                    "GB true-point start: GB_START_FACTOR=%g (multiplicative "
                    "x*(1 + factor*randn) scatter; 0 = exact truth).",
                    _start_factor,
                )
                setup_state_for_injection(
                    curr, state, source_type="GB", branch_name="gb",
                    subset_inds=gb_snr_subset_inds, priors=priors,
                    relative_spread=_start_factor,
                )

    # ========================= MATERIALIZE THE RECIPE =========================
    # The fit's Recipe names the stock GB moves to install;
    # ``build_gb_moves`` owns the GB reference recipe and is steered by
    # exactly those names.
    recipe: Recipe = curr.source_metadata["recipe"]

    # Search RJ cycle wiring (2026-08-01; GB_MODE=search only): insert the
    # replace / prior-removal moves right AFTER the fstat-birth move in its
    # stage. The stage's moves are wrapped in ONE GFCombineMove, and eryn's
    # CombineMove.propose runs them sequentially in list order every
    # iteration -- that is the ordering guarantee for the per-iteration
    # cycle fstat-birth -> fstat-REPLACE -> prior-REMOVAL. Anchor priority
    # (2026-08-24, USER directive "immediately after the RJ move" in
    # gb_search): ``rj_fstat_search`` first, so STAGED recipes carrying
    # both a gb_search and a full_pe stage get the cycle in the SEARCH
    # stage and full_pe stays untouched (PE installation is future work);
    # the pe-named anchors remain for the legacy single-stage search
    # campaigns that run their search through the pe-named moves.
    if getattr(gb_info, "mode", "pe") == "search":
        _names = recipe.move_names()
        _anchor = next(
            (n for n in ("rj_fstat_search", "rj_fstat_pe", "rj_prior")
             if n in _names), None
        )
        if _anchor is None and (
            getattr(gb_info, "search_rj_replace", False)
            or getattr(gb_info, "search_in_model", False)
            or getattr(gb_info, "search_prior_removal", False)
        ):
            logger.warning(
                "GB search RJ cycle knobs set but the recipe has no "
                "'rj_prior' move to anchor the cycle on; skipping the "
                "rj_replace / rj_prior_removal insertion."
            )
        if _anchor is not None:
            if (
                getattr(gb_info, "search_rj_replace", False)
                and "rj_replace" not in _names
            ):
                recipe.add_move("rj_replace", after=_anchor, branch="gb")
                _anchor = "rj_replace"
            # Pure in-model refinement BEFORE the removal move: freshly-born
            # sources get a full repeat block to climb the likelihood before
            # rj_prior_removal judges them for death.
            if (
                getattr(gb_info, "search_in_model", False)
                and "in_model" not in recipe.move_names()
            ):
                recipe.add_move("in_model", after=_anchor, branch="gb")
                _anchor = "in_model"
            if (
                getattr(gb_info, "search_prior_removal", False)
                and "rj_prior_removal" not in recipe.move_names()
            ):
                recipe.add_move("rj_prior_removal", after=_anchor, branch="gb")

    requested = recipe.stock_names()
    include_search = any(name.endswith("_search") for name in requested)
    include_refit = any(name.startswith("rj_refit") for name in requested)
    pe_names = [name for name in requested if not name.endswith("_search")]
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state,
        include_search=include_search,
        include_refit=include_refit,
        pe_move_names=pe_names or None,
    )
    stock_moves = {m.name: m for m in list(gb_search_moves) + list(gb_pe_moves)}
    # Legacy name aliases (2026-08-12 rename: rj_prior_search ->
    # rj_fstat_search, rj_prior -> rj_fstat_pe): recipes and runbooks
    # written before the rename keep resolving; the moves LOG under the
    # new names either way.
    for _legacy, _new in (("rj_prior_search", "rj_fstat_search"),
                          ("rj_prior", "rj_fstat_pe")):
        if _new in stock_moves and _legacy not in stock_moves:
            stock_moves[_legacy] = stock_moves[_new]
    return stock_moves


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``gb_no_fg`` (the run's ``setup_function``)."""
    general_info = curr.general_info
    stock_moves = setup_gb_moves(engine_info, curr, acs, priors, state)
    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state, stock_moves=stock_moves,
        ntemps=general_info.ntemps, nwalkers=general_info.nwalkers,
    )
    recipe.setup(ctx)


GBNoForegroundGlobalFit.default_setup_function = staticmethod(setup_recipe)
