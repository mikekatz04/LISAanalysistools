"""GB (galactic-binary) branch blocks for the Erebor recipe.

:class:`GBSettings` is the GB knob block (``fit.gb``; also the base of the
VGB block) and :class:`GBSetup` turns it into the built branch: the band
grid (:meth:`GBSetup.init_band_structure`), the sampling transform and
priors (:meth:`GBSetup.init_sampling_info`) and the GB sub-state / HDF
sub-backend. The sig-het in-model engine of the GB and VGB moves is built by
:func:`build_sighet_engine` (configuration from :func:`sighet_engine_kwargs`,
refusals in :func:`check_sighet_build_config`).

Pipeline position: ``variants/gb_no_fg.prepare_gb_branch`` (shared by
``gb_no_fg`` and the production ``all_sources``) fills the band, ``f0`` /
``fdot`` boxes, ``nleaves_max``, TDI setup and the ladder, then
:meth:`StockGlobalFit.build <lisatools.globalfit.stock.base.StockGlobalFit.build>`
constructs :class:`GBSetup`; at run start ``setup_gb_moves`` builds the
WDM comp and ``recipe.build_gb_moves`` the GB move stack. Production knobs
(9mo / 12mo launchers): ``GB_NTEMPS=24``, ``GB_SUBBAND_DIVISOR=8``
(uniform WDM edges, 8 sub-bands per layer), ``GB_BAND_UNIT_STRIDE=9``,
``GB_N_SUBBANDS=4096``, ``GB_CAP_DIVISOR=1`` + ``GB_CAP_STAGGER=1``,
``GB_WDM_BAND_SLAB_LAYERS=5``, ``GB_PSD_SHARED_MIRROR=1``, warm start + self
refit on, ``SIGHET_REF_BUILD=lookup`` / ``SIGHET_ANCHOR_ENGINE=lookup``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
import warnings
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.domains import DomainSettingsBase, FDSettings, WDMSettings
from lisatools.utils.constants import YRSID_SI

from ..base import env_default
from ...engine import Settings, Setup
from ...hdfbackend import GBHDFBackend
from ...loginfo import init_logger
from ...priors.gbpriors import get_fdot_mojito
from ...state import GBState
from .transforms import make_gb_transform_container


# TODO check: keep/remove/document -- only reachable with GB_BAND_EDGES_MODE=get_n (no launcher)
def get_n_based_band_edges(
    start_freq: float,
    end_freq: float,
    Tobs: float,
    layer_df: float,
    *,
    subband_divisor: int = 1,
    oversample: int = 4,
    extra_buffer: int = 5,
    target_count: int = 0,
    min_band_layers: typing.Optional[float] = None,
    unit_stride: int = 2,
    sep_factor: typing.Optional[float] = None,
    amp: float = 1e-30,
) -> np.ndarray:
    """FREE-FLOATING GB band edges at the ``2 * get_N`` minimum width.

    THE WIDTH RULE (user ruling, 2026-08-15): each sub-band's width is
    the MINIMUM the chunked-heterodyne bookkeeping allows,

        ``w_band = 2 * get_N(f_max_band) / Tobs``  (Hz),

    where :func:`gbgpu.utils.utility.get_N` is the installed production
    function that sizes the initial frequency-domain heterodyned
    waveform of a single GB chunked-het source (the same call the
    FD/chunked-het paths use for per-source windows -- reused, never
    reimplemented), evaluated at the maximum frequency IN the band. The
    builder MAXIMIZES the number of sub-bands: it packs minimum-width
    bands left-to-right across ``[start_freq, end_freq]`` and never
    widens a band beyond its own floor, with exactly two documented
    exceptions:

    * the FINAL band is clamped to ``end_freq`` and absorbs the
      remainder (merged into its neighbor when the remainder is below
      the top band's floor), and
    * the fixed-point resolution of the self-consistent width (below).

    Self-consistency: the width depends on the band's own upper edge,
    ``w = 2 * get_N(f_lo + w) / Tobs``. ``get_N`` is monotone
    non-decreasing in ``f`` and returns discrete values, so iterating
    ``w_{n+1} = 2 * get_N(f_lo + w_n) / Tobs`` from
    ``w_0 = 2 * get_N(f_lo) / Tobs`` is a monotone non-decreasing
    sequence over a finite value set: it converges EXACTLY to the least
    fixed point in a few steps (typically 1-2; hard-capped with a loud
    RuntimeError -- never a silent approximation).

    FREE-FLOATING EDGES (2026-08-15 user ruling): edges are NOT snapped
    to WDM layer (or fractional-layer) boundaries -- the frequency-based
    parallel scheduling is fully independent of the WDM pixelization.
    The outermost edges are ``start_freq`` / ``end_freq`` verbatim
    (callers clamp those to the WDM active band first). Only the STORAGE
    slabs are layer-derived (see ``SubBandBuffer.recommend_band_slab_layers``
    / ``_compute_slab_min_f``: whole-layer origins/spans covering each
    band's touched range plus max(leakage, FD-support) margins).
    ``layer_df`` is used here only for the summary log's
    layers-equivalent numbers.

    LEAKAGE / SEPARATION GUARD ("strong checks" ruling): a source at a
    band's top edge has FD support reaching ``get_N(f_hi)/Tobs`` Hz past
    the edge, so same-unit concurrency is guarded in SUPPORT terms by
    :func:`lisatools.globalfit.moves.gbbands.check_band_support_separation`
    (gap between same-unit bands >= ``GB_ORTHO_SEP_FACTOR`` x the sum of
    edge half-supports; the width rule guarantees stride 2 passes at
    factor 1.0, stride 3 adds a full band of clearance -- see that
    docstring for the derivation). The builder runs it on the final
    edges and raises loudly on failure, then logs a one-line summary
    (band count, width stats in FD bins and layer-widths, worst
    overhang, minimum safe stride).

    Knob semantics in this mode:

    * ``target_count`` (``GB_BAND_TARGET_COUNT``) > 0 -- optional
      COARSENING: adjacent minimum-width bands are merged (evenly
      spread) down to ~``target_count`` bands. Merged bands are wider
      than each member's floor, so the width rule's guarantees are
      preserved. The count saturates at the maximal packing (a target
      above it is a no-op); the default 0 IS the maximal packing.
    * ``min_band_layers`` (``GB_BAND_MIN_LAYERS``) has NO role here (the
      2*get_N rule is the authority) -- warned and ignored when set;
      it belongs to the legacy uniform mode's era only.
    * ``subband_divisor`` (``GB_SUBBAND_DIVISOR``) -- warned and
      ignored; it only defines the snap grid of the legacy ``uniform``
      mode.
    * ``extra_buffer`` -- unused by this rule (the legacy FD walk keeps
      it); accepted for signature compatibility.

    Returns:
        np.ndarray: ascending band edges (Hz), ``len(edges) - 1`` bands.
    """
    if int(subband_divisor) not in (0, 1):
        warnings.warn(
            "subband_divisor is ignored by the get_n band construction "
            "(free-floating edges, 2026-08-15 user ruling: scheduling is "
            "independent of the WDM pixelization); it only defines the "
            "snap grid of the legacy 'uniform' mode.",
            DeprecationWarning,
            stacklevel=2,
        )
    if min_band_layers is not None and float(min_band_layers) != 1.0:
        warnings.warn(
            "min_band_layers has no role in the get_n band construction "
            "(2026-08-15 user ruling: band widths are set by "
            "2*get_N(f_max_band)/Tobs, the minimum the chunked-het "
            "bookkeeping allows); the value is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    start_freq = float(start_freq)
    end_freq = float(end_freq)
    if end_freq <= start_freq:
        raise ValueError(
            f"empty GB frequency range [{start_freq:.4e}, {end_freq:.4e}]."
        )
    df = 1.0 / float(Tobs)

    def _n_at(f: float) -> float:
        return float(get_N(amp, f, Tobs, oversample=oversample).item())

    def _min_width(f_lo: float) -> float:
        # Least fixed point of w = 2 * get_N(f_lo + w) * df, iterated
        # from below (monotone: get_N non-decreasing => the sequence is
        # non-decreasing over get_N's finite value set => exact
        # convergence). Cap + loud failure rather than silent drift.
        w = 2.0 * _n_at(f_lo) * df
        for _ in range(32):
            w_next = 2.0 * _n_at(f_lo + w) * df
            if w_next == w:
                return w
            if w_next < w:  # get_N must be non-decreasing; never trust silently
                raise RuntimeError(
                    f"get_N not monotone at f={f_lo:.6e} Hz "
                    f"(w {w:.6e} -> {w_next:.6e}); cannot size bands."
                )
            w = w_next
        raise RuntimeError(
            f"band-width fixed point did not converge at f={f_lo:.6e} Hz "
            f"(last w={w:.6e} Hz); get_N is expected to be a step "
            "function with a handful of levels across one band."
        )

    # Maximal packing at the floor: every band exactly its own minimum
    # width, left-to-right.
    fs = [start_freq]
    f = start_freq
    while f < end_freq:
        f = min(f + _min_width(f), end_freq)
        fs.append(f)
    # The final band absorbs the remainder at the top edge: if the
    # clamped last band is below ITS OWN floor (2*get_N(end_freq)*df),
    # merge it into its neighbor (the one documented widening).
    if len(fs) >= 3 and fs[-1] - fs[-2] < 2.0 * _n_at(end_freq) * df:
        del fs[-2]
    edges = np.asarray(fs, dtype=float)

    if int(target_count) > 0 and int(target_count) < len(edges) - 1:
        # Optional coarsening: MERGE adjacent minimum-width bands, evenly
        # spread, down to ~target_count. Keeps the outer edges and the
        # relative get_N profile; merged widths exceed each member's
        # floor, so every guarantee of the width rule still holds.
        n = len(edges) - 1
        keep = np.unique(np.round(
            np.linspace(0, n, int(target_count) + 1)
        ).astype(int))
        edges = edges[keep]

    assert np.all(np.diff(edges) > 0)

    # SUPPORT-based separation guard on the final edges (raises loudly)
    # + the one-line bookkeeping summary (user "strong checks" ruling).
    from ...moves.gbbands import (
        band_support_halfwidths,
        check_band_support_separation,
    )

    sep = check_band_support_separation(
        edges, Tobs, int(unit_stride),
        sep_factor=sep_factor, oversample=oversample, amp=amp,
        context="get_n_based_band_edges", enforce=True,
    )
    widths = np.diff(edges)
    support = band_support_halfwidths(
        edges, Tobs, oversample=oversample, amp=amp)
    ldf = float(layer_df)
    logging.getLogger(__name__).info(
        "[GB_BAND_EDGES get_n] %d bands over [%.4e, %.4e] Hz; widths "
        "min/median/max = %.0f/%.0f/%.0f FD bins (%.3g/%.3g/%.3g "
        "layer-widths); worst edge-source overhang %.3g layer-widths "
        "(vs default slab margin max(leakage=2, support)+guard); "
        "min safe stride %s at sep_factor %.3g (running stride %d).",
        len(edges) - 1, edges[0], edges[-1],
        float(np.min(widths) / df), float(np.median(widths) / df),
        float(np.max(widths) / df),
        float(np.min(widths) / ldf), float(np.median(widths) / ldf),
        float(np.max(widths) / ldf),
        float(np.max(support) / ldf),
        str(sep["min_safe_stride"]), float(sep["sep_factor"]),
        int(unit_stride),
    )
    return edges


def sighet_engine_kwargs(info) -> dict:
    """``for_band_engine`` kwargs for a branch's sig-het in-model engine.

    ONE function for the GB and VGB branches so their engines can never
    drift apart: every knob is read from the branch's settings block (the
    ``sighet_*`` fields, which resolve the SHARED ``SIGHET_*`` env names).

    The Tukey alpha is passed EXPLICITLY (pinned-alpha ruling 2026-08-19).
    Left to ``None``, ``for_band_engine`` inherits the chunked delegate's
    0.05 -- a per-chunk stitching fraction that the sig-het kernels apply
    as a fraction of the WHOLE observation. At six months that tapers ~110
    WDM layers per side against a 60-layer time crop and suppresses the
    reference inside the active region (the engine only WARNS for an
    inherited alpha). The VGB wiring did exactly that until 2026-09-17
    (``VGBSettings`` had no ``sighet_tukey_alpha`` field), the likely
    source of the ``[GB_CELL_LL]`` growth that had ``VGB_SIGHET_INMODEL``
    pinned to 0 in the 6mo campaign.

    Production: ``SIGHET_REF_BUILD=lookup`` + ``SIGHET_ANCHOR_ENGINE=lookup``
    (the carrier reference and the anchor read the GB WDM lookup table,
    resolved by :func:`~lisatools.wdm_lookup_store.resolve_gb_lookup_table`),
    ``SIGHET_TUKEY_ALPHA=0.01``, ``SIGHET_NT_LAYER`` 180 (9mo) / 240 (12mo),
    ``SIGHET_N_CP=256``.

    Args:
        info (Settings): The GB or VGB block; reads the ``sighet_*`` fields
            and (lookup reference) ``lookup_table_path``,
            ``lookup_table_dir``, ``lookup_n_nodes``, ``lookup_k_coarse``.

    Returns:
        dict: ``tukey_alpha``, ``nt_layer``, ``n_sparse_fd``, ``max_r``,
        ``n_cp_build``, ``v3_n_nodes``, ``v4_knots``, ``v4_band``, plus ``v5``
        when non-zero and, on the lookup paths, ``lookup_table``,
        ``lookup_n_nodes``, ``lookup_k_coarse`` and ``anchor_engine``.

    Raises:
        ValueError: An unknown ``sighet_ref_build`` / ``sighet_anchor_engine``,
            or a lookup anchor without the lookup reference.
    """
    v5 = int(info.sighet_v5)
    kw = dict(
        tukey_alpha=float(info.sighet_tukey_alpha),
        nt_layer=int(info.sighet_nt_layer),
        n_sparse_fd=int(info.sighet_n_sparse_fd),
        max_r=float(info.sighet_max_r),
        n_cp_build=int(info.sighet_n_cp),
        v3_n_nodes=int(info.sighet_v3_nodes),
        v4_knots=int(info.sighet_v4_knots),
        v4_band=int(info.sighet_v4_band),
    )
    if v5:
        kw["v5"] = v5
    # Carrier reference from the GB lookup table (GBGPU >= 84f06bf) and the anchor
    # engine. Only passed when asked for, so an older GBGPU keeps building.
    ref = str(getattr(info, "sighet_ref_build", "fd") or "fd").lower()
    anchor = str(getattr(info, "sighet_anchor_engine", "chunked") or "chunked").lower()
    if ref not in ("fd", "lookup"):
        raise ValueError(f"SIGHET_REF_BUILD={ref!r}: 'fd' or 'lookup'.")
    if anchor not in ("chunked", "lookup"):
        raise ValueError(f"SIGHET_ANCHOR_ENGINE={anchor!r}: 'chunked' or 'lookup'.")
    if anchor == "lookup" and ref != "lookup":
        raise ValueError("SIGHET_ANCHOR_ENGINE=lookup needs SIGHET_REF_BUILD=lookup "
                         "(the anchor reads the same attached table).")
    if ref == "lookup":
        from lisatools.wdm_lookup_store import resolve_gb_lookup_table

        kw["lookup_table"] = resolve_gb_lookup_table(
            getattr(info, "lookup_table_path", "") or None,
            getattr(info, "lookup_table_dir", "") or None)
        kw["lookup_n_nodes"] = int(getattr(info, "lookup_n_nodes", -1))
        kw["lookup_k_coarse"] = bool(getattr(info, "lookup_k_coarse", True))
    if anchor == "lookup":
        kw["anchor_engine"] = "lookup"
    return kw


def check_sighet_build_config(info, comp, *, branch: str) -> None:
    """Refuse the silently-degrading sig-het configurations at build time.

    Shared by the GB and VGB wiring (``branch`` names the offender).

    * v5 is gated INSIDE the kernel selector on ``v4_knots``, and its
      phase-aliased arena additionally needs a non-zero band (``band_len
      == 0`` falls back to the flat carve, i.e. v5=2). Both degrade
      SILENTLY to a slower path, so a run that asked for v5 would look
      fine and quietly not be v5. Fail loudly instead.
    * THE EDGE-EXCLUSION INVARIANT (user ruling 2026-08-19): any region
      where taper error is created must be REMOVED by the WDM
      ``[min_time, max_time]`` crop -- the crop serves the taper, never
      the taper shrunk to fit the crop. Enforced here, at build time,
      where both knobs are on the table: either raise
      ``EDGE_CROP_WAVELETS`` or lower ``SIGHET_TUKEY_ALPHA``. (The engine
      carries a last-resort clamp for non-erebor callers, but this
      refusal is the real guard.)

    Args:
        info (Settings): The GB or VGB block (``sighet_v5``,
            ``sighet_v4_knots``, ``sighet_v4_band``, ``sighet_tukey_alpha``).
        comp: The chunked-het comp (its ``wdm_settings.Nt`` / ``ind_min_t``
            are read).
        branch: Branch name for the error messages.

    Raises:
        ValueError: v5 without v4 knots, ``SIGHET_V5=1`` without a v4 band,
            or a taper of ``ceil(0.5 * alpha * Nt) + 8`` layers exceeding the
            time crop ``ind_min_t``.
    """
    v5 = int(info.sighet_v5)
    if v5:
        knots = int(info.sighet_v4_knots)
        band = int(info.sighet_v4_band)
        if knots <= 0:
            raise ValueError(
                f"[{branch}] SIGHET_V5={v5} requires SIGHET_V4_KNOTS > 0 "
                f"(got {knots}): the v5 kernel is selected only when the v4 "
                "fixed-knot resample is active, so v5 would be silently "
                "ignored. Benchmarked config: SIGHET_V3_NODES=64 "
                "SIGHET_V4_KNOTS=128 SIGHET_V4_BAND=16."
            )
        if v5 == 1 and band <= 0:
            raise ValueError(
                f"[{branch}] SIGHET_V5=1 requires SIGHET_V4_BAND > 0 (got "
                f"{band}): with an empty band the kernel takes the flat "
                "carve, which is the v5=2 control arm, not the phase-aliased "
                "arena. Set SIGHET_V4_BAND=16, or ask for the control "
                "explicitly with SIGHET_V5=2."
            )

    alpha = float(info.sighet_tukey_alpha)
    wdm = getattr(comp, "wdm_settings", None)
    Nt = int(getattr(wdm, "Nt", 0) or 0)
    crop = int(getattr(wdm, "ind_min_t", 0) or 0)
    if Nt:
        taper = int(np.ceil(0.5 * alpha * Nt))
        if taper + 8 > crop:
            raise ValueError(
                f"[{branch}] sig-het reference taper (SIGHET_TUKEY_ALPHA="
                f"{alpha} -> {taper} WDM layers/side + 8 margin) is not "
                f"excluded by the time crop (ind_min_t={crop}). "
                "Error-created edges must be REMOVED by [min_time, "
                f"max_time]: raise EDGE_CROP_WAVELETS to >= {taper + 8} or "
                "lower SIGHET_TUKEY_ALPHA to <= "
                f"{max(0.0, 2.0 * (crop - 8) / Nt):.4f}."
            )


def build_gb_wdm_comp(info, general_info, *, branch: str):
    """Build a branch's WDM GB comp: ``GBWDMComputations``, or with
    ``likelihood="lookup"`` (``GB_LIKELIHOOD``) ``GBLookupWDMComputations``.

    ONE builder for the GB and VGB branches, like :func:`build_sighet_engine`,
    so their per-row likelihoods can never drift apart. The lookup comp IS the
    chunked-het comp for every purpose but ``get_ll_wdm`` (fills, swaps, the
    F-stat and the sig-het engine's delegate are unchanged); that one call --
    RJ births and removals, every score outside an in-model block -- runs
    GBGPU's fused direct-to-WDM lookup scorer, which reads each row's WDM
    coefficients from the GB lookup table (the table the sig-het reference,
    anchor and information matrix read). H100 harness, 2026-10-05: 4.4 / 7.0 /
    19 us per row at 180 / 360 / 720 d against 88 / 172 / 345 us chunked;
    median 1-3e-5 nats and at most 5e-3 nats from the chunked value. The
    production launchers export ``GB_LIKELIHOOD=lookup``; ``lookup_fill`` stays
    off (fills chunked-het).

    The domain settings' ``t0`` is moved to the absolute data start first: the
    het samples the orbits inside their span, and WDM coefficient addressing
    is t0-independent.

    Args:
        info (Settings): The GB or VGB block: ``likelihood``, ``t0``, ``nt_sub``,
            ``n_pad``, ``n_sparse``, ``n_cp_sig``, ``n_cp_orbit``, ``use_tdi2``
            and, on the lookup path, ``lookup_table_path``, ``lookup_table_dir``,
            ``lookup_n_nodes``, ``lookup_k_coarse``, ``lookup_fill``.
        general_info: The run's general block: ``domain_settings`` (a
            :class:`~lisatools.domains.WDMSettings`), ``data_t0``,
            ``gpu_orbits``, ``force_backend``.
        branch (str): Branch name for the log lines (``"gb"`` / ``"vgb"``).

    Returns:
        gbgpu.gbcomps.GBWDMComputations: The comp; a
        ``gbgpu.gblookupcomputations.GBLookupWDMComputations`` on the lookup
        path.

    Raises:
        ValueError: ``likelihood`` is neither ``"chunked"`` nor ``"lookup"``.
    """
    from gbgpu.gbcomps import GBWDMComputations

    log = logging.getLogger(__name__)
    tdi_gen = 2 if getattr(info, "use_tdi2", True) else 1
    tdi_gen_str = f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation"
    wdm = general_info.domain_settings
    orig_t0 = float(wdm.t0)
    wdm.t0 = float(getattr(general_info, "data_t0", 0.0))
    like = str(getattr(info, "likelihood", "chunked") or "chunked").lower()
    if like not in ("chunked", "lookup"):
        raise ValueError(f"[{branch}] GB_LIKELIHOOD={like!r}: 'chunked' or 'lookup'.")
    comp_cls, lookup_kw = GBWDMComputations, {}
    if like == "lookup":
        from gbgpu.gblookupcomputations import GBLookupWDMComputations
        from lisatools.wdm_lookup_store import resolve_gb_lookup_table

        comp_cls = GBLookupWDMComputations
        lookup_kw = dict(
            lookup_table=resolve_gb_lookup_table(
                getattr(info, "lookup_table_path", "") or None,
                getattr(info, "lookup_table_dir", "") or None),
            lookup_n_nodes=int(getattr(info, "lookup_n_nodes", -1)),
            lookup_k_coarse=bool(getattr(info, "lookup_k_coarse", True)),
            lookup_fill=bool(getattr(info, "lookup_fill", False)),
        )
    comp = comp_cls(
        wdm,
        t_ref=info.t0,
        Nt_sub=int(info.nt_sub),
        n_pad=int(info.n_pad),
        N_sparse=int(info.n_sparse),
        N_cp_sig=int(info.n_cp_sig),
        N_cp_orbit=int(info.n_cp_orbit),
        orbits=general_info.gpu_orbits,
        tdi_config=tdi_gen_str,
        force_backend=general_info.force_backend,
        tdi_type="XYZ",
        **lookup_kw,
    )
    if like == "lookup":
        log.info(
            "%s per-row likelihood: LOOKUP (GBLookupWDMComputations; table %s, "
            "n_nodes=%d, k_coarse=%s, fills %s) -- fills / swaps / F-stat stay "
            "chunked-het",
            branch.upper(), lookup_kw["lookup_table"], lookup_kw["lookup_n_nodes"],
            lookup_kw["lookup_k_coarse"],
            "LOOKUP" if lookup_kw["lookup_fill"] else "chunked-het")
    log.info(
        "Chunked-het %s likelihood: Nf=%d Nt=%d Nt_sub=%d N_sparse=%d "
        "N_cp_sig=%d N_cp_orbit=%d (domain t0 %.6e -> het t_obs_start=%.6e, "
        "t_ref=%.6e, chunk_t_starts=[%.6e, %.6e])",
        branch.upper(), wdm.Nf, wdm.Nt, comp.Nt_sub, comp.N_sparse, comp.N_cp_sig,
        comp.N_cp_orbit, orig_t0, comp.t_obs_start, info.t0,
        float(comp.chunk_t_starts.min()), float(comp.chunk_t_starts.max()),
    )
    return comp


def build_sighet_engine(info, comp, *, branch: str):
    """Wrap the chunked-het ``comp`` in the sig-het in-model engine.

    The single build path for both branches: the configuration checks,
    then ``GBSignalHetComputations.for_band_engine`` with
    :func:`sighet_engine_kwargs`. RJ / fills / swaps / information
    matrices keep delegating to ``comp`` (pure type dispatch).

    Called by the variants' ``setup_gb_moves`` / ``setup_vgb_moves`` when the
    branch's ``sighet_inmodel`` is on (production: GB and VGB both).

    Args:
        info (Settings): The GB or VGB block.
        comp: The chunked-het WDM comp to wrap.
        branch: Branch name for messages and logs.

    Returns:
        gbgpu.gbsignalhetcomputations.GBSignalHetComputations: The engine.

    Raises:
        ValueError: From :func:`check_sighet_build_config` /
            :func:`sighet_engine_kwargs`.
    """
    from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

    check_sighet_build_config(info, comp, branch=branch)
    kw = sighet_engine_kwargs(info)
    if "lookup_table" in kw:
        logging.getLogger(__name__).info(
            "[%s] sig-het carrier reference from the LOOKUP table %s (anchor: %s)",
            branch, kw["lookup_table"], kw.get("anchor_engine", "chunked"))
    return GBSignalHetComputations.for_band_engine(comp, **kw)


@dataclasses.dataclass
class GBSettings(Settings):
    """Settings dataclass describing the GB branch in an Erebor-style recipe.

    Mostly holds parameter limits, frequency-band oversampling, and the
    starting frequency / TDI configuration consumed by :class:`GBSetup`.

    The shared GB knob block, reached as ``fit.gb`` (and the base of
    :class:`~lisatools.globalfit.stock.erebor.vgb.VGBSettings`). Variants
    subclass it: ``GBNoFgGBSettings`` (``variants/gb_no_fg.py``; band
    bounds ``min_freq`` / ``max_freq`` / ``center_freq`` / ``n_layers``,
    ``mode`` / ``start``, the chunked-het and sig-het kernel sizes, the
    lookup-table paths, env-backed ``num_repeat_proposals`` /
    ``nleaves_max``, wide default prior boxes) and its production subclass
    ``AllSourcesGBSettings`` (``GB_MIN_FREQ`` / ``GB_MAX_FREQ``). Many fields
    are consumed by the GB moves (``moves/gbspecialstretch.py``,
    ``moves/gbbands.py``) through ``recipe.build_gb_moves``; many more GB
    knobs (``GB_INMODEL_*``, ``GB_TEMPER_*``, ``GB_FSTAT_*``, ``GB_RJ_*``,
    ``GB_SIGHET_*``, ...) are read by the moves directly from the environment
    and are not fields here. Field defaults below are this class's; the
    launcher values are those of ``submit_gf_9mo_v9.sh`` /
    ``submit_gf_12mo.sh``.

    Production sampling basis (9 columns, ``use_fdot_astro`` /
    ``use_distance`` on): ``dist`` (kpc), ``f0`` (mHz), ``Mc`` (Msol),
    ``phi0``, ``cos_iota``, ``psi``, ``alpha`` (RA, ICRS), ``sin_delta``,
    ``fdot_astro_ratio``, with the 3-D Milky Way ``(dist, alpha, sin_delta)``
    prior and the astrophysical ``(f0, Mc)`` GMM prior.

    Attributes:
        A_lims: ``[A_min, A_max]`` physical amplitude box: the ``ln A`` prior
            of the legacy (non-distance) bases and the ``ln A`` draw of the
            F-stat birth container when no distance box applies. Unused by the
            production distance basis. Default empty (variants fill it).
        f0_lims: ``[f_min, f_max]`` (Hz) of the ``f0`` prior; OVERWRITTEN by
            :meth:`GBSetup.init_band_structure` with the interior band edges.
        build_seed: Per-rank seed of the prior / birth objects, stamped at
            build by :meth:`StockGlobalFit.prepare_branch_settings
            <lisatools.globalfit.stock.base.StockGlobalFit.prepare_branch_settings>`;
            not a user or env knob. ``None`` = OS entropy.
        ntemps: GB ladder rungs (the engine runs cold-chain only). Env
            ``GB_NTEMPS``. Default 24; launchers 24.
        nleaves_max_cap: Ceiling of the automatic ``nleaves_max`` (2 x
            catalogue sources in band). Env ``GB_NLEAVES_MAX_CAP``. Default
            20000; bypassed in production (``GB_NLEAVES_MAX=20000`` sets
            ``nleaves_max`` directly).
        m_chirp_lims: ``[Mc_min, Mc_max]`` (Msol) chirp-mass box (prior box of
            the GMM / uniform prior and the transform inverse's clip box);
            empty -> ``(0.001, 1.0)`` where a default is needed.
        fdot_lims: ``fdot`` box (Hz/s) of the legacy fdot basis; overwritten by
            :meth:`GBSetup.init_band_structure` (mojito fdot range at the top
            band edge).
        phi0_lims: ``phi0`` box (rad). Default empty (variants:
            ``[0, 2 pi]``).
        iota_lims: Inclination box (rad); the prior is uniform in
            ``cos(iota)`` over it.
        psi_lims: Polarization box (rad).
        alpha_lims: Right-ascension box (rad, ICRS).
        delta_lims: Declination box (rad, ICRS); the prior is uniform in
            ``sin(delta)``.
        start_freq: Low edge (Hz) of the GB band; set by the branch prep from
            the band bounds and clamped to the WDM active band. Default 1e-4.
        end_freq: High edge (Hz) of the GB band. Default 0.025.
        band_edges_override: Explicit ascending band edges (Hz); replaces the
            derived grid when set (``fit.gb.band_edges_override = edges``).
        subband_divisor: ``uniform`` mode: sub-bands per WDM layer. Env
            ``GB_SUBBAND_DIVISOR``. Default 1; launchers 8.
        band_edges_mode: ``"uniform"`` (WDM-layer / divisor grid) or
            ``"get_n"`` (free-floating minimum-width bands,
            :func:`get_n_based_band_edges`). Env ``GB_BAND_EDGES_MODE``.
            Default ``"uniform"`` (launchers unset).
        band_target_count: ``get_n`` mode: merge bands down to ~this count
            (0 = maximal packing). Env ``GB_BAND_TARGET_COUNT``. Default 0.
        band_min_layers: Deprecated; ignored (warned when not 1). Env
            ``GB_BAND_MIN_LAYERS``.
        band_unit_stride: Band-unit stride ``k`` of the concurrent band
            scheduling (bands ``b`` with equal ``b % k`` run together); flows
            to the moves as ``band_units``. Env ``GB_BAND_UNIT_STRIDE``.
            Default 2; launchers 9.
        oversample: FD oversampling of ``get_N`` (band ``N`` values). Default
            4 (variants: ``None`` -> 4 on WDM, 2 on FD).
        extra_buffer: FD band-walk buffer (frequency bins). Default 5.
        start_resample_iter: Not read anywhere.
        iter_count_per_resample: Not read anywhere.
        num_repeat_proposals: In-model repeats per proposal. Plain default 100
            here; ``GBNoFgGBSettings`` makes it env-backed
            (``GB_NUM_REPEAT_PROPOSALS``; launchers 25).
        search_kwargs: Per-band serial MCMC search configuration (read only in
            the ``setup`` of ``GBSpecialRJSerialSearchMCMC`` / the refit move,
            legacy FD-only moves that no production stage runs); ``None`` ->
            defaults in :meth:`GBSetup.init_sampling_info`.
        group_proposal_kwargs: Constructor kwargs of the GB moves; ``None`` ->
            defaults; the slab / mirror / ``num_band_preload`` / ``band_units``
            keys are filled from the fields here unless already present.
        n_subbands: Peak resident (temperature, walker, band) sub-band buffer
            cells (the move's ``num_band_preload``); bounds peak GPU memory.
            Env ``GB_N_SUBBANDS``. Default 1024; launchers 4096.
        use_chirp_mass: Sample ``Mc`` (Msol) at slot 2 instead of ``fdot``.
            Env ``GB_USE_CHIRP_MASS``. Default ``True``.
        use_astrophysical_f0_mc_prior: Joint 6-component ``(f0, Mc)`` GMM prior
            (:class:`~lisatools.sampling.f0_mchirp_prior.F0McGMMSampling`)
            instead of separate uniforms; with ``use_chirp_mass`` it also turns
            on :attr:`use_fdot_astro` / :attr:`use_distance`. Env
            ``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR`` (legacy ``GB_F0MC_GMM_PRIOR``).
            Default ``True``.
        fdot_astro_ratio_max: Half-width ``M`` of the ``fdot_astro_ratio ~
            U[-M, M]`` prior (physical ``fdot = fdot_gr * (1 + r)``). Env
            ``GB_FDOT_ASTRO_RATIO_MAX``. Default 5.0.
        dist_lims: Distance box (kpc) of slot 0 in the distance basis (no env;
            set the field). Default ``[0.001, 40.0]``.
        sky_dist_distribution: Explicit joint ``(dist, alpha, sin_delta)``
            prior (eryn duck-typed; must pickle); wins over
            ``use_galaxy_prior``. Default ``None``.
        use_galaxy_prior: Build the 3-D Milky Way ``(dist, alpha, sin_delta)``
            prior (:mod:`lisatools.globalfit.priors.galaxy_sky_dist`) instead of
            independent uniforms. Env ``GB_USE_GALAXY_PRIOR``. Default ``True``;
            launchers 1.
        rj_birth_distribution: Custom birth distribution of the prior RJ
            moves over the GB sampling basis; ``None`` = global-prior births.
        warm_start_components: Path of the refereed warm-start components
            ``.npz`` (from a parent run); non-empty builds the ``rj_warm_search``
            / ``rj_warm_pe`` births. Env ``GB_WARM_START_COMPONENTS``. Default
            ``""``; launchers ``<STORE_DIR>/warmstart/<parent>_refereed.npz``
            (built automatically when missing).
        warm_start_weight: Reserved, parsed but never read. Env
            ``GB_WARM_START_WEIGHT``.
        warm_self: Self refit of the warm-start mixture from the run's own
            recent cold rows (:mod:`lisatools.globalfit.warmstart.self_refit`).
            Needs ``warm_start_components``. Env ``GB_WARM_SELF``. Default
            ``False``; launchers 1.
        warm_self_every_search: Stored rows between self refits in the
            numbered search stages (<= 0 = none). Env
            ``GB_WARM_SELF_EVERY_SEARCH``. Default 1; launchers 1.
        warm_self_every_pe: Stored rows between self refits in ``full_pe``
            (<= 0 = none). Env ``GB_WARM_SELF_EVERY_PE``. Default 25;
            launchers 25.
        warm_self_at_entry: Build each refit on the head at the warm move's
            entry (``False`` = on the saver after its saves). Env
            ``GB_WARM_SELF_AT_ENTRY``. Default ``True``; launchers 1.
        warm_self_last_k: Latest stored cold rows (x every walker) per refit.
            Env ``GB_WARM_SELF_LAST_K``. Default 50; launchers 50.
        warm_self_split_search: ``"original,full,filtered"`` draw weights of
            the three warm containers in search. Env
            ``GB_WARM_SELF_SPLIT_SEARCH``. Default ``"0.5,0,0.5"``; launchers
            the same.
        warm_self_split_pe: The same weights in PE. Env
            ``GB_WARM_SELF_SPLIT_PE``. Default ``"1,1,1"``; launchers the same.
        warm_self_consensus: Fraction above which a cluster held by every
            walker is dropped from the FILTERED container. Env
            ``GB_WARM_SELF_CONSENSUS``. Default 0.95; launchers 0.95.
        warm_self_adopt_lag: The build propose draws from the epoch held
            BEFORE its own build (one-draw adoption lag). Env
            ``GB_WARM_SELF_ADOPT_LAG``. Default ``True`` (launchers unset).
        para_search: Build ``rj_para_search_pe`` (per-band parallel residual
            search -> GMM births; ``docs/gb-para-search.md``). Env
            ``GB_PARA_SEARCH``. Default ``False`` (launchers unset: off).
        para_search_every_pe: PE iterations between searches. Env
            ``GB_PARA_SEARCH_EVERY_PE``. Default 100.
        para_search_ntemps: Stage-1 rungs per sub-band. Env
            ``GB_PARA_SEARCH_NTEMPS``. Default 8.
        para_search_nwalkers: Walkers per sub-band. Env
            ``GB_PARA_SEARCH_NWALKERS``. Default 16.
        para_search_stage2_ntemps: Stage-2 rungs. Env
            ``GB_PARA_SEARCH_STAGE2_NTEMPS``. Default 4.
        para_search_snr_min: Detection SNR floor. Env
            ``GB_PARA_SEARCH_SNR_MIN``. Default 5.0.
        para_search_det_frac: Minimum detection / optimal SNR ratio. Env
            ``GB_PARA_SEARCH_DET_FRAC``. Default 0.7.
        para_search_converge_tol: Convergence tolerance (nats). Env
            ``GB_PARA_SEARCH_CONVERGE_TOL``. Default 0.5.
        para_search_converge_window: Convergence window (steps). Env
            ``GB_PARA_SEARCH_CONVERGE_WINDOW``. Default 50.
        para_search_min_steps: Env ``GB_PARA_SEARCH_MIN_STEPS``. Default 100.
        para_search_max_steps: Env ``GB_PARA_SEARCH_MAX_STEPS``. Default 600.
        para_search_stage2_steps: Env ``GB_PARA_SEARCH_STAGE2_STEPS``.
            Default 100.
        para_search_split_bins: Cluster split / merge radius (1/Tobs bins).
            Env ``GB_PARA_SEARCH_SPLIT_BINS``. Default 4.0.
        para_search_cluster_rungs: Rungs pooled for clustering. Env
            ``GB_PARA_SEARCH_CLUSTER_RUNGS``. Default 2.
        para_search_scorer: Stage-1 F-stat (``"auto"`` / ``"chunked"``). Env
            ``GB_PARA_SEARCH_SCORER``. Default ``"auto"``.
        para_search_group_bands: Sub-bands per sampler run (0 = auto). Env
            ``GB_PARA_SEARCH_GROUP_BANDS``. Default 0.
        para_search_init: ``"comb"`` or ``"prior"`` stage-1 start. Env
            ``GB_PARA_SEARCH_INIT``. Default ``"comb"``.
        para_search_init_nsky_min: Env ``GB_PARA_SEARCH_INIT_NSKY_MIN``.
            Default 8.
        para_search_init_nsky_max: Env ``GB_PARA_SEARCH_INIT_NSKY_MAX``.
            Default 64.
        para_search_max_finds: Stage-2 finds kept per walker (<= 0 = no cap).
            Env ``GB_PARA_SEARCH_MAX_FINDS``. Default 512.
        pe_extrinsic_draw: PE F-stat distance births DRAW ``phi0`` /
            ``cos_iota`` / ``psi`` around their maximizers (exact densities)
            instead of pinning them. Env ``GB_PE_EXTRINSIC_DRAW``. Default
            ``True``; dormant in production (``GB_RJ_FSTAT_DIST_BIRTH=0``).
        search_rj_replace: Build ``rj_replace`` (exact-MH replacement after the
            F-stat birth in search). Env ``GB_SEARCH_RJ_REPLACE``. Default
            ``True``; launchers 0.
        pe_rj_replace: Build ``rj_replace_pe`` (the PE twin). Env
            ``GB_PE_RJ_REPLACE``. Default ``True``; launchers 0.
        search_prior_removal: Build ``rj_prior_removal``. Env
            ``GB_SEARCH_PRIOR_REMOVAL``. Default ``False``; production 1 (the
            driver's default).
        search_prior_removal_only: ``rj_prior_removal`` proposes deaths only
            (births force-rejected); ``False`` = births and deaths. Env
            ``GB_SEARCH_PRIOR_REMOVAL_ONLY``. Default ``True``; launchers 1.
        leaf_cap_iter_only: Leaf caps advance on a fixed iteration schedule
            only. Env ``GB_LEAF_CAP_ITER_ONLY``. Default ``False``.
        leaf_cap_per_walker: Per-walker leaf caps. Env
            ``GB_LEAF_CAP_PER_WALKER``. Default ``False``.
        search_stage_per_walker: Per-(walker, band) coarse -> fine opt-SNR
            search schedule. Env ``GB_SEARCH_STAGE_PER_WALKER``. Default
            ``False``; launchers 0.
        search_stage_min_iters: Unchanged-count iterations before a (walker,
            band) promotes. Env ``GB_SEARCH_STAGE_MIN_ITERS``. Default 20.
        opt_snr_limit_search_coarse: Coarse opt-SNR floor (``None`` = the
            move's resolved floor). Env ``GB_OPT_SNR_LIMIT_SEARCH_COARSE``.
        opt_snr_limit_search_fine: Fine opt-SNR floor. Env
            ``GB_OPT_SNR_LIMIT_SEARCH_FINE``. Default 5.0.
        search_shutoff_per_walker: Per-(walker, band) search RJ valve (no
            births / deaths / swaps once that pair's count stops growing in the
            current step). Env ``GB_SEARCH_BAND_SHUTOFF_PER_WALKER``. Default
            ``False``; launchers 1 (with ``GB_SEARCH_BAND_SHUTOFF_ALL_WALKERS=1``
            read by the move).
        search_shutoff_conv_iter: Window (iterations) of that valve test. Env
            ``GB_SEARCH_BAND_SHUTOFF_CONV_ITER``. Default 5; launchers 3.
        cap_divisor: Leaf-cap cells per sub-band (1 = the band grid). Changing
            it on a store needs ``migrate_gb_cap_grid.py``. Env
            ``GB_CAP_DIVISOR``. Default 8; launchers 1.
        cap_stagger: Shift the cap grid half a cell against the band grid; at
            ``cap_divisor == 1`` this gives the midpoint-to-midpoint grid
            (one cell per sub-band, each straddling one band seam; see
            :func:`~lisatools.globalfit.state.make_cap_edges`). Fresh store
            only. Env ``GB_CAP_STAGGER``. Default ``False``; launchers 1.
        cap_overlap_frac: Overlap of neighbouring cap cells, in ``[0, 0.5)``
            (0 = exact partition; forced off at divisor 1). Env
            ``GB_CAP_OVERLAP_FRAC``. Default 0.0; launchers 0.
        wdm_band_slab_layers: WDM layers of each per-band buffer slab: 0 =
            auto (band span + 2 x (leakage + guard)), ``N > 0`` explicit,
            ``None`` (programmatic only) = full active band. Env
            ``GB_WDM_BAND_SLAB_LAYERS``. Default 0; launchers 5.
        wdm_slab_guard_layers: Guard layers per side of the auto slab. Env
            ``GB_WDM_SLAB_GUARD_LAYERS``. Default 1.
        psd_shared_mirror: One per-device replica of the per-walker inverse
            covariance plane instead of a copy per buffer slot (WDM only;
            needs a matching GBGPU wheel). Env ``GB_PSD_SHARED_MIRROR``.
            Default ``False``; launchers 1.
        psd_mirror_parity_proposes: Shadow-check the mirror against the
            per-slot layout for the first ``N`` fills (0 = off). Env
            ``GB_PSD_MIRROR_PARITY_PROPOSES``. Default 0; launchers 0.
        psd_mirror_parity_rows: Rows sampled per parity check. Env
            ``GB_PSD_MIRROR_PARITY_ROWS``. Default 64.
        start_freq_ind: FD start index passed to the waveform kwargs. Default 0.
        t0: Phase / frequency reference epoch (s) of the GB parameters.
            Default 0.0 here; ``GBNoFgGBSettings`` defaults ``None``, which the
            branch prep resolves to ``GB_MOJITO_T_REF`` (mojito data) or the
            synthetic stream start.
        tdi_setup: TDI channel set (``"XYZ"``; ``"AET"`` / ``"AE"``); set from
            ``general.tdi_chan`` by the branch prep.
        use_tdi2: TDI 2nd generation; set from ``general.tdi_chan``.
        waveform_kwargs: GBGPU waveform kwargs; empty -> built from ``dt``,
            ``Tobs``, ``oversample``, ``start_freq_ind``, ``tdi_setup``,
            ``use_tdi2``.
        domain_settings: The run's analysis domain (filled from the general
            setup); WDM vs FD selects the band construction.
        gb_wdm_comp: Runtime WDM likelihood object (chunked-het comp, wrapped
            in the sig-het engine by :func:`build_sighet_engine` when the
            branch's ``sighet_inmodel`` is on, as in production); built at run
            start by the variant's ``setup_gb_moves`` (the branch prep resets
            it to ``None``).

    Example:
        .. code-block:: python

            fit = erebor.all_sources(nwalkers=4)
            fit.gb.ntemps = 24                  # env GB_NTEMPS
            fit.gb.subband_divisor = 8          # env GB_SUBBAND_DIVISOR
            fit.gb.cap_divisor = 1              # env GB_CAP_DIVISOR
            fit.gb.warm_start_components = "parent_refereed.npz"
    """

    A_lims: typing.List[float] = dataclasses.field(default_factory=list)
    f0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    # PER-RANK seed for the prior objects :meth:`GBSetup.init_sampling_info`
    # constructs with their OWN ``np.random.Generator`` (the (f0, Mc) heatmap
    # GMM and the 3-D galaxy sky/distance prior). Stamped at build time by
    # ``StockGlobalFit.prepare_branch_settings`` from
    # ``communication.ranks.rank_build_seed`` -- NOT a user knob and NOT an
    # env knob: a bare run seed here would give every walker block the same
    # prior draws. ``None`` (no ``general.random_seed``) keeps those objects
    # on OS entropy, exactly as before this field existed.
    # The RJ BIRTH container is seeded from it too, per F-stat epoch: the
    # recipe copies it into ``fstat_fit_kwargs["build_seed"]`` and
    # ``GBSpecialRJFStatGridMove._birth_seed(k)`` derives one stream per
    # (rank, epoch) for ``build_gb_birth_distribution(seed=...)``.
    build_seed: typing.Optional[int] = None
    # GB's own tempering ladder size (the engine runs cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("GB_NTEMPS", 24, int))
    # Ceiling on the DYNAMIC nleaves_max sizing (2x catalogue sources in
    # band). The fresh-start prior draw allocates
    # (ntemps, nwalkers, nleaves_max, ndim) float64 several times over, so
    # an uncapped full-band catalogue count OOMs the host at init (observed:
    # 83.8 GB peak against an 80 GB allocation on the 3-month combined run).
    # An explicitly set ``nleaves_max`` bypasses the resolver and this cap.
    nleaves_max_cap: int = dataclasses.field(
        default_factory=env_default("GB_NLEAVES_MAX_CAP", 20000, int)
    )
    m_chirp_lims: typing.List[float] = dataclasses.field(default_factory=list)
    fdot_lims: typing.List[float] = dataclasses.field(default_factory=list)
    phi0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    iota_lims: typing.List[float] = dataclasses.field(default_factory=list)
    psi_lims: typing.List[float] = dataclasses.field(default_factory=list)
    alpha_lims: typing.List[float] = dataclasses.field(default_factory=list)
    delta_lims: typing.List[float] = dataclasses.field(default_factory=list)
    start_freq: float = 0.0001  # this might get adjusted ?
    end_freq: float = 0.025
    # Explicit band_edges override (Hz, ascending, one value per band
    # boundary). If set (e.g. ``fit.gb.band_edges_override = np.array([...])``)
    # it REPLACES the per-WDM-layer edges derived in ``compute_band_edges``.
    band_edges_override: typing.Optional[typing.Any] = None
    # Subdivide each WDM layer into ``subband_divisor`` sub-bands when deriving
    # band_edges in the legacy ``uniform`` mode ONLY (2 -> half-layer_df
    # edges). Ignored if ``band_edges_override`` is set, and IGNORED (with a
    # DeprecationWarning) by the ``get_n`` mode -- free-floating edges have
    # no snap grid (2026-08-15 user ruling: the scheduling division is
    # independent of the WDM pixelization).
    subband_divisor: int = dataclasses.field(
        default_factory=env_default("GB_SUBBAND_DIVISOR", 1, int)
    )
    # WDM band-edge construction mode (ignored if ``band_edges_override`` set):
    #   "uniform" [default] -> today's per-layer edges (one band per
    #           layer_df/subband_divisor step), bit-for-bit unchanged.
    #   "get_n" -> FREE-FLOATING bands at the 2*get_N minimum width
    #           (:func:`get_n_based_band_edges`; 2026-08-15 user rulings):
    #           each band's width is its own MINIMUM
    #           ``2 * get_N(f_max_band) / Tobs`` (the installed chunked-het
    #           FD sizing function, self-consistent in the band's upper
    #           edge), packed left-to-right for the MAXIMAL band count; no
    #           snapping to layer boundaries -- scheduling is independent
    #           of the WDM pixelization (storage slabs stay layer-derived).
    #           Low-f bands (small N) come out narrow, high-f bands (large
    #           N) wide -> more parallel (temp, walker, band) cells where
    #           sources are dense. Env: ``GB_BAND_EDGES_MODE``.
    band_edges_mode: str = dataclasses.field(
        default_factory=env_default("GB_BAND_EDGES_MODE", "uniform", str)
    )
    # OPTIONAL COARSENING for ``band_edges_mode="get_n"``. 0 [default] ->
    # maximal packing at the 2*get_N floor (the authority; most bands, each
    # at its own minimum width). >0 -> adjacent minimum-width bands are
    # MERGED (evenly spread) down to ~this many bands -- merged bands are
    # wider than each member's floor, so the width rule's separation
    # guarantees are preserved. Saturates at the maximal packing (a target
    # above it is a no-op). Env: ``GB_BAND_TARGET_COUNT``.
    band_target_count: int = dataclasses.field(
        default_factory=env_default("GB_BAND_TARGET_COUNT", 0, int)
    )
    # DEPRECATED (2026-08-15 user ruling): band widths in ``get_n`` mode
    # are set by ``2 * get_N(f_max_band)/Tobs`` -- the minimum the
    # chunked-het bookkeeping allows -- so a layer-based minimum width has
    # NO role; a non-default value is warned about and ignored. The knob
    # never affected the ``uniform`` mode. Env: ``GB_BAND_MIN_LAYERS``.
    band_min_layers: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_BAND_MIN_LAYERS", None, float)
    )
    # Band-unit stride for the GB move's concurrent sub-band scheduling
    # (flows to the moves as ``band_units`` via ``group_proposal_kwargs``).
    # Stride k partitions bands into k units by ``band_index % k``; each
    # unit's bands are opened/scored CONCURRENTLY (independent buffer
    # cells), so same-unit bands always have k - 1 closed bands between
    # them. Default 2 = the legacy odd/even parity scheduling,
    # bit-identical to historical behavior. Honored by BOTH the proposal
    # unit loop and run_tempering. The orthogonality physics ruling
    # (FD inner product ~0 => WDM inner product ~0, even within one
    # wavelet layer) makes frequency separation -- not wavelet-pixel
    # support -- the concurrency constraint; the guard is SUPPORT-based
    # (gap between same-unit bands >= GB_ORTHO_SEP_FACTOR x the sum of
    # edge-source half-supports get_N(f_edge)/Tobs -- see
    # ``check_band_support_separation``). Under the 2*get_N width rule
    # stride 2 always passes at factor 1.0; stride 3 adds a full band of
    # clearance. Env: ``GB_BAND_UNIT_STRIDE``.
    band_unit_stride: int = dataclasses.field(
        default_factory=env_default("GB_BAND_UNIT_STRIDE", 2, int)
    )
    oversample: int = 4. # FD
    extra_buffer: int = 5
    # TODO check: keep/remove/document -- these two fields have no reader anywhere
    start_resample_iter: Optional[typing.Tuple[int]] = (-1,)  # -1 so that it starts right at the start of PE
    iter_count_per_resample: Optional[int] = 10
    num_repeat_proposals: int = 100
    search_kwargs: Optional[dict] = None
    group_proposal_kwargs: Optional[dict] = None
    # Peak GPU-memory cap for the GB special move. ``n_subbands`` is the
    # maximum number of (temp, walker, band) sub-band buffer cells resident
    # at once (the ``BandScheduler`` slot count / the move's
    # ``num_band_preload``). The move preloads at most this many cells and
    # swaps finished cells out for pending ones, so PEAK sub-band-buffer
    # memory scales with ``n_subbands``, not the total cell count
    # (ntemps*nwalkers*n_bands). Lower it to bound peak GPU memory on large
    # runs; the historical move default is 20000 (effectively "load every
    # cell"). Env: ``GB_N_SUBBANDS``. Flows to the move via
    # ``group_proposal_kwargs["num_band_preload"]`` (see ``__post_init__``).
    # Peak sub-band-buffer memory is ``n_subbands * per_slot`` once the cap
    # binds (total cells > n_subbands); ntemps/nwalkers/n_bands only decide
    # whether it binds and how many scheduler rounds run, NOT the peak. Default
    # 1024 (was an effectively-uncapped 20000): bounds peak on large multi-band
    # runs while leaving small runs -- whose cell count is already < 1024 --
    # untouched. Env: ``GB_N_SUBBANDS``.
    n_subbands: int = dataclasses.field(
        default_factory=env_default("GB_N_SUBBANDS", 1024, int)
    )
    # Sampling-basis knob: sample **chirp mass ``Mc``** at slot 2 (Msol)
    # instead of ``fdot``. The physical basis is unchanged (fdot); the
    # sampling->physical :class:`TransformContainer` gets a multi-parameter
    # ``(f0, Mc) -> (f0, fdot)`` transform via
    # :func:`gbgpu.utils.utility.get_fdot`, and ``key_map={"Mc": "fdot"}``
    # places the sampled Mc at the fdot slot before the transform runs.
    # **Default: True** -- Mc is the physically meaningful axis (fdot varies
    # by ~10 orders of magnitude across the GB band, while Mc is ~O(1) Msol
    # everywhere), and the F-stat proposal / astrophysical joint prior both
    # live natively in the Mc axis. Set ``GB_USE_CHIRP_MASS=0`` to fall back
    # to the legacy fdot sampling basis (requires ``fdot_lims`` populated).
    use_chirp_mass: bool = dataclasses.field(
        default_factory=env_default("GB_USE_CHIRP_MASS", True, bool)
    )
    # When ``use_chirp_mass`` is on, swap the separate uniform priors on
    # ``f0`` and ``Mc`` for the 6-component astrophysical GMM fit from
    # ``heatmap_GMMs.ipynb`` (combined heatmap of 102 population-synthesis
    # models; :class:`lisatools.sampling.f0_mchirp_prior.F0McGMMSampling`,
    # truncated to the run's (f0, Mc) box and renormalized). **Default:
    # True** -- every GB-carrying stock fit now samples the astrophysical
    # joint (f0, Mc) prior; set ``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR=0`` for
    # the legacy separate uniforms. Ignored (with the legacy fdot basis)
    # when ``use_chirp_mass=False``.
    use_astrophysical_f0_mc_prior: bool = dataclasses.field(
        default_factory=env_default("GB_USE_ASTROPHYSICAL_F0_MC_PRIOR", True, bool)
    )
    # Half-width M of the uniform prior on the 9th sampled parameter
    # ``fdot_astro_ratio`` (see :attr:`use_fdot_astro`): ``r = fdot_astro /
    # fdot_gr ~ U[-M, M]``, physical ``fdot = fdot_gr(f0, Mc) * (1 + r)``,
    # so ``r < -1`` represents interacting ``fdot < 0`` systems and the
    # total fdot is bounded to ``(1 +/- M) * fdot_gr`` by construction.
    # Catalogue facts (scripts/gb/fdot_astro_prior_study.py, full wdwd
    # catalogue): at truth Mc the whole population lies in ``r`` in
    # ``[-1.5, +5e-4]``; the mirror-convention seeds land at ``r = 0`` /
    # ``r = -2``; the most negative sources need ``r`` down to ``-4.83``
    # even at Mc = 1.0 -- so M >= 5 keeps every catalogue source
    # representable (M = 3 would exclude the loudest accretors). Env:
    # ``GB_FDOT_ASTRO_RATIO_MAX``.
    fdot_astro_ratio_max: float = dataclasses.field(
        default_factory=env_default("GB_FDOT_ASTRO_RATIO_MAX", 5.0, float)
    )
    # Distance box [d_min, d_max] in kpc for slot 0 of the distance basis
    # (see :attr:`use_distance`): the amplitude is DERIVED from the sampled
    # (dist, f0, Mc), so distance -- not lnA -- is the free parameter. The
    # 1 pc floor keeps A finite (A propto 1/d). PLACEHOLDER uniform box per
    # the incoming Galaxy 3-D (dist, sky) distribution; tighten when that
    # lands. Env: ``GB_DIST_LIMS`` (comma pair) is not wired -- set the
    # field directly. Units: kpc (gbgpu ``get_amplitude`` convention).
    dist_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.001, 40.0]
    )
    # Incoming 3-D joint prior over ``(dist, alpha, sin_delta)`` (the
    # Galaxy-shaped sky+distance distribution). An eryn duck-typed
    # distribution: ``rvs(size) -> (size, 3)`` columns (dist[kpc],
    # alpha[rad], sin_delta), ``logpdf((n, 3)) -> (n,)``; must pickle
    # (sprint rule). None (default) -> a placeholder nested
    # ``ProbDistContainer`` of independent uniforms (dist over ``dist_lims``,
    # alpha over ``alpha_lims``, sin_delta over ``sin(delta_lims)``) is built
    # in :meth:`GBSetup.init_sampling_info`. Swap this in for the real
    # distribution; the birth container's independent U(dist) then needs a
    # p(dist | sky) draw (documented follow-up).
    sky_dist_distribution: typing.Optional[typing.Any] = None
    # Build the 3-D Milky Way prior over (dist, alpha, sin_delta) instead of
    # the independent-uniform placeholder. Ignored when
    # ``sky_dist_distribution`` is set explicitly (that always wins).
    # See lisatools.globalfit.priors.galaxy_sky_dist; validate with
    # scripts/gb/check_galaxy_sky_dist.py.
    use_galaxy_prior: bool = dataclasses.field(
        # Default ON (2026-08-12 user ruling): the 3-D Milky Way joint is the
        # production prior; GB_USE_GALAXY_PRIOR=0 restores the independent
        # uniforms (dist x sky placeholder).
        default_factory=env_default("GB_USE_GALAXY_PRIOR", True, bool)
    )

    @property
    def use_fdot_astro(self) -> bool:
        """9-parameter basis switch (deliberately NOT a separate knob).

        The ``fdot_astro_ratio`` column is part of the astrophysical-prior
        package: active iff ``use_chirp_mass and
        use_astrophysical_f0_mc_prior``. ``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR
        =0`` (or ``GB_USE_CHIRP_MASS=0``) reverts to the 8-parameter basis.
        """
        return bool(self.use_chirp_mass and self.use_astrophysical_f0_mc_prior)

    @property
    def use_distance(self) -> bool:
        """Distance-basis switch (slot 0 = ``dist`` instead of ``lnA``).

        Part of the SAME astrophysical package as :attr:`use_fdot_astro`
        (also NOT a separate knob): once Mc is sampled the amplitude is
        derived, so distance is the natural free parameter and the prior can
        carry a 3-D (dist, sky) joint. Active exactly when
        :attr:`use_fdot_astro` is.
        """
        return self.use_fdot_astro
    # Custom RJ-birth distribution for the prior RJ moves. An eryn
    # duck-typed distribution over the FULL 8-column GB sampling basis
    # (``rvs(size) -> (size, 8)`` with f0 in mHz at column 1 and Mc at
    # column 2; ``logpdf((n, 8)) -> (n,)``). Build one from a 4-D intrinsic
    # proposal (e.g. :class:`lisatools.sampling.fstat_proposal
    # .FStatProposal4D`) with :func:`lisatools.sampling.fstat_proposal
    # .make_gb_rj_birth_container`; wrap narrow proposals in a
    # ``UniformFloorMixture`` so death factors stay finite wherever
    # refined leaves can drift. Must pickle/deepcopy (sprint rule) --
    # FStatProposal4D-backed containers do. None (default) keeps the
    # stock global-prior births.
    rj_birth_distribution: typing.Optional[typing.Any] = None
    # Warm-start components npz path (workstream B, user ruling 2026-08-24):
    # the clustered previous-run posterior components file produced by
    # lisatools.globalfit.warmstart.fit_from_store. Non-empty -> build_gb_moves
    # constructs the ``rj_warm_search`` RJ birth move (the prior-RJ move
    # class drawing full 9-column births from
    # ``lisatools.globalfit.warmstart.proposal.WarmStartComponents``: mixture
    # weights ~ inclusion probability p, f0 windows re-derived against the
    # NEW run's 1/Tobs, no phase maximization); the staged search recipes
    # order it IMMEDIATELY BEFORE ``rj_fstat_search``. Empty/unset (default)
    # -> the feature is entirely absent and runs are bit-identical. A plain
    # path string -> deepcopy/pickle safe. Env: GB_WARM_START_COMPONENTS.
    warm_start_components: str = dataclasses.field(
        default_factory=env_default("GB_WARM_START_COMPONENTS", "", str)
    )
    # RESERVED (v1 parse-only; documented in submit_gf_6mo_v1.sh's staged
    # warm-start block): the future MIXTURE WEIGHT of the warm-start
    # component against the F-stat proposal once both feed ONE birth move
    # (v2). v1 runs them as separate sequential moves (rj_warm_search then
    # rj_fstat_search), so this parses but is unused. Env:
    # GB_WARM_START_WEIGHT.
    # TODO check: keep/remove/document -- warm_start_weight has no reader anywhere
    warm_start_weight: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_WARM_START_WEIGHT", None, float)
    )
    # -- warm-start SELF refit (user rulings 2026-10-13;
    # lisatools.globalfit.warmstart.self_refit) --------------------------
    # "Starting in gb search 2 (and in full pe). Let's rebuild our warm
    # start proposal with the last 50 samples (x4 walkers)." The HEAD, at a
    # warm move's entry once the cadence has elapsed (warm_self_at_entry;
    # the saver after its saves when off), refits the warm-start mixture
    # from the run's OWN last ``warm_self_last_k`` cold rows x every walker
    # (the parent build's fit -> referee -> apply, at the run's Tobs, then
    # the consensus filter) into ``<run_dir>/warmstart/self/epoch_NNNN/``;
    # every rank adopts the newest complete epoch, and rj_warm_search /
    # rj_warm_pe then draw from the ORIGINAL parent warm start, the FULL self
    # refit and the FILTERED one by ``warm_self_split_search`` /
    # ``warm_self_split_pe`` ("We should keep the original warm start in the
    # fit"). Needs ``warm_start_components`` (no warm move otherwise).
    #
    # warm_self: master switch. False (default) -> no builder, no mixer, the
    # warm moves hold their original container: bit-identical runs.
    # Env: GB_WARM_SELF.
    warm_self: bool = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF", False, bool)
    )
    # warm_self_every_search: STORED ROWS between self refits in the numbered
    # search stages, gb_search_1 on (user rulings 2026-10-14, after the first
    # PE epoch's cross-walker births were seen to carry 84 % of the lnL gain:
    # "update the warm start refit in gb search 2 to every iteration" (was 5),
    # then "gb search 1 and gb search 2 will have warm start refits every
    # iteration" (gb_search_1 did not build before)). The clock is
    # stage-relative (the stage's start row, or the latest complete epoch's
    # built_at row, whichever is later) and read from disk, so a resume never
    # rebuilds early. <= 0 -> no refit in search. Env: GB_WARM_SELF_EVERY_SEARCH.
    warm_self_every_search: int = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_EVERY_SEARCH", 1, int)
    )
    # warm_self_every_pe: stored rows (= iterations) between self refits in
    # full_pe (2026-10-15: "Set the default refitting duration for full pe
    # ... to be 25 iterations ... both the relaunch of 9mo and full year";
    # was 150, before that 250). With the head build (warm_self_at_entry)
    # it is the minimum number of rows since the last build, checked when
    # rj_warm_pe is drawn. A periodic approximation. <= 0 -> no refit in PE.
    # Env: GB_WARM_SELF_EVERY_PE.
    warm_self_every_pe: int = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_EVERY_PE", 25, int)
    )
    # warm_self_at_entry: build every refit ON THE HEAD at the warm move's
    # entry (user ruling 2026-10-15: "Let's just build everything on the
    # head at entry time for the refit ... Make that the default."). The
    # head keeps the last ``warm_self_last_k`` stored cold rows in memory
    # (filled from the store at start-up, appended at each save) and, when
    # the cadence above has elapsed, runs fit -> referee -> apply -> the
    # consensus filter before the propose that draws from it; every rank
    # waits (~75 s on the 9mo). False -> the saver rank builds after its
    # saves instead (the 2026-10-13 design). Env: GB_WARM_SELF_AT_ENTRY.
    warm_self_at_entry: bool = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_AT_ENTRY", True, bool)
    )
    # warm_self_last_k: how many of the run's latest stored cold rows feed a
    # refit, every walker of each (50 x 4 walkers = 200 posterior samples).
    # Env: GB_WARM_SELF_LAST_K.
    warm_self_last_k: int = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_LAST_K", 50, int)
    )
    # warm_self_split_search / warm_self_split_pe: weights "original, full,
    # filtered" over the three containers a warm move draws from once a self
    # epoch exists (100 % original before): the ORIGINAL parent warm start
    # (never changes), the FULL self refit, and the FILTERED self refit
    # (the full one minus the clusters every walker already holds, see
    # warm_self_consensus). One build makes both self containers. Each
    # propose draws one at random, probability ~ weight (1,1,1 = one third
    # each); it serves the births AND every density of the propose. User
    # rulings 2026-10-15: search "100% refit on sources that do not exist in
    # all four (with high confidence)" -> "0,0,1", then "do the split between
    # the old run fit warm start and the refit filtered warm start during gb
    # search 1/2 so we make sure not to leave out leaves from the previous
    # run ... 20 % old run refit warmstart / 80% self refit warmstart
    # filtered" -> "0.2,0,0.8", then 2026-10-16 "Let's split them 0.5/0/0.5 in search"
    # -> "0.5,0,0.5" (the 1yr tar 4: the original still held ~1,000 sources
    # no walker had); PE "split the refits three ways ... each 1/3
    # of the time" -> "1,1,1". These replace GB_WARM_SELF_FRACTION /
    # _FRACTION_SEARCH (0.75 / 1.0, two containers).
    # Env: GB_WARM_SELF_SPLIT_SEARCH, GB_WARM_SELF_SPLIT_PE.
    warm_self_split_search: str = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_SPLIT_SEARCH", "0.5,0,0.5", str)
    )
    warm_self_split_pe: str = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_SPLIT_PE", "1,1,1", str)
    )
    # warm_self_consensus: a refit cluster is "in all walkers with high
    # confidence" -- and dropped from the FILTERED container -- when EVERY
    # walker holds it in >= this fraction of its refit rows AND >= this
    # fraction of the samples hold the same number of its leaves (a blend
    # one walker resolves and another does not is kept). Measured on the
    # 9mo (rows 82-89 and 103-110): ~2,160 of ~3,230 source groups are
    # consensus and took 83 % of the full mixture's draws; the filter takes
    # the share of draws landing on a source the walker already holds from
    # 93 % to 57 %. Env: GB_WARM_SELF_CONSENSUS.
    warm_self_consensus: float = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_CONSENSUS", 0.95, float)
    )
    # warm_self_adopt_lag: the ONE-DRAW ADOPTION LAG of the head's entry
    # refit (user ruling 2026-10-16: "refit on the current data holding our
    # previous within-run refit information, use the previous information on
    # the current proposal and then switch it out"). A warm propose whose
    # entry builds epoch k draws from the epoch held before the build (the
    # original warm start on the run's first build) and ships THAT epoch to
    # every rank; the next warm draw adopts k. On the 9mo job 767 every draw
    # from an epoch built at that same entry lost 15-20 leaves and 100-527
    # nats per walker, while draws on epochs >= 13 rows old were the job's
    # best. Epochs already complete on disk are adopted at once either way.
    # False -> the build propose draws from its own new epoch (pre-10-16).
    # Env: GB_WARM_SELF_ADOPT_LAG.
    warm_self_adopt_lag: bool = dataclasses.field(
        default_factory=env_default("GB_WARM_SELF_ADOPT_LAG", True, bool)
    )
    # ---- GB PARALLEL RESIDUAL SEARCH (rj_para_search_pe; user ruling
    # 2026-10-16, docs/gb-para-search.md). "build the paraensemble setup to
    # sample 1 GB per sub-band per walker (cold-chain only) ... run the
    # parallel-tempered fixed-dimensional MCMC on each cold (walker, band)
    # until they all converge per sub-band in max logL. Then determine if any
    # source has been found at all above optimal SNR of 5 or 7 (probably 5),
    # turn each sub-band's surviving source into a GMM, combine them into a
    # large GMM, and then we slot that in and sample from it."
    # para_search: master switch. True -> build_gb_moves builds
    # rj_para_search_pe and setup_gb_moves composes it into the PE stage(s)
    # carrying rj_prior_pe; its PE draw-one share is GB_PE_RJ_PARA_FRACTION
    # (default 0: outside the draw, every PE iteration). False (default) -> no
    # move object exists; the run is bit-identical. Env: GB_PARA_SEARCH.
    para_search: bool = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH", False, bool)
    )
    # para_search_every_pe: PE iterations between two searches (the move's
    # journaled clock, <run_dir>/warmstart/parasearch/clock.json; the first
    # build runs at the move's first PE entry; <= 0 -> that first build only).
    # Env: GB_PARA_SEARCH_EVERY_PE.
    para_search_every_pe: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_EVERY_PE", 100, int)
    )
    # para_search_ntemps / _nwalkers: chains per sub-band (one GB each) of the
    # stage-1 F-stat PT; stage 2 runs para_search_stage2_ntemps rungs of the
    # same walkers. Env: GB_PARA_SEARCH_NTEMPS, GB_PARA_SEARCH_NWALKERS,
    # GB_PARA_SEARCH_STAGE2_NTEMPS (<= 0 -> the stage-1 count).
    para_search_ntemps: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_NTEMPS", 8, int)
    )
    para_search_nwalkers: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_NWALKERS", 16, int)
    )
    para_search_stage2_ntemps: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_STAGE2_NTEMPS", 4, int)
    )
    # para_search_snr_min / _det_frac: the detection gate. Stage 1 keeps a
    # cluster whose best sqrt(2F) >= snr_min; stage 2 keeps a find whose best
    # sample has opt SNR sqrt(<h|h>) >= snr_min and det SNR
    # <r|h>/sqrt(<h|h>) >= det_frac x opt. Env: GB_PARA_SEARCH_SNR_MIN,
    # GB_PARA_SEARCH_DET_FRAC.
    para_search_snr_min: float = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_SNR_MIN", 5.0, float)
    )
    para_search_det_frac: float = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_DET_FRAC", 0.7, float)
    )
    # para_search_converge_tol / _window / _min_steps / _max_steps: a sub-band
    # stops once its cold-rung running max lnL has not improved by more than
    # tol nats over window steps, after >= min_steps and at most max_steps.
    # Env: GB_PARA_SEARCH_CONVERGE_TOL, GB_PARA_SEARCH_CONVERGE_WINDOW,
    # GB_PARA_SEARCH_MIN_STEPS, GB_PARA_SEARCH_MAX_STEPS.
    para_search_converge_tol: float = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_CONVERGE_TOL", 0.5, float)
    )
    para_search_converge_window: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_CONVERGE_WINDOW", 50, int)
    )
    para_search_min_steps: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_MIN_STEPS", 100, int)
    )
    para_search_max_steps: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_MAX_STEPS", 600, int)
    )
    # para_search_stage2_steps: the short full-likelihood stage per find (the
    # GMM is fitted to the cold rung's last half). Env:
    # GB_PARA_SEARCH_STAGE2_STEPS.
    para_search_stage2_steps: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_STAGE2_STEPS", 100, int)
    )
    # para_search_split_bins: 2+ sources per sub-band. The pooled cold +
    # warm-rung samples (para_search_cluster_rungs rungs) and comb cells split
    # into clusters further apart than this many 1/Tobs bins; each cluster
    # that passes the gate is its own find. Also the merge radius across
    # walkers and band edges. Env: GB_PARA_SEARCH_SPLIT_BINS,
    # GB_PARA_SEARCH_CLUSTER_RUNGS.
    para_search_split_bins: float = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_SPLIT_BINS", 4.0, float)
    )
    para_search_cluster_rungs: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_CLUSTER_RUNGS", 2, int)
    )
    # para_search_scorer: the stage-1 F-stat ("auto" = the shared-reference
    # sig-het F-stat when the GB comp has it, else "chunked", the exact
    # per-candidate one); para_search_group_bands: sub-bands per sampler run
    # (0 = as many as one sig-het reference block holds; all with chunked).
    # Env: GB_PARA_SEARCH_SCORER, GB_PARA_SEARCH_GROUP_BANDS.
    para_search_scorer: str = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_SCORER", "auto", str)
    )
    para_search_group_bands: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_GROUP_BANDS", 0, int)
    )
    # para_search_init: "comb" (default) seeds the stage-1 cold + warm rungs
    # at the best cells of a 1-bin f0 comb x n_sky sky points per sub-band
    # (n_sky = the F-stat fit's Doppler law, clipped to
    # [para_search_init_nsky_min, para_search_init_nsky_max]); "prior" starts
    # every rung from the prior. Env: GB_PARA_SEARCH_INIT,
    # GB_PARA_SEARCH_INIT_NSKY_MIN, GB_PARA_SEARCH_INIT_NSKY_MAX.
    para_search_init: str = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_INIT", "comb", str)
    )
    para_search_init_nsky_min: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_INIT_NSKY_MIN", 8, int)
    )
    para_search_init_nsky_max: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_INIT_NSKY_MAX", 64, int)
    )
    # para_search_max_finds: stage-2 cost bound per walker (the loudest
    # stage-1 finds are kept; <= 0 = no cap). Env: GB_PARA_SEARCH_MAX_FINDS.
    para_search_max_finds: int = dataclasses.field(
        default_factory=env_default("GB_PARA_SEARCH_MAX_FINDS", 512, int)
    )
    # PE-stage extrinsic DRAW (user design ruling 2026-08-25). STAGE
    # SPLIT: the SEARCH RJ stages (rj_fstat_search / rj_prior_removal /
    # rj_replace) PIN phi0/cos_iota/psi deterministically at the F-stat
    # maximizers and charge them as uniform constants — unchanged,
    # bit-identical. The PE stages (rj_fstat_pe / rj_prior_pe, when their
    # F-stat distance-birth path is active) instead DRAW each extrinsic
    # from a genuine distribution centered on its maximizer — von Mises
    # for phi0 (period 2 pi), von Mises on the DOUBLED angle for psi
    # (period pi), truncated Gaussian on [-1, 1] for cos iota — each
    # eps-floor-mixed with the uniform law on its domain, and charge the
    # real forward AND reverse densities in the RJ factors (exact
    # detailed balance; deaths evaluate the reverse density around the
    # dead row's OWN maximizers). Widths sigma = geom/SNR from the F-stat
    # curvature; the (phi0 + pi, psi + pi/2) F-stat identity is handled
    # by summing the density over both representatives. See
    # ``lisatools.sampling.fstat_proposal.pe_extrinsic_rvs`` /
    # ``pe_extrinsic_logpdf`` and ``GBSpecialBase._pe_or_pin_extrinsics``.
    # Default ON; =0 restores today's PE pin + uniform-wash behavior
    # bit-identically. Companion env-only knobs:
    # GB_PE_EXTRINSIC_FLOOR_EPS (default 0.05) and
    # GB_PE_EXTRINSIC_SIGMA_GEOM (default 2.0).
    # Env: GB_PE_EXTRINSIC_DRAW.
    pe_extrinsic_draw: bool = dataclasses.field(
        default_factory=env_default("GB_PE_EXTRINSIC_DRAW", True, bool)
    )
    # -- search RJ cycle knobs (2026-08-01; search mode ONLY, i.e.
    # GB_MODE=search) -- when on, the variant's recipe setup inserts the
    # matching stock move(s) right after the fstat-birth prior move
    # (``rj_prior``) in its stage, so the stage's GFCombineMove runs the
    # per-iteration cycle fstat-birth -> fstat-REPLACE -> prior-REMOVAL in
    # guaranteed order (eryn CombineMove proposes its moves sequentially in
    # list order). PE recipes are untouched. Plain bools -> deepcopy/pickle
    # safe.
    #
    # search_rj_replace: build + install the fixed-dimension REPLACEMENT
    # move ``rj_replace`` directly AFTER the fstat-birth move in the
    # gb_search cycle (2026-08-24 exact-MH reinstatement, USER directive):
    # full 9-column candidates from the F-stat grid container + epoch
    # center table (stored maximizing extrinsics), scored at their EXACT
    # likelihood -- never phase-maximized -- so a higher-SNR table draw
    # can replace a mis-seated leaf in one accepted move. See
    # GBSpecialBase._run_replace_step for the acceptance derivation.
    # Default ON (search recipes only; full_pe is untouched -- PE
    # installation is future work). Env: GB_SEARCH_RJ_REPLACE.
    search_rj_replace: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_RJ_REPLACE", True, bool)
    )
    # pe_rj_replace: the PE-mode sibling of ``search_rj_replace`` (USER
    # directive 2026-08-28, "we need a PE replace that also uses the same
    # machinery as fstat pe"): build + install ``rj_replace_pe`` directly
    # AFTER the pe-named F-stat birth move (``rj_fstat_pe``) in the PE
    # cycle. SAME move, PE-flavored: PE repeat/SNR defaults, PE flip
    # fraction, the shared PE tempering cadence, and the pe-strict cap
    # config -- and, critically, the SAME extrinsic-center machinery the
    # pe-named F-stat moves use, the epoch CENTER TABLE (the install
    # stamps ``replace_pe_stage``, which is what
    # GBSpecialBase._replace_ctr_mode resolves "table" from; the SEARCH
    # install keeps per-row centers bit-identically).
    #
    # EXACT DETAILED BALANCE, unlike the search install: the PE move does
    # NOT carry ``replace_search_stage`` and its name has no "search" in
    # it, so GBSpecialBase._replace_fstat_max resolves False and slot 0 is
    # genuinely DRAWN (truncated + floor-mixed lognormal about the epoch
    # center) and priced on both sides -- the maximize-then-pretend search
    # exception never arms here, where it would maximization-bias the
    # amplitude posterior. See GBSpecialBase._run_replace_step for the
    # acceptance derivation.
    #
    # Default ON (the move is exact-MH, so it is posterior-valid); set
    # GB_PE_RJ_REPLACE=0 to build a PE cycle without it. Env:
    # GB_PE_RJ_REPLACE.
    pe_rj_replace: bool = dataclasses.field(
        default_factory=env_default("GB_PE_RJ_REPLACE", True, bool)
    )
    # search_prior_removal: build + install the prior-container RJ move
    # ``rj_prior_removal`` (GBSpecialRJPriorMove with the PRIOR container;
    # what it proposes is ``search_prior_removal_only`` below). Env:
    # GB_SEARCH_PRIOR_REMOVAL.
    search_prior_removal: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_PRIOR_REMOVAL", False, bool)
    )
    # search_prior_removal_only: what ``rj_prior_removal`` proposes. True
    # (default, user ruling 2026-09-22): the original pruning move --
    # births force-rejected, only deaths are judged against the prior.
    # False: the prior container proposes BOTH births and deaths -- the
    # configuration the 6mo production run has used since stored iteration
    # 300 (2026-09-21, first as a hand edit in recipe.py); the 6mo submit
    # script exports 0 for that reason. Env: GB_SEARCH_PRIOR_REMOVAL_ONLY.
    search_prior_removal_only: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_PRIOR_REMOVAL_ONLY", True, bool)
    )
    # leaf_cap_iter_only: per-band progressive leaf caps advance on a FIXED
    # schedule -- the only gate is ``iters >= GB_LEAF_CAP_MIN_ITERS`` (the
    # lnL-plateau and occupancy tests are skipped; the cap < nleaves_max
    # guard stays). Env: GB_LEAF_CAP_ITER_ONLY.
    leaf_cap_iter_only: bool = dataclasses.field(
        default_factory=env_default("GB_LEAF_CAP_ITER_ONLY", False, bool)
    )
    # leaf_cap_per_walker: give EVERY WALKER its own per-cap-cell leaf cap
    # (user request 2026-09-22). The cap family
    # (``cap_cell_leaf_cap``/``_iters``/``_best_ll``) gains a walker axis:
    # each walker earns its allowance from its OWN lnL plateau and its OWN
    # occupancy-at-cap, and every enforcement gate -- RJ birth, in-model
    # drift, the replacement move, the pick pool, the tempering swap gate --
    # scores a row against ``cap[its walker, its cell]``.
    #
    # WHY. The gate collapses the walker axis the moment it runs
    # (``cur_max = lls.max(axis=0)``), so the LEADING walker's evidence
    # ramps the allowance for everyone, including laggards that never
    # filled the cap they already had. On the 6-month production run that
    # is a 400-1000 leaf spread across four cold walkers.
    #
    # DEFAULT FALSE, and off is a HARD guarantee, not a convention: no
    # per-walker array is allocated, no new HDF5 dataset is written and no
    # new branch is taken, so the live 6-month run can be relaunched from a
    # build carrying this change and behave exactly as before
    # (``tests/test_gb_cap_per_walker.py`` asserts it). Turning it ON over
    # an existing store needs no migration -- every walker is seeded by
    # broadcast from the stored shared cap.
    #
    # Subsumes GB_LEAF_CAP_ALL_WALKERS, which is ignored (with a warning)
    # while this is on. Diagnostics: ``[GB_CAP_PW]`` log lines, plus
    # ``GB_CAP_PW_DIAG=1`` for the per-increment decision tuple.
    # Env: GB_LEAF_CAP_PER_WALKER.
    leaf_cap_per_walker: bool = dataclasses.field(
        default_factory=env_default("GB_LEAF_CAP_PER_WALKER", False, bool)
    )
    # ------------------------------------------------------------------
    # PER-(WALKER, BAND) SEARCH SCHEDULE (user request 2026-09-23)
    # ------------------------------------------------------------------
    # search_stage_per_walker: the SEARCH's opt-SNR floor stops being one
    # number for the whole run and becomes a function of a (walker, band)
    # STAGE. Stage 0 = COARSE (``opt_snr_limit_search_coarse``, hunting
    # bright sources), stage 1 = FINE (``opt_snr_limit_search_fine``,
    # digging the faint tail). A band promotes once THAT WALKER's
    # cold-chain source count there has been unchanged for
    # ``search_stage_min_iters`` consecutive iterations.
    #
    # WHY. A band where a walker has already assembled its sources wants a
    # LOWER floor; a band still finding bright ones wants the HIGH floor,
    # or the model fills with noise births. Those two states coexist at the
    # same iteration, in different bands, and in the SAME band on different
    # walkers -- which one global number cannot express.
    #
    # The F-stat peak floor follows per BAND but not per walker: there is
    # ONE catalog, swept against ONE reference walker's residual, so a band
    # takes the LOOSER floor once any walker has promoted
    # (FSTAT_PEAK_MIN_SNR_COARSE / _FINE). The per-walker half of the
    # schedule is carried entirely by the opt-SNR boundary, which IS per
    # row. Phase maximization is NOT part of this (deferred 2026-09-24).
    #
    # DEFAULT FALSE, and off is a HARD guarantee: no array allocated, no
    # dataset written, the floor stays the scalar float it has always been.
    # Env: GB_SEARCH_STAGE_PER_WALKER.
    search_stage_per_walker: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_STAGE_PER_WALKER", False, bool)
    )
    # Consecutive unchanged-source-count iterations a (walker, band) must
    # post before it promotes. Env: GB_SEARCH_STAGE_MIN_ITERS.
    search_stage_min_iters: int = dataclasses.field(
        default_factory=env_default("GB_SEARCH_STAGE_MIN_ITERS", 20, int)
    )
    # The two opt-SNR floors. COARSE defaults to None = "whatever the
    # move's floor already resolved to", so arming the schedule with no
    # other knob reproduces today's behaviour until the first promotion.
    # Env: GB_OPT_SNR_LIMIT_SEARCH_COARSE / GB_OPT_SNR_LIMIT_SEARCH_FINE.
    opt_snr_limit_search_coarse: typing.Optional[float] = dataclasses.field(
        default_factory=env_default(
            "GB_OPT_SNR_LIMIT_SEARCH_COARSE", None, float)
    )
    opt_snr_limit_search_fine: float = dataclasses.field(
        default_factory=env_default("GB_OPT_SNR_LIMIT_SEARCH_FINE", 5.0, float)
    )
    # search_shutoff_per_walker: a SEARCH-only per-(walker, band) RJ valve,
    # ADDITIONAL to the existing per-band one (which is untouched; the two
    # compose with OR). Once a (walker, band)'s cold-chain source count has
    # stopped GROWING within the CURRENT RECIPE STEP -- the per-pair form
    # of the nleaves-max window test ``RJRecipeStep.stopping_function``
    # uses to end the step -- that pair takes no further RJ (births OR
    # deaths, and no tempering swaps) until the next recipe step begins.
    #
    # Scoped to the step and RELEASED when the next one starts: the next
    # step changes the moves, the caps and the floors, so "nothing left to
    # find here" is a statement about a configuration that has just been
    # replaced. Env: GB_SEARCH_BAND_SHUTOFF_PER_WALKER.
    search_shutoff_per_walker: bool = dataclasses.field(
        default_factory=env_default(
            "GB_SEARCH_BAND_SHUTOFF_PER_WALKER", False, bool)
    )
    # Window length for that test, i.e. ``RJRecipeStep.convergence_iter``'s
    # per-pair twin. Env: GB_SEARCH_BAND_SHUTOFF_CONV_ITER.
    search_shutoff_conv_iter: int = dataclasses.field(
        default_factory=env_default(
            "GB_SEARCH_BAND_SHUTOFF_CONV_ITER", 5, int)
    )
    # cap_divisor: the LEAF-CAP CELL grid (user design 2026-08-15). Each
    # sub-band is split into this many equal pieces and the per-cell leaf
    # caps are enforced on THOSE, not on the sub-band. Sub-band widths are
    # set by what the likelihood engine can run concurrently; the scale at
    # which two GB sources actually get confused is the posterior width,
    # which is much narrower. NOTHING ELSE MOVES -- scheduling, units,
    # buffers, tempering, band shutoff and the cell-ll credit all stay on
    # the band grid, and every cap cell is contained in exactly one band.
    #
    # DEFAULT 8, from the snap11 3-month store (154 uniform 138.9 uHz bands
    # = 1080 FD bins each; 1147 coexisting same-band cold pairs): the pair
    # separation distribution is bimodal -- ~10% of pairs sit within 8 FD
    # bins (genuinely degenerate, no subdivision separates them) and the
    # rest spread near-uniformly across the band. Directly assigning cells,
    # the fraction of pairs SHARING a cap cell is 100% at K=1, 48.9% at
    # K=2, 23.1% at K=4, 16.5% at K=8 and 12.8% at K=16: 4 -> 8 removes 76
    # shared cells, 8 -> 16 only another 42 for double the bookkeeping. A
    # K=8 cell is 17.4 uHz / 135 FD bins, still far wider than a source's
    # own 3-month frequency extent (~+-2 uHz of Doppler sidebands), so cell
    # membership stays stable under in-model f0 moves; K=16 starts to
    # approach that scale. 8 is the knee.
    #
    # ``GB_CAP_DIVISOR=1`` reproduces the pre-2026-08-15 per-band caps
    # BIT-IDENTICALLY (the cap grid IS the band grid). Changing this on an
    # existing run requires scripts/fstat_proposal/migrate_gb_cap_grid.py --
    # the resume guard refuses a mismatched store. Env: GB_CAP_DIVISOR.
    cap_divisor: int = dataclasses.field(
        default_factory=env_default("GB_CAP_DIVISOR", 8, int)
    )
    # cap_stagger: STAGGER the cap-cell grid against the sub-band grid
    # (user design 2026-08-20, the v5 grid). Interior cap edges shift by
    # half a cell width so NO cap edge coincides with a band edge -- the
    # leaf-cap seams and the band seams (scheduling / serial-within-band /
    # F-stat fit interior) decouple, and no source can sit on both at
    # once. Band b still owns cells b*K..b*K+K-1 (sizes, reshapes and
    # storage unchanged); the cell at each band seam physically straddles
    # it. At cap_divisor == 1 it is NOT ignored: state.make_cap_edges then
    # builds the midpoint-to-midpoint grid (one cell per sub-band, each
    # straddling one band seam; the production setting). Changing it on an
    # existing store is refused by the resume guard -- fresh store only.
    # Env: GB_CAP_STAGGER.
    cap_stagger: bool = dataclasses.field(
        default_factory=env_default("GB_CAP_STAGGER", False, bool)
    )
    # cap_overlap_frac: OVERLAP the leaf-cap cells (user design 2026-08-23).
    # Each cap cell's enforcement SPAN widens symmetrically so it shares
    # this fraction of its own width with EACH neighbour: 0.25 gives the
    # 1/4-overlap / 1/2-alone / 1/4-overlap layout (cell width w = s/(1-p)
    # for stride s; extension x = (w-s)/2 = s/6 at p=0.25). The stored EDGE
    # grid never changes -- same count, same stride, same stagger -- so
    # resume guards and every stored cap array keep their shapes, and a
    # rewound store picks the new semantics up with no migration. A leaf in
    # an overlap zone is a member of BOTH covering cells: the occupancy
    # census counts it in both, and a location is AT CAP when ANY covering
    # cell is at its cap (AND-headroom for births / cap-drift entries).
    # Purpose: prevents FORMATION of split sources at cap-cell edges (the
    # flagship 20.38 mHz double-count straddled a cell edge -- each
    # fragment lived under its own cell's cap while the pair shared one
    # posterior mode). Only acts while caps are armed (search stages).
    # Valid range [0, 0.5); 0 (default) is bit-identical to the exact
    # partition. Meaningless at cap_divisor == 1 (forced off there).
    # Env: GB_CAP_OVERLAP_FRAC.
    cap_overlap_frac: float = dataclasses.field(
        default_factory=env_default("GB_CAP_OVERLAP_FRAC", 0.0, float)
    )
    # Task-b: narrow per-band WDM slabs. Each per-band sub-band-buffer slab
    # spans a few WDM layers centered on the band instead of the full analysis
    # band ``Nf_active``, cutting the dominant buffer memory term by
    # ~Nf_active/slab_Nf. WDM path only (FD is already per-band narrow).
    # Requires the chunked-het backend built with the task-b per-slab origin
    # (the default CUDA/CPU build). Env: GB_WDM_BAND_SLAB_LAYERS.
    #   0    -> AUTO: band layer span + 2*(leakage + wdm_slab_guard_layers),
    #           leakage=2 (recommended-Tukey estimate) -> ~band_span + 6. [default]
    #   N>0  -> EXPLICIT N layers.
    #   None -> OFF (full active band; bit-identical to pre-task-b). Not
    #           reachable from the env knob (which casts to int); set it
    #           programmatically (``fit.gb.wdm_band_slab_layers = None``) to
    #           restore the pre-task-b full-band buffers.
    #
    # AUTO is the default because the full-band slab is the dominant sub-band
    # buffer term: a cell spans the whole analysis band (e.g. 180 WDM layers)
    # when the GB band itself occupies only a few, so AUTO cuts per-cell
    # memory by ~Nf_active/slab_Nf (~20x at Nf_active=180) on both the device
    # buffers and the host staging copies in the ACA repack.
    wdm_band_slab_layers: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GB_WDM_BAND_SLAB_LAYERS", 0, int)
    )
    # Adjustable guard (extra WDM layers each side) used by the AUTO slab size
    # (wdm_band_slab_layers=0). Env: GB_WDM_SLAB_GUARD_LAYERS.
    wdm_slab_guard_layers: int = dataclasses.field(
        default_factory=env_default("GB_WDM_SLAB_GUARD_LAYERS", 1, int)
    )
    # Shared-psd MIRROR for the GB sub-band buffers (2026-09-09). Env:
    # GB_PSD_SHARED_MIRROR (1/0/true/false/yes/no/on/off). Default OFF.
    #   False -> every buffer slot stores its own inverse-covariance slab,
    #            copied out of the parent ACA's per-walker psd plane on every
    #            unit fill (3.07 of the 4.09 MB per slot at 1 yr; the
    #            template twin allocates a second, never-read copy).
    #   True  -> no per-slot invC: each device holds ONE replica of the
    #            parent's per-walker plane (nwalkers x 110 MB at 1 yr), every
    #            slot carries its walker row + absolute layer origin, and the
    #            chunked-het kernels / sig-het builds index the replica
    #            directly. Values, dtype and the time axis are untouched --
    #            deduplication of copies only, bit-identical by construction.
    #            Requires a GBGPU wheel built against the mirror kernel args
    #            (hard error otherwise; never a silent fallback). WDM-only.
    # Rollback = unset the env var; no rebuild needed.
    psd_shared_mirror: bool = dataclasses.field(
        default_factory=env_default("GB_PSD_SHARED_MIRROR", False, bool)
    )
    # Runtime parity gate for the mirror: for the first N unit fills
    # (proposes) of every buffer, every get_ll / swap_ll / F-stat / sig-het
    # setup on that buffer is shadow-checked against the per-slot layout on a
    # row sample and the run RAISES on the first differing bit (one INFO line
    # per check; a retire line once N is exhausted). 0 = gate off. Meaningful
    # only with psd_shared_mirror=True (logs "disarmed" otherwise).
    # Env: GB_PSD_MIRROR_PARITY_PROPOSES.
    psd_mirror_parity_proposes: int = dataclasses.field(
        default_factory=env_default("GB_PSD_MIRROR_PARITY_PROPOSES", 0, int)
    )
    # Rows sampled per parity check (bounds the shadow scratch: rows x 4 MB
    # at 1 yr). Env: GB_PSD_MIRROR_PARITY_ROWS.
    psd_mirror_parity_rows: int = dataclasses.field(
        default_factory=env_default("GB_PSD_MIRROR_PARITY_ROWS", 64, int)
    )
    start_freq_ind: Optional[int] = 0  # goes into GPU for start of data stream
    t0: Optional[float] = 0.0
    tdi_setup: Optional[str] = "XYZ" # other options are AET and AE.
    use_tdi2: Optional[bool] = True
    waveform_kwargs: dict = dataclasses.field(default_factory=dict)
    # Domain selection for this branch. The user passes the same
    # ``DomainSettingsBase`` that lives on the parent ``GeneralSetup``
    # (FDSettings, STFTSettings, WDMSettings, ...). The band structure /
    # waveform-engine selection branches on ``isinstance(domain_settings, ...)``
    # — no string mode flag.
    domain_settings: Optional[DomainSettingsBase] = None
    # Optional WDM-domain likelihood object (a
    # ``gbgpu.gbcomps.GBWDMComputations`` instance). Required when
    # ``domain_settings`` is a :class:`WDMSettings`; ignored otherwise. The
    # user builds this once their WDM grid + lookup table are known
    # (see global-fit input scripts).
    gb_wdm_comp: typing.Any = None



class GBSetup(Setup, GBSettings):
    """:class:`Setup` for galactic binaries in the Erebor recipe.

    Construction copies the settings, opens the ``gb_setup.log`` logger
    (under ``log_dir``) and runs :meth:`init_setup`: band structure, then
    sampling info, then the GB sub-state / sub-backend. Built once per
    branch by :meth:`StockGlobalFit.build
    <lisatools.globalfit.stock.base.StockGlobalFit.build>` from the block
    completed by ``prepare_gb_branch``.

    Args:
        gb_settings: Settings dataclass holding GB parameter ranges and
            domain configuration.

    Attributes:
        band_edges: ``(num_sub_bands + 1,)`` ascending sub-band edges (Hz).
        band_N_vals: ``(num_sub_bands,)`` FD ``get_N`` per band (shape parity
            only on WDM).
        num_sub_bands: Number of sub-bands.
        logger: The branch-setup logger.
    """

    def __init__(self, gb_settings: GBSettings):

        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, gb_settings)

        level = logging.DEBUG
        name = "GBSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None), shared_console=False)

        self.init_setup()

    def _build_sub_seeds(self, n: int):
        """``n`` independent sub-seeds off :attr:`GBSettings.build_seed`.

        Returns ``[None] * n`` when no build seed was stamped (no
        ``general.random_seed``), which leaves every consumer on OS entropy
        -- the pre-2026-09-16 behaviour. With a seed, the values are the
        deterministic ``SeedSequence(build_seed).spawn(n)`` children, so the
        objects seeded from them are reproducible AND mutually independent.
        ``build_seed`` itself is already per rank (``rank_build_seed``), so
        two walker blocks never share a prior stream.
        """
        seed = getattr(self, "build_seed", None)
        if seed is None:
            return [None] * int(n)
        return [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in np.random.SeedSequence(int(seed)).spawn(int(n))
        ]

    def init_sampling_info(self):
        """Build the GB :class:`TransformContainer`, prior, periodicity, and waveform kwargs.

        Plain parameter names (same convention as
        :func:`make_gb_transform_container`) with ICRS sky angles
        (``alpha`` / ``sin_delta``) -- the run frame is ICRS. ``A`` is
        sampled in ``ln A`` (the ``np.exp`` transform maps to the physical
        amplitude), ``f0`` in mHz.

        When :attr:`GBSettings.use_chirp_mass` is on, slot 2 of the sampling
        basis carries ``Mc`` (chirp mass, Msol) instead of ``fdot``. A
        multi-parameter ``(f0, Mc) -> (f0, fdot)`` transform via
        :func:`gbgpu.utils.utility.get_fdot` recovers the physical fdot after
        ``fill_values`` moves the sampled ``Mc`` to the fdot slot (via
        ``key_map={"Mc": "fdot"}``). When
        :attr:`GBSettings.use_astrophysical_f0_mc_prior` is also on, the
        separate ``f0`` and ``Mc`` uniforms are replaced by the 6-component
        heatmap GMM from :func:`~lisatools.sampling.f0_mchirp_prior
        .F0McGMMSampling.from_heatmap` under a tuple key ``("f0", "Mc")``,
        AND the basis gains a 9th sampled column ``fdot_astro_ratio``
        (:attr:`GBSettings.use_fdot_astro`): ``r = fdot_astro / fdot_gr ~
        U[-M, M]`` (M = :attr:`GBSettings.fdot_astro_ratio_max`), physical
        ``fdot = fdot_gr(f0, Mc) * (1 + r)`` with ``fddot`` exactly 0 --
        interacting ``fdot < 0`` systems live at ``r < -1``. With the
        distance basis (production) slot 0 is the distance (kpc), the
        amplitude is derived, and ``(dist, alpha, sin_delta)`` carry the joint
        sky / distance prior (the 3-D Milky Way prior under
        :attr:`GBSettings.use_galaxy_prior`).

        Fills, each only when unset: ``transform``
        (:func:`~lisatools.globalfit.stock.erebor.transforms.make_gb_transform_container`),
        ``periodic`` (``phi0`` and ``alpha`` period ``2 pi``, ``psi`` period
        ``pi``), ``priors`` (``{"gb": ProbDistContainer}``), ``betas``,
        ``other_tempering_kwargs`` (``adaptation_time=2, permute=True``),
        ``initialize_kwargs``, ``waveform_kwargs``, ``group_proposal_kwargs``
        (plus its ``setdefault`` keys) and ``search_kwargs``. The ``betas``
        fallback (a 24-rung ladder with the ``GB_LADDER_*`` env knobs) is
        never reached on the stock paths, because ``prepare_gb_branch`` /
        ``prepare_vgb_branch`` always fill ``betas`` first (the GB stock
        ladder: ``GB_NTEMPS`` rungs at ratio 1.2 with the top rung pinned to
        ``beta = 1e-4``).

        Raises:
            ValueError: ``use_chirp_mass`` without the astrophysical prior and
                with empty ``m_chirp_lims``.
        """
        third_name = "Mc" if self.use_chirp_mass else "fdot"
        # Slot 0 is the luminosity DISTANCE (kpc) in the distance basis --
        # amplitude is derived from the sampled (dist, f0, Mc) -- else lnA.
        slot0_name = "dist" if self.use_distance else "A"
        input_basis = [slot0_name, "f0", third_name, "phi0",
                       "cos_iota", "psi", "alpha", "sin_delta"]
        if self.use_fdot_astro:
            # 9th sampled column: r = fdot_astro / fdot_gr (dimensionless;
            # physical fdot = fdot_gr(f0, Mc) * (1 + r), fddot stays 0).
            input_basis.append("fdot_astro_ratio")

        if self.transform is None:
            # THE single GB transform factory (phi0 sign convention lives
            # there, nowhere else).
            self.transform = make_gb_transform_container(
                use_chirp_mass=self.use_chirp_mass,
                use_fdot_astro=self.use_fdot_astro,
                use_distance=self.use_distance,
                mc_lims=(
                    tuple(self.m_chirp_lims) if self.m_chirp_lims
                    else (0.001, 1.0)
                ),
            )

        if self.periodic is None:
            self.periodic = {"gb": {"phi0": 2*np.pi, "psi": np.pi, "alpha": 2 * np.pi}}

        # Sub-seeds for the prior objects below that own a private
        # ``np.random.Generator``. ``build_seed`` is already PER RANK
        # (``rank_build_seed``); the spawn keeps the three streams
        # independent of each other. ``None`` -> every ``default_rng``
        # below falls back to OS entropy, exactly as before.
        _sub_seeds = self._build_sub_seeds(3)

        if self.priors is None:
            if self.use_chirp_mass and self.use_astrophysical_f0_mc_prior:
                # Joint tuple-key prior on (f0[mHz], Mc[Msol]) from the
                # heatmap 6-component GMM fit.
                from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling

                # Truncate the heatmap GMM to the run's sampled box and
                # renormalize (RJ birth/death compares prior masses across
                # leaf counts, so the truncated prior must integrate to 1
                # over exactly the space the sampler covers). Gaussian
                # tails keep the density finite anywhere in the box, even
                # outside the original heatmap support.
                _f0_mc_prior = F0McGMMSampling.from_heatmap(
                    f0_lims_mHz=tuple(np.asarray(self.f0_lims) * 1e3),
                    mc_lims=tuple(self.m_chirp_lims) if self.m_chirp_lims
                    else None,
                    seed=_sub_seeds[0],
                )
                # Slot 0 is DISTANCE (kpc) in this branch (use_distance): the
                # sky and distance share a 3-D JOINT prior over
                # (dist, alpha, sin_delta) -- the Galaxy-shaped distribution
                # is incoming, so this is a tuple-key slot. For now the joint
                # is a placeholder nested ProbDistContainer of independent
                # uniforms (dist over dist_lims, sky uniform); swap
                # sky_dist_distribution for the real 3-D distribution.
                sky_dist = self.sky_dist_distribution
                if sky_dist is None and getattr(
                    self, "use_galaxy_prior", False
                ):
                    from ...priors.galaxy_sky_dist import (
                        build_gb_galaxy_sky_dist,
                    )

                    sky_dist = build_gb_galaxy_sky_dist(
                        dist_lims=self.dist_lims,
                        # two SEPARATE ``default_rng`` sites: the sky/dist
                        # prior's own rejection sampler and the nested
                        # GalaxyPrior3D it draws from
                        rng=_sub_seeds[1],
                        galaxy_rng=_sub_seeds[2],
                    )
                    self.logger.info(
                        "GB sky/distance prior: 3-D Milky Way "
                        "(GB_USE_GALAXY_PRIOR=1), dist_lims=%s kpc.",
                        list(self.dist_lims),
                    )
                if sky_dist is None:
                    sky_dist = ProbDistContainer({
                        0: uniform_dist(self.dist_lims[0], self.dist_lims[1]),
                        1: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                        2: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                    })
                priors_gb = {
                    # tuple members get CONSECUTIVE columns by insertion
                    # order; reset_key_order below remaps them to the real
                    # basis columns (dist->0, alpha->6, sin_delta->7).
                    ("dist", "alpha", "sin_delta"): sky_dist,
                    ("f0", "Mc"): _f0_mc_prior,
                    "phi0": uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    "cos_iota": uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    "psi": uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    # 9th column: r ~ U[-M, M]. Prior defined in the SAMPLING
                    # basis (no in-sampler Jacobian); induced physical prior
                    # at fixed (f0, Mc) is uniform in fdot -- see
                    # McDistFdotAstroQuad.
                    "fdot_astro_ratio": uniform_dist(
                        -self.fdot_astro_ratio_max, self.fdot_astro_ratio_max
                    ),
                }
            elif self.use_chirp_mass:
                if not self.m_chirp_lims:
                    raise ValueError(
                        "use_chirp_mass=True requires GBSettings.m_chirp_lims "
                        "to be set (a two-element [Mc_min, Mc_max] in Msol)."
                    )
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),
                    input_basis[2]: uniform_dist(self.m_chirp_lims[0], self.m_chirp_lims[1]),
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: sort defensively.
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                }
            else:
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),  # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
                    input_basis[2]: uniform_dist(self.fdot_lims[0], self.fdot_lims[1]),
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: cos(iota_lims) comes out
                    # (max, min), so sort into increasing order before handing
                    # it to uniform_dist -- never rely on the dist silently
                    # swapping reversed bounds. (Same defensive sort on the
                    # sin(delta) line even though sin is increasing there.)
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                }

            _gb_prior = ProbDistContainer(priors_gb)
            # The distance branch inserts the ("dist","alpha","sin_delta")
            # joint FIRST, so its members grabbed consecutive columns 0/1/2;
            # remap by name to the real basis columns (dist->0, alpha->6,
            # sin_delta->7). A no-op for the other branches (already in
            # column order), so it is safe to apply whenever the input basis
            # is available.
            if self.use_distance:
                _gb_prior.reset_key_order(list(input_basis))
            self.priors = {"gb": _gb_prior}

        if self.betas is None:
            # TODO check: keep/remove/document -- unreachable on stock paths; GB_LADDER_* inert
            # snrs_ladder = np.array(
            #     [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
            #      15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2]
            # )
            ntemps_pe = 24 # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            #
            # TWO-SLOPE LADDER (user ruling 2026-09-27). Defaults below
            # reproduce the historical single-slope ladder EXACTLY
            # (ratio 1.2 throughout, last rung pinned to 1e-4), so nothing
            # changes for any run that does not set the knobs.
            #
            # WHY A SECOND SLOPE IS AVAILABLE. Measured on 6mo v9 job 650:
            # adjacent-rung swap acceptance climbs monotonically 31.2%
            # (T0-T1) -> 71.1% (T20-T21), i.e. the hot half is spaced far
            # too FINELY -- replica exchange wants ~20-30%, so rungs
            # T10-T22 buy 60-71% where 30% would do. Then T22 (beta
            # 1.765e-2, T=55.3) jumps straight to the pinned T23 (beta
            # 1e-4, T=10,000), a 176x gap, and that rung holds 64.5
            # sources/walker against T22's 933 -- 93% shed -- and couples
            # at 15.4%. The 3mo run measured the same cliff at 181x with
            # 95.8% shed.
            #
            # Fitting sources/walker against ln(T) over T0..T22 gives
            # ``2284 - 313*ln(T)``, which reaches zero near T ~ 1460. The
            # hottest working rung is T=55, so roughly a factor of 26 of
            # USABLE temperature is skipped before the pinned rung
            # overshoots it by 7x. Spending the hot half's excess
            # acceptance on range closes both gaps at once.
            #
            #   GB_LADDER_RATIO       cold-half ratio           (1.2)
            #   GB_LADDER_HOT_RATIO   hot-half ratio, 0 = off   (0 -> same)
            #   GB_LADDER_HOT_FROM    first rung of the hot half (10)
            #   GB_LADDER_PIN_LAST    beta for the top rung, 0 = leave it
            #                         on the geometric progression (1e-4)
            #
            # ⚠ A RESUME IGNORES ALL OF THIS. ``band_temps`` lives in the
            # store and ``initialize_band_information`` restores it, so
            # changing these knobs only affects a run that starts fresh or
            # whose store has had its ladder migrated. The launcher's
            # LADDER PREFLIGHT compares the rung COUNT, not the values, so
            # a value change is otherwise a silent no-op.
            def _env_f(name, default):
                try:
                    return float(os.environ.get(name, "") or default)
                except ValueError:
                    return float(default)

            _ratio = _env_f("GB_LADDER_RATIO", 1.2)
            _hot_ratio = _env_f("GB_LADDER_HOT_RATIO", 0.0)
            _hot_from = int(_env_f("GB_LADDER_HOT_FROM", 10))
            _pin_last = _env_f("GB_LADDER_PIN_LAST", 1e-4)

            if _hot_ratio > 0.0 and 0 < _hot_from < ntemps_pe:
                _steps = np.zeros(ntemps_pe)
                _steps[1:_hot_from + 1] = np.log(_ratio)
                _steps[_hot_from + 1:] = np.log(_hot_ratio)
                betas = np.exp(-np.cumsum(_steps))
            else:
                betas = 1 / _ratio ** np.arange(ntemps_pe)
            if _pin_last > 0.0:
                betas[-1] = _pin_last
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(adaptation_time=2, permute=True)

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        # GBGPU FD waveform kwargs. The WDM-domain engine in
        # ``gb_likelihood.WDMBandLikelihoodEngine`` ignores
        # ``start_freq_ind`` / ``oversample`` (it dispatches to the WDM C
        # kernel via ``GBWDMComputations``), but we keep the same dict so
        # the down-stream Buffer's ``tdi_channel_setup`` read still works
        # on either path. The settings file may pre-populate
        # ``waveform_kwargs`` (stft_tof field); this default is built from
        # the other fields only when it is left empty.
        if not self.waveform_kwargs:
            self.waveform_kwargs = dict(
                dt=self.dt,
                T=self.Tobs,
                use_c_implementation=True,
                oversample=self.oversample,
                start_freq_ind=self.start_freq_ind,
                tdi_channel_setup=self.tdi_setup,
                tdi2=self.use_tdi2
            )

        if self.group_proposal_kwargs is None:
            self.group_proposal_kwargs: typing.Dict[str, Any] = dict(
                n_iter_update=1, live_dangerously=True, a=1.75, num_repeat_proposals=200
            )
        # Peak sub-band-buffer memory cap. Injected here (not only in the
        # None-default dict above) so an explicit ``group_proposal_kwargs``
        # still gets the ``n_subbands`` knob unless it names one itself.
        self.group_proposal_kwargs.setdefault("num_band_preload", int(self.n_subbands))
        # Task-b narrow per-band WDM slabs (None -> full active band).
        self.group_proposal_kwargs.setdefault(
            "wdm_band_slab_layers", self.wdm_band_slab_layers
        )
        self.group_proposal_kwargs.setdefault(
            "wdm_slab_guard_layers", self.wdm_slab_guard_layers
        )
        # Shared-psd mirror + its parity gate (plumbed like the slab knobs:
        # move ctor kwargs -> BandSorter -> SubBandBuffer).
        self.group_proposal_kwargs.setdefault(
            "psd_shared_mirror", bool(self.psd_shared_mirror)
        )
        self.group_proposal_kwargs.setdefault(
            "psd_mirror_parity_proposes", int(self.psd_mirror_parity_proposes)
        )
        self.group_proposal_kwargs.setdefault(
            "psd_mirror_parity_rows", int(self.psd_mirror_parity_rows)
        )
        # Band-unit stride for the concurrent sub-band scheduling
        # (GB_BAND_UNIT_STRIDE; default 2 = legacy parity, bit-identical).
        # Flows to every GBSpecial* move as the ``band_units`` ctor kwarg.
        self.group_proposal_kwargs.setdefault(
            "band_units", int(getattr(self, "band_unit_stride", 2))
        )

        if self.search_kwargs is None:
            # Configuration for the stft_tof per-band serial GB search
            # (consumed by ``GBSpecialRJSerialSearchMCMC.setup`` and
            # ``GBSpecialRJRefitMove.setup``):
            #   nwalkers / ntemps          — per-band ParaEnsembleSampler size
            #   burn_1 / nsteps_1          — stage 1: F-stat (phase-maximized)
            #                                MCMC, Gibbs on the 4 intrinsic-
            #                                like params
            #   snr_threshold              — band "found a source" gate on the
            #                                stage-1 chain's min optimal SNR
            #   burn_2 / nsteps_2          — stage 2: full-likelihood refine
            #                                from stage-1's last sample; its
            #                                samples feed the GMM fit that
            #                                becomes the RJ proposal
            #   shutoff_band_iteration     — shut a band off after this many
            #                                consecutive source-less search
            #                                iterations (all-off reverts the
            #                                RJ proposal to the global prior)
            #   shutoff_frequency_threshold— only bands above this frequency
            #                                are shutoff-eligible (None = all)
            #   refit_start_iteration      — samples kept per leaf for the
            #                                GMM refit in GBSpecialRJRefitMove
            self.search_kwargs: typing.Dict[str, Any] = dict(
                nwalkers = 32,
                ntemps = 24,
                shutoff_band_iteration = 5,
                shutoff_frequency_threshold = None, # 4e-3
                burn_1 = 200,
                nsteps_1 = 200,
                snr_threshold = 8.0,
                burn_2 = 500,
                nsteps_2 = 500,
                refit_start_iteration = 5
            )

    # def __getattr__(self, attr: str) -> typing.Any:
    #     if hasattr(self.gb_settings, attr):
    #         return getattr(self.gb_settings, attr)

    def init_setup(self):
        """Run the band-structure, sampling-info, and state-backend init helpers."""
        self.init_band_structure()
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the branch state / backend to :class:`GBState` / :class:`GBHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = GBState

        if self.branch_backend is None:
            self.branch_backend = GBHDFBackend

    def init_band_structure(self):
        """Compute :attr:`band_edges` and :attr:`band_N_vals` from the GB frequency range.

        Both FD and WDM domains use a frequency-banded layout. The edges
        are derived from ``df = 1/Tobs`` (always meaningful for either
        basis) and the FD-oversampled per-band N. For WDM runs the
        per-band ``band_N_vals`` is unused by the WDM likelihood engine
        (which sizes its buffers from ``WDMSettings.Nf_active`` /
        ``Nt_active``) but is preserved for shape parity.

        Order of precedence on WDM: ``band_edges_override``, else
        ``band_edges_mode`` (``"uniform"``: edges at multiples of
        ``layer_df / subband_divisor`` inside ``[start_freq, end_freq]``,
        production 8 per layer; ``"get_n"``: :func:`get_n_based_band_edges`);
        the GB band is first clamped to the WDM active band. On FD the bands
        are walked down from ``end_freq`` in ``2 N + extra_buffer`` bins. A
        support-separation check at ``band_unit_stride`` is logged (not
        enforced) for the WDM ``uniform`` grid; the ``get_n`` builder enforces
        it itself, and an override grid is not checked. Sets ``band_edges``,
        ``band_N_vals``, ``num_sub_bands``, and overwrites ``f0_lims``
        (interior edges) and ``fdot_lims`` (mojito fdot range at the top
        edge).

        Raises:
            ValueError: No overlap with the WDM active band, an unknown
                ``band_edges_mode``, or a band narrower than one sub-band.
        """
        # band separation setup
        if self.oversample is None and self.Tobs < YRSID_SI / 2.0:
            self.oversample = 2
        elif self.oversample is None:
            self.oversample = 4

        assert self.oversample >= 1

        # Clamp the GB band to the WDM active band when running on a WDM
        # grid. The WDM kernel only carries layers in [ind_min_f, ind_max_f];
        # putting band edges outside that range silently produces zero-fill
        # behaviour, so we trim to stay inside.
        if isinstance(self.domain_settings, WDMSettings):
            wdm = self.domain_settings
            wdm_min = float(wdm.ind_min_f * wdm.layer_df)
            wdm_max = float(wdm.ind_max_f * wdm.layer_df)
            self.start_freq = max(self.start_freq, wdm_min)
            self.end_freq = min(self.end_freq, wdm_max)
            if self.start_freq >= self.end_freq:
                raise ValueError(
                    "WDM active band [{:.4e}, {:.4e}] does not overlap GB "
                    "frequency range; widen WDMSettings.min_freq/max_freq "
                    "or the GB start/end_freq.".format(wdm_min, wdm_max)
                )

        _edges_override = getattr(self, "band_edges_override", None)
        if _edges_override is not None and np.size(_edges_override) >= 2:
            # Explicit user override: use the supplied band_edges verbatim
            # (Hz, ascending). ``fit.gb.band_edges_override = np.array([...])``.
            self.band_edges = np.asarray(_edges_override, dtype=float)
            self.band_N_vals = np.asarray(
                [
                    get_N(1e-30, edge, self.Tobs, oversample=self.oversample).item()
                    for edge in self.band_edges[:-1]
                ]
            )
        elif isinstance(self.domain_settings, WDMSettings):
            wdm = self.domain_settings
            layer_df = float(wdm.layer_df)
            div = max(1, int(getattr(self, "subband_divisor", 1)))
            _mode = str(getattr(self, "band_edges_mode", "uniform")).lower()
            if _mode == "get_n":
                # FREE-FLOATING bands at the 2*get_N minimum width
                # (GB_BAND_EDGES_MODE=get_n; 2026-08-15 user rulings): each
                # band's width is 2*get_N(f_max_band)/Tobs, maximal
                # packing, no layer snapping. See
                # :func:`get_n_based_band_edges` for the knob semantics.
                # subband_divisor / band_min_layers are passed only so the
                # builder can warn-and-ignore when a user set them: they
                # have no role in the 2*get_N width rule (min_band_layers)
                # or the free-floating construction (subband_divisor).
                self.band_edges = get_n_based_band_edges(
                    self.start_freq,
                    self.end_freq,
                    self.Tobs,
                    layer_df,
                    subband_divisor=div,
                    oversample=int(self.oversample),
                    target_count=int(getattr(self, "band_target_count", 0)),
                    min_band_layers=getattr(self, "band_min_layers", None),
                    unit_stride=int(getattr(self, "band_unit_stride", 2)),
                )
            elif _mode != "uniform":
                raise ValueError(
                    "GB_BAND_EDGES_MODE / GBSettings.band_edges_mode must be "
                    f"'uniform' or 'get_n', got {_mode!r}."
                )
            else:
                # WDM path (default): one band per WDM frequency layer, or per
                # 1/``subband_divisor`` layer (GB_SUBBAND_DIVISOR=2 -> half-layer
                # edges). Edges land on ``k * layer_df / div`` boundaries so the
                # band grid aligns with the WDM grid; ``band_N_vals`` is unused by
                # the WDM likelihood engine but is preserved for shape parity.
                k_lo = int(np.ceil(self.start_freq / layer_df)) * div
                k_hi = int(np.floor(self.end_freq / layer_df)) * div
                if k_hi <= k_lo:
                    raise ValueError(
                        "GB frequency range [{:.4e}, {:.4e}] spans fewer than "
                        "one WDM sub-band (layer_df/div={:.4e}).".format(
                            self.start_freq, self.end_freq, layer_df / div
                        )
                    )
                self.band_edges = np.asarray(
                    [k * layer_df / div for k in range(k_lo, k_hi + 1)]
                )
            self.band_N_vals = np.asarray(
                [
                    get_N(1e-30, edge, self.Tobs, oversample=self.oversample).item()
                    for edge in self.band_edges[:-1]
                ]
            )
            # SUPPORT-based separation diagnostic on the ACTUAL edges
            # (2026-08-15 rulings: the guard is expressed in FD-SUPPORT
            # terms -- gap between same-unit bands vs the sum of the
            # edge-source half-supports get_N(f_edge)/Tobs -- replacing
            # the earlier layer-width formulas). For get_n grids the
            # BUILDER already enforced it (raise); the legacy uniform /
            # override grids are NOT refused (uniform stride-2 is the
            # measured-safe production default even though high-f
            # edge-source FD supports overlap by this conservative
            # envelope -- the WDM likelihood windows there are +-1 layer
            # and the [GB_ORTHO_LL] runtime monitor, default ON, is the
            # operative accuracy check), so the diagnostic only LOGS the
            # minimum safe stride for them.
            if _mode != "get_n":
                from ...moves.gbbands import check_band_support_separation

                _sep = check_band_support_separation(
                    self.band_edges,
                    self.Tobs,
                    int(getattr(self, "band_unit_stride", 2)),
                    oversample=int(self.oversample),
                    context="GBSetup.init_band_structure",
                    enforce=False,
                )
                self.logger.info(
                    "GB band grid (mode=%s): support-based same-unit "
                    "separation at stride %d passes=%s; min safe stride "
                    "%s at sep_factor %.3g.",
                    _mode, int(getattr(self, "band_unit_stride", 2)),
                    _sep["passes"], str(_sep["min_safe_stride"]),
                    float(_sep["sep_factor"]),
                )
        else:
            # FD path: bands sized in multiples of ``df = 1/Tobs`` using the
            # FD-oversampled per-band N. Walks down from ``end_freq``
            # (stft_tof refinements: half-bin start, min_N stop guard, and
            # edge trim against out-of-bound indexing).
            # TODO: assign to binned f or leave general? probably better to be general
            band_edges_in_reverse_order = [self.end_freq]
            current_N = get_N(1e-30, self.end_freq, self.Tobs, oversample=self.oversample).item()
            min_N = get_N(1e-30, self.start_freq, self.Tobs, oversample=self.oversample).item()
            band_N_vals_reverse_order = [current_N]

            current_freq = self.end_freq - self.df / 2
            last_freq = self.end_freq
            while current_freq > self.start_freq + min_N * self.df:
                current_freq = last_freq - (current_N * 2 + self.extra_buffer) * self.df
                band_edges_in_reverse_order.append(current_freq)
                current_N = get_N(1e-30, current_freq, self.Tobs, oversample=self.oversample).item()
                band_N_vals_reverse_order.append(current_N)
                last_freq = current_freq

            band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
            band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]

            # trim edges to avoid out of bound indexing
            self.band_edges = band_edges[2:-1]
            self.band_N_vals = band_N_vals[2:-1]

        self.f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]

        self.fdot_lims = [
            get_fdot_mojito(self.f0_lims[1], sign="-"), 
            get_fdot_mojito(self.f0_lims[1], sign="+"), 
        ]

        self.num_sub_bands = len(self.band_edges) - 1
        
        self.logger.info(
            f"GB f0 prior range is set from {round(self.f0_lims[0],7)} to {round(self.f0_lims[1],7)}"
        )
        self.logger.info(f"The number of subbands is {self.num_sub_bands}")
        self.logger.info(f"Min freq of subbands is {self.band_edges.min()}")
        self.logger.info(f"Max freq of subbands is {self.band_edges.max()}")

