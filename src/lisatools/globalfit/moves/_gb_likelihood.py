"""Domain-agnostic likelihood engines for the GB special moves.

Two implementations live here:

* :class:`FDBandLikelihoodEngine` -- wraps :class:`gbgpu.gbgpu.GBGPU` (frequency
  domain). Ports the inlined logic that used to live inside
  :meth:`Buffer.get_swap_ll` / :meth:`Buffer.adjust_sources_in_band_buffer`.

* :class:`WDMBandLikelihoodEngine` -- wraps
  :class:`fastlisaresponse.gbcomps.GBWDMComputations` (WDM time-frequency
  domain). Uses :func:`GBWDMComputations.get_ll_wdm`,
  :func:`get_swap_ll_wdm`, :func:`fill_global_wdm`.

The :class:`BandLikelihoodEngine` protocol is the contract both implementations
honour. :class:`Buffer` dispatches on its ``basis_settings`` to pick one and
then talks to the engine via :class:`AnalysisContainerArray` only -- the move
itself never reaches into ``self.gb`` or C-side pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

from ...analysiscontainer import AnalysisContainerArray
from ...domains import DomainSettingsBase, FDSettings, WDMSettings


@dataclass
class SwapLLResult:
    """Outputs of one batched swap-likelihood call.

    All arrays are 1-D, length ``num_proposals``. Out-of-bounds proposals get
    ``ll_diff = -1e300`` (callers don't need to mask separately).

    Attributes
    ----------
    ll_diff
        Posterior log-likelihood difference for the swap proposal, summed over
        bands. For FD this is the legacy
        ``gbgpu.swap_likelihood_difference`` output. For WDM it is
        ``-0.5 * ( <h_add|h_add> - 2<d|h_add> ) - ( -0.5 * ( <h_remove|h_remove> - 2<d|h_remove> ) )``
        which is the standard
        ``(d_h_add - d_h_remove) - 0.5*(h_add_h_add - h_remove_h_remove) - cross_term``
        algebra, computed from the five swap inner products.
    d_h_add, d_h_remove, hh_add, hh_remove, hh_cross
        The raw inner products. WDM populates all five; FD currently returns
        only those it computes natively, leaving the rest as ``None``.
    opt_snr_add
        Optimal SNR of the proposed (added) template: ``sqrt(<h_add|h_add>)``.
        Used by :meth:`Buffer.get_swap_ll` for the rejection-sampling clamp.
    phase_angle
        When ``phase_marginalize=True``, the per-proposal phase rotation that
        the engine applied to maximise over phase. Otherwise ``None``. Callers
        subtract this from the proposed ``phi0`` parameter when an accept lands
        on a phase-maximised draw.
    kept
        Boolean mask (length ``num_proposals``) identifying the proposals that
        passed engine-internal bounds checks. Rejected proposals already have
        ``ll_diff = -1e300`` and ``opt_snr_add = 0``.
    """

    ll_diff: Any
    d_h_add: Any
    d_h_remove: Any
    hh_add: Any
    hh_remove: Any
    hh_cross: Any
    opt_snr_add: Any
    phase_angle: Optional[Any]
    kept: Any


class BandLikelihoodEngine(Protocol):
    """Interface implemented by both the FD and WDM band engines.

    Implementations take an :class:`AnalysisContainerArray` (the per-band
    residual + inverse-PSD buffers) and the move's physical-parameter arrays;
    they return either inner products (:meth:`get_ll`) or a
    :class:`SwapLLResult` (:meth:`get_swap_ll`). The :meth:`fill_template`
    method writes (or removes) templates into the linear data buffer of the
    same ACA.

    All engines must expose ``basis_settings``, ``nchannels``, and
    ``tdi_channel_setup`` so the Buffer can sanity-check shapes at construction
    time.
    """

    basis_settings: DomainSettingsBase
    nchannels: int
    tdi_channel_setup: str

    def fill_template(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        params_index,
        N_vals,
        *,
        factor: int,
        waveform_kwargs: dict,
    ) -> None:
        """Add (``factor=+1``) or remove (``factor=-1``) sources from
        ``buffer_aca.linear_data_arr[0]`` in-place.

        ``params_phys`` are already transformed to physical units.
        ``params_index`` maps each source to a band buffer index.
        """
        ...

    def get_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        waveform_kwargs: dict,
    ) -> tuple:
        """Compute (``d_h``, ``h_h``) for each source against the per-band data
        in ``buffer_aca``. Returns two length-``len(params_phys)`` arrays on the
        engine's xp module.
        """
        ...

    def get_swap_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_marginalize: bool,
        waveform_kwargs: dict,
    ) -> SwapLLResult:
        """Compute the five swap inner products and the resulting
        ``ll_diff`` for each row of the two parameter arrays.
        """
        ...


# ---------------------------------------------------------------------------
# Frequency-domain engine (wraps gbgpu.GBGPU)
# ---------------------------------------------------------------------------


class FDBandLikelihoodEngine:
    """FD engine wrapping a :class:`gbgpu.gbgpu.GBGPU` instance.

    Ports the inlined logic that used to live in
    :meth:`Buffer.get_swap_ll` (gbspecialstretch.py:787-952) and
    :meth:`Buffer.adjust_sources_in_band_buffer` (gbspecialstretch.py:1055-1099)
    behind the :class:`BandLikelihoodEngine` protocol.

    Required fields exposed on ``aca`` (the per-band ACA passed in at call
    time): ``linear_data_arr[0]``, ``linear_psd_arr[0]``, ``start_freq_ind``,
    ``data_length``, ``gpu_map`` (one entry per band in the ACA).
    """

    def __init__(
        self,
        gb,
        basis_settings: FDSettings,
        nchannels: int,
        tdi_channel_setup: str,
        df: float,
        start_freq_inds,
        data_length: int,
        opt_snr_rej_samp_limit: float = 5.0,
    ):
        self.gb = gb
        self.basis_settings = basis_settings
        self.nchannels = nchannels
        self.tdi_channel_setup = tdi_channel_setup
        self.df = df
        self.start_freq_inds = start_freq_inds  # per-band start FD bin
        self.data_length = data_length
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit

    @property
    def xp(self):
        return cp

    # ---------- fill_template ------------------------------------------------

    def fill_template(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        params_index,
        N_vals,
        *,
        factor: int,
        waveform_kwargs: dict,
    ) -> None:
        assert factor in (-1, +1)
        wave_kwargs = waveform_kwargs.copy()
        wave_kwargs.pop("start_freq_ind", None)

        flat_band = buffer_aca.linear_data_arr[0]
        # Per-band data buffer shape: (num_bands, nchannels, data_length).
        num_bands = buffer_aca.data_shaped[0].shape[0]

        factors_change = factor * cp.ones_like(params_index, dtype=float)
        gpu0 = self.gb.gpus[0] if getattr(self.gb, "gpus", None) else 0
        self.gb.generate_global_template(
            params_phys,
            params_index,
            flat_band,
            data_length=self.data_length,
            factors=factors_change,
            data_splits=np.full(num_bands, gpu0),
            N=N_vals,
            start_freq_ind=self.start_freq_inds,
            **wave_kwargs,
        )

    # ---------- get_ll -------------------------------------------------------

    def get_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        waveform_kwargs: dict,
    ):
        wave_kwargs = waveform_kwargs.copy()
        wave_kwargs.pop("start_freq_ind", None)

        band_buffer = buffer_aca.data_shaped[0]
        num_bands = band_buffer.shape[0]
        flat_band = buffer_aca.linear_data_arr[0]
        flat_psd = buffer_aca.linear_psd_arr[0]

        self.gb.get_ll(
            params_phys,
            flat_band,
            flat_psd,
            start_freq_ind=self.start_freq_inds,
            data_index=data_index,
            noise_index=noise_index,
            N=N_vals,
            data_length=band_buffer.shape[-1],
            data_splits=np.full(num_bands, self.gb.gpus[0] if getattr(self.gb, "gpus", None) else 0),
            phase_marginalize=False,
            return_cupy=True,
            **wave_kwargs,
        )
        d_h = cp.asarray(self.gb.d_h).copy()
        h_h = cp.asarray(self.gb.h_h).copy()
        return d_h, h_h

    # ---------- get_swap_ll --------------------------------------------------

    def get_swap_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_marginalize: bool,
        waveform_kwargs: dict,
    ) -> SwapLLResult:
        wave_kwargs = waveform_kwargs.copy()
        wave_kwargs.pop("start_freq_ind", None)

        band_buffer = buffer_aca.data_shaped[0]
        num_bands = band_buffer.shape[0]
        flat_band = buffer_aca.linear_data_arr[0]
        flat_psd = buffer_aca.linear_psd_arr[0]
        data_length_bins = band_buffer.shape[-1]

        # Reject proposals whose ±N/2 window would fall outside the per-band
        # FD buffer. This is the FD-only sanity check; the kernel would crash
        # on out-of-bounds access otherwise.
        keep = ~(
            (
                (params_remove_phys[:, 1] / self.df).astype(int)
                - self.start_freq_inds[data_index]
                - (N_vals / 2)
                < 0
            )
            | (
                (params_add_phys[:, 1] / self.df).astype(int)
                - self.start_freq_inds[data_index]
                - (N_vals / 2)
                < 0
            )
            | (
                (params_remove_phys[:, 1] / self.df).astype(int)
                - self.start_freq_inds[data_index]
                + (N_vals / 2)
                > data_length_bins
            )
            | (
                (params_add_phys[:, 1] / self.df).astype(int)
                - self.start_freq_inds[data_index]
                + (N_vals / 2)
                > data_length_bins
            )
        )

        ll_diff = cp.full(keep.shape[0], -1e300)
        opt_snr = cp.zeros(keep.shape[0])

        if cp.any(keep):
            ll_diff_keep = cp.asarray(
                self.gb.swap_likelihood_difference(
                    params_remove_phys[keep],
                    params_add_phys[keep],
                    flat_band,
                    flat_psd,
                    start_freq_ind=self.start_freq_inds,
                    data_index=data_index[keep],
                    noise_index=noise_index[keep],
                    adjust_inplace=False,
                    N=N_vals[keep],
                    data_length=data_length_bins,
                    data_splits=np.full(num_bands, self.gb.gpus[0] if getattr(self.gb, "gpus", None) else 0),
                    phase_marginalize=phase_marginalize,
                    return_cupy=True,
                    **wave_kwargs,
                )
            )
            ll_diff[keep] = ll_diff_keep
            opt_snr[keep] = self.gb.add_add.real ** (1.0 / 2.0)

        phase_angle = self.gb.phase_angle if phase_marginalize else None

        return SwapLLResult(
            ll_diff=ll_diff,
            d_h_add=None,
            d_h_remove=None,
            hh_add=None,
            hh_remove=None,
            hh_cross=None,
            opt_snr_add=opt_snr,
            phase_angle=phase_angle,
            kept=keep,
        )


# ---------------------------------------------------------------------------
# WDM-domain engine (wraps fastlisaresponse.GBWDMComputations)
# ---------------------------------------------------------------------------


class WDMBandLikelihoodEngine:
    """WDM engine wrapping a
    :class:`fastlisaresponse.gbcomps.GBWDMComputations` instance.

    Three calls all route through the same ``gb_comps`` object: fill, get_ll,
    and get_swap_ll. The per-band ACA is forwarded directly -- the GB WDM
    Python API already speaks AnalysisContainerArray.

    The bounds-keep mask is WDM-flavoured: each source's
    ``layer_m = int(f0 / layer_df)`` must land within the WDM
    ``[ind_min_f, ind_max_f]`` active band.
    """

    def __init__(
        self,
        gb_comps,
        basis_settings: WDMSettings,
        nchannels: int,
        tdi_channel_setup: str,
        opt_snr_rej_samp_limit: float = 5.0,
    ):
        self.gb_comps = gb_comps
        self.basis_settings = basis_settings
        self.nchannels = nchannels
        self.tdi_channel_setup = tdi_channel_setup
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit

    @property
    def xp(self):
        return self.gb_comps.xp

    # ---------- fill_template ------------------------------------------------

    def fill_template(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        params_index,
        N_vals,
        *,
        factor: int,
        waveform_kwargs: dict,
    ) -> None:
        assert factor in (-1, +1)
        # GBWDMComputations.fill_global_wdm needs the params in the layout it
        # expects (num_bin, 9). data_index controls which per-band buffer each
        # source writes to. The factors argument was added to the kernel for
        # this purpose so add/remove is one C call.
        xp = self.gb_comps.xp
        num_bin = params_phys.shape[0]
        factors_arr = xp.full(num_bin, float(factor), dtype=xp.float64)

        # The Python signature expects the *templates* buffer (the flat WDM
        # template array). buffer_aca.linear_data_arr[0] is that buffer.
        self.gb_comps.fill_global_wdm(
            buffer_aca.linear_data_arr[0],
            params_phys,
            buffer_aca,
            convert_to_ra_dec=False,
            data_index=params_index,
            factors=factors_arr,
        )

    # ---------- get_ll -------------------------------------------------------

    def get_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        waveform_kwargs: dict,
    ):
        # Returns the log-likelihood; the inner products are stashed on the
        # gb_comps object as d_h_out / h_h_out.
        _ = self.gb_comps.get_ll_wdm(
            params_phys,
            buffer_aca,
            data_index=data_index,
            noise_index=noise_index,
            convert_to_ra_dec=False,
        )
        return (
            self.gb_comps.d_h_out.copy(),
            self.gb_comps.h_h_out.copy(),
        )

    # ---------- get_swap_ll --------------------------------------------------

    def get_swap_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_marginalize: bool,
        waveform_kwargs: dict,
    ) -> SwapLLResult:
        xp = self.gb_comps.xp

        if phase_marginalize:
            # The current WDM kernel does not implement phase maximisation;
            # surfacing this loudly is better than silently returning the
            # un-maximised result.
            raise NotImplementedError(
                "WDMBandLikelihoodEngine.get_swap_ll does not yet support "
                "phase_marginalize=True. The WDM swap_ll kernel needs a phase-"
                "maximisation pass first."
            )

        # WDM bounds-keep: each source's central frequency layer must sit in
        # the active band. The kernel internally skips OOB layers, but the
        # move needs to set ll_diff = -1e300 on those proposals.
        layer_df = self.basis_settings.layer_df
        m_min = self.basis_settings.ind_min_f
        m_max = self.basis_settings.ind_max_f
        layer_add = (params_add_phys[:, 1] / layer_df).astype(int)
        layer_remove = (params_remove_phys[:, 1] / layer_df).astype(int)
        keep = (
            (layer_add >= m_min)
            & (layer_add <= m_max)
            & (layer_remove >= m_min)
            & (layer_remove <= m_max)
        )

        num_prop = keep.shape[0]
        ll_diff = xp.full(num_prop, -1e300, dtype=xp.float64)
        opt_snr = xp.zeros(num_prop, dtype=xp.float64)
        d_h_add = xp.zeros(num_prop, dtype=xp.float64)
        d_h_remove = xp.zeros(num_prop, dtype=xp.float64)
        hh_add = xp.zeros(num_prop, dtype=xp.float64)
        hh_remove = xp.zeros(num_prop, dtype=xp.float64)
        hh_cross = xp.zeros(num_prop, dtype=xp.float64)

        if bool(keep.any() if hasattr(keep, "any") else keep.any()):
            keep_idx = keep
            (
                like_add,
                like_remove,
                d_h_a,
                d_h_r,
                aa,
                rr,
                ar,
            ) = self.gb_comps.get_swap_ll_wdm(
                params_add_phys[keep_idx],
                params_remove_phys[keep_idx],
                buffer_aca,
                data_index=data_index[keep_idx],
                noise_index=noise_index[keep_idx],
                convert_to_ra_dec=False,
            )

            # ll_diff = like_add - like_remove + cross-term correction.
            # The standard swap-likelihood difference is
            #   Δ log L = (<d|h_add> - <d|h_remove>)
            #            - 0.5 * (<h_add|h_add> - <h_remove|h_remove>)
            #            - (<h_add|h_remove> - <h_remove|h_remove>)
            # In the presence of an existing remove-template that is already
            # part of the residual, the cross term is what survives. For the
            # simplest case (independent proposal) it reduces to:
            #   Δ log L = (d_h_add - d_h_remove) - 0.5*(hh_add - hh_remove)
            #            - (hh_cross - hh_remove)
            # which is the form gbgpu's swap_likelihood_difference returns.
            ll_diff_keep = (
                (d_h_a - d_h_r)
                - 0.5 * (aa - rr)
                - (ar - rr)
            )
            ll_diff[keep_idx] = ll_diff_keep
            d_h_add[keep_idx] = d_h_a
            d_h_remove[keep_idx] = d_h_r
            hh_add[keep_idx] = aa
            hh_remove[keep_idx] = rr
            hh_cross[keep_idx] = ar
            opt_snr[keep_idx] = xp.sqrt(xp.maximum(aa, 0.0))

        return SwapLLResult(
            ll_diff=ll_diff,
            d_h_add=d_h_add,
            d_h_remove=d_h_remove,
            hh_add=hh_add,
            hh_remove=hh_remove,
            hh_cross=hh_cross,
            opt_snr_add=opt_snr,
            phase_angle=None,
            kept=keep,
        )


def make_band_likelihood_engine(
    basis_settings: DomainSettingsBase,
    *,
    gb=None,
    gb_wdm_comp=None,
    nchannels: int,
    tdi_channel_setup: str,
    df: Optional[float] = None,
    start_freq_inds=None,
    data_length: Optional[int] = None,
    opt_snr_rej_samp_limit: float = 5.0,
) -> BandLikelihoodEngine:
    """Construct the right engine for the supplied basis-domain settings.

    Dispatch keys off ``isinstance(basis_settings, ...)`` -- no string-level
    mode flag. The caller passes whichever ``gb`` / ``gb_wdm_comp`` is
    appropriate for the domain it has selected; the missing one stays ``None``.
    """
    if isinstance(basis_settings, FDSettings):
        if gb is None:
            raise ValueError(
                "FDBandLikelihoodEngine requires a gbgpu.GBGPU instance "
                "(pass gb=...)."
            )
        if df is None or start_freq_inds is None or data_length is None:
            raise ValueError(
                "FDBandLikelihoodEngine requires df, start_freq_inds, and "
                "data_length."
            )
        return FDBandLikelihoodEngine(
            gb=gb,
            basis_settings=basis_settings,
            nchannels=nchannels,
            tdi_channel_setup=tdi_channel_setup,
            df=df,
            start_freq_inds=start_freq_inds,
            data_length=data_length,
            opt_snr_rej_samp_limit=opt_snr_rej_samp_limit,
        )
    if isinstance(basis_settings, WDMSettings):
        if gb_wdm_comp is None:
            raise ValueError(
                "WDMBandLikelihoodEngine requires a "
                "fastlisaresponse.GBWDMComputations instance (pass "
                "gb_wdm_comp=...)."
            )
        return WDMBandLikelihoodEngine(
            gb_comps=gb_wdm_comp,
            basis_settings=basis_settings,
            nchannels=nchannels,
            tdi_channel_setup=tdi_channel_setup,
            opt_snr_rej_samp_limit=opt_snr_rej_samp_limit,
        )
    raise NotImplementedError(
        f"No BandLikelihoodEngine for basis domain {type(basis_settings).__name__}."
    )
