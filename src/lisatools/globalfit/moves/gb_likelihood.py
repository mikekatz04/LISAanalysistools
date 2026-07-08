"""Domain-agnostic band likelihood engines for the GB special moves.

(Public module as of the 2026-06 stft_tof merge; formerly the private
``_gb_likelihood``. The engines are intended for documented, repeatable
use by any band-based GB sampler component, not just the in-tree moves.)

Three implementations live here:

* :class:`FDBandLikelihoodEngine` -- wraps :class:`gbgpu.gbgpu.GBGPU` (frequency
  domain). Ports the inlined logic that used to live inside
  :meth:`Buffer.get_swap_ll` / :meth:`Buffer.adjust_sources_in_band_buffer`.

* :class:`WDMBandLikelihoodEngine` -- wraps
  :class:`gbgpu.gbcomps.GBWDMComputations` (WDM time-frequency
  domain). Uses :func:`GBWDMComputations.get_ll_wdm`,
  :func:`get_swap_ll_wdm`, :func:`fill_global_wdm`.

* :class:`STFTBandLikelihoodEngine` -- wraps
  :class:`gbgpu.gbcomps.STFTGBComputations` (STFT/Fresnel time-frequency
  domain). Uses :func:`STFTGBComputations.get_ll_stft`,
  :func:`get_swap_ll_stft`, :func:`fill_global_stft`, :func:`get_ll_grad_stft`.

The :class:`BandLikelihoodEngine` protocol is the contract both implementations
honour. :class:`Buffer` dispatches on its ``basis_settings`` (a
:class:`~lisatools.domains.DomainSettingsBase` child -- never a string flag)
to pick one and then talks to the engine via
:class:`AnalysisContainerArray` only -- the move itself never reaches into
``self.gb`` or C-side pointers.

Engine guarantees:

* ``get_swap_ll`` performs the frequency-bounds rejection internally:
  proposals whose ``f0/df +- N_vals/2`` fall outside the band buffer are
  clamped to ``ll_diff = -1e300`` and reported via ``SwapLLResult.kept``
  (this is the bounds check that previously lived inline in
  ``gbspecialstretch``).
* The compute backend (CPU C++ / CUDA / JAX) is fixed at engine
  construction; per the sprint-wide rule there is no runtime ``backend=``
  kwarg on any method.
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
from ...domains import DomainSettingsBase, FDSettings, STFTSettings, WDMSettings


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

    def get_ll_grad(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        param_eps=None,
        chunk: Optional[int] = None,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source gradient of ``L = <d|h> - 0.5<h|h>``.

        Returns ``(num_proposals, nparams)``. Only the chunked-het
        backend implements this; the FD legacy path raises. The
        compute backend (C++ central-FD vs JAX autograd) is fixed at
        engine construction; per the sprint-wide rule there is no
        runtime ``backend=`` kwarg.
        """
        ...

    def hessian(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        chunk: Optional[int] = None,
        psd_fix: bool = False,
        psd_floor_rel: float = 1e-30,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source Hessian of ``L = <d|h> - 0.5<h|h>``.

        Returns ``(num_proposals, nparams, nparams)``. When
        ``psd_fix=True`` returns ``M = |-H|`` ready to feed to a NUTS
        sampler as a mass matrix. Only the chunked-het backend
        implements this; the FD legacy path raises. The compute
        backend (currently JAX autograd only) is fixed at engine
        construction.
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

    @staticmethod
    def _data_splits(buffer_aca: AnalysisContainerArray):
        """Per-band GPU assignment array consumed by ``GBGPU.*`` calls.

        Mirrors the shape of ``params_index`` / ``data_index`` semantics:
        ``data_splits[band_i] = gpu_id_owning_band_i``. Sourced directly from
        ``buffer_aca.gpu_map`` so the GBGPU kernel-dispatch loop picks the
        right per-GPU buffer from the list we hand it.
        """
        return np.asarray(buffer_aca.gpu_map, dtype=int)

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

        # Multi-GPU contract: pass the full per-GPU buffer list and the
        # per-band GPU assignment as data_splits. GBGPU's kernel dispatch
        # loop at gbgpu.py:1546 iterates over self.gb.gpus and indexes
        # templates_in[gpu_i] -- so when len(gpus) > 1, the list-per-GPU
        # form is required. When gpus is None or single-GPU, a single-
        # element list still works.
        flat_bands = buffer_aca.linear_data_arr
        data_splits = self._data_splits(buffer_aca)

        factors_change = factor * cp.ones_like(params_index, dtype=float)
        self.gb.generate_global_template(
            params_phys,
            params_index,
            flat_bands,
            data_length=self.data_length,
            factors=factors_change,
            data_splits=data_splits,
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

        flat_bands = buffer_aca.linear_data_arr
        flat_psds = buffer_aca.linear_psd_arr
        data_splits = self._data_splits(buffer_aca)
        # Per-band frequency-bin count is constant across bands in this engine,
        # so any shard's reshape is fine for the data_length argument.
        data_length_bins = buffer_aca.data_shaped[0].shape[-1]

        self.gb.get_ll(
            params_phys,
            flat_bands,
            flat_psds,
            start_freq_ind=self.start_freq_inds,
            data_index=data_index,
            noise_index=noise_index,
            N=N_vals,
            data_length=data_length_bins,
            data_splits=data_splits,
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

        flat_bands = buffer_aca.linear_data_arr
        flat_psds = buffer_aca.linear_psd_arr
        data_splits = self._data_splits(buffer_aca)
        # Per-band frequency-bin count is constant across bands; any shard works.
        data_length_bins = buffer_aca.data_shaped[0].shape[-1]

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
                    flat_bands,
                    flat_psds,
                    start_freq_ind=self.start_freq_inds,
                    data_index=data_index[keep],
                    noise_index=noise_index[keep],
                    adjust_inplace=False,
                    N=N_vals[keep],
                    data_length=data_length_bins,
                    data_splits=data_splits,
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

    # ---------- get_ll_grad / hessian (NOT YET implemented for FD) -----------
    #
    # The FD analogues (``GBFDComputations.get_ll_grad_fd`` / a future
    # ``hessian_fd``) would slot in here once the JAX-FD autograd path
    # is wired. For now the FD legacy generator lacks a Hessian method,
    # and the NUTS-in-globalfit work uses the WDM/chunked-het backend
    # exclusively. These stubs make the Protocol contract explicit:
    # caller will get a clear error rather than AttributeError.
    def get_ll_grad(self, *_args, **_kwargs):
        raise NotImplementedError(
            "FDBandLikelihoodEngine.get_ll_grad is not implemented yet. "
            "Use the WDM/chunked-het Buffer for gradient (NUTS) moves; "
            "an FD JAX-autograd path is planned."
        )

    def hessian(self, *_args, **_kwargs):
        raise NotImplementedError(
            "FDBandLikelihoodEngine.hessian is not implemented yet. "
            "Use the WDM/chunked-het Buffer for NUTS metric construction; "
            "an FD JAX-autograd path is planned."
        )


# ---------------------------------------------------------------------------
# WDM-domain engine (wraps gbgpu.gbcomps.GBWDMComputations)
# ---------------------------------------------------------------------------


class WDMBandLikelihoodEngine:
    """WDM engine wrapping a
    :class:`gbgpu.gbcomps.GBWDMComputations` instance.

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

        # Post-Phase-3L.7p, ``fill_global_wdm`` signature is
        # ``(params, templates, ...)`` -- the flat WDM-template buffer is the
        # only positional after params, no third ``wdm_holder`` slot.
        self.gb_comps.fill_global_wdm(
            params_phys,
            buffer_aca.linear_data_arr[0],
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

    # ---------- get_ll_grad / hessian (chunked-het backends only) ----------
    #
    # These two methods are present on ``gb_comps`` only when the
    # underlying generator is :class:`GBWDMComputations` (the chunked-het
    # backend). The legacy lookup-table ``GBWDMComputations`` does not
    # expose ``hessian_wdm`` (its FD ``get_ll_grad_wdm`` is implemented
    # but the in-the-kernel info-matrix path it served has been retired).
    # We probe at call time so a global fit running on the legacy
    # generator fails loudly rather than silently masking a config bug.

    def _require_chunked_het(self, method_name: str):
        if not hasattr(self.gb_comps, method_name):
            raise NotImplementedError(
                f"WDMBandLikelihoodEngine.{method_name.replace('_wdm','')}: "
                f"underlying gb_comps ({type(self.gb_comps).__name__}) "
                f"does not expose {method_name!r}. The gradient / Hessian "
                "paths require a GBWDMComputations backend; rebuild the "
                "Buffer with the chunked-het generator."
            )

    def get_ll_grad(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        param_eps=None,
        chunk: Optional[int] = None,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source gradient of ``L = <d|h> - 0.5 <h|h>`` w.r.t. params.

        Returns ``(num_proposals, nparams)`` -- one row per source.
        The compute backend (C++ central-FD vs JAX autograd) is fixed
        at the ``GBWDMComputations``'s construction-time
        ``force_backend``; per the sprint-wide rule no ``backend=``
        runtime kwarg is taken.
        """
        self._require_chunked_het("get_ll_grad_wdm")
        # Post-Phase-3L.7p, ``get_ll_grad_wdm`` no longer accepts
        # ``param_eps`` / ``chunk``; the JAX-autograd backend handles the
        # full-precision derivative internally and the C++ kernel does
        # not (yet) split the gradient over sub-chunks. Engine swallows
        # the kwargs for caller backward-compat.
        del N_vals, waveform_kwargs, param_eps, chunk
        return self.gb_comps.get_ll_grad_wdm(
            params_phys, buffer_aca,
            data_index=data_index, noise_index=noise_index,
        )

    def hessian(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        chunk: Optional[int] = None,
        psd_fix: bool = False,
        psd_floor_rel: float = 1e-30,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source Hessian of ``L = <d|h> - 0.5 <h|h>``.

        Returns ``(num_proposals, nparams, nparams)``. When
        ``psd_fix=True`` returns ``M = |−H|`` (eigendecompose +
        ``|lambda|``), ready to feed to ``NUTSSampler(metric=M)``. The
        compute backend (JAX autograd only at present) is fixed at
        ``GBWDMComputations`` construction; calling this on a non-JAX
        chunked-het instance raises ``NotImplementedError`` inside
        ``hessian_wdm``.
        """
        self._require_chunked_het("hessian_wdm")
        del N_vals, waveform_kwargs
        return self.gb_comps.hessian_wdm(
            params_phys, buffer_aca,
            data_index=data_index, noise_index=noise_index,
            chunk=chunk, psd_fix=psd_fix, psd_floor_rel=psd_floor_rel,
        )


# ---------------------------------------------------------------------------
# STFT/Fresnel-domain engine (wraps gbgpu.gbcomps.STFTGBComputations)
# ---------------------------------------------------------------------------


class STFTBandLikelihoodEngine:
    """STFT (Fresnel) engine wrapping a
    :class:`gbgpu.gbcomps.STFTGBComputations` instance.

    Unlike the WDM computations object (whose methods take the holder per
    call), ``STFTGBComputations`` reads its data / inverse-covariance through
    the ``STFTComputationGroup`` bound to ``gb_stft_comp.stft_comps``. The
    engine therefore REBINDS that attribute per call to the band ACA's own
    per-split strategy (``buffer_aca.cpp_splits[s]``): the ACA builds and
    caches those groups itself (``window_alpha`` / ``use_midpoint`` /
    ``tdi_type`` via its ``domain_group_kwargs``), and the group's C++ domain
    reads the LIVE ``linear_data_arr`` / ``linear_psd_arr`` buffers zero-copy,
    so in-place residual updates between calls stay visible without any
    rebuild.

    Multi-split dispatch: proposal rows are partitioned by the split owning
    their band (``buffer_aca.split_map``), band indices are remapped to
    intra-split indices (``buffer_aca.ac_to_intra``), and each split's kernel
    call runs under that split's device context. The single-split case (all
    the WDM engine supports today) degenerates to one iteration with an
    identity remap. Cross-device gathers of the per-split outputs into the
    full-length result arrays are untested on multi-GPU.

    Engine outputs are ``(d|d)``-independent -- :meth:`get_ll` returns the raw
    ``d_h`` / ``h_h`` and :meth:`get_swap_ll` assembles ``ll_diff`` from the
    five raw swap terms -- so the group's ``d_d`` snapshot going stale as the
    Buffer rewrites residuals in place is harmless here.

    Serial-use assumption: rebinding mutates the shared ``gb_stft_comp``
    (same constraint as the WDM engine's stashed ``d_h_out`` attributes) --
    one engine call at a time per gb object.

    The Fresnel knobs (``n_side_bins`` / ``window_factor`` /
    ``freq_from_tdi_phase`` / ``T`` / ``t_ref`` / orbits / TDI config) are
    fixed on ``gb_stft_comp`` at its construction; per the sprint-wide rule
    there is no runtime override here.
    """

    def __init__(
        self,
        gb_stft_comp,
        basis_settings: STFTSettings,
        nchannels: int,
        tdi_channel_setup: str,
        opt_snr_rej_samp_limit: float = 5.0,
    ):
        self.gb_stft_comp = gb_stft_comp
        self.basis_settings = basis_settings
        self.nchannels = nchannels
        self.tdi_channel_setup = tdi_channel_setup
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit

    @property
    def xp(self):
        return self.gb_stft_comp.xp

    # ---------- per-split dispatch -------------------------------------------

    def _split_plan(self, buffer_aca: AnalysisContainerArray, data_index, noise_index=None):
        """Partition proposal rows by the ACA split owning their data band.

        Returns ``[(split, row_mask, intra_data_idx, intra_noise_idx, device),
        ...]`` for every split referenced by ``data_index`` (a row's split is
        keyed on its data band). ``intra_noise_idx`` is ``None`` when
        ``noise_index`` is. Raises if any row's noise band lives in a
        different split than its data band -- per-split kernel dispatch needs
        them co-located (always true in the Buffer, where both are the
        band's AC).
        """
        xp = self.xp
        split_map = xp.asarray(buffer_aca.split_map)
        ac_to_intra = xp.asarray(buffer_aca.ac_to_intra)
        d_idx = xp.asarray(data_index).astype(int)
        n_idx = None if noise_index is None else xp.asarray(noise_index).astype(int)
        if n_idx is not None and bool((split_map[d_idx] != split_map[n_idx]).any()):
            raise ValueError(
                "STFTBandLikelihoodEngine: a proposal row references a data "
                "band and a noise band owned by different ACA splits; "
                "per-split kernel dispatch requires them to be co-located."
            )
        row_split = split_map[d_idx]
        plan = []
        for s in range(len(buffer_aca.cpp_splits)):
            mask = row_split == s
            if not bool(mask.any()):
                continue
            device = buffer_aca.gpus[s] if buffer_aca.gpus is not None else None
            intra_d = ac_to_intra[d_idx[mask]].astype(xp.int32)
            intra_n = (
                None if n_idx is None else ac_to_intra[n_idx[mask]].astype(xp.int32)
            )
            plan.append((s, mask, intra_d, intra_n, device))
        return plan

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
        xp = self.xp
        params_phys = xp.atleast_2d(xp.asarray(params_phys))
        # N_vals / waveform_kwargs are FD-specific; the Fresnel kernel's band
        # support is n_side_bins, fixed on gb_stft_comp at construction.
        for s, mask, intra_d, _, device in self._split_plan(buffer_aca, params_index):
            with buffer_aca.device_context(device):
                self.gb_stft_comp.stft_comps = buffer_aca.cpp_splits[s]
                factors_arr = xp.full(
                    int(mask.sum()), float(factor), dtype=xp.float64
                )
                # The split's flat buffer IS the (num_bands, nchannels, NT,
                # NF_active) template stack the kernel scatters into;
                # intra-split data_index addresses the band slot.
                self.gb_stft_comp.fill_global_stft(
                    params_phys[mask],
                    buffer_aca.linear_data_arr[s],
                    data_index=intra_d,
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
        xp = self.xp
        params_phys = xp.atleast_2d(xp.asarray(params_phys))
        num_bin = params_phys.shape[0]
        d_h = xp.zeros(num_bin, dtype=xp.complex128)
        h_h = xp.zeros(num_bin, dtype=xp.complex128)
        for s, mask, intra_d, intra_n, device in self._split_plan(
            buffer_aca, data_index, noise_index
        ):
            with buffer_aca.device_context(device):
                self.gb_stft_comp.stft_comps = buffer_aca.cpp_splits[s]
                self.gb_stft_comp.get_ll_stft(
                    params_phys[mask], data_index=intra_d, noise_index=intra_n
                )
                d_h[mask] = self.gb_stft_comp.d_h_out
                h_h[mask] = self.gb_stft_comp.h_h_out
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
        xp = self.xp

        if phase_marginalize:
            # get_ll_stft/get_swap_ll_stft have no phase-maximisation pass;
            # surfacing this loudly beats silently returning the un-maximised
            # result (same policy as the WDM engine).
            raise NotImplementedError(
                "STFTBandLikelihoodEngine.get_swap_ll does not yet support "
                "phase_marginalize=True. The STFT swap kernel needs a phase-"
                "maximisation pass first."
            )

        pa = xp.atleast_2d(xp.asarray(params_add_phys))
        pr = xp.atleast_2d(xp.asarray(params_remove_phys))

        # STFT bounds-keep (WDM parity): each source's central frequency bin
        # must sit in the active band. The kernels clamp every pixel access to
        # the grid, so this is proposal semantics (reject band-escapees), not
        # crash safety.
        df = self.basis_settings.df
        ind_min = self.basis_settings.ind_min
        ind_max = self.basis_settings.ind_max
        bin_add = (pa[:, 1] / df).astype(int)
        bin_remove = (pr[:, 1] / df).astype(int)
        keep = (
            (bin_add >= ind_min)
            & (bin_add <= ind_max)
            & (bin_remove >= ind_min)
            & (bin_remove <= ind_max)
        )

        num_prop = keep.shape[0]
        ll_diff = xp.full(num_prop, -1e300, dtype=xp.float64)
        opt_snr = xp.zeros(num_prop, dtype=xp.float64)
        d_h_add = xp.zeros(num_prop, dtype=xp.float64)
        d_h_remove = xp.zeros(num_prop, dtype=xp.float64)
        hh_add = xp.zeros(num_prop, dtype=xp.float64)
        hh_remove = xp.zeros(num_prop, dtype=xp.float64)
        hh_cross = xp.zeros(num_prop, dtype=xp.float64)

        if bool(keep.any()):
            rows_kept = xp.where(keep)[0]
            d_idx = xp.asarray(data_index)[keep]
            n_idx = xp.asarray(noise_index)[keep]
            for s, sub_mask, intra_d, intra_n, device in self._split_plan(
                buffer_aca, d_idx, n_idx
            ):
                rows = rows_kept[sub_mask]
                with buffer_aca.device_context(device):
                    self.gb_stft_comp.stft_comps = buffer_aca.cpp_splits[s]
                    (
                        _like_add,
                        _like_rem,
                        d_h_a,
                        d_h_r,
                        aa,
                        rr,
                        ar,
                    ) = self.gb_stft_comp.get_swap_ll_stft(
                        pa[rows],
                        pr[rows],
                        data_index=intra_d,
                        noise_index=intra_n,
                    )
                    # Same swap algebra as the WDM engine; the STFT kernel
                    # returns the raw complex inner products, so take the
                    # real parts here.
                    ll_diff[rows] = (
                        (d_h_a.real - d_h_r.real)
                        - 0.5 * (aa.real - rr.real)
                        - (ar.real - rr.real)
                    )
                    d_h_add[rows] = d_h_a.real
                    d_h_remove[rows] = d_h_r.real
                    hh_add[rows] = aa.real
                    hh_remove[rows] = rr.real
                    hh_cross[rows] = ar.real
                    opt_snr[rows] = xp.sqrt(xp.maximum(aa.real, 0.0))

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

    # ---------- get_ll_grad / hessian ---------------------------------------

    def get_ll_grad(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        param_eps=None,
        chunk: Optional[int] = None,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source gradient of ``L = <d|h> - 0.5 <h|h>`` w.r.t. params.

        Returns ``(num_proposals, nparams)``. Routed through the central-FD
        kernel behind :meth:`STFTGBComputations.get_ll_grad_stft`, which
        (unlike the WDM chunked-het path) accepts ``param_eps`` natively --
        it is forwarded. ``chunk`` has no STFT analog and is swallowed for
        caller backward-compat.
        """
        xp = self.xp
        del N_vals, chunk, waveform_kwargs
        params_phys = xp.atleast_2d(xp.asarray(params_phys))
        num_bin = params_phys.shape[0]
        grad = xp.zeros(
            (num_bin, self.gb_stft_comp.num_params), dtype=xp.float64
        )
        for s, mask, intra_d, intra_n, device in self._split_plan(
            buffer_aca, data_index, noise_index
        ):
            with buffer_aca.device_context(device):
                self.gb_stft_comp.stft_comps = buffer_aca.cpp_splits[s]
                grad[mask] = self.gb_stft_comp.get_ll_grad_stft(
                    params_phys[mask],
                    param_eps=param_eps,
                    data_index=intra_d,
                    noise_index=intra_n,
                )
        return grad

    def hessian(self, *_args, **_kwargs):
        raise NotImplementedError(
            "STFTBandLikelihoodEngine.hessian is not implemented: "
            "STFTGBComputations has no hessian_stft kernel. Use the "
            "WDM/chunked-het Buffer for NUTS metric construction."
        )


def make_band_likelihood_engine(
    basis_settings: DomainSettingsBase,
    *,
    gb=None,
    gb_wdm_comp=None,
    gb_stft_comp=None,
    nchannels: int,
    tdi_channel_setup: str,
    df: Optional[float] = None,
    start_freq_inds=None,
    data_length: Optional[int] = None,
    opt_snr_rej_samp_limit: float = 5.0,
) -> BandLikelihoodEngine:
    """Construct the right engine for the supplied basis-domain settings.

    Dispatch keys off ``isinstance(basis_settings, ...)`` -- no string-level
    mode flag. The caller passes whichever ``gb`` / ``gb_wdm_comp`` /
    ``gb_stft_comp`` is appropriate for the domain it has selected; the
    others stay ``None``.
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
                "gbgpu.gbcomps.GBWDMComputations instance (pass "
                "gb_wdm_comp=...)."
            )
        return WDMBandLikelihoodEngine(
            gb_comps=gb_wdm_comp,
            basis_settings=basis_settings,
            nchannels=nchannels,
            tdi_channel_setup=tdi_channel_setup,
            opt_snr_rej_samp_limit=opt_snr_rej_samp_limit,
        )
    if isinstance(basis_settings, STFTSettings):
        if gb_stft_comp is None:
            raise ValueError(
                "STFTBandLikelihoodEngine requires a "
                "gbgpu.gbcomps.STFTGBComputations instance (pass "
                "gb_stft_comp=...)."
            )
        return STFTBandLikelihoodEngine(
            gb_stft_comp=gb_stft_comp,
            basis_settings=basis_settings,
            nchannels=nchannels,
            tdi_channel_setup=tdi_channel_setup,
            opt_snr_rej_samp_limit=opt_snr_rej_samp_limit,
        )
    raise NotImplementedError(
        f"No BandLikelihoodEngine for basis domain {type(basis_settings).__name__}."
    )
