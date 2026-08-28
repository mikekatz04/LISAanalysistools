"""Coarse-grained likelihood utilities for real WDM noise inference.

The data-dependent sample covariance is computed once by averaging products of
fine WDM coefficients over adjacent time columns.  Noise models are evaluated
on the matching coarse grid, reducing covariance construction, inversion, and
likelihood reduction by approximately the coarse factor ``Q``.

Welch--Satterthwaite effective degrees of freedom are frozen at a fiducial
fine-grid covariance.  Consequently, normalization terms depending only on the
data and frozen degrees of freedom are omitted.  The returned likelihood is
valid for MCMC and tempering, but not as an absolutely normalized likelihood
for evidence calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .domains import CoarseWDMSettings, WDMSignal
from .utils.utility import get_array_module


def _coarse_sample_covariance(signal: WDMSignal, settings: CoarseWDMSettings):
    if not isinstance(signal, WDMSignal):
        raise TypeError("signal must be a WDMSignal.")
    if getattr(signal, "is_complex", False) or np.iscomplexobj(signal.arr):
        raise ValueError("Coarse WDM likelihood currently supports real WDM data only.")
    if signal.is_batched:
        raise ValueError("CoarseWDMStatistic requires an unbatched WDM signal.")
    if signal.nchannels != 3:
        raise ValueError(f"Coarse WDM likelihood requires 3 XYZ channels; got {signal.nchannels}.")
    if signal.settings != settings.fine_settings:
        raise ValueError("signal settings do not match coarse_settings.fine_settings.")

    arr = signal.arr
    xp = get_array_module(arr)
    cells = []
    for start, size in zip(settings.cell_starts, settings.cell_sizes):
        block = arr[..., int(start) : int(start + size)]
        cells.append(xp.einsum("amk,bmk->abm", block, block) / float(size))
    return xp.stack(cells, axis=-1)


@dataclass
class CoarseWDMStatistic:
    """Precomputed real-WDM sample covariance and frozen effective dof."""

    P: object
    Qeff: object
    settings: CoarseWDMSettings
    Qeff_channels: Optional[object] = None

    def __post_init__(self):
        if not isinstance(self.settings, CoarseWDMSettings):
            raise TypeError("settings must be CoarseWDMSettings.")
        expected_p = (3, 3) + tuple(self.settings.basis_shape_active)
        expected_q = tuple(self.settings.basis_shape_active)
        if tuple(self.P.shape) != expected_p:
            raise ValueError(f"P has shape {self.P.shape}; expected {expected_p}.")
        if tuple(self.Qeff.shape) != expected_q:
            raise ValueError(f"Qeff has shape {self.Qeff.shape}; expected {expected_q}.")
        if self.Qeff_channels is not None:
            expected_channels = (3,) + expected_q
            if tuple(self.Qeff_channels.shape) != expected_channels:
                raise ValueError(
                    f"Qeff_channels has shape {self.Qeff_channels.shape}; "
                    f"expected {expected_channels}."
                )

    @property
    def xp(self):
        """Array module derived from the settings backend (never pickled directly)."""
        return self.settings.xp

    @classmethod
    def from_wdm_signal(
        cls,
        signal: WDMSignal,
        coarse_settings: CoarseWDMSettings,
        *,
        fiducial_sens_mat_fine=None,
        use_ws: bool = True,
        qeff=None,
        qeff_channels=None,
    ) -> "CoarseWDMStatistic":
        """Build the statistic once from fine WDM data and a fiducial model."""
        P = _coarse_sample_covariance(signal, coarse_settings)
        if qeff is None:
            Qeff, channels = compute_qeff(
                fiducial_sens_mat_fine,
                coarse_settings,
                use_ws=use_ws,
                return_channels=True,
            )
        else:
            if not use_ws:
                raise ValueError("qeff override is only valid when use_ws=True.")
            Qeff = coarse_settings.xp.asarray(qeff)
            channels = (
                None
                if qeff_channels is None
                else coarse_settings.xp.asarray(qeff_channels)
            )
        return cls(P=P, Qeff=Qeff, settings=coarse_settings, Qeff_channels=channels)

    def update_from_residual(self, residual_signal: WDMSignal) -> None:
        """Recompute the data statistic in place from a fine residual signal."""
        updated = _coarse_sample_covariance(residual_signal, self.settings)
        self.P[...] = updated


def _fine_covariance_array(fiducial_sens_mat_fine):
    if fiducial_sens_mat_fine is None:
        return None
    return getattr(fiducial_sens_mat_fine, "sens_mat", fiducial_sens_mat_fine)


def compute_qeff(
    fiducial_sens_mat_fine,
    coarse_settings: CoarseWDMSettings,
    *,
    use_ws: bool = True,
    return_channels: bool = False,
):
    """Compute frozen channelwise Welch--Satterthwaite effective dof.

    Each diagonal channel is moment-matched independently, then the three dof
    values are averaged.  Computing the ratio after first averaging X/Y/Z would
    hide opposing channel drifts and is deliberately not used.
    """
    if not isinstance(coarse_settings, CoarseWDMSettings):
        raise TypeError("coarse_settings must be CoarseWDMSettings.")

    sizes = coarse_settings.xp.asarray(coarse_settings.cell_sizes, dtype=float)
    shape = tuple(coarse_settings.basis_shape_active)
    if not use_ws:
        qeff = coarse_settings.xp.broadcast_to(sizes[None, :], shape).copy()
        channels = coarse_settings.xp.broadcast_to(qeff[None, ...], (3,) + shape).copy()
        return (qeff, channels) if return_channels else qeff

    covariance = _fine_covariance_array(fiducial_sens_mat_fine)
    if covariance is None:
        raise ValueError("WS coarse graining requires a fiducial fine covariance.")
    xp = get_array_module(covariance)
    covariance = xp.asarray(covariance)
    expected = (3, 3, coarse_settings.Nf_active, coarse_settings.Nt_active)
    if tuple(covariance.shape) != expected:
        raise ValueError(
            f"fiducial fine covariance has shape {covariance.shape}; expected {expected}."
        )

    diagonal = xp.real(xp.stack([covariance[a, a] for a in range(3)], axis=0))
    per_cell = []
    for start, size in zip(coarse_settings.cell_starts, coarse_settings.cell_sizes):
        block = diagonal[..., int(start) : int(start + size)]
        s1 = xp.sum(block, axis=-1)
        s2 = xp.sum(block * block, axis=-1)
        valid = xp.isfinite(s1) & xp.isfinite(s2) & (s2 > 0.0)
        ratio = xp.where(valid, (s1 * s1) / xp.where(valid, s2, 1.0), 0.0)
        ratio = xp.minimum(ratio, float(size))
        per_cell.append(ratio)
    channels = xp.stack(per_cell, axis=-1)
    # One invalid channel means a shared multivariate dof is undefined.  Mark
    # the cell inactive rather than silently averaging only the surviving axes.
    all_channels_valid = xp.all(channels > 0.0, axis=0)
    qeff = xp.where(all_channels_valid, xp.mean(channels, axis=0), 0.0)
    return (qeff, channels) if return_channels else qeff


def coarse_wdm_log_likelihood_terms(stat: CoarseWDMStatistic, sens_mat):
    """Return ``(quadratic_term, logdet_term)`` for the coarse likelihood."""
    if not isinstance(stat, CoarseWDMStatistic):
        raise TypeError("stat must be a CoarseWDMStatistic.")
    settings = getattr(sens_mat, "basis_settings", None)
    if settings != stat.settings:
        raise ValueError("sensitivity matrix basis does not match the coarse statistic.")

    invC = sens_mat.invC
    detC = sens_mat.detC
    xp = get_array_module(invC)
    if tuple(invC.shape) != tuple(stat.P.shape):
        raise ValueError(f"invC has shape {invC.shape}; expected {stat.P.shape}.")

    qeff = xp.asarray(stat.Qeff)
    P = xp.asarray(stat.P)
    valid = (
        (qeff > 0.0)
        & xp.isfinite(qeff)
        & xp.isfinite(detC)
        & (detC != 0.0)
        & xp.all(xp.isfinite(invC), axis=(0, 1))
        & xp.all(xp.isfinite(P), axis=(0, 1))
    )
    weights = xp.where(valid, qeff, 0.0)
    quadratic_pixel = xp.where(
        valid, xp.real(xp.einsum("abmq,abmq->mq", invC, P)), 0.0
    )
    safe_det = xp.where(valid, xp.abs(detC), 1.0)
    logdet_pixel = xp.log(safe_det)
    quadratic = -0.5 * xp.sum(weights * quadratic_pixel)
    logdet = -0.5 * xp.sum(weights * logdet_pixel)
    return quadratic, logdet


def coarse_wdm_log_likelihood(
    stat: CoarseWDMStatistic,
    sens_mat,
    *,
    noise_only: bool = False,
):
    """Evaluate the coarse real-WDM noise likelihood.

    ``noise_only=True`` returns only the weighted log-determinant term, matching
    :meth:`AnalysisContainer.likelihood`'s existing public semantics.
    """
    quadratic, logdet = coarse_wdm_log_likelihood_terms(stat, sens_mat)
    out = logdet if noise_only else quadratic + logdet
    try:
        return out.item()
    except AttributeError:
        return out


def coarse_wdm_log_likelihood_batch_frequency_terms(
    stat: CoarseWDMStatistic,
    covariance,
    *,
    noise_only: bool = False,
    frequency_indices=None,
):
    """Score a covariance batch and retain one likelihood term per layer.

    Parameters
    ----------
    stat:
        Shared data statistic.
    covariance:
        Array shaped ``(batch, 3, 3, Nf_active, Ncoarse)``.  When
        ``frequency_indices`` is supplied, the frequency axis instead has
        ``len(frequency_indices)`` entries in that order.
    frequency_indices:
        Optional active-frequency indices selecting the matching slices of the
        shared data statistic and effective degrees of freedom.  This supports
        an exact-baseline likelihood correction when only one additive noise
        component changes in a strict subband.

    Notes
    -----
    The returned shape is ``(batch, n_frequency)``.  Retaining the frequency
    axis lets a split-component move cache an exact full-band baseline and
    replace only the layers touched by its variable component.  Use
    :func:`coarse_wdm_log_likelihood_batch` for the ordinary scalar-per-row
    reduction.
    """
    if not isinstance(stat, CoarseWDMStatistic):
        raise TypeError("stat must be a CoarseWDMStatistic.")
    xp = get_array_module(covariance)
    covariance = xp.asarray(covariance)
    if frequency_indices is None:
        frequency_indices_xp = None
        nfreq = stat.settings.Nf_active
    else:
        frequency_indices_xp = xp.asarray(frequency_indices, dtype=np.int64)
        if frequency_indices_xp.ndim != 1:
            raise ValueError("frequency_indices must be one-dimensional.")
        nfreq = int(frequency_indices_xp.size)
    expected = (3, 3, nfreq, stat.settings.Ncoarse)
    if covariance.ndim != 5 or tuple(covariance.shape[1:]) != expected:
        raise ValueError(
            f"covariance has shape {covariance.shape}; expected "
            f"(batch, {', '.join(map(str, expected))})."
        )

    # _mat3x3_det_inv keeps channel axes first.  Put batch immediately after
    # them so every algebraic temporary remains one contiguous native-array
    # operation over (batch, frequency, coarse-time).
    from .sensitivity import _mat3x3_det_inv

    matrices = xp.moveaxis(covariance, (1, 2), (0, 1))
    detC, invC = _mat3x3_det_inv(matrices, xp)

    # Match SensitivityMatrixBase._setup_det_and_inv's sanitization exactly.
    bad_inv = ~xp.isfinite(invC)
    if bool(xp.any(bad_inv)):
        invC = xp.where(bad_inv, xp.zeros_like(invC), invC)
    bad_det = ~xp.isfinite(detC)
    if bool(xp.any(bad_det)):
        detC = xp.where(bad_det, xp.ones_like(detC), detC)

    qeff = xp.asarray(stat.Qeff)
    P = xp.asarray(stat.P)
    if frequency_indices_xp is not None:
        qeff = qeff[frequency_indices_xp]
        P = P[:, :, frequency_indices_xp]
    valid = (
        (qeff[None, ...] > 0.0)
        & xp.isfinite(qeff)[None, ...]
        & xp.isfinite(detC)
        & (detC != 0.0)
        & xp.all(xp.isfinite(invC), axis=(0, 1))
        & xp.all(xp.isfinite(P), axis=(0, 1))[None, ...]
    )
    weights = xp.where(valid, qeff[None, ...], 0.0)
    quadratic_pixel = xp.where(
        valid,
        xp.real(xp.einsum("abxmq,abmq->xmq", invC, P)),
        0.0,
    )
    safe_det = xp.where(valid, xp.abs(detC), 1.0)
    quadratic = -0.5 * xp.sum(weights * quadratic_pixel, axis=2)
    logdet = -0.5 * xp.sum(weights * xp.log(safe_det), axis=2)
    return logdet if noise_only else quadratic + logdet


def coarse_wdm_log_likelihood_batch(
    stat: CoarseWDMStatistic,
    covariance,
    *,
    noise_only: bool = False,
    frequency_indices=None,
):
    """Score a batch of coarse 3x3 covariances in one native array pass.

    This is the coarse-WDM counterpart of the FD/STFT batched likelihood
    route.  It avoids constructing one ``SensitivityMatrixBase`` and making
    one ``AnalysisContainer.likelihood`` call per proposal row.  The 3x3
    adjugate, quadratic form, and log-determinant reduction are vectorized
    across the complete batch and run inside NumPy/CuPy's native loops.
    """
    terms = coarse_wdm_log_likelihood_batch_frequency_terms(
        stat,
        covariance,
        noise_only=noise_only,
        frequency_indices=frequency_indices,
    )
    return get_array_module(terms).sum(terms, axis=1)


def _coarsen_covariance(fine_sens_mat, coarse_settings: CoarseWDMSettings):
    from .sensitivity import SensitivityMatrixBase

    covariance = _fine_covariance_array(fine_sens_mat)
    coarse = SensitivityMatrixBase(coarse_settings)
    coarse.sens_mat = coarse_settings.cell_mean(covariance)
    return coarse


def coarse_q_scan(
    fine_settings,
    fiducial_sens_mat,
    Q_list: Sequence[int],
    data: WDMSignal,
    comparison_sens_mat=None,
    *,
    use_ws: bool = True,
):
    """Measure coarse-likelihood error and nominal speedup for candidate Q."""
    from .diagnostic import residual_full_source_and_noise_likelihood

    fine_fiducial = residual_full_source_and_noise_likelihood(data, fiducial_sens_mat)
    fine_comparison = None
    if comparison_sens_mat is not None:
        fine_comparison = residual_full_source_and_noise_likelihood(data, comparison_sens_mat)

    covariance = _fine_covariance_array(fiducial_sens_mat)
    xp = get_array_module(covariance)
    diagonal = xp.real(xp.stack([covariance[a, a] for a in range(3)], axis=0))
    results = []

    def _masked_summary(values, valid):
        selected = values[valid]
        if int(selected.size) == 0:
            return float("nan"), float("nan")
        return float(xp.min(selected).item()), float(xp.median(selected).item())

    for Q in Q_list:
        coarse_settings = CoarseWDMSettings.from_fine(fine_settings, int(Q))
        stat = CoarseWDMStatistic.from_wdm_signal(
            data,
            coarse_settings,
            fiducial_sens_mat_fine=fiducial_sens_mat,
            use_ws=use_ws,
        )
        coarse_fiducial_mat = _coarsen_covariance(fiducial_sens_mat, coarse_settings)
        coarse_fiducial = coarse_wdm_log_likelihood(stat, coarse_fiducial_mat)

        variations = []
        for start, size in zip(coarse_settings.cell_starts, coarse_settings.cell_sizes):
            block = diagonal[..., int(start) : int(start + size)]
            mean = xp.mean(block, axis=-1)
            span = xp.max(block, axis=-1) - xp.min(block, axis=-1)
            variations.append(xp.where(mean != 0.0, xp.abs(span / mean), 0.0))
        variation = xp.stack(variations, axis=-1)
        sizes = xp.asarray(coarse_settings.cell_sizes, dtype=float)
        qratio_channels = stat.Qeff_channels / sizes[None, None, :]
        qratio = stat.Qeff / sizes[None, :]
        qeff_min, qeff_median = _masked_summary(qratio, stat.Qeff > 0.0)
        channel_summaries = [
            _masked_summary(qratio_channels[a], stat.Qeff_channels[a] > 0.0)
            for a in range(3)
        ]
        variation_max = []
        for a in range(3):
            valid_variation = xp.isfinite(variation[a]) & (stat.Qeff_channels[a] > 0.0)
            selected = variation[a][valid_variation]
            variation_max.append(
                float(xp.max(selected).item()) if int(selected.size) else float("nan")
            )

        result = {
            "Q": int(Q),
            "Ncoarse": coarse_settings.Ncoarse,
            "nominal_speedup": fine_settings.Nt_active / coarse_settings.Ncoarse,
            "qeff_ratio_min": qeff_min,
            "qeff_ratio_median": qeff_median,
            "qeff_channel_ratio_min": np.asarray([item[0] for item in channel_summaries]),
            "qeff_channel_ratio_median": np.asarray(
                [item[1] for item in channel_summaries]
            ),
            "worst_diagonal_fractional_variation": np.asarray(variation_max),
            "fiducial_logl_gap": float(coarse_fiducial - fine_fiducial),
        }
        if comparison_sens_mat is not None:
            coarse_comparison_mat = _coarsen_covariance(
                comparison_sens_mat, coarse_settings
            )
            coarse_comparison = coarse_wdm_log_likelihood(stat, coarse_comparison_mat)
            result["delta_logl_gap"] = float(
                (coarse_comparison - coarse_fiducial)
                - (fine_comparison - fine_fiducial)
            )
        results.append(result)
    return results
