"""Reusable utilities for the per-source TD-vs-fast validation chain.

THREE source-agnostic helpers, lifted from the verified blocks in
``scripts/gb_chunked_het/gb_chunked_prior_draws.py`` (duck-type lines 64-92;
fast-vs-dense calibration lines 415-552) and the dense-logL idiom repeated in
``scripts/{mbh,sobbh}/*_injection_final.py``. They reuse existing lisatools
machinery only (``TDSignal``/``AnalysisContainer``/``XYZ2SensitivityMatrix``/
``WDMSettings``) and take the source generators / fast-likelihood objects via
injection -- nothing source-specific is imported here.

Written for review under LISAanalysistools/plot_tests/; promotable verbatim into
``src/lisatools/validation/`` (only stdlib + lisatools deps) if approved.
"""
import numpy as np

from lisatools.domains import TDSignal, WDMSettings, WDMSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix


# ---------------------------------------------------------------------------
# Utility 1 -- TD array -> domain logL + overlap (the dense ground truth).
# ---------------------------------------------------------------------------
def domain_logL_and_overlap(data_td, template_td, target_settings, *,
                            sens_model, td_settings, window=None):
    """Push a TD data array + TD template through one domain and return the
    lisatools log-likelihood + overlap. Works for ``FDSettings`` *or*
    ``WDMSettings`` (this is the TD->FD/WDM ground truth for every source).

    Args:
        data_td, template_td (np.ndarray): ``(nchan, N)`` time-domain TDI.
        target_settings: ``FDSettings`` or ``WDMSettings`` to transform into.
        sens_model (str): e.g. ``"scirdv1"`` / ``"mrdv1"``.
        td_settings (TDSettings): the time-domain settings of the inputs.
        window (np.ndarray, optional): window applied in ``.transform``.

    Returns:
        dict: ``logL`` (=-1/2<d-h|d-h>), ``overlap`` (complex, normalized),
        ``mm`` (1-|O|), ``reO`` (1-Re(O)), ``snr_opt``, ``snr_det``, plus the
        ``data_sig``/``template_sig``/``analysis`` objects for reuse (e.g. d_d).
    """
    d = TDSignal(data_td, td_settings).transform(target_settings, window=window)
    t = TDSignal(template_td, td_settings).transform(target_settings, window=window)
    ac = AnalysisContainer(d, XYZ2SensitivityMatrix(target_settings, model=sens_model))
    O = ac.template_inner_product(t, normalize=True, complex=True)
    opt, det = ac.template_snr(t)
    logL = float(np.real(ac.template_likelihood(t)))
    return {
        "logL": logL, "overlap": complex(O), "mm": float(1.0 - abs(O)),
        "reO": float(1.0 - O.real), "snr_opt": float(opt), "snr_det": float(det),
        "data_sig": d, "template_sig": t, "analysis": ac,
    }


# ---------------------------------------------------------------------------
# Utility 2 -- WDM narrowband (mm5/mm2) slice  +  the get_ll_wdm holder.
# ---------------------------------------------------------------------------
def wdm_narrowband(parent_wdm_signal, f0, *, kind="mm5", backend=None):
    """Slice a parent ``WDMSignal`` onto the canonical mm5 / mm2 band around
    ``f0`` (the layer-df bounds documented in CLAUDE.md). Lifted from
    ``_make_wdm_signal_slice`` + the band math.

    kind="mm5": ``[f0-3*layer_df, f0+2*layer_df]``.
    kind="mm2": ``[(m_floor-0.5)*layer_df, (m_floor+1.5)*layer_df]``.
    """
    s = parent_wdm_signal.settings
    df = s.layer_df
    if kind == "mm5":
        lo, hi = f0 - 3.0 * df, f0 + 2.0 * df
    elif kind == "mm2":
        m = int(f0 / df)
        lo, hi = (m - 0.5) * df, (m + 1.5) * df
    else:
        raise ValueError("kind must be 'mm5' or 'mm2'")
    band = WDMSettings(s.Nf, s.Nt, s.data_dt, min_freq=lo, max_freq=hi,
                       min_time=s.min_time, max_time=s.max_time,
                       force_backend=backend if backend is not None else s.backend.name.split("_")[-1])
    arr = np.asarray(parent_wdm_signal.arr)[
        :, band.ind_min_f - s.ind_min_f: band.ind_max_f - s.ind_min_f + 1]
    return WDMSignal(arr, band)


class WDMHolder:
    """Minimal duck-type for ``WDMComputationsBase.get_ll_wdm``'s
    ``wdm_holder`` argument: exposes ``linear_data_arr[0]`` /
    ``linear_psd_arr[0]`` / ``__len__`` over full-grid ``(3, Nf, Nt)`` data +
    diagonal invC buffers. Promotion of ``_FullGridWDMHolder``.
    """
    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# Utility 3 -- fast get_ll_wdm  vs  dense TD-WDM logL (with the ll_scale calib).
# ---------------------------------------------------------------------------
def fast_vs_dense_wdm_logL(fast_comp, params, wdm_data_active, sens_mat_active,
                           wdm_settings, *, d_d, dense_logL,
                           convert_to_ra_dec=False):
    """Run a fast ``WDMComputationsBase`` subclass's ``get_ll_wdm``, reconstruct
    its logL on lisatools' normalization, and diff against the dense TD-WDM
    ground-truth logL. Encodes the empirical normalization-matching from
    ``gb_chunked_prior_draws.py:470-541``:

      the kernel returns BARE sums ``d_h_out``/``h_h_out``; lisatools'
      inner product is ``4*df*Sum(...*invC)``. With the same active-band data
      zero-filled onto the full ``(3,Nf,Nt)`` grid, the scale is fixed by the
      known ``<d|d>``:  ``ll_scale = d_d / Sum(d*d*invC)``;
      then ``logL = -1/2 (d_d + ll_scale*h_h - 2*ll_scale*d_h)``.

    Args:
        fast_comp: a ``GBWDMComputations`` / ``SOBBHWDMComputations`` instance
            (its ``_NPARAMS`` sets the param count; ``d_h_out``/``h_h_out`` are
            read after the call).
        params (array-like): the source params (length ``fast_comp._NPARAMS``).
        wdm_data_active (np.ndarray): mojito WDM data on the ACTIVE band,
            shape ``(nchan, Nf_active, Nt_active)``.
        sens_mat_active (np.ndarray): the WDM sensitivity (diagonal
            ``(nchan, Nf_a, Nt_a)`` or cross ``(nchan, nchan, Nf_a, Nt_a)``).
        wdm_settings (WDMSettings): the parent grid (Nf, Nt, ind_min_f, ...).
        d_d (float): lisatools ``<d|d>`` over the same band/sens.
        dense_logL (float): the TD-WDM ground-truth logL to compare against.

    Returns:
        dict: ``logL_fast``, ``logL_dense``, ``abs_diff``, ``rel_diff``,
        ``d_h``, ``h_h``, ``ll_scale``.
    """
    nch = int(np.asarray(wdm_data_active).shape[0])
    Nf, Nt = int(wdm_settings.Nf), int(wdm_settings.Nt)
    psd = np.asarray(sens_mat_active)
    psd_diag = np.stack([psd[c, c] for c in range(nch)], axis=0) if psd.ndim == 4 else psd
    with np.errstate(divide="ignore", invalid="ignore"):
        invC = 1.0 / np.where(np.isfinite(psd_diag) & (psd_diag > 0), psd_diag, np.inf)
    invC = np.where(np.isfinite(invC), invC, 0.0)

    data_full = np.zeros((nch, Nf, Nt), dtype=float)
    invC_full = np.zeros_like(data_full)
    ilo, ihi = int(wdm_settings.ind_min_f), int(wdm_settings.ind_max_f) + 1
    if int(wdm_settings.Nt_active) == Nt:
        data_full[:, ilo:ihi, :] = np.asarray(wdm_data_active)
        invC_full[:, ilo:ihi, :] = invC
    else:
        ts = wdm_settings.active_slice_t
        data_full[:, ilo:ihi, ts] = np.asarray(wdm_data_active)
        invC_full[:, ilo:ihi, ts] = invC

    holder = WDMHolder(data_full, invC_full)
    sum_dd = float(np.sum(data_full * data_full * invC_full))
    ll_scale = d_d / max(sum_dd, 1e-300)

    npar = int(getattr(fast_comp, "_NPARAMS", len(np.ravel(params))))
    fast_comp.get_ll_wdm(np.asarray(params, dtype=float).reshape(1, npar), holder,
                         convert_to_ra_dec=convert_to_ra_dec)
    d_h = ll_scale * float(np.asarray(fast_comp.d_h_out)[0])
    h_h = ll_scale * float(np.asarray(fast_comp.h_h_out)[0])
    logL_fast = -0.5 * (d_d + h_h - 2.0 * d_h)
    return {
        "logL_fast": float(logL_fast), "logL_dense": float(dense_logL),
        "abs_diff": float(abs(logL_fast - dense_logL)),
        "rel_diff": float(abs(logL_fast - dense_logL) / max(abs(dense_logL), 1e-300)),
        "d_h": d_h, "h_h": h_h, "ll_scale": ll_scale,
    }
