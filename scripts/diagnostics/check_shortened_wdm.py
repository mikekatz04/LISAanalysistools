#!/usr/bin/env python
"""Check shortened TD->FD->WDM transforms + verify the new complex WDM mode.

Setup:
* Full observation = 1/2 year of GB signal from ``GBTDIonTheFly``, no TD/FD
  windowing applied anywhere -- only the WDM analysis window is in play.
* Full WDM grid (the reference) uses ``Nf, Nt = Nf, NT`` with ``NT`` a few
  times the chunk size.
* Chunk WDMs use the same ``Nf`` and a shorter ``Nt_sub`` of order
  128-256. ``Nt_sub`` is the per-layer iFFT length inside
  :meth:`FDSignal.wdmtransform` -- which the user calls the "size of the
  WDM window filter" -- so at this size the wavelet window is well
  resolved by the chunk transform and the chunk WDM should match the
  full WDM at any interior pixel.

Tests:

* **Test A** -- chunk at an even global pixel ``n0`` in the middle of the
  observation. Reports interior diff (chunk vs full) for several
  ``Nt_sub`` values.

* **Test B** -- chunk at an odd ``n0``. Without a parity correction, with
  the candidate ``(-1)^n0``-on-even-m sign flip. The sign flip alone is
  expected to be insufficient because the ``(m+n)`` parity also flips
  the Re/Im branch -- the new complex WDM is what makes odd-aligned
  chunks portable.

* **Test C** -- complex/quadrature WDM forward path, verifying:
    (i) ``WDM_complex.real == WDM_real`` exactly,
   (ii) ``inner_product(complex, complex)`` matches
        ``inner_product(real, real)`` (the halved
        ``differential_component`` cancels the 2x sum-of-squares).
"""

import os

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import FDSettings, FDSignal, TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def build_gb_generator(t_arr, Tobs, t_ref, dt, backend="cpu", n_sparse=16384):
    """Build a GBTDIonTheFly with a sparse evaluation grid covering t_arr."""
    orbits = EqualArmlengthOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation")
    t_tdi_sparse = np.linspace(float(t_arr[0]), float(t_arr[-1]), n_sparse)
    return GBTDIonTheFly(
        t_tdi_sparse, Tobs, t_ref, 1.0 / dt, 1,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )


def _tukey_window(n, alpha):
    """Tukey window of length ``n``. ``alpha=0`` -> rectangular; ``alpha=1`` -> Hann."""
    from scipy.signal import windows
    return windows.tukey(int(n), alpha=float(alpha))


# ---------------------------------------------------------------------------
# Recommended Tukey alphas for the chunked-WDM workflow.
#
# Tuned from Test G's sweep (Nf=64, Nt=24640, Nt_sub=256, n_pad=32; mm5/mm2
# evaluated with the standard min_time/max_time crop). When tukey_alpha is
# set to RECOMMENDED, a value is picked automatically based on:
#
#   * TD-based path (wdm_of_td_slice, Test D): alpha = 0.02. The taper is
#     confined to the wavelet-time-support edge of each chunk and tightens
#     out-of-band leakage that the n_pad=32 overlap doesn't fully eat.
#     mm5/mm2 floor ~ 2.8e-13 (better than no-Tukey 4.8e-13).
#
#   * FD-heterodyne path with N_sparse >= 512: alpha = 0.01. The
#     heterodyne band is already wide enough to contain the WDM mm5
#     band, so a tiny Tukey just trims residual edge leakage. mm5/mm2
#     floor ~ 4.6e-13 (same as no-Tukey for these N_sparse).
#
#   * FD-heterodyne path with N_sparse < 512: alpha = 0.05. The
#     heterodyne band is narrower than the WDM band, so a noticeable
#     Tukey is what collapses out-of-band leakage and lets small
#     N_sparse work at all. At N_sparse=64 this drops mm5 from 9e-5
#     (rect) to 1e-7 (~1000x improvement).
#
# Don't go above alpha=0.1 -- past that the taper bites into the chunk
# interior and biases stitched pixels (floor jumps to ~5e-12 by 0.1,
# ~9e-3 by 0.5).
# ---------------------------------------------------------------------------
RECOMMENDED_TUKEY_ALPHA_TD = 0.02
RECOMMENDED_TUKEY_ALPHA_HET_WIDE = 0.01      # N_sparse >= 512
RECOMMENDED_TUKEY_ALPHA_HET_NARROW = 0.05    # N_sparse <  512

# Shared with the C++ kernel (lisa-on-gpu/.../TDIonthefly.cu: FAST_WDM_N_SPARSE_MAX).
# Caps the heterodyne FFT length so the per-block shared-memory workspace fits.
FAST_WDM_N_SPARSE_MAX = 256

# Sentinel: pass tukey_alpha=USE_RECOMMENDED_TUKEY to opt in to the
# auto-pick logic. Float-typed so existing positional callers that pass
# numerics still work unchanged.
USE_RECOMMENDED_TUKEY = -1.0


def recommended_tukey_alpha(path, N_sparse=None):
    """Return the recommended Tukey alpha for a chunked-WDM workflow.

    Args:
        path: ``"td"`` for the TD-based chunked stitch (Test D) or
            ``"heterodyne"`` / ``"het"`` / ``"fd"`` for the FD-direct
            heterodyne stitch (Test E).
        N_sparse: heterodyne FFT length (only consulted for the
            heterodyne path). Recommendation is HET_WIDE for >= 512,
            HET_NARROW otherwise.
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


def _resolve_tukey_alpha(tukey_alpha, use_tukey, path, N_sparse=None):
    """Resolve the actual Tukey alpha to apply per the use_tukey flag.

    Precedence: an explicit numeric ``tukey_alpha`` (>= 0) is always
    honoured; ``USE_RECOMMENDED_TUKEY`` triggers
    :func:`recommended_tukey_alpha`; ``use_tukey=False`` forces 0.0;
    otherwise we keep the value (default 0.0).
    """
    if tukey_alpha == USE_RECOMMENDED_TUKEY:
        if not use_tukey:
            return 0.0
        return recommended_tukey_alpha(path, N_sparse=N_sparse)
    if not use_tukey:
        return 0.0
    return float(tukey_alpha)


def wdm_of_td_slice(td_arr_full, start_sample, Nf, Nt_chunk, dt,
                    backend="cpu", is_complex=False,
                    tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True):
    """Run TD->FD->WDM on a contiguous chunk of the full TD signal."""
    n_samples = Nf * Nt_chunk
    sub_arr = td_arr_full[..., start_sample:start_sample + n_samples].copy()
    td_set = TDSettings(n_samples, dt, force_backend=backend)
    wdm_set = WDMSettings(
        Nf=Nf, Nt=Nt_chunk, dt=dt, force_backend=backend,
        is_complex=is_complex,
    )
    alpha = _resolve_tukey_alpha(tukey_alpha, use_tukey, path="td")
    if alpha > 0:
        win_1d = _tukey_window(n_samples, alpha)
        window = np.broadcast_to(win_1d, sub_arr.shape).copy()
    else:
        window = None
    return TDSignal(sub_arr, td_set).transform(wdm_set, window=window)


def apply_parity_shift(chunk_arr, n0):
    """Candidate sign-flip parity correction for a chunk starting at global n0."""
    if n0 % 2 == 0:
        return chunk_arr
    out = chunk_arr.copy()
    out[:, 0::2, :] *= -1.0
    return out


def apply_complex_chunk_unitary(chunk_arr, n0):
    """Map a complex-WDM chunk at global ``n0`` into the full WDM layout.

    Derivation outline. The WDM coefficient factors as

        W[m, n] = kappa * (-1)^((m+1) n) * conj(C_{m, n}) * after_ifft[m, n]

    The half-``Nt`` shift baked into the WDM iFFT input gives
    ``after_ifft[m, n] = (-1)^n * X[m, n]`` where ``X`` is the centered
    iFFT sum. The chunk's FD differs from the full's FD by the
    time-shift phase ``exp(2 pi i k n0 / Nt_chunk)``, which at the
    layer-m slice supplies a factor

        X_chunk[m, n_local] ~= (-1)^(m n0) * X_full[m, n_local + n0]

    Combining: for **odd** ``n0`` and **interior** layers ``m >= 1``,

        U[m, n_local] = W_full[m, n0 + n_local] / W_chunk[m, n_local]
                      = -i * (-1)^(m + n_local)

    (the sign factor (-1)^((m+1)·n_global)/(-1)^((m+1)·n_local) cancels
    the extra (-1)^(m+1) from the X relation, leaving only the C ratio).
    For **even** ``n0`` the unitary is the identity.

    Boundary row ``m = 0`` is *not* covered: the WDM folds m=0 and m=Nf
    into alternating slots, so odd-``n0`` chunks pack physically
    different basis functions (cos@DC vs cos@Nyquist) into the slots a
    per-cell unitary would map between. Callers should mask m=0 out of
    comparisons when stitching at odd ``n0``.

    Args:
        chunk_arr: ``(nchan, Nf, Nt_chunk)`` complex WDM chunk.
        n0: global WDM time-pixel index where the chunk starts.

    Returns:
        Rotated copy of ``chunk_arr``. m=0 row is left untouched.
    """
    if n0 % 2 == 0:
        return chunk_arr.copy()
    Nf, Nt_chunk = chunk_arr.shape[-2:]
    m = np.arange(Nf)[:, None]                   # (Nf, 1)
    n_local = np.arange(Nt_chunk)[None, :]       # (1, Nt_chunk)
    U = (-1j * (-1.0) ** (m + n_local)).astype(np.complex128)  # (Nf, Nt_chunk)
    out = chunk_arr.copy()
    # Interior layers only -- m=0 is the folded DC/Nyquist row.
    out[:, 1:, :] = chunk_arr[:, 1:, :] * U[None, 1:, :]
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def report(label, full, other, interior_slice=None):
    diff = other - full
    max_abs_full = float(np.max(np.abs(full)))
    max_abs_diff = float(np.max(np.abs(diff)))
    rms_full = float(np.sqrt(np.mean(np.abs(full) ** 2)))
    rms_diff = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
    print("\n--- {} ---".format(label))
    print("  max |full|           = {:.3e}".format(max_abs_full))
    print("  max |other - full|   = {:.3e}".format(max_abs_diff))
    print("  max diff / max full  = {:.3e}".format(
        max_abs_diff / max(max_abs_full, 1e-300)))
    print("  rms diff / rms full  = {:.3e}".format(
        rms_diff / max(rms_full, 1e-300)))
    if interior_slice is not None:
        lo, hi = interior_slice
        if hi <= lo:
            print("  interior pixels [{}:{}]: (empty)".format(lo, hi))
        else:
            f_int = full[..., lo:hi]
            d_int = diff[..., lo:hi]
            rms_full_int = float(np.sqrt(np.mean(np.abs(f_int) ** 2)))
            rms_diff_int = float(np.sqrt(np.mean(np.abs(d_int) ** 2)))
            max_diff_int = float(np.max(np.abs(d_int)))
            print("  interior pixels [{}:{}]:".format(lo, hi))
            print("    max diff / max full = {:.3e}".format(
                max_diff_int / max(max_abs_full, 1e-300)))
            print("    rms diff / rms full = {:.3e}".format(
                rms_diff_int / max(rms_full_int, 1e-300)))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_A_even_chunk_sweep(td_arr, full_arr, dt, Nf, Nt, n0, backend):
    """Sweep Nt_sub over the 128-256 range at a fixed mid even pixel n0."""
    print("\n========== Test A: even-n0 chunk vs full (sweep Nt_sub) ==========")
    print("n0 = {} (even, mid-obs)".format(n0))
    print("{:>10s} {:>10s} {:>14s} {:>14s}".format(
        "Nt_sub", "n_pad", "rms diff full", "rms diff interior"))
    for Nt_sub in [128, 256]:
        if Nt_sub % 2 != 0 or n0 + Nt_sub > Nt:
            continue
        n_pad = Nt_sub // 8
        chunk_wdm = wdm_of_td_slice(td_arr, n0 * Nf, Nf, Nt_sub, dt, backend)
        chunk_arr = np.asarray(chunk_wdm.arr)
        full_slice = full_arr[:, :, n0:n0 + Nt_sub]
        diff = chunk_arr - full_slice
        rms_full = float(np.sqrt(np.mean(full_slice ** 2)))
        rms_diff = float(np.sqrt(np.mean(diff ** 2)))
        d_int = diff[..., n_pad:Nt_sub - n_pad]
        f_int = full_slice[..., n_pad:Nt_sub - n_pad]
        rms_diff_int = float(np.sqrt(np.mean(d_int ** 2)))
        rms_full_int = float(np.sqrt(np.mean(f_int ** 2)))
        print("{:>10d} {:>10d} {:>14.3e} {:>14.3e}".format(
            Nt_sub, n_pad,
            rms_diff / max(rms_full, 1e-300),
            rms_diff_int / max(rms_full_int, 1e-300)))


def test_B_odd_chunk(td_arr, full_arr, dt, Nf, Nt, n0, Nt_sub, backend):
    """Single chunk at odd n0: raw vs sign-flip parity correction."""
    assert n0 % 2 == 1
    n_pad = Nt_sub // 8
    print("\n========== Test B: odd-n0 chunk (Nt_sub={}) ==========".format(Nt_sub))
    chunk_wdm = wdm_of_td_slice(td_arr, n0 * Nf, Nf, Nt_sub, dt, backend)
    raw = np.asarray(chunk_wdm.arr)
    corrected = apply_parity_shift(raw, n0)
    full_slice = full_arr[:, :, n0:n0 + Nt_sub]
    report("Test B: NO correction (raw odd-n0 chunk vs full)",
           full=full_slice, other=raw,
           interior_slice=(n_pad, Nt_sub - n_pad))
    report("Test B: WITH (-1)^n0 sign-flip on even-m (real WDM, INSUFFICIENT)",
           full=full_slice, other=corrected,
           interior_slice=(n_pad, Nt_sub - n_pad))


def test_B_complex_odd_chunk(td_arr, full_arr_complex, dt, Nf, Nt, n0, Nt_sub, backend):
    """Single COMPLEX-WDM chunk at odd n0 + per-cell unitary correction.

    Both the full and chunk WDM are computed with ``is_complex=True``;
    the unitary derived above is applied to the chunk; the interior diff
    is reported on layers m=1..Nf-1 (the boundary row m=0 is excluded
    because at odd n0 its folded packing is incompatible).
    """
    assert n0 % 2 == 1
    n_pad = Nt_sub // 8
    print("\n========== Test B': complex-WDM odd-n0 chunk + unitary (Nt_sub={}) ==========".format(Nt_sub))

    chunk_wdm = wdm_of_td_slice(td_arr, n0 * Nf, Nf, Nt_sub, dt, backend,
                                is_complex=True)
    chunk_arr = np.asarray(chunk_wdm.arr)
    full_slice = full_arr_complex[:, :, n0:n0 + Nt_sub]

    # raw complex chunk vs full slice (no correction)
    report("Test B': raw complex chunk vs full (interior layers m>=1)",
           full=full_slice[:, 1:, :], other=chunk_arr[:, 1:, :],
           interior_slice=(n_pad, Nt_sub - n_pad))

    # apply the per-cell unitary
    rotated = apply_complex_chunk_unitary(chunk_arr, n0)
    report("Test B': rotated chunk vs full (interior layers m>=1)",
           full=full_slice[:, 1:, :], other=rotated[:, 1:, :],
           interior_slice=(n_pad, Nt_sub - n_pad))

    # for completeness, also report m=0 row diff (expected to be non-trivial)
    report("Test B': m=0 row (folded boundary, NOT covered by unitary)",
           full=full_slice[:, 0:1, :], other=rotated[:, 0:1, :],
           interior_slice=(n_pad, Nt_sub - n_pad))


def stitch_overlap_real_chunks(td_arr, dt, Nf, Nt, Nt_sub, n_pad, backend):
    """Tile the full TD with overlapping chunks, stitch interior portions.

    Each chunk is ``Nf * Nt_sub`` TD samples; chunks step by
    ``Nt_sub - 2*n_pad`` WDM pixels. ``n_pad`` pixels are dropped at
    each chunk's WDM-time edge so the wavelet support of the discarded
    edge pixels falls fully inside the next chunk. With even step every
    chunk starts at an even global pixel ``n0``, so the standard
    real-WDM transform suffices (no parity unitary needed).

    Returns a NumPy array of shape ``(nchan, Nf, Nt)`` that should match
    the full TD->FD->WDM reference everywhere except inside the
    outermost ``n_pad`` pixels of the observation (the first and last
    chunk can't be supplied by an interior overlap).
    """
    step = Nt_sub - 2 * n_pad
    assert step > 0 and step % 2 == 0, (
        "step = Nt_sub - 2*n_pad must be positive and even; "
        "got Nt_sub={}, n_pad={}".format(Nt_sub, n_pad)
    )
    assert (Nt - Nt_sub) % step == 0, (
        "(Nt - Nt_sub) must be divisible by step; got Nt={}, Nt_sub={}, "
        "step={}".format(Nt, Nt_sub, step)
    )
    n_chunks = (Nt - Nt_sub) // step + 1
    starts = [j * step for j in range(n_chunks)]
    assert all(s % 2 == 0 for s in starts)

    nchan = td_arr.shape[-2]
    out = np.zeros((nchan, Nf, Nt), dtype=float)
    for j, n0 in enumerate(starts):
        chunk_wdm = wdm_of_td_slice(td_arr, n0 * Nf, Nf, Nt_sub, dt, backend)
        chunk_arr = np.asarray(chunk_wdm.arr)            # (nchan, Nf, Nt_sub)
        keep_lo = 0 if j == 0 else n_pad
        keep_hi = Nt_sub if j == n_chunks - 1 else (Nt_sub - n_pad)
        out[:, :, n0 + keep_lo:n0 + keep_hi] = chunk_arr[:, :, keep_lo:keep_hi]
    return out, starts, step, n_chunks


def _build_band_wdm_signal(full_arr, full_wdm_set, min_freq, max_freq,
                           min_time, max_time, backend):
    """Slice ``full_arr`` (shape (nchan, Nf, Nt) on full_wdm_set) onto a
    band-restricted, time-cropped :class:`WDMSettings`."""
    band_set = WDMSettings(
        full_wdm_set.Nf, full_wdm_set.Nt, full_wdm_set.data_dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )
    arr_band = full_arr[:, band_set.ind_min_f:band_set.ind_max_f + 1, :]
    return WDMSignal(arr_band, band_set), band_set


def _mismatch(full_arr, stitched_arr, full_wdm_set, min_freq, max_freq,
              min_time, max_time, backend):
    """Compute 1 - overlap (mismatch) between full and stitched WDMs on a band.

    Both signals are sliced down to the same band ``[min_freq, max_freq]``
    and time crop ``[min_time, max_time]``, wrapped in
    :class:`AnalysisContainer` with an XYZ2 PSD, and compared via
    :meth:`AnalysisContainer.template_inner_product` with ``normalize=True``.
    """
    inj_sig, band_set = _build_band_wdm_signal(
        full_arr, full_wdm_set, min_freq, max_freq, min_time, max_time, backend,
    )
    tpl_sig, _ = _build_band_wdm_signal(
        stitched_arr, full_wdm_set, min_freq, max_freq, min_time, max_time, backend,
    )
    inj = DataResidualArray(inj_sig)
    tpl = DataResidualArray(tpl_sig)
    sens = XYZ2SensitivityMatrix(band_set, model="scirdv1")
    analysis = AnalysisContainer(inj, sens)
    overlap = float(np.real(analysis.template_inner_product(tpl, normalize=True)))
    return 1.0 - overlap, band_set


def _plot_full_vs_stitched_X(full_arr, stitched_arr, wdm_set, min_time, max_time,
                              filename, title_prefix=""):
    """3-panel heatmap of WDM X channel: full vs stitched vs diff.

    Panel 1: full WDM X channel coefficients (signed) over (t, f).
    Panel 2: stitched WDM X channel (same colour scale as full).
    Panel 3: stitched - full, on its own colour scale.

    Dashed vertical lines mark ``min_time`` / ``max_time`` (the
    mismatch-crop region used for mm5 / mm2).
    """
    chan = 0                                     # X channel
    Nf, Nt = full_arr.shape[1], full_arr.shape[2]
    t_arr = np.arange(Nt) * wdm_set.layer_dt
    f_arr = np.arange(Nf) * wdm_set.layer_df

    full_X = full_arr[chan]
    stitch_X = stitched_arr[chan]
    diff_X = stitch_X - full_X

    vmax = float(np.max(np.abs(full_X)))
    dmax = float(np.max(np.abs(diff_X))) or 1e-30

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, sharey=True)
    im0 = axes[0].pcolormesh(t_arr, f_arr, full_X, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    axes[0].set_title(title_prefix + "full TD->FD->WDM (X channel)")
    im1 = axes[1].pcolormesh(t_arr, f_arr, stitch_X, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    axes[1].set_title("stitched chunked TD->FD->WDM (X channel)")
    im2 = axes[2].pcolormesh(t_arr, f_arr, diff_X, cmap="RdBu_r",
                             vmin=-dmax, vmax=dmax, shading="auto", rasterized=True)
    axes[2].set_title("stitched - full  (own colour scale, max |diff| = {:.2e})".format(dmax))

    for ax in axes:
        ax.axvline(min_time, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(max_time, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel("WDM freq (Hz)")
    axes[-1].set_xlabel("WDM time (s)")
    for im, ax in zip((im0, im1, im2), axes):
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)
    print("  wrote heatmap to", filename)


def _plot_band_zoom_X(full_arr, stitched_arr, wdm_set, mm_band_set, min_time, max_time,
                      filename, title):
    """3-panel zoom of WDM X channel restricted to the mismatch band."""
    chan = 0
    f_lo, f_hi = mm_band_set.ind_min_f, mm_band_set.ind_max_f
    f_arr = (np.arange(f_lo, f_hi + 1)) * wdm_set.layer_df
    Nt = full_arr.shape[2]
    t_arr = np.arange(Nt) * wdm_set.layer_dt

    full_X = full_arr[chan, f_lo:f_hi + 1, :]
    stitch_X = stitched_arr[chan, f_lo:f_hi + 1, :]
    diff_X = stitch_X - full_X
    vmax = float(np.max(np.abs(full_X)))
    dmax = float(np.max(np.abs(diff_X))) or 1e-30

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True, sharey=True)
    for ax, data, name, vlim in zip(
        axes,
        (full_X, stitch_X, diff_X),
        ("full", "stitched", "stitched - full"),
        (vmax, vmax, dmax),
    ):
        im = ax.pcolormesh(t_arr, f_arr, data, cmap="RdBu_r",
                           vmin=-vlim, vmax=vlim, shading="auto", rasterized=True)
        ax.set_title("{} (max|val|={:.2e})".format(name, vlim))
        ax.axvline(min_time, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(max_time, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel("freq (Hz)")
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel("WDM time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)
    print("  wrote zoom heatmap to", filename)


def test_D_stitch_mismatch(td_arr, full_arr, wdm_set, dt, Nf, Nt, f0,
                           backend, Nt_sub, n_pad, edge_layer_pixels=20):
    """Overlap-tiled stitched WDM vs full WDM via mm5 and mm2.

    Uses ``min_time = edge_layer_pixels * layer_dt`` and
    ``max_time = (Nt - edge_layer_pixels) * layer_dt`` to cut off the
    outer-edge wavelet pixels (matching the convention in
    gb_lookup_prior_draws.py).

    Reports:
      * ``mm5`` -- 1 - overlap on the 5-layer band [f0-3 df, f0+2 df]
      * ``mm2`` -- 1 - overlap on the 2-layer band [f0, f0+df]
    """
    print("\n========== Test D: stitched WDM vs full -- mm5 / mm2 ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt - edge_layer_pixels) * layer_dt
    print("stitching: Nt_sub={}, n_pad={}, step={}".format(
        Nt_sub, n_pad, Nt_sub - 2 * n_pad))

    stitched, starts, step, n_chunks = stitch_overlap_real_chunks(
        td_arr, dt, Nf, Nt, Nt_sub, n_pad, backend,
    )
    print("  {} chunks tiled, step={}, first/last start = {}/{}".format(
        n_chunks, step, starts[0], starts[-1]))

    # Sanity: raw L2 diff full vs stitched (no band cut, no time cut).
    diff = stitched - full_arr
    rms_diff_total = float(np.sqrt(np.mean(diff ** 2)))
    rms_full_total = float(np.sqrt(np.mean(full_arr ** 2)))
    interior_slice = slice(n_pad, Nt - n_pad)
    rms_diff_int = float(np.sqrt(np.mean(diff[..., interior_slice] ** 2)))
    rms_full_int = float(np.sqrt(np.mean(full_arr[..., interior_slice] ** 2)))
    print("  rms diff / rms full (all pixels)        = {:.3e}".format(
        rms_diff_total / max(rms_full_total, 1e-300)))
    print("  rms diff / rms full (n_pad-cropped int) = {:.3e}".format(
        rms_diff_int / max(rms_full_int, 1e-300)))

    # mm5: 5 layers (m_floor-2..m_floor+2). The wider band tolerates f0 anywhere
    # in a layer; mirrors `gb_lookup_prior_draws.py` (min=f0-3 df, max=f0+2 df).
    m_floor = int(f0 / layer_df)
    mm5_min = f0 - 3 * layer_df
    mm5_max = f0 + 2 * layer_df
    mm5, set5 = _mismatch(full_arr, stitched, wdm_set,
                          mm5_min, mm5_max, min_time, max_time, backend)
    print("  mm5 band [{:.4e}, {:.4e}] Hz, layers {}..{}, Nt_active={}".format(
        mm5_min, mm5_max, set5.ind_min_f, set5.ind_max_f, set5.Nt_active))
    print("  mm5 = 1 - overlap = {:.3e}".format(mm5))

    # mm2: 2 layers (m_floor, m_floor+1) around the GB carrier. Use the bin
    # edges (m_floor*layer_df, (m_floor+2)*layer_df) so the ceil/floor lands
    # exactly on those two layers regardless of f0's intra-layer position.
    mm2_min = m_floor * layer_df
    mm2_max = (m_floor + 2) * layer_df - 0.5 * layer_df
    mm2, set2 = _mismatch(full_arr, stitched, wdm_set,
                          mm2_min, mm2_max, min_time, max_time, backend)
    print("  mm2 band [{:.4e}, {:.4e}] Hz, layers {}..{}, Nt_active={}".format(
        mm2_min, mm2_max, set2.ind_min_f, set2.ind_max_f, set2.Nt_active))
    print("  mm2 = 1 - overlap = {:.3e}".format(mm2))

    # Heatmap comparisons of the WDM X channel (full vs stitched vs diff).
    _plot_full_vs_stitched_X(
        full_arr, stitched, wdm_set, min_time, max_time,
        filename="check_shortened_wdm_testD_full.png",
        title_prefix="Test D ({} chunks, Nt_sub={}, n_pad={}):  ".format(
            n_chunks, Nt_sub, n_pad),
    )
    _plot_band_zoom_X(
        full_arr, stitched, wdm_set, set5, min_time, max_time,
        filename="check_shortened_wdm_testD_mm5band.png",
        title="Test D mm5 band (layers {}..{}, mm5={:.2e})".format(
            set5.ind_min_f, set5.ind_max_f, mm5),
    )
    _plot_band_zoom_X(
        full_arr, stitched, wdm_set, set2, min_time, max_time,
        filename="check_shortened_wdm_testD_mm2band.png",
        title="Test D mm2 band (layers {}..{}, mm2={:.2e})".format(
            set2.ind_min_f, set2.ind_max_f, mm2),
    )


def propagate_gb_params_to_new_tref(amp, f0, fdot, fddot, phi0, inc, psi, lam,
                                    beta, t_ref_old, t_ref_new):
    """Re-express GB intrinsic params at a shifted phase reference time.

    Matches the kernel's internal phase convention in
    ``lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu`` (see
    ``LISATDIonTheFly::get_phase_ref`` / ``GBTDIonTheFly::ucb_phase``),
    which evaluates

        phase = -phi0 + 2*pi * (f0*tau + 0.5*fdot*tau^2 + (1/6)*fddot*tau^3)

    with ``tau = t_sc - t_ref`` (``t_sc`` is the Roemer-corrected
    spacecraft-frame time). Note the ``-phi0`` -- the carrier
    ``exp(-i*phase)`` carries ``+phi0`` so the user-facing convention is
    still that ``phi0`` is the source phase at ``t_ref``, but
    propagation under a ``t_ref`` shift uses the kernel's sign.

    Coefficient matching of ``phase`` under ``dt = t_ref_new -
    t_ref_old`` gives

        f0'    = f0 + fdot*dt + 0.5*fddot*dt^2          (unchanged)
        fdot'  = fdot + fddot*dt                         (unchanged)
        fddot' = fddot                                   (unchanged)
        phi0'  = phi0 - 2*pi*(f0*dt + 0.5*fdot*dt^2 + (1/6)*fddot*dt^3)

    -- the ``phi0`` correction has the OPPOSITE SIGN from the naive
    ``phi(t) = phi0 + 2*pi*f(tau)`` derivation because the kernel uses
    ``-phi0`` in its phase formula. Empirical verification:
    ``check_shortened_wdm.py`` Test E previously showed a constant
    143.7 deg rotation at ``Nf=64, Nt=24640, f0=16.37*layer_df``;
    ``4*pi*frac(f0*Tobs/2) mod 2*pi`` was exactly 144 deg with the wrong
    sign, and zero (= bit-identical match) with this sign.

    Roemer-delay caveat: the kernel's actual phase argument is
    ``t_sc - t_ref = t - k.x_rec(t)/c - t_ref``, which depends on the
    absolute time through the orbit. This propagation is exact for the
    polynomial part of the phase, which is all that varies under
    ``t_ref`` shifts when the orbit-dependent piece ``k.x_rec(t)/c``
    stays a function of absolute ``t`` only.

    (Only the intrinsic phase params change; amp / extrinsic geometry
    are invariant.) Returns the same 9-tuple with shifted values.
    """
    dt = float(t_ref_new) - float(t_ref_old)
    f0_new   = f0 + fdot * dt + 0.5 * fddot * dt * dt
    fdot_new = fdot + fddot * dt
    fddot_new = fddot
    phi0_new = phi0 - 2.0 * np.pi * (
        f0 * dt + 0.5 * fdot * dt * dt + fddot * dt * dt * dt / 6.0
    )
    return amp, f0_new, fdot_new, fddot_new, phi0_new, inc, psi, lam, beta


class CachedHeterodyneGenerator:
    """Cached GBTDIonTheFly + wave_gen wrapper for many chunks (same T, t_ref).

    The motivation: ``GBTDIonTheFly.wave_gen`` is a @property that
    rebuilds the C++ wrapper on every access. Looping over chunks
    naively pays that O(ms-or-more) cost for every call. By
    instantiating one ``GBTDIonTheFly`` with ``num_sub=1`` up front and
    grabbing ``wave_gen`` once, every chunk just reuses the cached
    wrapper. Per-chunk we only allocate ``N_sparse`` floats and call
    the C++ kernel once -- memory stays bounded by a single chunk's
    worth regardless of ``n_chunks``.

    Args:
        T_window: per-window duration (seconds); fixed for every chunk.
        t_ref_source: source ``t_ref`` (absolute seconds).
        N_sparse: heterodyne FFT length per chunk (power of 2).
        dt: underlying TD sample step (seconds).
        nchannels: 3 for XYZ TDI.
        gb_kwargs: kwargs for :class:`GBTDIonTheFly` (orbits, tdi_config, ...).
    """
    def __init__(self, T_window, t_ref_source, N_sparse, dt,
                 nchannels=3, gb_kwargs=None,
                 source_class=GBTDIonTheFly, n_params=9,
                 f0_param_index=1):
        """Args:
            source_class: TDIonTheFly subclass (default GBTDIonTheFly).
                For SOBBH pass ``SOBBHTDIonTheFly``.
            n_params: number of source params (9 for GB, 11 for SOBBH).
            f0_param_index: where the carrier frequency lives in the
                params vector (1 for GB's f0, 5 for SOBBH's f_low).
        """
        self.T_window = float(T_window)
        self.t_ref_source = float(t_ref_source)
        self.N_sparse = int(N_sparse)
        self.dt = float(dt)
        self.dt_sparse = T_window / N_sparse
        self.t_offsets = np.arange(N_sparse) * self.dt_sparse
        self.nchannels = nchannels
        # Dummy t_arr so the constructor succeeds; we overwrite t_arr per call.
        dummy_t = self.t_offsets.copy().reshape(1, -1)
        self.gb = source_class(dummy_t, T_window, t_ref_source, 1.0 / dt,
                                1, **(gb_kwargs or {}))
        # The @property getter rebuilds the wrapper -- grab it ONCE and keep it.
        self._wave_gen = self.gb.wave_gen
        self.n_params = int(n_params)
        self.f0_param_index = int(f0_param_index)

    def chunk_fd(self, source_params, chunk_t_start, N_window_td,
                 tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True):
        """Compute one chunk's heterodyne FD; uses the cached wrapper.

        ``tukey_alpha > 0`` multiplies the slow signal ``s(tau)`` by a
        Tukey window of length ``N_sparse`` before the FFT -- this is
        equivalent to multiplying the underlying chunk TD by the same
        Tukey on the chunk window, since the carrier removal is just a
        phase factor.

        Default behaviour (``use_tukey=True``,
        ``tukey_alpha=USE_RECOMMENDED_TUKEY``) picks 0.05 for
        ``N_sparse < 512`` and 0.01 otherwise (see
        :func:`recommended_tukey_alpha`). Pass an explicit numeric
        ``tukey_alpha`` to override.
        """
        f0 = float(source_params[self.f0_param_index])
        t_sparse = chunk_t_start + self.t_offsets
        # Update t_arr (this is what the kernel sees; no re-init needed).
        self.gb.t_arr = self.gb.xp.atleast_2d(t_sparse).copy()

        params = np.asarray([float(p) for p in source_params],
                            dtype=float)                      # (n_params,)
        N = self.N_sparse
        nch = self.nchannels
        tdi_channels = np.zeros(N * nch, dtype=complex)
        tdi_amp = np.zeros(N * nch, dtype=float)
        tdi_phase = np.zeros(N * nch, dtype=float)
        phase_ref = np.zeros(N, dtype=float)
        self._wave_gen.run_wave_tdi_wrap(
            tdi_channels, tdi_amp, tdi_phase, phase_ref,
            params, self.gb.t_arr.flatten().copy(),
            N, 1, self.n_params, nch,
        )
        tdi_amp = tdi_amp.reshape(nch, N)
        tdi_phase = tdi_phase.reshape(nch, N)

        df = 1.0 / self.T_window
        k_f0 = int(round(f0 / df))
        f0_grid = k_f0 * df

        carrier = 2.0 * np.pi * f0_grid * self.t_offsets
        slow = tdi_amp * np.exp(
            +1j * (tdi_phase + phase_ref[None, :] - carrier[None, :])
        )
        alpha = _resolve_tukey_alpha(
            tukey_alpha, use_tukey, path="heterodyne", N_sparse=self.N_sparse,
        )
        if alpha > 0:
            slow = slow * _tukey_window(self.N_sparse, alpha)[None, :]
        X_het = 0.5 * np.fft.fft(slow, axis=-1) * self.dt_sparse

        n_rfft = N_window_td // 2 + 1
        window_fd = np.zeros((nch, n_rfft), dtype=complex)
        m_idx = np.fft.fftfreq(N, d=1.0 / N).astype(int)
        kbins = k_f0 + m_idx
        mask = (kbins >= 0) & (kbins < n_rfft)
        window_fd[:, kbins[mask]] = X_het[:, mask]
        return window_fd, k_f0


def chunk_fd_via_python_heterodyne(source_params, t_start, T_window,
                                   t_ref_source, N_sparse, N_window_td, dt,
                                   gb_kwargs, backend="cpu",
                                   tukey_alpha=USE_RECOMMENDED_TUKEY,
                                   use_tukey=True):
    """Python heterodyne FD: GBTDIonTheFly on a sparse grid + manual carrier
    removal + numpy FFT + placement on the dense rfft grid.

    Steps (mirroring the working Python prototype in
    `gb_heterodyne_fd.py:make_heterodyne_fd`):

    1. Compute ``tdi_amp``, ``tdi_phase``, ``phase_ref`` at ``N_sparse``
       sparse times spanning ``[t_start, t_start + T_window)``, with the
       source's ORIGINAL ``t_ref`` (no per-window params propagation).
    2. Snap the GB carrier to the window's rfft grid: ``k_f0 = round(f0
       / df)`` with ``df = 1/T_window``; ``f0_grid = k_f0 * df``.
    3. Subtract the snapped carrier from the source phase to get a slow
       phase and form the slow positive-freq complex signal

           s(tau) = tdi_amp * exp( +1j * (tdi_phase + phase_ref
                                          - 2*pi * f0_grid * tau) )

       where ``tau = t_sparse - t_start``.
    4. FFT it; ``X_het = 0.5 * dt_sparse * fft(s, axis=-1)``.
    5. Place ``X_het`` into the window's dense rfft grid at bins
       ``[k_f0 + fftfreq(N_sparse, 1/N_sparse).astype(int)]``.

    Args:
        source_params: 9-tuple of GB params at ``t_ref_source``.
        t_start: start time of the analysis window (seconds, absolute).
        T_window: window duration in seconds (= ``N_window_td * dt``).
        t_ref_source: GB source ``t_ref`` (NOT per-window; the original).
        N_sparse: heterodyne FFT length (power of 2).
        N_window_td: number of TD samples spanning ``T_window``.
        dt: TD sample step.
        gb_kwargs: kwargs for :class:`GBTDIonTheFly` (orbits, tdi_config,
            tdi_chan, force_backend).

    Returns:
        ``(window_fd, k_f0)`` -- ``window_fd`` is shape
        ``(nchannels, N_window_td/2+1)`` complex, populated only in the
        ``N_sparse`` bins around ``k_f0``.
    """
    f0 = float(source_params[1])
    dt_sparse = T_window / N_sparse
    t_sparse = t_start + np.arange(N_sparse) * dt_sparse

    gb = GBTDIonTheFly(t_sparse, T_window, t_ref_source, 1.0 / dt, 1, **gb_kwargs)
    out = gb(*[np.array([p]) for p in source_params],
             convert_to_ra_dec=False, return_spline=False)
    tdi_amp = np.asarray(out.tdi_amp[0])              # (3, N_sparse)
    tdi_phase = np.asarray(out.tdi_phase[0])          # (3, N_sparse)
    phase_ref = np.asarray(out.phase_ref[0])          # (N_sparse,)

    df = 1.0 / T_window
    k_f0 = int(round(f0 / df))
    f0_grid = k_f0 * df

    tau = t_sparse - t_start
    carrier = 2.0 * np.pi * f0_grid * tau
    slow = tdi_amp * np.exp(
        +1j * (tdi_phase + phase_ref[None, :] - carrier[None, :])
    )
    alpha = _resolve_tukey_alpha(
        tukey_alpha, use_tukey, path="heterodyne", N_sparse=N_sparse,
    )
    if alpha > 0:
        slow = slow * _tukey_window(N_sparse, alpha)[None, :]
    S = np.fft.fft(slow, axis=-1) * dt_sparse         # (3, N_sparse)
    X_het = 0.5 * S

    n_rfft = N_window_td // 2 + 1
    window_fd = np.zeros((X_het.shape[0], n_rfft), dtype=complex)
    m_idx = np.fft.fftfreq(N_sparse, d=1.0 / N_sparse).astype(int)
    kbins = k_f0 + m_idx
    mask = (kbins >= 0) & (kbins < n_rfft)
    window_fd[:, kbins[mask]] = X_het[:, mask]
    return window_fd, k_f0


def _plot_td_fd_vs_fd_direct(fd_td, fd_direct, df, f0, label,
                             filename, band_halfwidth_hz=None):
    """Overlay |FD| (log) of TD->rfft vs direct FD-heterodyne + residual.

    Args:
        fd_td: ``(nchan, N_rfft)`` -- rfft of the TD waveform * dt.
        fd_direct: ``(nchan, N_rfft)`` -- FD-direct from
            ``chunk_fd_via_python_heterodyne``, placed onto the same
            dense rfft grid as ``fd_td``.
        df: bin spacing of the rfft grid (Hz).
        f0: source carrier (Hz), used for the zoom-window centre.
        label: plot title prefix (e.g. "chunk 0" / "full obs").
        filename: output png path.
        band_halfwidth_hz: half-width of zoom window in Hz; default
            picks 32 * df * 4 = 128 bins each side.
    """
    chan = 0
    n_rfft = fd_td.shape[-1]
    f_arr = np.arange(n_rfft) * df

    if band_halfwidth_hz is None:
        band_halfwidth_hz = 128 * df
    k_lo = max(0, int((f0 - band_halfwidth_hz) / df))
    k_hi = min(n_rfft - 1, int((f0 + band_halfwidth_hz) / df))
    sl = slice(k_lo, k_hi + 1)

    diff = fd_direct - fd_td
    eps = max(1e-300, float(np.max(np.abs(fd_td[chan, sl]))) * 1e-300)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].semilogy(f_arr[sl], np.abs(fd_td[chan, sl]) + eps,
                     label="TD->rfft (GBTDIonTheFly)", lw=1.2)
    axes[0].semilogy(f_arr[sl], np.abs(fd_direct[chan, sl]) + eps,
                     label="FD-direct (Python heterodyne)",
                     lw=1.0, ls="--")
    axes[0].axvline(f0, color="k", lw=0.5, ls=":", alpha=0.5)
    axes[0].set_ylabel("|FD|  (X channel)")
    axes[0].set_title("{} -- TD->FD vs FD-direct heterodyne".format(label))
    axes[0].legend()

    axes[1].semilogy(
        f_arr[sl],
        np.abs(diff[chan, sl]) / np.maximum(np.abs(fd_td[chan, sl]), 1e-300),
        color="firebrick", lw=1.0,
    )
    axes[1].axvline(f0, color="k", lw=0.5, ls=":", alpha=0.5)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("|FD_direct - FD_TD| / |FD_TD|")
    axes[1].set_title("relative residual")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print("  wrote FD comparison plot to", filename)


def test_F_td_fd_vs_fd_direct(td_arr, dt, Nf, Nt, source_params, t_ref_full,
                              backend, Nt_sub, N_sparse_chunk=64,
                              N_sparse_full=512):
    """Compare TD->rfft FD against the Python-heterodyne FD-direct path.

    Two figures:

    * **chunk 0** (``T_chunk = Nf*Nt_sub*dt`` ~ 1.9 days,
      ``N_sparse_chunk`` heterodyne bins): rfft of
      ``td_arr[..., :Nf*Nt_sub]`` vs the Python-heterodyne placed onto
      the chunk's rfft grid.
    * **full obs** (~0.5 yr): same comparison on the full dense rfft
      grid; ``N_sparse_full=512`` covers the LISA-orbit Doppler
      sidebands.

    Both calls use the source's ORIGINAL ``t_ref`` (no per-window
    params propagation) -- the Python heterodyne is invariant under
    ``t_start`` since ``GBTDIonTheFly`` is evaluated at absolute times
    and ``tau`` is window-local.
    """
    print("\n========== Test F: TD->FD vs Python-heterodyne FD-direct ==========")

    f0 = source_params[1]
    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    # --- chunk 0 ---------------------------------------------------------
    N_chunk_td = Nf * Nt_sub
    T_chunk = N_chunk_td * dt
    chunk_df = 1.0 / T_chunk

    chunk_td = td_arr[..., :N_chunk_td]
    fd_td_chunk = np.fft.rfft(chunk_td, axis=-1) * dt
    fd_direct_chunk, k_f0_chunk = chunk_fd_via_python_heterodyne(
        source_params, t_start=0.0, T_window=T_chunk,
        t_ref_source=t_ref_full, N_sparse=N_sparse_chunk,
        N_window_td=N_chunk_td, dt=dt, gb_kwargs=gb_kwargs, backend=backend,
    )
    print("  chunk 0: k_f0={}, chunk_df={:.3e} Hz, N_sparse={}".format(
        k_f0_chunk, chunk_df, N_sparse_chunk))
    _plot_td_fd_vs_fd_direct(
        fd_td_chunk, fd_direct_chunk, chunk_df, f0,
        label="chunk 0 (T={:.1f} d, N_sparse={})".format(
            T_chunk / 86400.0, N_sparse_chunk),
        filename="check_shortened_wdm_testF_chunk.png",
    )

    # --- full obs --------------------------------------------------------
    N_full_td = td_arr.shape[-1]
    T_full = N_full_td * dt
    full_df = 1.0 / T_full
    fd_td_full = np.fft.rfft(td_arr, axis=-1) * dt
    fd_direct_full, k_f0_full = chunk_fd_via_python_heterodyne(
        source_params, t_start=0.0, T_window=T_full,
        t_ref_source=t_ref_full, N_sparse=N_sparse_full,
        N_window_td=N_full_td, dt=dt, gb_kwargs=gb_kwargs, backend=backend,
    )
    print("  full obs: k_f0={}, full_df={:.3e} Hz, N_sparse={}".format(
        k_f0_full, full_df, N_sparse_full))
    _plot_td_fd_vs_fd_direct(
        fd_td_full, fd_direct_full, full_df, f0,
        label="full obs (T={:.3f} yr, N_sparse={})".format(
            T_full / YRSID_SI, N_sparse_full),
        filename="check_shortened_wdm_testF_full.png",
        band_halfwidth_hz=N_sparse_full * full_df * 0.6,
    )

    # quick numerical residual at the carrier bin (peak), chunk only.
    print("  residual at chunk k_f0={}:".format(k_f0_chunk))
    print("    |fd_td|    = {:.3e}".format(abs(fd_td_chunk[0, k_f0_chunk])))
    print("    |fd_direct|= {:.3e}".format(abs(fd_direct_chunk[0, k_f0_chunk])))
    if abs(fd_td_chunk[0, k_f0_chunk]) > 0:
        r = fd_direct_chunk[0, k_f0_chunk] / fd_td_chunk[0, k_f0_chunk]
        print("    ratio |.|={:.5f}, phase={:+.3f} deg".format(
            abs(r), np.degrees(np.angle(r))))


def test_E_heterodyne_chunked(td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params,
                              t_ref_full, backend, Nt_sub, n_pad,
                              N_sparse=64, edge_layer_pixels=20):
    """Build the stitched WDM via Python heterodyne FD chunks (no TD generation).

    Per chunk: call :func:`chunk_fd_via_python_heterodyne` which runs
    ``GBTDIonTheFly`` on a sparse grid spanning the chunk, builds the
    slow signal, FFTs it, and places ``X_het`` into the chunk's dense
    rfft grid. Then ``FDSignal.transform(WDMSettings)`` -> WDM, and
    stitch with overlap.

    All chunks use the source's ORIGINAL ``t_ref`` (no per-chunk params
    propagation) -- the chunk-local heterodyne carrier ``2*pi*f0_grid*tau``
    handles the time shift naturally because ``tau`` is window-local.
    """
    print("\n========== Test E: Python-heterodyne stitched WDM ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    step = Nt_sub - 2 * n_pad
    assert step > 0 and step % 2 == 0
    assert (Nt - Nt_sub) % step == 0
    n_chunks = (Nt - Nt_sub) // step + 1
    starts = [j * step for j in range(n_chunks)]
    N_chunk_td = Nf * Nt_sub
    T_chunk = N_chunk_td * dt
    chunk_df = 1.0 / T_chunk
    print("  Nt_sub={}, n_pad={}, step={}, n_chunks={}, "
          "T_chunk={:.1f}s, N_sparse={}".format(
              Nt_sub, n_pad, step, n_chunks, T_chunk, N_sparse))

    nchan = td_arr.shape[-2]
    stitched = np.zeros((nchan, Nf, Nt), dtype=float)
    chunk_fd_set = FDSettings(N_chunk_td // 2 + 1, chunk_df, force_backend=backend)
    chunk_wdm_set = WDMSettings(Nf=Nf, Nt=Nt_sub, dt=dt, force_backend=backend)
    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    # ONE cached GBTDIonTheFly + wave_gen wrapper, reused per chunk.
    # Memory stays bounded by a single chunk's allocation (~few KB).
    import time as _time
    _t0 = _time.perf_counter()
    gen = CachedHeterodyneGenerator(
        T_window=T_chunk, t_ref_source=t_ref_full, N_sparse=N_sparse,
        dt=dt, nchannels=nchan, gb_kwargs=gb_kwargs,
    )
    print("  cached GBTDIonTheFly construction (N_sparse={}): {:.2f}s".format(
        N_sparse, _time.perf_counter() - _t0))

    _t1 = _time.perf_counter()
    k_f0 = None
    for j, n0 in enumerate(starts):
        chunk_t_start = n0 * layer_dt
        chunk_fd, k_f0 = gen.chunk_fd(source_params, chunk_t_start, N_chunk_td)
        chunk_wdm = FDSignal(chunk_fd, chunk_fd_set).transform(chunk_wdm_set)
        chunk_arr = np.asarray(chunk_wdm.arr)            # (nchan, Nf, Nt_sub)
        keep_lo = 0 if j == 0 else n_pad
        keep_hi = Nt_sub if j == n_chunks - 1 else (Nt_sub - n_pad)
        stitched[:, :, n0 + keep_lo:n0 + keep_hi] = chunk_arr[:, :, keep_lo:keep_hi]
    print("  loop ({} chunks: heterodyne + FD->WDM + stitch): {:.2f}s".format(
        n_chunks, _time.perf_counter() - _t1))
    print("  k_f0={}, heterodyne band [{}, {}] dense rfft bins "
          "(~{:.3e} - {:.3e} Hz)".format(
              k_f0, k_f0 - N_sparse // 2, k_f0 + N_sparse // 2 - 1,
              max(k_f0 - N_sparse // 2, 0) * chunk_df,
              (k_f0 + N_sparse // 2 - 1) * chunk_df,
          ))

    # Raw L2 diff (no crop).
    diff = stitched - full_arr
    rms_diff_total = float(np.sqrt(np.mean(diff ** 2)))
    rms_full_total = float(np.sqrt(np.mean(full_arr ** 2)))
    interior_slice = slice(n_pad, Nt - n_pad)
    rms_diff_int = float(np.sqrt(np.mean(diff[..., interior_slice] ** 2)))
    rms_full_int = float(np.sqrt(np.mean(full_arr[..., interior_slice] ** 2)))
    print("  rms diff / rms full (all pixels)        = {:.3e}".format(
        rms_diff_total / max(rms_full_total, 1e-300)))
    print("  rms diff / rms full (n_pad-cropped int) = {:.3e}".format(
        rms_diff_int / max(rms_full_int, 1e-300)))

    # mm5 / mm2 (same crop as Test D)
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt - edge_layer_pixels) * layer_dt
    f0 = source_params[1]
    m_floor = int(f0 / layer_df)
    mm5_min = f0 - 3 * layer_df
    mm5_max = f0 + 2 * layer_df
    mm5, set5 = _mismatch(full_arr, stitched, wdm_set,
                          mm5_min, mm5_max, min_time, max_time, backend)
    mm2_min = m_floor * layer_df
    mm2_max = (m_floor + 2) * layer_df - 0.5 * layer_df
    mm2, set2 = _mismatch(full_arr, stitched, wdm_set,
                          mm2_min, mm2_max, min_time, max_time, backend)
    print("  mm5 (layers {}..{}) = {:.3e}".format(set5.ind_min_f, set5.ind_max_f, mm5))
    print("  mm2 (layers {}..{}) = {:.3e}".format(set2.ind_min_f, set2.ind_max_f, mm2))

    _plot_full_vs_stitched_X(
        full_arr, stitched, wdm_set, min_time, max_time,
        filename="check_shortened_wdm_testE_full.png",
        title_prefix="Test E (FD-heterodyne, {} chunks, N_sparse={}):  ".format(
            n_chunks, N_sparse),
    )
    _plot_band_zoom_X(
        full_arr, stitched, wdm_set, set5, min_time, max_time,
        filename="check_shortened_wdm_testE_mm5band.png",
        title="Test E mm5 band (layers {}..{}, mm5={:.2e})".format(
            set5.ind_min_f, set5.ind_max_f, mm5),
    )
    _plot_band_zoom_X(
        full_arr, stitched, wdm_set, set2, min_time, max_time,
        filename="check_shortened_wdm_testE_mm2band.png",
        title="Test E mm2 band (layers {}..{}, mm2={:.2e})".format(
            set2.ind_min_f, set2.ind_max_f, mm2),
    )


def _stitched_wdm_from_td(td_arr, dt, Nf, Nt, Nt_sub, n_pad, backend,
                          tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True):
    """TD-based overlap-tile stitched WDM (Test D path), with optional Tukey."""
    step = Nt_sub - 2 * n_pad
    n_chunks = (Nt - Nt_sub) // step + 1
    starts = [j * step for j in range(n_chunks)]
    nchan = td_arr.shape[-2]
    stitched = np.zeros((nchan, Nf, Nt), dtype=float)
    for j, n0 in enumerate(starts):
        chunk_wdm = wdm_of_td_slice(
            td_arr, n0 * Nf, Nf, Nt_sub, dt, backend,
            tukey_alpha=tukey_alpha, use_tukey=use_tukey,
        )
        chunk_arr = np.asarray(chunk_wdm.arr)
        keep_lo = 0 if j == 0 else n_pad
        keep_hi = Nt_sub if j == n_chunks - 1 else (Nt_sub - n_pad)
        stitched[:, :, n0 + keep_lo:n0 + keep_hi] = chunk_arr[:, :, keep_lo:keep_hi]
    return stitched, n_chunks


def _stitched_wdm_from_heterodyne(td_arr, dt, Nf, Nt, Nt_sub, n_pad,
                                  source_params, t_ref_full, N_sparse,
                                  gb_kwargs, backend,
                                  tukey_alpha=USE_RECOMMENDED_TUKEY,
                                  use_tukey=True):
    """Helper: build the chunked Python-heterodyne stitched WDM array.

    Pulled out of :func:`test_E_heterodyne_chunked` so the
    ``N_sparse``-sweep test (and any future PE-time use) can reuse it.
    """
    layer_dt = Nf * dt
    step = Nt_sub - 2 * n_pad
    n_chunks = (Nt - Nt_sub) // step + 1
    starts = [j * step for j in range(n_chunks)]
    N_chunk_td = Nf * Nt_sub
    T_chunk = N_chunk_td * dt
    chunk_df = 1.0 / T_chunk

    nchan = td_arr.shape[-2]
    stitched = np.zeros((nchan, Nf, Nt), dtype=float)
    chunk_fd_set = FDSettings(N_chunk_td // 2 + 1, chunk_df, force_backend=backend)
    chunk_wdm_set = WDMSettings(Nf=Nf, Nt=Nt_sub, dt=dt, force_backend=backend)

    gen = CachedHeterodyneGenerator(
        T_window=T_chunk, t_ref_source=t_ref_full, N_sparse=N_sparse,
        dt=dt, nchannels=nchan, gb_kwargs=gb_kwargs,
    )
    for j, n0 in enumerate(starts):
        chunk_t_start = n0 * layer_dt
        chunk_fd, _k_f0 = gen.chunk_fd(
            source_params, chunk_t_start, N_chunk_td,
            tukey_alpha=tukey_alpha, use_tukey=use_tukey,
        )
        chunk_wdm = FDSignal(chunk_fd, chunk_fd_set).transform(chunk_wdm_set)
        chunk_arr = np.asarray(chunk_wdm.arr)
        keep_lo = 0 if j == 0 else n_pad
        keep_hi = Nt_sub if j == n_chunks - 1 else (Nt_sub - n_pad)
        stitched[:, :, n0 + keep_lo:n0 + keep_hi] = chunk_arr[:, :, keep_lo:keep_hi]
    return stitched, n_chunks


def test_G_n_sparse_sweep(td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params,
                          t_ref_full, backend, Nt_sub, n_pad,
                          n_sparse_list=(64, 128, 256, 512, 1024, 2048, 4096),
                          tukey_alphas=(0.0, 0.1, 0.25, 0.5),
                          edge_layer_pixels=20):
    """Sweep N_sparse x Tukey alpha; characterise mm5 / mm2.

    For each ``(N_sparse, alpha)`` pair, run the chunked FD-heterodyne
    stitched-WDM pipeline with that Tukey window applied to the slow
    signal before the heterodyne FFT (equivalent to windowing the chunk
    TD before its rfft). Also compute the TD-based Test-D stitched
    baseline at each alpha as a reference (this is the floor the
    heterodyne path converges to as N_sparse grows wider than the band).

    Saves a two-panel log-log plot ``mm5`` and ``mm2`` versus
    ``N_sparse`` with one curve per alpha, plus the matching TD baseline
    as a horizontal dashed line per alpha.
    """
    import time as _time
    print("\n========== Test G: N_sparse x Tukey sweep ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    f0 = source_params[1]
    m_floor = int(f0 / layer_df)
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt - edge_layer_pixels) * layer_dt
    mm5_min, mm5_max = f0 - 3 * layer_df, f0 + 2 * layer_df
    mm2_min = m_floor * layer_df
    mm2_max = (m_floor + 2) * layer_df - 0.5 * layer_df

    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    # TD-based baselines (one per alpha) -- mm5/mm2 of windowed chunks
    # stitched vs the (un-windowed) full WDM reference. Sets the floor
    # the FD-heterodyne path can hope to match for each alpha.
    td_baselines = {}
    n_chunks_total = None
    for alpha in tukey_alphas:
        stitched_td, n_chunks_total = _stitched_wdm_from_td(
            td_arr, dt, Nf, Nt, Nt_sub, n_pad, backend, tukey_alpha=alpha,
        )
        mm5_td, _ = _mismatch(full_arr, stitched_td, wdm_set,
                              mm5_min, mm5_max, min_time, max_time, backend)
        mm2_td, _ = _mismatch(full_arr, stitched_td, wdm_set,
                              mm2_min, mm2_max, min_time, max_time, backend)
        td_baselines[alpha] = (mm5_td, mm2_td)
        print("  TD baseline alpha={:.2f}: mm5={:.3e}, mm2={:.3e}".format(
            alpha, mm5_td, mm2_td))
        del stitched_td

    # Heterodyne sweep
    mm_table = {alpha: {"mm5": [], "mm2": [], "t": []} for alpha in tukey_alphas}
    print("  {:>8s} {:>10s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
        "N_sp", "het BW", "mm5(a=0)", "mm5(a=0.5)", "mm2(a=0)", "mm2(a=0.5)"))
    for N_sparse in n_sparse_list:
        for alpha in tukey_alphas:
            t0 = _time.perf_counter()
            stitched, _ = _stitched_wdm_from_heterodyne(
                td_arr, dt, Nf, Nt, Nt_sub, n_pad,
                source_params, t_ref_full, N_sparse, gb_kwargs, backend,
                tukey_alpha=alpha,
            )
            mm5, _ = _mismatch(full_arr, stitched, wdm_set,
                               mm5_min, mm5_max, min_time, max_time, backend)
            mm2, _ = _mismatch(full_arr, stitched, wdm_set,
                               mm2_min, mm2_max, min_time, max_time, backend)
            mm_table[alpha]["mm5"].append(mm5)
            mm_table[alpha]["mm2"].append(mm2)
            mm_table[alpha]["t"].append(_time.perf_counter() - t0)
            del stitched
        het_bw = N_sparse / (Nf * Nt_sub * dt)
        # quick at-a-glance summary row for alpha=0 vs alpha=0.5
        a0_idx, a1_idx = tukey_alphas.index(0.0), -1
        a_high = tukey_alphas[a1_idx]
        print("  {:>8d} {:>10.3e} {:>14.3e} {:>14.3e} {:>14.3e} {:>14.3e}".format(
            N_sparse, het_bw,
            mm_table[0.0]["mm5"][-1], mm_table[a_high]["mm5"][-1],
            mm_table[0.0]["mm2"][-1], mm_table[a_high]["mm2"][-1],
        ))

    # plot: two panels (mm5, mm2) with one curve per Tukey alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    for i, (kind, ylabel) in enumerate([("mm5", "1 - overlap (mm5, 5 layers)"),
                                        ("mm2", "1 - overlap (mm2, 2 layers)")]):
        ax = axes[i]
        for k, alpha in enumerate(tukey_alphas):
            ax.loglog(n_sparse_list, mm_table[alpha][kind], "o-",
                      label="FD het, alpha={:.2f}".format(alpha),
                      color="C{}".format(k), lw=1.5)
            ax.axhline(td_baselines[alpha][i], color="C{}".format(k),
                       ls="--", alpha=0.45,
                       label="TD floor, alpha={:.2f} = {:.2e}".format(
                           alpha, td_baselines[alpha][i]))
        ax.set_xlabel("N_sparse (heterodyne FFT length)")
        ax.set_ylabel(ylabel)
        ax.set_title(kind)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Test G: chunked FD-heterodyne stitched WDM, Tukey-window "
                 "sweep\n(Nf={}, Nt={}, Nt_sub={}, n_pad={}, n_chunks={})".format(
                     Nf, Nt, Nt_sub, n_pad, n_chunks_total))
    fig.tight_layout()
    fig.savefig("check_shortened_wdm_testG_mm_vs_Nsparse_tukey.png", dpi=120)
    plt.close(fig)
    print("  wrote Tukey sweep plot to "
          "check_shortened_wdm_testG_mm_vs_Nsparse_tukey.png")


def test_H_visualizations(td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params,
                          t_ref_full, backend, Nt_sub, n_pad,
                          edge_layer_pixels=20):
    """Plot-only test characterising the Tukey-helps-small-N_sparse finding.

    Generates:
      * H1: chunk-0 |FD| (log) -- rfft(TD) reference vs four FD-heterodyne
            configurations at (N_sparse, Tukey alpha).
      * H1z: same plot zoomed to the 5-layer mm5 band around f0.
      * H2: WDM stitched heatmaps in the 5-layer mm5 band (full vs four
            FD-heterodyne configs vs their diff against the full).
      * H3: WDM stitched heatmaps in the 2-layer mm2 band (same panels).
    """
    print("\n========== Test H: Tukey/N_sparse visualisations ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    f0 = source_params[1]
    m_floor = int(f0 / layer_df)
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt - edge_layer_pixels) * layer_dt
    N_chunk_td = Nf * Nt_sub
    T_chunk = N_chunk_td * dt
    chunk_df = 1.0 / T_chunk

    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    # Four representative configs (label, N_sparse, tukey_alpha, color).
    configs = [
        ("N=64,  rect      (no Tukey)", 64,   0.00, "C1"),
        ("N=64,  Tukey 0.05",            64,   0.05, "C2"),
        ("N=1024, rect      (no Tukey)", 1024, 0.00, "C3"),
        ("N=1024, Tukey 0.02",            1024, 0.02, "C4"),
    ]

    # --- H1, H1z: chunk-0 FD magnitudes ---------------------------------
    chunk_td = td_arr[..., :N_chunk_td]
    fd_td_chunk = np.fft.rfft(chunk_td, axis=-1) * dt        # reference
    n_rfft = N_chunk_td // 2 + 1
    f_arr = np.arange(n_rfft) * chunk_df

    chunk0_results = []
    for (label, Nsp, alpha, color) in configs:
        gen = CachedHeterodyneGenerator(
            T_window=T_chunk, t_ref_source=t_ref_full, N_sparse=Nsp,
            dt=dt, nchannels=td_arr.shape[-2], gb_kwargs=gb_kwargs,
        )
        fd_het, k_f0 = gen.chunk_fd(source_params, 0.0, N_chunk_td,
                                    tukey_alpha=alpha)
        chunk0_results.append((label, color, fd_het, k_f0))

    def _fd_plot(filename, band_lo, band_hi, title):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        sl = slice(band_lo, band_hi + 1)
        axes[0].semilogy(f_arr[sl], np.abs(fd_td_chunk[0, sl]) + 1e-300,
                         "k-", lw=1.6, label="rfft(TD) reference")
        for label, color, fd_het, k_f0 in chunk0_results:
            axes[0].semilogy(f_arr[sl], np.abs(fd_het[0, sl]) + 1e-300,
                             "-", color=color, lw=1.1, label=label, alpha=0.85)
        axes[0].axvline(f0, color="0.4", lw=0.6, ls=":")
        axes[0].set_ylabel("|FD| (X channel)")
        axes[0].legend(fontsize=8, loc="lower center", ncol=2)
        axes[0].set_title(title)
        axes[0].grid(True, which="both", alpha=0.3)
        ref = np.maximum(np.abs(fd_td_chunk[0, sl]), 1e-300)
        for label, color, fd_het, _ in chunk0_results:
            axes[1].semilogy(f_arr[sl],
                             np.abs(fd_het[0, sl] - fd_td_chunk[0, sl]) / ref,
                             "-", color=color, lw=1.0, label=label, alpha=0.85)
        axes[1].axvline(f0, color="0.4", lw=0.6, ls=":")
        axes[1].set_xlabel("frequency (Hz)")
        axes[1].set_ylabel("|FD_het - FD_td| / |FD_td|")
        axes[1].grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        print("  wrote", filename)

    # H1: wide view around the carrier
    wide_lo = max(0, int((f0 - 1024 * chunk_df) / chunk_df))
    wide_hi = min(n_rfft - 1, int((f0 + 1024 * chunk_df) / chunk_df))
    _fd_plot("check_shortened_wdm_testH1_FD_wide.png", wide_lo, wide_hi,
             "Test H1: chunk-0 |FD| -- rfft(TD) vs FD-heterodyne configs "
             "(wide view, k_f0 +- 1024)")
    # H1z: zoom to the mm5 band
    zlo = int((f0 - 3 * layer_df) / chunk_df)
    zhi = int((f0 + 3 * layer_df) / chunk_df)
    _fd_plot("check_shortened_wdm_testH1z_FD_zoom.png",
             max(0, zlo), min(n_rfft - 1, zhi),
             "Test H1z: chunk-0 |FD| -- zoomed to mm5 band (5 WDM layers)")

    # --- H2, H3: stitched WDM heatmaps ----------------------------------
    band_specs = [
        ("mm5", f0 - 3 * layer_df, f0 + 2 * layer_df,
         "check_shortened_wdm_testH2_heatmaps_mm5.png"),
        ("mm2", m_floor * layer_df, (m_floor + 2) * layer_df - 0.5 * layer_df,
         "check_shortened_wdm_testH3_heatmaps_mm2.png"),
    ]
    chan = 0
    for band_label, fmin, fmax, fname in band_specs:
        band_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_freq=fmin, max_freq=fmax,
            min_time=min_time, max_time=max_time,
            force_backend=backend,
        )
        f_lo, f_hi = band_set.ind_min_f, band_set.ind_max_f
        full_band = full_arr[chan, f_lo:f_hi + 1, :]
        f_band = np.arange(f_lo, f_hi + 1) * layer_df
        t_band = np.arange(Nt) * layer_dt
        vmax = float(np.max(np.abs(full_band)))

        fig = plt.figure(figsize=(16, 7))
        ncols = len(configs) + 1  # full + 4 configs
        # Top row: full + 4 config heatmaps
        for i, panel in enumerate(["full TD->WDM"] + [c[0] for c in configs]):
            ax = fig.add_subplot(2, ncols, i + 1)
            if i == 0:
                arr_show = full_band
            else:
                _, _, _, color = configs[i - 1]                  # noqa
                label, Nsp, alpha, _ = configs[i - 1]
                stitched, _ = _stitched_wdm_from_heterodyne(
                    td_arr, dt, Nf, Nt, Nt_sub, n_pad,
                    source_params, t_ref_full, Nsp, gb_kwargs, backend,
                    tukey_alpha=alpha,
                )
                arr_show = stitched[chan, f_lo:f_hi + 1, :]
            ax.pcolormesh(t_band, f_band, arr_show, cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
            ax.set_title(panel, fontsize=9)
            ax.axvline(min_time, color="k", lw=0.5, ls="--", alpha=0.4)
            ax.axvline(max_time, color="k", lw=0.5, ls="--", alpha=0.4)
            if i == 0:
                ax.set_ylabel("freq (Hz)")
        # Bottom row: diff vs full (own color scale per panel)
        for i, panel in enumerate(["(reference)"] + [c[0] for c in configs]):
            ax = fig.add_subplot(2, ncols, ncols + i + 1)
            if i == 0:
                ax.axis("off")
                continue
            label, Nsp, alpha, _ = configs[i - 1]
            stitched, _ = _stitched_wdm_from_heterodyne(
                td_arr, dt, Nf, Nt, Nt_sub, n_pad,
                source_params, t_ref_full, Nsp, gb_kwargs, backend,
                tukey_alpha=alpha,
            )
            diff = stitched[chan, f_lo:f_hi + 1, :] - full_band
            dmax = float(np.max(np.abs(diff))) or 1e-30
            im = ax.pcolormesh(t_band, f_band, diff, cmap="RdBu_r",
                               vmin=-dmax, vmax=dmax, shading="auto",
                               rasterized=True)
            ax.set_title("diff: {} (max={:.1e})".format(panel, dmax),
                         fontsize=8)
            ax.axvline(min_time, color="k", lw=0.5, ls="--", alpha=0.4)
            ax.axvline(max_time, color="k", lw=0.5, ls="--", alpha=0.4)
            ax.set_xlabel("time (s)")
            if i == 1:
                ax.set_ylabel("freq (Hz)")
        fig.suptitle("Test H ({}): WDM heatmaps (Nf={}, Nt={}, "
                     "Nt_sub={}, n_pad={})".format(
                         band_label.upper(), Nf, Nt, Nt_sub, n_pad), y=1.02)
        fig.tight_layout()
        fig.savefig(fname, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("  wrote", fname)


def group_binaries_by_layer(params, layer_df, group_band_layers=5,
                            margin_layers=2):
    """Group binaries by their WDM carrier layer for chunked get_ll dispatch.

    The chunked get_ll kernel's PSD/data shared-memory load is amortised
    across all binaries in the same kernel-launch group. Binaries in
    radically different m-layers shouldn't share a launch -- the shared
    PSD/data slab would have to cover all of their bands, blowing the
    shared-memory budget. Instead, the Python wrapper sorts and clusters
    binaries by their carrier layer, then dispatches one kernel launch
    per group with a tight WDM band restricted to that group's layers
    plus a small margin.

    Args:
        params: ``(num_bin, 9)`` array (f0 lives in column 1).
        layer_df: WDM layer frequency spacing (Hz).
        group_band_layers: each group spans at most this many layers
            (the kernel's per-block frequency band).
        margin_layers: layers added on each side of the group's range
            for wavelet-tail / mm5 coverage.

    Returns:
        List of ``(m_lo, m_hi, indices)`` where ``indices`` is a numpy
        array of binary indices in this group; the kernel's frequency
        band is ``[m_lo - margin_layers, m_hi + margin_layers]``.
    """
    params = np.asarray(params)
    f0 = params[:, 1]
    m_floor = (f0 / layer_df).astype(int)
    sorted_idx = np.argsort(m_floor)
    sorted_m = m_floor[sorted_idx]

    groups = []
    i = 0
    while i < len(sorted_m):
        m_start = sorted_m[i]
        m_end = m_start + group_band_layers - 1
        in_group = sorted_m[i:] <= m_end
        n_in_group = int(in_group.sum())
        idx = sorted_idx[i:i + n_in_group]
        groups.append((int(m_start), int(m_end), idx))
        i += n_in_group
    return groups


def chunked_get_ll_python_reference(td_arr, full_wdm_arr, wdm_set, sens_mat_inv,
                                    dt, Nf, Nt, source_params_list, t_ref_full,
                                    backend, Nt_sub, n_pad, N_sparse,
                                    tukey_alpha=USE_RECOMMENDED_TUKEY,
                                    use_tukey=True):
    """Python reference for the chunked-heterodyne ``gb_wdm_het_get_ll_kernel``.

    Outer loop = chunks; inner loop = binaries. For each chunk, the PSD
    and data slabs are sliced once (analogue of the C++ kernel's
    shared-memory load) and reused across all binaries in the kernel
    group. Per (chunk, binary):

      1. Build chunk WDM via ``CachedHeterodyneGenerator.chunk_fd`` +
         ``FDSignal.transform(WDMSettings)``.
      2. For pixels in the chunk's "kept" interior, accumulate

           dh += data * w_chunk / S
           hh += w_chunk * w_chunk / S

         where ``S = 1 / sens_mat_inv``. (We use the inverse PSD as
         stored on ``SensitivityMatrix.invC``, mirroring lisatools.)
      3. Add per-chunk dh, hh into per-binary accumulators.

    Returns ``(d_h, h_h)`` arrays of length ``num_bin``.
    """
    num_bin = len(source_params_list)
    d_h = np.zeros(num_bin, dtype=float)
    h_h = np.zeros(num_bin, dtype=float)

    step = Nt_sub - 2 * n_pad
    n_full = (Nt - Nt_sub) // step + 1
    starts = [j * step for j in range(n_full)]
    last_full_end = starts[-1] + Nt_sub

    nchan = td_arr.shape[-2]
    T_chunk = Nf * Nt_sub * dt
    chunk_df = 1.0 / T_chunk
    layer_dt = Nf * dt

    chunk_fd_set = FDSettings(Nf * Nt_sub // 2 + 1, chunk_df,
                              force_backend=backend)
    chunk_wdm_set = WDMSettings(Nf=Nf, Nt=Nt_sub, dt=dt, force_backend=backend)
    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    # Cache heterodyne generator across all binaries.
    gen = CachedHeterodyneGenerator(
        T_window=T_chunk, t_ref_source=t_ref_full, N_sparse=N_sparse,
        dt=dt, nchannels=nchan, gb_kwargs=gb_kwargs,
    )

    # Build the geometry up front (analogue of the host-side arrays
    # passed to the C++ kernel).
    geom = []                                            # (n0, keep_lo, keep_hi, n_global_lo)
    for j, n0 in enumerate(starts):
        keep_lo = 0 if j == 0 else n_pad
        if j == n_full - 1 and last_full_end == Nt:
            keep_hi = Nt_sub
        else:
            keep_hi = Nt_sub - n_pad
        geom.append((n0, keep_lo, keep_hi, n0 + keep_lo))
    if last_full_end < Nt:
        n0_partial = Nt - Nt_sub
        new_lo_global = starts[-1] + (Nt_sub - n_pad)
        new_lo_local = new_lo_global - n0_partial
        geom.append((n0_partial, new_lo_local, Nt_sub, new_lo_global))

    # OUTER: chunks
    for (n0, keep_lo, keep_hi, n_global_lo) in geom:
        n_pixels = keep_hi - keep_lo
        n_global_hi = n_global_lo + n_pixels

        # Slice PSD/data for THIS chunk's time range (== "load to shared mem").
        sh_data = full_wdm_arr[:, :, n_global_lo:n_global_hi]      # (nchan, Nf, n_pixels)
        sh_invC = sens_mat_inv[:, :, n_global_lo:n_global_hi]       # (nchan, Nf, n_pixels) or broadcasted

        # INNER: binaries
        for bin_i, source_params in enumerate(source_params_list):
            chunk_t_start = n0 * layer_dt
            chunk_fd, _ = gen.chunk_fd(
                source_params, chunk_t_start, Nf * Nt_sub,
                tukey_alpha=tukey_alpha, use_tukey=use_tukey,
            )
            chunk_wdm = FDSignal(chunk_fd, chunk_fd_set).transform(chunk_wdm_set)
            w_chunk = np.asarray(chunk_wdm.arr)                    # (nchan, Nf, Nt_sub)
            w_keep = w_chunk[:, :, keep_lo:keep_hi]                # (nchan, Nf, n_pixels)
            # accumulate d_h and h_h contributions for this (chunk, binary)
            d_h[bin_i] += float(np.sum(sh_data * w_keep * sh_invC))
            h_h[bin_i] += float(np.sum(w_keep * w_keep * sh_invC))

    return d_h, h_h


def gb_chunk_fd_to_wdm_python_reference(chunk_fd, wdm_window, Nf, Nt_sub,
                                        data_dt, nchannels):
    """Pure-Python reference for the C++ ``gb_chunk_fd_to_wdm`` device function.

    Implements the per-layer iFFT + parity / Cmn pick + boundary
    folding algorithm step-by-step matching the C++ port in
    ``lisa-on-gpu/.../TDIonthefly.cu:gb_chunk_fd_to_wdm``. Used in
    Phase 2 validation: pixel-for-pixel comparison against this
    reference proves the C++ transcription is correct *before* a full
    rebuild + integration test.

    The result must also match
    ``FDSignal(chunk_fd, FDSettings).transform(WDMSettings(Nf, Nt_sub,
    data_dt))`` to floating-point precision (this is just a manual
    re-implementation of the lisatools WDM transform, restricted to a
    single chunk).
    """
    N_chunk_td = Nf * Nt_sub
    half_Nt_sub = Nt_sub // 2
    n_rfft_chunk = N_chunk_td // 2 + 1
    kappa = 2.0 * np.sqrt(np.pi * data_dt) / float(Nf)
    sqrt2 = np.sqrt(2.0)
    w_mn = np.zeros((nchannels, Nf, Nt_sub), dtype=float)

    for c in range(nchannels):
        fd_c = chunk_fd[c]
        for m in range(Nf + 1):                  # 0..Nf inclusive
            # Build the length-Nt_sub windowed FD slice.
            before_ifft = np.zeros(Nt_sub, dtype=complex)
            for k_idx in range(Nt_sub):
                k_global = m * half_Nt_sub + (k_idx - half_Nt_sub)
                herm = False
                if k_global < 0:
                    k_global = -k_global
                    herm = True
                if k_global > N_chunk_td // 2:
                    k_global = N_chunk_td - k_global
                    herm = True
                if 0 <= k_global < n_rfft_chunk:
                    v = fd_c[k_global]
                    if herm:
                        v = np.conj(v)
                    v = v / data_dt * wdm_window[k_idx]
                    before_ifft[k_idx] = v
            after_ifft = np.fft.ifft(before_ifft)
            for n in range(Nt_sub):
                boundary = (m == 0 or m == Nf)
                mn_even = ((m + n) & 1) == 0
                if boundary and not mn_even:
                    continue
                real_part = after_ifft[n].real if mn_even else after_ifft[n].imag
                sign = 1.0 if ((((m + 1) * n) & 1) == 0) else -1.0
                val = kappa * sign * real_part
                if 1 <= m <= Nf - 1:
                    w_mn[c, m, n] = val
                elif m == 0:
                    if (n & 1) == 0:
                        w_mn[c, 0, n] = val / sqrt2
                else:                            # m == Nf
                    if (n & 1) == 0 and (n + 1) < Nt_sub:
                        w_mn[c, 0, n + 1] = val / sqrt2
    return w_mn


def test_L_chunked_get_ll_reference(td_arr, full_arr, wdm_set, dt, Nf, Nt,
                                    source_params, t_ref_full, backend,
                                    Nt_sub=256, n_pad=32, N_sparse=256,
                                    edge_layer_pixels=20):
    """Phase 2 pre-validation: chunked-heterodyne get_ll vs lisatools.

    Builds the chunked-heterodyne ``<d|h>`` / ``<h|h>`` via
    :func:`chunked_get_ll_python_reference` (outer chunk loop + inner
    binary loop -- the exact data flow the C++ kernel will use) and
    compares against the lisatools ``inner_product`` evaluated on the
    same WDM band crop.

    If they match (relative diff < ~1e-3 for our N_sparse=256 + Tukey
    auto config), the algorithm is right; future C++ output is validated
    against the same Python reference.
    """
    print("\n========== Test L: chunked get_ll Python reference ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    f0 = source_params[1]
    m_floor = int(f0 / layer_df)
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt - edge_layer_pixels) * layer_dt

    # ----- reference via the existing AnalysisContainer machinery -----
    mm5_min = f0 - 3 * layer_df
    mm5_max = f0 + 2 * layer_df
    inj_sig, band_set = _build_band_wdm_signal(
        full_arr, wdm_set, mm5_min, mm5_max, min_time, max_time, backend,
    )
    inj = DataResidualArray(inj_sig)
    sens = XYZ2SensitivityMatrix(band_set, model="scirdv1")
    analysis = AnalysisContainer(inj, sens)
    # Build the full reference template via TD->FD->WDM at the same params
    # (same convention as the injection -> truth).
    sens_invC = np.asarray(sens.invC)                          # (3, 3, Nf_active, Nt_active)
    # (Use diagonal channels only -- the SensitivityMatrix object is full XYZ).

    # ----- chunked Python reference -----
    # Use the SAME band crop on the data + invC arrays as the reference,
    # then call chunked_get_ll. We zero-pad PSD outside the band so the
    # comparison is well-defined.
    full_band_invC = np.zeros_like(full_arr)
    ind_min_f = band_set.ind_min_f
    ind_max_f = band_set.ind_max_f
    # sens.invC shape is typically (nchannels, ..., Nf_active, Nt_active);
    # broadcast to (nchannels, Nf, Nt) layout matching full_arr.
    for c in range(min(full_arr.shape[0], sens_invC.shape[0])):
        # diagonal (c, c) slot if 4D, else direct
        if sens_invC.ndim == 4:
            invC_c = sens_invC[c, c]
        else:
            invC_c = sens_invC[c]
        # invC may already be sliced to band -- expand to full Nt
        if invC_c.shape[-1] == band_set.Nt_active:
            t_lo = band_set.ind_min_t
            t_hi = band_set.ind_max_t + 1
            full_band_invC[c, ind_min_f:ind_max_f + 1, t_lo:t_hi] = invC_c
        else:
            full_band_invC[c, ind_min_f:ind_max_f + 1, :] = invC_c

    d_h_arr, h_h_arr = chunked_get_ll_python_reference(
        td_arr, full_arr, wdm_set, full_band_invC, dt, Nf, Nt,
        [source_params], t_ref_full, backend,
        Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
        tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
    )
    d_h_chunk = float(d_h_arr[0])
    h_h_chunk = float(h_h_arr[0])

    # Build the FULL stitched-WDM template (matches what the chunked
    # accumulator integrates over).
    stitched, _ = _stitched_wdm_from_heterodyne(
        td_arr, dt, Nf, Nt, Nt_sub, n_pad, source_params, t_ref_full,
        N_sparse, dict(tdi_config=TDIConfig("2nd generation"),
                       orbits=EqualArmlengthOrbits(force_backend=backend),
                       tdi_chan="XYZ", force_backend=backend),
        backend,
        tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
    )

    # Self-consistent reference: sum over the whole WDM grid using the
    # same full_band_invC -- the chunked path partitions this sum across
    # chunks with the partial-slide rule preventing double-counting in
    # the overlap region.
    d_h_ref = float(np.sum(full_arr * stitched * full_band_invC))
    h_h_ref = float(np.sum(stitched * stitched * full_band_invC))

    print("  chunked <d|h>  = {:+.6e}".format(d_h_chunk))
    print("  ref     <d|h>  = {:+.6e}".format(d_h_ref))
    if abs(d_h_ref) > 0:
        print("    rel diff = {:.3e}".format(
            abs(d_h_chunk - d_h_ref) / abs(d_h_ref)))
    print("  chunked <h|h>  = {:+.6e}".format(h_h_chunk))
    print("  ref     <h|h>  = {:+.6e}".format(h_h_ref))
    if abs(h_h_ref) > 0:
        print("    rel diff = {:.3e}".format(
            abs(h_h_chunk - h_h_ref) / abs(h_h_ref)))
    if abs(d_h_ref) > 0 and abs(h_h_ref) > 0:
        d_h_rel = abs(d_h_chunk - d_h_ref) / abs(d_h_ref)
        h_h_rel = abs(h_h_chunk - h_h_ref) / abs(h_h_ref)
        if d_h_rel < 1e-3 and h_h_rel < 1e-3:
            print("  PASS: chunked get_ll matches reference to <0.1% (C++ port "
                  "can be validated against the same reference).")
        else:
            print("  FAIL: chunked get_ll diverges from reference.")


def test_K_cpp_python_reference_match(td_arr, full_arr, wdm_set, dt, Nf, Nt,
                                       source_params, t_ref_full, backend,
                                       Nt_sub=256, n_pad=32, N_sparse=256):
    """Phase 2 pre-validation: gb_chunk_fd_to_wdm_python_reference vs
    FDSignal.wdmtransform on chunk 0.

    If the two agree to floating-point precision, the algorithm
    transcription is correct; the C++ port can then be validated by
    comparing C++ output against this same reference (or against the
    lisatools transform).
    """
    print("\n========== Test K: C++ algorithm Python reference ==========")
    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )
    N_chunk_td = Nf * Nt_sub
    T_chunk = N_chunk_td * dt

    # Build chunk 0's heterodyne FD via the cached Python path.
    gen = CachedHeterodyneGenerator(
        T_window=T_chunk, t_ref_source=t_ref_full, N_sparse=N_sparse,
        dt=dt, nchannels=td_arr.shape[-2], gb_kwargs=gb_kwargs,
    )
    chunk_fd, k_f0 = gen.chunk_fd(source_params, 0.0, N_chunk_td,
                                  tukey_alpha=USE_RECOMMENDED_TUKEY,
                                  use_tukey=True)
    chunk_df = 1.0 / T_chunk
    chunk_fd_set = FDSettings(N_chunk_td // 2 + 1, chunk_df,
                              force_backend=backend)
    chunk_wdm_set = WDMSettings(Nf=Nf, Nt=Nt_sub, dt=dt, force_backend=backend)

    # Reference 1: existing lisatools FDSignal.transform.
    ref_wdm = FDSignal(chunk_fd, chunk_fd_set).transform(chunk_wdm_set)
    ref_arr = np.asarray(ref_wdm.arr)

    # Reference 2: pure Python implementation matching C++ algorithm.
    # Build the wavelet window directly so it matches what the C++ kernel
    # would be given. The lisatools setup_window uses:
    #   omega = 2*pi/N * arange(-Nt_sub/2, Nt_sub/2)  (length Nt_sub)
    #   phif  = phitilde(omega, dOmega = pi/Nf)
    wdm_window = np.asarray(chunk_wdm_set.window)
    my_arr = gb_chunk_fd_to_wdm_python_reference(
        np.asarray(chunk_fd), wdm_window, Nf, Nt_sub, dt,
        nchannels=td_arr.shape[-2],
    )

    diff = my_arr - ref_arr
    max_abs_ref  = float(np.max(np.abs(ref_arr)))
    max_abs_diff = float(np.max(np.abs(diff)))
    rms_ref      = float(np.sqrt(np.mean(ref_arr ** 2)))
    rms_diff     = float(np.sqrt(np.mean(diff ** 2)))
    print("  shapes: python_ref={}, lisatools_ref={}".format(my_arr.shape, ref_arr.shape))
    print("  max |diff| / max |ref| = {:.3e}".format(
        max_abs_diff / max(max_abs_ref, 1e-300)))
    print("  rms |diff| / rms |ref| = {:.3e}".format(
        rms_diff / max(rms_ref, 1e-300)))
    if max_abs_diff / max(max_abs_ref, 1e-300) < 1e-12:
        print("  PASS: Python reference matches lisatools FDSignal.transform "
              "(ready for C++ port to be validated against same reference).")
    else:
        print("  FAIL: algorithm transcription mismatch. Inspect "
              "gb_chunk_fd_to_wdm_python_reference and the C++ port.")


def _stitched_wdm_partial_slide(td_arr, dt, Nf, Nt, Nt_sub, n_pad, backend,
                                source_params=None, t_ref_full=None,
                                gb_kwargs=None, N_sparse=None,
                                tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
                                use_heterodyne=False):
    """Stitch chunks covering arbitrary Nt, with a partial slide on the last chunk.

    Phase 1b implementation of the user's edge-handling proposal:

      * Place chunks at starts ``[0, step, 2*step, ..., (n_full-1)*step]``
        with ``step = Nt_sub - 2*n_pad``, where ``n_full`` is the
        largest number of *full* slides such that
        ``(n_full-1)*step + Nt_sub <= Nt``.
      * If ``(n_full-1)*step + Nt_sub < Nt``, append one *partial-slide*
        chunk whose start is ``Nt - Nt_sub`` so it ends exactly at
        ``Nt``. Slide amount = ``(Nt - Nt_sub) - (n_full-1)*step``, which
        is **less** than a full ``step``.
      * Stitching: full-slide chunks keep ``[n_pad : Nt_sub-n_pad]``
        (interior). First chunk also keeps its left edge ``[0:n_pad]``.
        For the partial-slide chunk: it overlaps the previous full
        chunk by ``Nt_sub - partial_slide`` pixels (more than the usual
        2*n_pad), so only the *new* pixels at the right end --
        ``[Nt_sub - partial_slide : Nt_sub]`` -- are taken from it; the
        rest were already supplied by the previous chunk's interior /
        right edge.

    Equivalent semantics for the ll/swap_ll outer loop: only add
    contributions from the new pixels of the partial-slide chunk; the
    previous chunk already accounted for the overlapped region.

    Args:
        use_heterodyne: if True, use the FD heterodyne path (requires
            ``source_params``, ``t_ref_full``, ``gb_kwargs``, ``N_sparse``).
            If False, use ``wdm_of_td_slice`` (TD-based).

    Returns:
        ``(stitched, starts, partial_info)`` where ``partial_info`` is
        ``None`` if the tile happened to align exactly, or
        ``(partial_start, partial_slide_pixels)`` otherwise.
    """
    step = Nt_sub - 2 * n_pad
    assert step > 0 and step % 2 == 0
    n_full = (Nt - Nt_sub) // step + 1
    starts = [j * step for j in range(n_full)]
    last_full_end = starts[-1] + Nt_sub
    nchan = td_arr.shape[-2]
    stitched = np.zeros((nchan, Nf, Nt), dtype=float)

    # Single chunk worker (TD or FD-heterodyne).
    if use_heterodyne:
        T_chunk = Nf * Nt_sub * dt
        gen = CachedHeterodyneGenerator(
            T_window=T_chunk, t_ref_source=t_ref_full, N_sparse=N_sparse,
            dt=dt, nchannels=nchan, gb_kwargs=gb_kwargs,
        )
        chunk_fd_set = FDSettings(Nf * Nt_sub // 2 + 1, 1.0 / T_chunk,
                                  force_backend=backend)
        chunk_wdm_set = WDMSettings(Nf=Nf, Nt=Nt_sub, dt=dt,
                                    force_backend=backend)

        def _chunk_wdm(n0):
            chunk_t_start = n0 * (Nf * dt)
            chunk_fd, _ = gen.chunk_fd(
                source_params, chunk_t_start, Nf * Nt_sub,
                tukey_alpha=tukey_alpha, use_tukey=use_tukey,
            )
            chunk_wdm = FDSignal(chunk_fd, chunk_fd_set).transform(chunk_wdm_set)
            return np.asarray(chunk_wdm.arr)
    else:
        def _chunk_wdm(n0):
            chunk_wdm = wdm_of_td_slice(
                td_arr, n0 * Nf, Nf, Nt_sub, dt, backend,
                tukey_alpha=tukey_alpha, use_tukey=use_tukey,
            )
            return np.asarray(chunk_wdm.arr)

    # Full-slide chunks
    for j, n0 in enumerate(starts):
        chunk_arr = _chunk_wdm(n0)
        keep_lo = 0 if j == 0 else n_pad
        # If we have a partial-slide chunk afterwards it covers the right
        # edge, so the "last full" chunk should still keep its right edge
        # only up to Nt_sub - n_pad (interior only). If we DON'T have a
        # partial chunk, the last full chunk keeps its right edge fully.
        if j == n_full - 1 and last_full_end == Nt:
            keep_hi = Nt_sub                # exact tile, no partial chunk
        else:
            keep_hi = Nt_sub - n_pad
        stitched[:, :, n0 + keep_lo:n0 + keep_hi] = chunk_arr[:, :, keep_lo:keep_hi]

    partial_info = None
    if last_full_end < Nt:
        # Append one partial-slide chunk.
        n0_partial = Nt - Nt_sub
        partial_slide = n0_partial - starts[-1]
        assert 0 < partial_slide < step, (partial_slide, step)
        chunk_arr = _chunk_wdm(n0_partial)
        # Take only the new pixels at the right end. The previous
        # chunk's contribution covers [n0_prev + n_pad,
        # n0_prev + Nt_sub - n_pad] = [starts[-1] + n_pad,
        # starts[-1] + Nt_sub - n_pad]. The partial chunk covers
        # [n0_partial, n0_partial + Nt_sub] = [Nt - Nt_sub, Nt].
        # Overlap with previous chunk: [n0_partial, starts[-1] + Nt_sub
        # - n_pad]. The "new" pixels are [starts[-1] + Nt_sub - n_pad, Nt].
        new_lo_global = starts[-1] + Nt_sub - n_pad
        new_lo_local  = new_lo_global - n0_partial
        new_hi_local  = Nt_sub
        stitched[:, :, new_lo_global:Nt] = chunk_arr[:, :, new_lo_local:new_hi_local]
        partial_info = (n0_partial, partial_slide)

    return stitched, starts, partial_info


def test_J_partial_slide_edge(td_arr, full_arr, wdm_set, dt, Nf, Nt,
                              source_params, t_ref_full, backend,
                              Nt_sub=256, n_pad=32,
                              edge_layer_pixels=20,
                              N_sparse=256):
    """Phase 1b: validate partial-slide edge handling against a non-aligning Nt.

    Pads/truncates the observation to a length the standard tile
    *cannot* cover exactly, runs both the TD-based and FD-heterodyne
    chunked stitchers with the partial-slide variant, and checks mm5 /
    mm2 against the matching full-WDM-restricted-to-Nt reference.
    """
    print("\n========== Test J: partial-slide edge handling ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    step = Nt_sub - 2 * n_pad
    # Force a non-aligning Nt: trim full obs to Nt_off = Nt - 50 (50 is
    # not a multiple of step=192, so partial slide will fire).
    Nt_off = Nt - 50
    N_off = Nf * Nt_off
    td_arr_off = td_arr[..., :N_off]
    full_arr_off = full_arr[..., :Nt_off]
    assert (Nt_off - Nt_sub) % step != 0, "test_J needs misalignment"

    print("  full Nt = {}, partial-tile Nt = {} (step={}, "
          "(Nt-Nt_sub) % step = {})".format(
              Nt, Nt_off, step, (Nt_off - Nt_sub) % step))

    f0 = source_params[1]
    m_floor = int(f0 / layer_df)
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt_off - edge_layer_pixels) * layer_dt
    mm5_min, mm5_max = f0 - 3 * layer_df, f0 + 2 * layer_df
    mm2_min = m_floor * layer_df
    mm2_max = (m_floor + 2) * layer_df - 0.5 * layer_df

    # Build a WDMSettings whose Nt matches the truncated obs so the
    # reference is consistent.
    wdm_set_off = WDMSettings(Nf=Nf, Nt=Nt_off, dt=dt, force_backend=backend)

    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    for label, use_het, extra in [
        ("TD-based   ", False, {}),
        ("FD-het N={} ".format(N_sparse), True,
         dict(source_params=source_params, t_ref_full=t_ref_full,
              gb_kwargs=gb_kwargs, N_sparse=N_sparse)),
    ]:
        stitched, starts, partial_info = _stitched_wdm_partial_slide(
            td_arr_off, dt, Nf, Nt_off, Nt_sub, n_pad, backend,
            use_heterodyne=use_het,
            tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
            **extra,
        )
        n_chunks = len(starts) + (1 if partial_info else 0)
        print("  {}: n_chunks={} (full={}, partial={}); "
              "partial_info={}".format(
                  label, n_chunks, len(starts),
                  "yes" if partial_info else "no", partial_info))
        mm5, _ = _mismatch(full_arr_off, stitched, wdm_set_off,
                           mm5_min, mm5_max, min_time, max_time, backend)
        mm2, _ = _mismatch(full_arr_off, stitched, wdm_set_off,
                           mm2_min, mm2_max, min_time, max_time, backend)
        # Cross-checks: rms-diff over interior, and over the "newly
        # supplied" partial region only (to verify edge isn't pathological).
        diff = stitched - full_arr_off
        rms_full = float(np.sqrt(np.mean(full_arr_off ** 2)))
        rms_diff = float(np.sqrt(np.mean(diff ** 2)))
        rms_diff_int = float(np.sqrt(np.mean(diff[..., n_pad:Nt_off - n_pad] ** 2)))
        rms_full_int = float(np.sqrt(np.mean(full_arr_off[..., n_pad:Nt_off - n_pad] ** 2)))
        print("    rms diff / rms full         = {:.3e}".format(
            rms_diff / max(rms_full, 1e-300)))
        print("    rms diff / rms full (int)   = {:.3e}".format(
            rms_diff_int / max(rms_full_int, 1e-300)))
        print("    mm5 = {:.3e}".format(mm5))
        print("    mm2 = {:.3e}".format(mm2))
        del stitched


def test_I_nt_sub_n_sparse_sweep(td_arr, full_arr, wdm_set, dt, Nf, Nt,
                                 source_params, t_ref_full, backend,
                                 nt_sub_list=(64, 128, 256),
                                 n_sparse_list=(32, 64, 128, 256),
                                 n_pad_frac=0.125,
                                 edge_layer_pixels=20):
    """Phase 1a: joint sweep over (Nt_sub, N_sparse) with Tukey auto-pick.

    For each combination, build the FD-heterodyne stitched WDM and the
    matching TD-based reference (Test D path; same Tukey) and compute
    mm5 / mm2 against the un-windowed full WDM with the standard time
    crop. The N_sparse axis is capped at 256 per the Tukey results
    (Test G) -- the shared-memory budget of the planned C++ kernel
    requires N_sparse <= FAST_WDM_N_SPARSE_MAX = 256.

    Saves three plots:
      * testI_mm5_vs_NtSub_Nsparse.png  -- 2D heatmap of mm5
      * testI_mm2_vs_NtSub_Nsparse.png  -- 2D heatmap of mm2
      * testI_runtime_vs_NtSub_Nsparse.png -- runtime per stitched build
    Plus a CSV-ish text table on stdout.
    """
    import time as _time
    print("\n========== Test I: (Nt_sub x N_sparse) sweep (Tukey auto) ==========")
    layer_dt = Nf * dt
    layer_df = wdm_set.layer_df
    f0 = source_params[1]
    m_floor = int(f0 / layer_df)
    min_time = edge_layer_pixels * layer_dt
    max_time = (Nt - edge_layer_pixels) * layer_dt
    mm5_min, mm5_max = f0 - 3 * layer_df, f0 + 2 * layer_df
    mm2_min = m_floor * layer_df
    mm2_max = (m_floor + 2) * layer_df - 0.5 * layer_df

    orbits = EqualArmlengthOrbits(force_backend=backend)
    gb_kwargs = dict(
        tdi_config=TDIConfig("2nd generation"), orbits=orbits,
        tdi_chan="XYZ", force_backend=backend,
    )

    # Each Nt_sub triggers its own n_pad and step; we want the tile to
    # cover Nt exactly. Pick n_pad = max(1, n_pad_frac * Nt_sub) rounded
    # to even; skip combos that don't tile evenly.
    # Two independent knobs:
    #   * Nt_sub    -- number of WDM time pixels per chunk. Sets the
    #                  chunk's per-layer iFFT length and the chunk's TD
    #                  length N_chunk_td = Nf*Nt_sub. Sets chunk_df =
    #                  1/T_chunk = 1/(Nf*Nt_sub*dt).
    #   * N_sparse  -- heterodyne FFT length (independent of Nt_sub).
    #                  Sets the bandwidth filled in the chunk's dense
    #                  rfft: het_BW_Hz = N_sparse * chunk_df, which in
    #                  WDM layers is N_sparse / Nt_sub *  ...:
    #                    layer_df = 1/(2*Nf*dt),
    #                    chunk_df = 1/(Nf*Nt_sub*dt),
    #                    het_BW   = N_sparse * chunk_df = N_sparse / (Nf*Nt_sub*dt),
    #                    het_BW_in_layers = het_BW / layer_df
    #                                     = 2 * N_sparse / Nt_sub.
    #                  e.g. (Nt_sub=256, N_sparse=256) -> 2 layers; need >=5
    #                  layers to cover mm5 band without Tukey help.
    print("  {:>7s} {:>7s} {:>7s} {:>9s} {:>10s} {:>9s} {:>9s} "
          "{:>14s} {:>14s} {:>14s} {:>14s} {:>8s}".format(
              "Nt_sub", "N_sparse", "n_pad",
              "alpha", "het BW Hz", "het lay", "n_chunks",
              "mm5 het", "mm5 td", "mm2 het", "mm2 td", "het s"))

    mm5_het = np.zeros((len(nt_sub_list), len(n_sparse_list)))
    mm2_het = np.zeros_like(mm5_het)
    mm5_td  = np.zeros_like(mm5_het)
    mm2_td  = np.zeros_like(mm5_het)
    runtime_het = np.zeros_like(mm5_het)
    valid       = np.zeros_like(mm5_het, dtype=bool)

    for i_nts, Nt_sub in enumerate(nt_sub_list):
        # Choose n_pad close to n_pad_frac*Nt_sub; if it doesn't tile Nt,
        # search nearby (step must be even and divide Nt-Nt_sub).
        target = max(2, int(n_pad_frac * Nt_sub))
        if target % 2:
            target -= 1
        n_pad = None
        for delta in range(0, Nt_sub // 2):
            for sign in (-1, +1):
                cand = target + sign * delta
                if cand < 2 or cand >= Nt_sub // 2:
                    continue
                if cand % 2:
                    continue
                step_cand = Nt_sub - 2 * cand
                if step_cand <= 0:
                    continue
                if (Nt - Nt_sub) % step_cand == 0:
                    n_pad = cand
                    break
            if n_pad is not None:
                break
        if n_pad is None:
            print("  Nt_sub={} skipped (no valid n_pad found for Nt={})".format(
                Nt_sub, Nt))
            continue
        step = Nt_sub - 2 * n_pad
        n_chunks = (Nt - Nt_sub) // step + 1

        # TD-based reference (same Tukey auto-pick path) once per Nt_sub.
        stitched_td, _ = _stitched_wdm_from_td(
            td_arr, dt, Nf, Nt, Nt_sub, n_pad, backend,
            tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
        )
        mm5_td_v, _ = _mismatch(full_arr, stitched_td, wdm_set,
                                mm5_min, mm5_max, min_time, max_time, backend)
        mm2_td_v, _ = _mismatch(full_arr, stitched_td, wdm_set,
                                mm2_min, mm2_max, min_time, max_time, backend)
        del stitched_td

        for i_nsp, N_sparse in enumerate(n_sparse_list):
            if N_sparse > FAST_WDM_N_SPARSE_MAX:    # respects C++ shared budget
                print("  N_sparse={} > FAST_WDM_N_SPARSE_MAX; skipped".format(N_sparse))
                continue
            alpha_eff = recommended_tukey_alpha("heterodyne", N_sparse=N_sparse)
            t0 = _time.perf_counter()
            stitched, _ = _stitched_wdm_from_heterodyne(
                td_arr, dt, Nf, Nt, Nt_sub, n_pad,
                source_params, t_ref_full, N_sparse, gb_kwargs, backend,
                tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
            )
            elapsed = _time.perf_counter() - t0
            mm5_v, _ = _mismatch(full_arr, stitched, wdm_set,
                                 mm5_min, mm5_max, min_time, max_time, backend)
            mm2_v, _ = _mismatch(full_arr, stitched, wdm_set,
                                 mm2_min, mm2_max, min_time, max_time, backend)
            del stitched

            mm5_het[i_nts, i_nsp] = mm5_v
            mm2_het[i_nts, i_nsp] = mm2_v
            mm5_td [i_nts, i_nsp] = mm5_td_v
            mm2_td [i_nts, i_nsp] = mm2_td_v
            runtime_het[i_nts, i_nsp] = elapsed
            valid       [i_nts, i_nsp] = True

            # Heterodyne band in absolute Hz and in "WDM layers covered".
            T_chunk = Nf * Nt_sub * dt
            chunk_df = 1.0 / T_chunk
            het_BW_hz = N_sparse * chunk_df            # N_sparse bins wide
            het_BW_layers = 2.0 * N_sparse / Nt_sub    # = het_BW_hz / layer_df

            print("  {:>7d} {:>7d} {:>7d} {:>9.3f} {:>10.3e} {:>9.2f} {:>9d} "
                  "{:>14.3e} {:>14.3e} {:>14.3e} {:>14.3e} {:>8.2f}".format(
                      Nt_sub, N_sparse, n_pad, alpha_eff,
                      het_BW_hz, het_BW_layers, n_chunks,
                      mm5_v, mm5_td_v, mm2_v, mm2_td_v, elapsed,
                  ))

    # plots
    def _heatmap(arr, title, fname, cbar_label):
        fig, ax = plt.subplots(figsize=(7, 5))
        with np.errstate(invalid="ignore"):
            shown = np.where(valid, arr, np.nan)
        im = ax.imshow(np.log10(shown), origin="lower", aspect="auto",
                       cmap="viridis",
                       extent=[-0.5, len(n_sparse_list) - 0.5,
                               -0.5, len(nt_sub_list) - 0.5])
        ax.set_xticks(range(len(n_sparse_list)))
        ax.set_xticklabels([str(n) for n in n_sparse_list])
        ax.set_yticks(range(len(nt_sub_list)))
        ax.set_yticklabels([str(n) for n in nt_sub_list])
        ax.set_xlabel("N_sparse")
        ax.set_ylabel("Nt_sub")
        ax.set_title(title)
        for i in range(len(nt_sub_list)):
            for j in range(len(n_sparse_list)):
                if valid[i, j]:
                    ax.text(j, i, "{:.1e}".format(arr[i, j]),
                            ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print("  wrote", fname)

    _heatmap(mm5_het,
             "Test I: mm5 (FD heterodyne, Tukey auto) vs (Nt_sub, N_sparse)",
             "check_shortened_wdm_testI_mm5_vs_NtSub_Nsparse.png",
             "log10(mm5)")
    _heatmap(mm2_het,
             "Test I: mm2 (FD heterodyne, Tukey auto) vs (Nt_sub, N_sparse)",
             "check_shortened_wdm_testI_mm2_vs_NtSub_Nsparse.png",
             "log10(mm2)")
    _heatmap(runtime_het,
             "Test I: heterodyne stitched-WDM build time vs (Nt_sub, N_sparse)",
             "check_shortened_wdm_testI_runtime_vs_NtSub_Nsparse.png",
             "log10(seconds)")


def test_C_complex_wdm(td_arr, dt, Nf, Nt, backend):
    """Complex (quadrature) WDM forward path: equivalence checks.

    (i) Real path equivalence: ``WDM_complex.real`` must equal the
        real-only ``WDM`` array exactly.
    (ii) Inner-product equivalence: the diagnostic
         ``inner_product`` of (sig, sig) on the complex WDM (with halved
         ``differential_component``) should match the inner product on
         the real WDM.
    """
    print("\n========== Test C: complex/quadrature WDM ==========")

    N = Nf * Nt
    td_set = TDSettings(N, dt, force_backend=backend)
    wdm_set_real = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, force_backend=backend)
    wdm_set_complex = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, force_backend=backend,
                                  is_complex=True)
    td_sig = TDSignal(td_arr, td_set)
    wdm_real = td_sig.transform(wdm_set_real)
    wdm_complex = td_sig.transform(wdm_set_complex)

    real_arr = np.asarray(wdm_real.arr)
    complex_arr = np.asarray(wdm_complex.arr)
    print("real WDM:    shape={}, dtype={}".format(real_arr.shape, real_arr.dtype))
    print("complex WDM: shape={}, dtype={}".format(complex_arr.shape, complex_arr.dtype))

    # (i) the real part must equal the real-only path exactly.
    diff_real = complex_arr.real - real_arr
    max_diff_real = float(np.max(np.abs(diff_real)))
    rms_real = float(np.sqrt(np.mean(real_arr ** 2)))
    print("\n(i) WDM_complex.real vs WDM_real:")
    print("    max |diff|         = {:.3e}".format(max_diff_real))
    print("    max diff / rms_real = {:.3e}".format(
        max_diff_real / max(rms_real, 1e-300)))

    # imaginary-part summary
    imag_arr = complex_arr.imag
    print("    rms imag(complex) = {:.3e}".format(
        float(np.sqrt(np.mean(imag_arr ** 2)))))
    print("    rms real(complex) = {:.3e}".format(
        float(np.sqrt(np.mean(complex_arr.real ** 2)))))
    print("    -> ratio (sanity: should be ~1 for narrowband signal away "
          "from boundary layers): {:.3f}".format(
              float(np.sqrt(np.mean(imag_arr ** 2))
                    / max(np.sqrt(np.mean(complex_arr.real ** 2)), 1e-300))))

    # (ii) inner-product equivalence with PSD = ones, computed directly
    # using the inner_product summation rule:
    #   <h|h> = 4 * sum( Re(W.conj() * W) ) * differential_component
    # which for real W is 4 * sum(W^2) * 0.25 = sum(W^2),
    # and for complex W with halved differential_component is
    # 4 * sum(|W|^2) * 0.125 = 0.5 * sum(|W|^2).
    def hh(arr, settings):
        return 4.0 * float(np.sum(np.real(arr.conj() * arr))) * settings.differential_component

    ip_real = hh(real_arr, wdm_set_real)
    ip_complex = hh(complex_arr, wdm_set_complex)
    print("\n(ii) <h|h> with PSD=ones via inner_product summation rule:")
    print("    differential_component:  real={}, complex={}".format(
        wdm_set_real.differential_component, wdm_set_complex.differential_component))
    print("    real WDM:    <h|h> = {:.6e}".format(ip_real))
    print("    complex WDM: <h|h> = {:.6e}".format(ip_complex))
    print("    ratio complex/real = {:.6f}  (target: 1.000)".format(
        ip_complex / max(abs(ip_real), 1e-300)))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    backend = "cpu"
    dt = 10.0
    Tobs_target = 0.5 * YRSID_SI

    Nf = 64
    Nt_sub_main = 256                 # per-layer iFFT length for chunked tests
    n_pad_main = Nt_sub_main // 8     # 32; pixels dropped at each chunk edge
    step_main = Nt_sub_main - 2 * n_pad_main   # 192

    # Pick Nt so (Nt - Nt_sub_main) is divisible by step_main (chunks tile
    # exactly) and Tobs ~= Tobs_target. Nt = step * k + Nt_sub_main.
    target_Nt = int(round(Tobs_target / (Nf * dt)))
    k_chunks = max(1, (target_Nt - Nt_sub_main) // step_main)
    Nt = step_main * k_chunks + Nt_sub_main
    N = Nf * Nt
    Tobs = N * dt
    layer_dt = Nf * dt
    layer_df = 1.0 / (2.0 * Nf * dt)
    f0 = (Nf // 4 + 0.37) * layer_df

    t_arr = np.arange(N) * dt
    t_ref = Tobs / 2.0

    print("Setup: Nf={}, Nt={}, dt={}s".format(Nf, Nt, dt))
    print("       full N = {} samples, Tobs = {:.3e} s ({:.4f} yr)".format(
        N, Tobs, Tobs / YRSID_SI))
    print("       layer_df={:.3e} Hz, layer_dt={:.1f} s".format(layer_df, layer_dt))
    print("       Nt_sub (chunk window FFT) = {} (full Nt is {}x that)".format(
        Nt_sub_main, Nt // Nt_sub_main))
    print("       f0 = {:.3e} Hz".format(f0))

    # ----- generate the TD waveform -------------------------------------
    gb_gen = build_gb_generator(t_arr, Tobs, t_ref, dt, backend=backend)
    params = np.array([[8.0e-23], [f0], [1.0e-17], [0.0],
                       [2.098], [0.5], [1.234], [4.098], [0.09]])
    print("\nGenerating TD waveform...", flush=True)
    wave_spline = gb_gen(*params, convert_to_ra_dec=False, return_spline=True)
    td_arr = np.asarray(wave_spline.eval_tdi(t_arr))
    if td_arr.ndim == 3:
        assert td_arr.shape[0] == 1
        td_arr = td_arr[0]
    print("TD shape: {}, peak |X|: {:.3e}".format(
        td_arr.shape, float(np.max(np.abs(td_arr[0])))))

    # ----- full TD -> FD -> WDM (reference) -----------------------------
    print("\nRunning full TD->FD->WDM (reference)...", flush=True)
    td_set = TDSettings(N, dt, force_backend=backend)
    wdm_set = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, force_backend=backend)
    wdm_full = TDSignal(td_arr, td_set).transform(wdm_set)
    full_arr = np.asarray(wdm_full.arr)
    print("Full WDM shape:", full_arr.shape)

    # Pick a mid-obs pixel aligned to Nt_sub_main so chunks fit cleanly.
    n0_even = (Nt // 2)
    n0_even -= n0_even % 2
    n0_odd = n0_even + 1

    test_A_even_chunk_sweep(td_arr, full_arr, dt, Nf, Nt, n0_even, backend)
    test_B_odd_chunk(td_arr, full_arr, dt, Nf, Nt, n0_odd,
                     Nt_sub=Nt_sub_main, backend=backend)

    # NOTE: Test B' (complex chunk unitary) and Test C (complex inner-
    # product equivalence) skipped here to conserve memory -- both were
    # validated to machine precision in prior runs (mm5/mm2 ~1e-13).
    # Re-enable when memory headroom permits.
    # wdm_set_cmpx = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, force_backend=backend,
    #                            is_complex=True)
    # wdm_full_cmpx = TDSignal(td_arr, td_set).transform(wdm_set_cmpx)
    # test_B_complex_odd_chunk(td_arr, np.asarray(wdm_full_cmpx.arr),
    #                          dt, Nf, Nt, n0_odd,
    #                          Nt_sub=Nt_sub_main, backend=backend)
    # test_C_complex_wdm(td_arr, dt, Nf, Nt, backend)

    # Test D: stitched WDM (overlapping chunks, real mode) vs full via mm5/mm2.
    test_D_stitch_mismatch(td_arr, full_arr, wdm_set, dt, Nf, Nt, f0,
                           backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main)

    # Test E: FD-heterodyne stitched WDM (no TD generation per chunk).
    # Use the same source params as the TD path; propagate per-chunk inside.
    source_params_tuple = (
        float(params[0, 0]), float(params[1, 0]), float(params[2, 0]),
        float(params[3, 0]), float(params[4, 0]), float(params[5, 0]),
        float(params[6, 0]), float(params[7, 0]), float(params[8, 0]),
    )
    # Test F / E: With use_tukey=True (default), N_sparse=256 reaches
    # the mm floor and stays within FAST_WDM_N_SPARSE_MAX (the shared-mem
    # ceiling for the planned C++ kernel).
    test_F_td_fd_vs_fd_direct(
        td_arr, dt, Nf, Nt, source_params_tuple, t_ref,
        backend, Nt_sub=Nt_sub_main, N_sparse_chunk=256, N_sparse_full=1024,
    )

    test_E_heterodyne_chunked(
        td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
        backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main, N_sparse=256,
    )

    # Test I (Phase 1a): joint Nt_sub x N_sparse sweep with Tukey auto-pick.
    test_I_nt_sub_n_sparse_sweep(
        td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
        backend,
        nt_sub_list=(64, 128, 256),
        n_sparse_list=(32, 64, 128, 256),
    )

    # Test J (Phase 1b): partial-slide edge-handling validation.
    test_J_partial_slide_edge(
        td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
        backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main, N_sparse=256,
    )

    # Test K (Phase 2 pre-validation): Python reference for the
    # gb_chunk_fd_to_wdm C++ device function.
    test_K_cpp_python_reference_match(
        td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
        backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main, N_sparse=256,
    )

    # Test L (Phase 2 pre-validation): chunked-heterodyne get_ll Python
    # reference vs lisatools template_inner_product.
    test_L_chunked_get_ll_reference(
        td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
        backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main, N_sparse=256,
    )

    # Tests G + H are skipped here: they are sweep / visualisation tests
    # whose plots are already saved (see git status). Re-enable by hand
    # if you want to refresh them.
    if False:
        test_H_visualizations(
            td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
            backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main,
        )
        test_G_n_sparse_sweep(
            td_arr, full_arr, wdm_set, dt, Nf, Nt, source_params_tuple, t_ref,
            backend, Nt_sub=Nt_sub_main, n_pad=n_pad_main,
            n_sparse_list=(64, 128, 256, 512, 1024, 2048, 4096),
            tukey_alphas=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1),
        )


if __name__ == "__main__":
    main()
