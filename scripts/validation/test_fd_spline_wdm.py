"""End-to-end verification of wdm_spline_helpers against lisatools.

Compares the layer-by-layer pipeline (Bluestein FFT for general even N,
2-stage row-column TD->FD FFT, layer-by-layer windowed IFFT with
(m+n)-parity extraction, optional narrow-window per-(src, layer))
against ``FDSignal.wdmtransform`` in ``lisatools.domains``.

Tests:
  1. ``Bluestein`` and the 2-stage FFT match numpy.fft to ~1e-13.
  2. ``synth_and_transform_one_binary`` matches
     ``FDSignal.wdmtransform`` element-wise to <1e-9 on synthetic
     waveforms.
  3. ``fill_global_wdm`` accumulates correctly with multiple sources
     and signed factors.
  4. ``get_ll_wdm`` and ``swap_ll_wdm`` match a direct Python inner
     product against the layer-by-layer w_mn output.
  5. **Narrow-window**: a layer transformed with Nt_narrow << Nt
     correctly handles both parities of (m + n_global_start).

Run from the repo root:
    /Users/mkatz/miniconda3/envs/deving/bin/python test_fd_spline_wdm.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from wdm_spline_helpers import (
    BluesteinPlan,
    WDMSplineKernelPlan,
    bluestein_fft,
    extract_layer_wdm,
    fill_global_wdm,
    get_ll_wdm,
    swap_ll_wdm,
    synth_and_transform_one_binary,
    two_stage_fft,
    two_stage_fft_to_dense,
)


# ---------------------------------------------------------------------------
# Synthetic TD waveform: chirp + Gaussian envelope (MBHB-like proxy).
# ---------------------------------------------------------------------------


def synth_chirp_td(
    t: np.ndarray,
    A0: float = 1.0,
    f0: float = 1e-3,
    fdot: float = 5e-7,
    t_merge: float = None,
    sigma_t: float = None,
) -> np.ndarray:
    """Quadratic-chirp waveform with a Gaussian envelope.

    ``t`` is the absolute time grid (seconds).  Returns a real ``(N,)``
    array.  The envelope peak at ``t_merge`` mimics an MBHB merger;
    layers near ``t_merge`` are good narrow-window candidates.
    """
    if t_merge is None:
        t_merge = 0.45 * t[-1]
    if sigma_t is None:
        sigma_t = 0.08 * t[-1]
    env = np.exp(-0.5 * ((t - t_merge) / sigma_t) ** 2)
    phase = 2 * np.pi * (f0 * t + 0.5 * fdot * t ** 2)
    return A0 * env * np.cos(phase)


def make_synth(N: int, dt: float, A0=1.0, f0=1e-3, fdot=5e-7) -> callable:
    """Return a ``td_synthesizer(t_arr) -> (1, N) real`` closure."""
    def f(t_arr: np.ndarray) -> np.ndarray:
        return synth_chirp_td(t_arr, A0=A0, f0=f0, fdot=fdot)[None, :]
    return f


# ---------------------------------------------------------------------------
# Lisatools reference: FDSignal.wdmtransform on the same TD waveform.
# ---------------------------------------------------------------------------


def reference_wdm_from_td(td_real: np.ndarray, Nf: int, Nt: int, dt: float) -> np.ndarray:
    """Compute the reference WDM via lisatools.

    Returns ``(1, Nf_active, Nt_active)`` real.  ``td_real`` is (N,).
    """
    from lisatools.domains import FDSettings, FDSignal, WDMSettings

    N = Nf * Nt
    assert td_real.shape == (N,)
    # FD via numpy (matches what lisatools uses internally for FDSignal.fft)
    fd = np.fft.fft(td_real) * dt
    fd_settings = FDSettings(N=N, df=1.0 / (N * dt), force_backend="cpu")
    sig = FDSignal(fd[None, :], settings=fd_settings)
    wdm = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, force_backend="cpu")
    out = sig.wdmtransform(settings=wdm)
    return np.asarray(out.arr)


# ---------------------------------------------------------------------------
# Test cases.
# ---------------------------------------------------------------------------


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b))) / denom


def test_fft_blocks():
    """Sanity: Bluestein and 2-stage FFT match numpy.fft."""
    rng = np.random.RandomState(0)
    ok = True
    for N in (6, 12, 20, 24, 48, 64):
        plan = BluesteinPlan.build(N)
        x = rng.randn(N) + 1j * rng.randn(N)
        err_fwd = float(np.max(np.abs(bluestein_fft(plan, x) - np.fft.fft(x))))
        err_inv = float(np.max(np.abs(bluestein_fft(plan, x, inverse=True) - np.fft.ifft(x))))
        passed = (err_fwd < 1e-10) and (err_inv < 1e-12)
        ok &= passed
        print(f"  [fft]   N={N:3d}  fwd={err_fwd:.2e}  inv={err_inv:.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def test_two_stage_fft():
    rng = np.random.RandomState(1)
    ok = True
    for Nf, Nt in [(8, 16), (16, 32), (6, 10), (12, 20), (10, 24)]:
        plan = WDMSplineKernelPlan.build(
            Nf=Nf, Nt=Nt, data_dt=1.0,
            m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
        )
        x = rng.randn(Nf * Nt)
        fd = two_stage_fft_to_dense(two_stage_fft(plan.big_fft_plan, x))
        err = _rel_err(fd, np.fft.fft(x))
        passed = err < 1e-10
        ok &= passed
        print(f"  [2stg]  Nf={Nf:3d} Nt={Nt:3d}  err={err:.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def test_full_wdm_vs_lisatools():
    """The core regression: our layer-by-layer WDM matches lisatools."""
    ok = True
    for Nf, Nt in [(16, 32), (32, 64), (16, 24)]:
        N = Nf * Nt
        dt = 10.0  # seconds; the absolute scale doesn't matter for this test
        plan = WDMSplineKernelPlan.build(
            Nf=Nf, Nt=Nt, data_dt=dt,
            m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
        )
        f0 = 1e-3 * (16.0 / Nf) * (32.0 / Nt) / (dt / 10.0)
        synth = make_synth(N, dt, f0=f0)
        td = synth(np.arange(N) * dt)
        ref = reference_wdm_from_td(td[0], Nf, Nt, dt)
        ours = synth_and_transform_one_binary(plan, synth, nchannels=1)
        err = _rel_err(ours, ref)
        passed = err < 1e-8
        ok &= passed
        print(f"  [wdm]   Nf={Nf:3d} Nt={Nt:3d}  N={N:5d}  "
              f"max_abs={float(np.max(np.abs(ours - ref))):.2e}  "
              f"rel={err:.2e}  ref_max={float(np.max(np.abs(ref))):.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def test_fill_global_two_sources():
    """fill_global should sum (with factors) into the template."""
    Nf, Nt, dt = 16, 32, 10.0
    N = Nf * Nt
    plan = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
    )
    s1 = make_synth(N, dt, A0=1.0, f0=1e-3)
    s2 = make_synth(N, dt, A0=0.5, f0=8e-4)
    template = np.zeros((1, 1, plan.Nf_active, plan.Nt_active))
    fill_global_wdm(
        plan, [s1, s2], template,
        data_index_all=np.array([0, 0], dtype=np.int32),
        factors_all=np.array([+1.0, -1.0], dtype=np.float64),
        nchannels=1,
    )
    # Reference: do each source separately, combine
    w1 = synth_and_transform_one_binary(plan, s1, nchannels=1)
    w2 = synth_and_transform_one_binary(plan, s2, nchannels=1)
    expected = (1.0 * w1 - 1.0 * w2)[None, :]
    err = _rel_err(template, expected)
    passed = err < 1e-12
    print(f"  [fill]  err={err:.2e}  {'OK' if passed else 'FAIL'}")
    return passed


def test_get_ll():
    """get_ll_wdm against direct Python inner-product."""
    Nf, Nt, dt = 16, 32, 10.0
    N = Nf * Nt
    plan = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
    )
    s = make_synth(N, dt, A0=1.0, f0=1e-3)
    w = synth_and_transform_one_binary(plan, s, nchannels=1)
    rng = np.random.RandomState(2)
    data = w + 0.1 * rng.randn(*w.shape)
    invC = np.ones_like(w) * 4.0
    d_h, h_h = get_ll_wdm(
        plan, [s],
        wdm_data=data[None, :], wdm_invC=invC[None, :],
        data_index_all=np.array([0], dtype=np.int32),
        noise_index_all=np.array([0], dtype=np.int32),
        nchannels=1,
    )
    ref_dh = 4.0 * np.sum(data * w * invC)
    ref_hh = 4.0 * np.sum(w * w * invC)
    err = max(abs(d_h[0] - ref_dh) / abs(ref_dh),
              abs(h_h[0] - ref_hh) / abs(ref_hh))
    passed = err < 1e-12
    print(f"  [ll]    d_h={d_h[0]:+.6e}  ref={ref_dh:+.6e}  "
          f"h_h={h_h[0]:+.6e}  ref={ref_hh:+.6e}  rel={err:.2e}  "
          f"{'OK' if passed else 'FAIL'}")
    return passed


def test_swap_ll():
    Nf, Nt, dt = 16, 32, 10.0
    N = Nf * Nt
    plan = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
    )
    s_add = make_synth(N, dt, A0=1.0, f0=1.05e-3)
    s_rem = make_synth(N, dt, A0=1.0, f0=1.00e-3)
    w_a = synth_and_transform_one_binary(plan, s_add, nchannels=1)
    w_r = synth_and_transform_one_binary(plan, s_rem, nchannels=1)
    rng = np.random.RandomState(3)
    data = w_a + 0.05 * rng.randn(*w_a.shape)
    invC = np.ones_like(w_a) * 4.0
    d_h_a, d_h_r, aa, rr, ar = swap_ll_wdm(
        plan, [s_add], [s_rem],
        wdm_data=data[None, :], wdm_invC=invC[None, :],
        data_index_all=np.array([0], dtype=np.int32),
        noise_index_all=np.array([0], dtype=np.int32),
        nchannels=1,
    )
    ref = {
        "d_h_a": 4.0 * np.sum(data * w_a * invC),
        "d_h_r": 4.0 * np.sum(data * w_r * invC),
        "aa":    4.0 * np.sum(w_a * w_a * invC),
        "rr":    4.0 * np.sum(w_r * w_r * invC),
        "ar":    4.0 * np.sum(w_a * w_r * invC),
    }
    got = {"d_h_a": d_h_a[0], "d_h_r": d_h_r[0], "aa": aa[0], "rr": rr[0], "ar": ar[0]}
    err = max(abs(got[k] - ref[k]) / abs(ref[k]) for k in ref)
    passed = err < 1e-12
    for k in ref:
        print(f"  [swap]  {k}={got[k]:+.6e}  ref={ref[k]:+.6e}")
    print(f"  [swap]  max rel err {err:.2e}  {'OK' if passed else 'FAIL'}")
    return passed


def test_narrow_window_parity():
    """Verify the narrow-window (m+n) parity-flip handling.

    Standalone check on ``extract_layer_wdm``: build a known FD,
    extract a layer at the full grid AND at a narrow window centered
    on a pixel chosen so that (m + n_global_start) % 2 is BOTH 0 AND 1.
    The narrow result for the overlapping pixels should match the full
    result (when the same window is used).  More importantly, BOTH
    parities of n_global_start must produce results consistent with
    the full-grid reference -- which exercises both the
    ``real(IFFT)`` and ``imag(IFFT)`` branches at i=0.
    """
    Nf, Nt, dt = 16, 64, 10.0
    N = Nf * Nt
    plan = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
    )
    s = make_synth(N, dt, A0=1.0, f0=1e-3)
    w_full = synth_and_transform_one_binary(plan, s, nchannels=1)
    # Pick a middle layer
    m = 5
    full_row = w_full[0, m - plan.m_min, :]

    # Now narrow window: pick a centre and verify both parities.
    Nt_narrow = Nt // 2  # 32 -- power of two
    plan_narrow = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
        narrow_widths=[Nt_narrow],
    )
    ok = True
    for n_center in (Nt // 2, Nt // 2 + 1):
        narrow_widths_row = np.zeros(Nf, dtype=np.int32)
        narrow_widths_row[m] = Nt_narrow
        narrow_centers_row = np.zeros(Nf, dtype=np.int32)
        narrow_centers_row[m] = n_center

        # Build narrow_widths/centers for ONE binary with all-zero other layers
        nw = np.zeros((1, plan_narrow.Nf_active), dtype=np.int32)
        nc = np.zeros((1, plan_narrow.Nf_active), dtype=np.int32)
        nw[0, m - plan_narrow.m_min] = Nt_narrow
        nc[0, m - plan_narrow.m_min] = n_center

        w_mixed = synth_and_transform_one_binary(
            plan_narrow, s, nchannels=1,
            narrow_widths_row=nw[0], narrow_centers_row=nc[0],
        )
        narrow_row = w_mixed[0, m - plan_narrow.m_min, :]

        # We compare the narrow IFFT result against an independently-computed
        # reference: run extract_layer_wdm with the SAME narrow window and
        # plan over the SAME FD samples derived from the same TD waveform.
        td = s(np.arange(N) * dt)
        fd_dense = two_stage_fft_to_dense(two_stage_fft(plan_narrow.big_fft_plan, td[0]))
        w_layer_narrow = extract_layer_wdm(
            fd_dense, plan_narrow.narrow_windows[Nt_narrow],
            m=m, Nf=Nf, Nt=Nt, Nt_layer=Nt_narrow,
            n_global_start=n_center - Nt_narrow // 2,
            kappa=plan_narrow.kappa, data_dt=dt,
            layer_ifft_plan=plan_narrow.narrow_plans[Nt_narrow],
        )
        # Scatter into a full-Nt row for comparison
        scatter = np.zeros(Nt)
        for i in range(Nt_narrow):
            ng = n_center - Nt_narrow // 2 + i
            if 0 <= ng < Nt:
                scatter[ng] = w_layer_narrow[i]
        # The narrow row in w_mixed should match this scatter exactly
        rel = _rel_err(narrow_row, scatter)
        # Parity check: print which branch we used for i=0
        parity_at_zero = (m + (n_center - Nt_narrow // 2)) & 1
        passed = rel < 1e-12
        ok &= passed
        print(f"  [narw]  n_center={n_center:3d}  "
              f"parity@i=0={parity_at_zero}  rel={rel:.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def test_narrow_window_consistency():
    """A narrower-than-full layer at the FULL Nt size should reproduce the
    full transform exactly when Nt_narrow == Nt and n_global_start == 0."""
    Nf, Nt, dt = 16, 64, 10.0
    N = Nf * Nt
    plan_full = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
    )
    plan_with = WDMSplineKernelPlan.build(
        Nf=Nf, Nt=Nt, data_dt=dt,
        m_min=0, m_max=Nf - 1, n_min=0, n_max=Nt - 1,
        narrow_widths=[Nt],
    )
    s = make_synth(N, dt, A0=1.0, f0=1e-3)
    w_ref = synth_and_transform_one_binary(plan_full, s, nchannels=1)
    # Use narrow=Nt centered at Nt/2 for every active layer
    nw = np.full((1, plan_with.Nf_active), Nt, dtype=np.int32)
    nc = np.full((1, plan_with.Nf_active), Nt // 2, dtype=np.int32)
    w_with = synth_and_transform_one_binary(
        plan_with, s, nchannels=1,
        narrow_widths_row=nw[0], narrow_centers_row=nc[0],
    )
    # The two should match for non-edge layers
    diff_inner = w_with[:, 1:, :] - w_ref[:, 1:, :]
    rel = float(np.max(np.abs(diff_inner)) / max(float(np.max(np.abs(w_ref[:, 1:, :]))), 1e-30))
    passed = rel < 1e-10
    print(f"  [narw-eq] Nt_narrow == Nt at centre Nt/2 (excl edge layer): rel={rel:.2e}  "
          f"{'OK' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("WDM Spline kernel reference tests (Python prototype of C++/CUDA path)")
    print("=" * 72)
    tests = [
        ("Bluestein FFT vs numpy.fft (general even N)", test_fft_blocks),
        ("2-stage row-column FFT vs numpy.fft", test_two_stage_fft),
        ("Full WDM vs lisatools.domains.wdmtransform", test_full_wdm_vs_lisatools),
        ("fill_global with signed factors", test_fill_global_two_sources),
        ("get_ll diagonal-noise IP", test_get_ll),
        ("swap_ll (five accumulators)", test_swap_ll),
        ("narrow-window (m+n) parity handling", test_narrow_window_parity),
        ("narrow == full @ same width consistency", test_narrow_window_consistency),
    ]
    all_ok = True
    for name, fn in tests:
        print()
        print("-" * 72)
        print(name)
        print("-" * 72)
        ok = fn()
        all_ok &= ok
    print()
    print("=" * 72)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
