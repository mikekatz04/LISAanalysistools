"""Isolated debug of the WDM transform on td_sig.npy.

Skips SangriaProcessing entirely. Loads the saved TD signal,
builds the WDMSettings used by the lookup table, walks the
transform step by step, and reports where NaN first appears.
"""

import sys

import numpy as np
from scipy import special

# noqa: F401 — needed for backend registration when DomainSettingsBase resolves
# Phase 3L.7l: import fastlisaresponse removed (no longer registers backends).
# If a script needs the backend registry, import lisatools / gbgpu / bbhx as appropriate.
from lisatools.domains import (
    TDSettings,
    TDSignal,
    WDMSettings,
    WDMLookupTable,
)


TD_NPY = "/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools/td_sig.npy"
LOOKUP_H5 = "/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools/wdm_lookup_n_ref_NF720_NT2160_3mo.h5"


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def report_array(name: str, arr, axis_for_per_channel: int = 0) -> None:
    arr = np.asarray(arr)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} nan={n_nan} inf={n_inf}")
    if arr.ndim >= 1 and arr.shape[0] in (1, 2, 3, 4):
        for i in range(arr.shape[0]):
            ai = arr[i]
            n_nan_i = int(np.isnan(ai).sum())
            print(f"  ch{i}: nan={n_nan_i} max|a|={np.nanmax(np.abs(ai)) if ai.size else 0:.3e}")


def main() -> None:
    section("Load TD signal")
    td_arr = np.load(TD_NPY)
    print("td_arr shape:", td_arr.shape, "dtype:", td_arr.dtype)
    print(f"td nan={np.isnan(td_arr).sum()}, inf={np.isinf(td_arr).sum()}, "
          f"max|a|={np.abs(td_arr).max():.3e}")

    N = td_arr.shape[-1]
    dt = 5.0

    # --- WDM settings matching the lookup table (same as build_wdm_lookup_3mo.py) ---
    Nf, Nt = 720, 2160
    expected_N = Nf * Nt
    print(f"expected N = Nf*Nt = {expected_N}, actual N = {N}")
    if N != expected_N:
        print("!! TD signal length does not match Nf*Nt — WDM transform expects exactly Nf*Nt samples.")
        return

    section("Build WDMSettings")
    wdm_settings = WDMSettings(
        Nf=Nf,
        Nt=Nt,
        dt=dt,
        oversample=16,
        min_freq=1e-4,
        max_freq=2.5e-2,
        min_time=20 * 3600.0,
        max_time=(Nt - 20) * 3600.0,
        force_backend="cpu",
    )
    print(f"layer_dt={wdm_settings.layer_dt} layer_df={wdm_settings.layer_df}")
    print(f"ind_min_f={wdm_settings.ind_min_f} ind_max_f={wdm_settings.ind_max_f} "
          f"Nf_active={wdm_settings.Nf_active}")
    print(f"ind_min_t={wdm_settings.ind_min_t} ind_max_t={wdm_settings.ind_max_t} "
          f"Nt_active={wdm_settings.Nt_active}")
    print(f"A={wdm_settings.A}, dOmega_s=pi/Nf={np.pi/Nf}, "
          f"omega shape={wdm_settings.omega.shape}, "
          f"omega range=[{wdm_settings.omega.min():.6e},{wdm_settings.omega.max():.6e}]")

    section("Inspect WDM window (settings.window)")
    win = np.asarray(wdm_settings.window)
    report_array("window", win.reshape(1, -1))
    nan_mask = np.isnan(win)
    if nan_mask.any():
        idx = np.where(nan_mask)[0]
        print(f"window NaN indices ({len(idx)}): {idx}")
        print(f"omega at NaN: {wdm_settings.omega[idx]}")
        A = wdm_settings.A
        dOmega = np.pi / Nf
        B = dOmega - 2 * A
        print(f"A={A:.16e}, B={B:.16e}, A+B={A+B:.16e}, dOmega_s={dOmega:.16e}")
        # Reproduce the exact mask used in phitilde
        omega = wdm_settings.omega
        mask = (np.abs(omega) >= A) & (np.abs(omega) <= A + B)
        x = (np.abs(omega[mask]) - A) / B
        y = special.betainc(4, 4, x)
        print(f"beta_inc_calc covers {mask.sum()} samples of {omega.size}")
        print(f"x min={x.min():.16e}, x max={x.max():.16e}")
        print(f"any x > 1.0? {(x > 1.0).any()} count={int((x>1.0).sum())}")
        print(f"any x == 1.0? {(x == 1.0).any()} count={int((x==1.0).sum())}")
        ynan_idx = np.where(np.isnan(y))[0]
        print(f"NaN in y inside beta_inc_calc: {len(ynan_idx)} indices, x at NaN: {x[ynan_idx]}")
        for xv in [0.999999999999, 1.0, 1.0 + 1e-16, 1.0 + 2e-16, 1.0 + 1e-14]:
            print(f"  scipy.special.betainc(4,4,{xv!r}) = {special.betainc(4,4,xv)}")
    else:
        print("window has NO NaN.")

    section("Step 1: TDSignal → FFT (FDSignal)")
    td_settings = TDSettings(N, dt)
    td_signal = TDSignal(td_arr, td_settings)
    fd_signal = td_signal.fft(settings=None, window=np.ones_like(td_arr))
    report_array("fd_signal.arr", fd_signal.arr)

    section("Step 2: FDSignal.wdmtransform — manual walk through")
    settings = wdm_settings
    nch = fd_signal.arr.shape[0]

    m = np.repeat(np.arange(0, settings.Nf)[:, None], settings.Nt, axis=-1)
    n = np.tile(np.arange(settings.Nt), (settings.Nf, 1))
    m_special = np.repeat(np.arange(0, settings.Nf + 1)[:, None], settings.Nt, axis=-1)
    k = settings.get_shift_map(m_special)
    neg_k = (k < 0)
    over_k = (k > int(settings.N / 2))
    k[neg_k] = np.abs(k[neg_k])
    k[over_k] = settings.N - k[over_k]
    base_window = settings.window[:]
    print(f"base_window: nan={int(np.isnan(base_window).sum())}, shape={base_window.shape}")
    print(f"k: shape={k.shape}, min={int(k.min())}, max={int(k.max())}, "
          f"fd_signal.arr last axis len={fd_signal.arr.shape[-1]}")

    arr_in = fd_signal.arr.copy()
    before_ifft = arr_in[:, k] / settings.data_dt
    report_array("before_ifft (pre-hermitian)", before_ifft)

    herm = neg_k | over_k
    if herm.any():
        before_ifft[:, herm] = np.conj(before_ifft[:, herm])
    report_array("before_ifft (post-hermitian)", before_ifft)

    before_ifft[:] *= base_window[None, None, :]
    report_array("before_ifft (post-window mul)", before_ifft)

    after_ifft = np.fft.ifft(before_ifft, axis=-1)
    report_array("after_ifft", after_ifft)

    section("Compare against the in-tree wdmtransform")
    wdm = fd_signal.wdmtransform(settings=settings)
    report_array("wdm.arr (active slice)", wdm.arr)
    print(f"wdm shape: {wdm.arr.shape} "
          f"(expected ({nch},{settings.Nf_active},{settings.Nt_active}))")

    section("Load lookup table and report fields")
    try:
        table = WDMLookupTable.from_file(LOOKUP_H5, force_backend="cpu")
        print("loaded lookup table OK")
        for name in ("table_cos", "table_sin", "f_vals", "m_diffs", "norm_freq_single_layer"):
            if hasattr(table, name):
                a = np.asarray(getattr(table, name))
                print(f"  {name}: shape={a.shape} dtype={a.dtype} "
                      f"nan={int(np.isnan(a).sum()) if a.dtype.kind == 'f' else 'n/a'}")
    except Exception as exc:  # noqa: BLE001
        print(f"lookup table load failed: {exc}")


if __name__ == "__main__":
    main()
