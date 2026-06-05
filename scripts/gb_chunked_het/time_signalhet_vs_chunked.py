#!/usr/bin/env python
"""Wall-clock timing for signal-het vs chunked-het kernels.

Times ``fill_global`` and ``get_ll`` on each backend, batched over
``NUM_BIN`` candidates per call. Reuses the kernel-construction code in
``compare_signalhet_vs_chunked_mcmc.build_pack`` so this script adds
nothing but the timing harness.

Backend-switchable: ``BACKEND=cpu`` / ``cuda11x`` / ``cuda12x`` /
``cuda13x``. Defaults to ``cpu``. The pack constructor imports the
matching ``fastlisaresponse_backend_*`` so both kernel families run on
the same device.

Env knobs:
    BACKEND      default "cpu"
    NUM_BIN      default 16
    REPEATS      default 20
    F0_MHZ       default 14.22
    SNR_TARGET   default 50.0
    SEED         default 42
    NT_LAYER     default 64
    N_SPARSE_FD  default 1024
    NT_SUB       default 256
    N_SPARSE     default 256
    N_PAD        default NT_SUB // 8
    N_CP_SIG     default 48
    N_CP_ORBIT   default 32
    TUKEY_ALPHA  default 0.05    (kept in validated [0.01, 0.05] range)
    MAX_R        default 5.0
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from compare_signalhet_vs_chunked_mcmc import build_pack


def _sync(is_gpu: bool):
    if not is_gpu:
        return
    try:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _timer(fn, repeats: int, is_gpu: bool, warmup: int = 2):
    """Run ``fn()`` ``warmup + repeats`` times; return per-call times (s)."""
    for _ in range(warmup):
        fn()
        _sync(is_gpu)
    out = np.empty(repeats, dtype=np.float64)
    for k in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(is_gpu)
        out[k] = time.perf_counter() - t0
    return out


def _stats(label: str, ts_s: np.ndarray, num_bin: int) -> dict:
    ts_ms = ts_s * 1e3
    return dict(
        label=label,
        median_ms=float(np.median(ts_ms)),
        p5_ms=float(np.percentile(ts_ms, 5)),
        p95_ms=float(np.percentile(ts_ms, 95)),
        min_ms=float(np.min(ts_ms)),
        per_source_us=float(np.median(ts_ms)) * 1e3 / num_bin,
    )


def main():
    BACKEND_NAME = os.environ.get("BACKEND", "cpu")
    NUM_BIN      = int(os.environ.get("NUM_BIN", "16"))
    REPEATS      = int(os.environ.get("REPEATS", "20"))
    F0_MHZ       = float(os.environ.get("F0_MHZ", "14.22"))
    SNR_TARGET   = float(os.environ.get("SNR_TARGET", "50.0"))
    SEED         = int(os.environ.get("SEED", "42"))
    NT_LAYER     = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD  = int(os.environ.get("N_SPARSE_FD", "1024"))
    NT_SUB       = int(os.environ.get("NT_SUB", "256"))
    N_SPARSE     = int(os.environ.get("N_SPARSE", "256"))
    N_PAD_ENV    = os.environ.get("N_PAD", "")
    N_PAD        = int(N_PAD_ENV) if N_PAD_ENV else None
    N_CP_SIG     = int(os.environ.get("N_CP_SIG", "48"))
    N_CP_ORBIT   = int(os.environ.get("N_CP_ORBIT", "32"))
    TUKEY_ALPHA  = float(os.environ.get("TUKEY_ALPHA", "0.05"))
    MAX_R        = float(os.environ.get("MAX_R", "5.0"))

    # All construction goes through the shared helper.
    pack = build_pack(
        backend_name=BACKEND_NAME,
        f0_mhz=F0_MHZ, snr_target=SNR_TARGET, seed=SEED,
        nt_layer=NT_LAYER, n_sparse_fd=N_SPARSE_FD,
        nt_sub=NT_SUB, n_sparse=N_SPARSE, n_pad=N_PAD,
        n_cp_sig=N_CP_SIG, n_cp_orbit=N_CP_ORBIT,
        tukey_alpha=TUKEY_ALPHA, max_r=MAX_R,
    )
    xp = pack.xp
    is_gpu = pack.is_gpu

    # Candidate batch -- NUM_BIN samples drawn near the injection. We don't
    # need a valid MCMC step; we just need realistic-ish input shapes.
    rs = np.random.RandomState(SEED + 1)
    spread = np.array([
        pack.amp_inj * 1e-3, 1e-3 * pack.layer_df, 1e-18, 0.0,
        1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
    ])
    cand_host = (pack.params_inj[None, :]
                 + spread[None, :] * rs.standard_normal((NUM_BIN, 9)))
    cand_host[:, 3] = 0.0  # fddot frozen
    params_cand_all = xp.asarray(cand_host.astype(np.float64))
    data_index_all  = xp.zeros(NUM_BIN, dtype=np.int32)
    factors_all     = xp.ones(NUM_BIN, dtype=np.float64)

    # Pre-allocated output buffers (zeroed before each fill_global call).
    d_h_out          = xp.zeros(NUM_BIN, dtype=np.float64)
    h_h_out          = xp.zeros(NUM_BIN, dtype=np.float64)
    template_fill_sh = xp.zeros((1, 3, pack.Nf, pack.Nt), dtype=np.float64)
    template_fill_ch = xp.zeros((3, pack.Nf, pack.Nt), dtype=np.float64)

    # ----- four zero-arg closures bound to the actual kernel calls -----
    def call_signalhet_get_ll():
        pack.cpp.gb_signal_het_get_ll_in_kernel(
            pack.tdi_wrap, d_h_out, h_h_out,
            pack.c0_sparse_all,
            pack.A0_all, pack.A1_all, pack.B0_all, pack.B1_all,
            pack.window_full, pack.n_sparse_local,
            params_cand_all, pack.params_ref_all, data_index_all,
            NUM_BIN, 1, 9, 1, 2,
            pack.Nf, pack.Nt, pack.Nf_active, pack.Nt_active,
            pack.Nt_layer, pack.N_sparse_t, pack.stride,
            pack.ind_min_t, pack.ind_min_f, 2,
            pack.layer_df, pack.dt, pack.Tobs, pack.t_start,
            3, 0, pack.n_sparse_fd,
            pack.tukey_alpha, pack.max_r,
        )

    def call_signalhet_fill_global():
        template_fill_sh[:] = 0.0
        pack.cpp.gb_signal_het_fill_global_in_kernel(
            pack.tdi_wrap, template_fill_sh,
            pack.c0_sparse_all, pack.c0_dense_cmplx_all,
            pack.window_full, pack.n_sparse_local,
            params_cand_all, pack.params_ref_all,
            factors_all, data_index_all,
            NUM_BIN, 1, 9, 1, 2,
            pack.Nf, pack.Nt, pack.Nf_active, pack.Nt_active,
            pack.Nt_layer, pack.N_sparse_t, pack.stride,
            pack.ind_min_t, pack.ind_min_f, 2,
            pack.layer_df, pack.dt, pack.Tobs, pack.t_start,
            3, pack.n_sparse_fd, pack.tukey_alpha, pack.max_r,
        )

    def call_chunked_get_ll():
        pack.gb_wdm_comp.get_ll_wdm(
            params_cand_all, pack.holder,
            convert_to_ra_dec=False,
            use_layer_groups=True, group_band_layers=5, margin_layers=0,
        )

    def call_chunked_fill_global():
        template_fill_ch[:] = 0.0
        pack.gb_wdm_comp.fill_global_wdm(
            params_cand_all, template_fill_ch,
            convert_to_ra_dec=False, factors=None,
        )

    # ----- time each kernel -----
    print(f"\n[time] REPEATS={REPEATS}  NUM_BIN={NUM_BIN}  "
          f"warmup=2 implicit", flush=True)
    rows = []
    for label, fn in [
        ("signal-het  get_ll",       call_signalhet_get_ll),
        ("signal-het  fill_global",  call_signalhet_fill_global),
        ("chunked-het get_ll",       call_chunked_get_ll),
        ("chunked-het fill_global",  call_chunked_fill_global),
    ]:
        try:
            ts = _timer(fn, REPEATS, is_gpu)
        except Exception as e:
            print(f"[time] {label:<26s} FAILED: {e}", flush=True)
            rows.append(dict(label=label, median_ms=float("nan"),
                             p5_ms=float("nan"), p95_ms=float("nan"),
                             min_ms=float("nan"),
                             per_source_us=float("nan")))
            continue
        rows.append(_stats(label, ts, NUM_BIN))
        r = rows[-1]
        print(f"[time] {label:<26s}  "
              f"median={r['median_ms']:7.2f} ms  "
              f"min={r['min_ms']:7.2f} ms  "
              f"p95={r['p95_ms']:7.2f} ms  "
              f"per-source={r['per_source_us']:7.1f} us",
              flush=True)

    # ----- summary tables -----
    print(f"\n## Timing summary  ({pack.backend_name},  NUM_BIN={NUM_BIN},  "
          f"REPEATS={REPEATS})\n", flush=True)
    print(f"| kernel                     | median ms |   min ms |   p95 ms |"
          f" per-source us |", flush=True)
    print(f"|----------------------------|----------:|---------:|---------:|"
          f"--------------:|", flush=True)
    for r in rows:
        print(f"| {r['label']:<26s} | "
              f"{r['median_ms']:9.2f} | "
              f"{r['min_ms']:8.2f} | "
              f"{r['p95_ms']:8.2f} | "
              f"{r['per_source_us']:13.1f} |", flush=True)

    by_label = {r["label"]: r for r in rows}
    sh_get = by_label.get("signal-het  get_ll")
    ch_get = by_label.get("chunked-het get_ll")
    sh_fil = by_label.get("signal-het  fill_global")
    ch_fil = by_label.get("chunked-het fill_global")
    print(f"\n## Speedup (chunked-het / signal-het)", flush=True)
    if sh_get and ch_get and np.isfinite(sh_get["median_ms"]) \
                          and np.isfinite(ch_get["median_ms"]):
        print(f"  get_ll       : "
              f"{ch_get['median_ms'] / sh_get['median_ms']:6.2f}x", flush=True)
    if sh_fil and ch_fil and np.isfinite(sh_fil["median_ms"]) \
                          and np.isfinite(ch_fil["median_ms"]):
        print(f"  fill_global  : "
              f"{ch_fil['median_ms'] / sh_fil['median_ms']:6.2f}x", flush=True)


if __name__ == "__main__":
    sys.exit(main())
