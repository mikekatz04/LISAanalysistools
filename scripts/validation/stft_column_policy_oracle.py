"""Byte-identity oracle + CPU micro-bench for the STFT column-producer seam.

Purpose: prove that a refactor of the STFT/Fresnel kernel internals
(`lisatools/cutils/lat_stft_kernels.hh`) changed NOTHING — every kernel output
bit-for-bit identical — and did not cost performance on CPU.

Covers all six kernel entry points through ``gbgpu.gbcomps.STFTGBComputations``
(get_ll, swap_ll, fill_global, get_ll_grad, get_swap_ll_grad, get_fstat_ll) on
three configurations that exercise every code path:

  * ``rect``      -- window_alpha=0 (unwindowed Fresnel), freq_from_tdi_phase=True
  * ``tukey_mid`` -- window_alpha=0.3 (7-term windowed path), use_midpoint=True
  * ``astro``     -- window_alpha=0, freq_from_tdi_phase=False (astro fallback)

The source batch includes carriers near the active-band edges so the
out-of-band side-bin clamps are exercised.

Usage:
    # BEFORE the refactor (current build):
    python stft_column_policy_oracle.py --capture /tmp/stft_oracle_pre.npz
    python stft_column_policy_oracle.py --bench

    # AFTER the refactor + rebuild of BOTH wheels:
    python stft_column_policy_oracle.py --compare /tmp/stft_oracle_pre.npz
    python stft_column_policy_oracle.py --bench

``--compare`` exits 0 only if every array is bit-for-bit identical
(``np.array_equal``); otherwise it reports per-array max-abs and max-ulp
differences and exits 1.
"""

import argparse
import sys
import time

import numpy as np

import lisatools
from gbgpu.gbcomps import STFTGBComputations
from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import STFTSettings

NCH = 3
SEED = 20260710


def _shim_group(settings, window_alpha, use_midpoint, data, invC, backend_name="cpu"):
    """Minimal stft_comps stand-in (the STFTEngineAccuracy pattern): raw
    STFTDomainWrap/STFTFresnelWrap around caller-OWNED complex128 buffers
    (dangling-pointer rule: keep them alive on the shim)."""
    backend = lisatools.get_backend(backend_name)
    tdi_type = backend.TDITypeDict["XYZ"]
    NT, NF = settings.NT, settings.NF_active
    s = settings
    assert data.dtype == np.complex128 and invC.dtype == np.complex128
    assert data.flags["C_CONTIGUOUS"] and invC.flags["C_CONTIGUOUS"]
    domain = backend.STFTDomainWrap(NT, NF, NCH, s.t0, s.min_freq, s.max_freq,
                                    s.dt, s.df, data.reshape(-1),
                                    invC.reshape(-1), 1, 1, tdi_type)
    fres = backend.STFTFresnelWrap(NT, NF, NCH, s.t0, s.min_freq, s.max_freq,
                                   s.dt, s.df, window_alpha=window_alpha,
                                   use_midpoint=use_midpoint)
    shim = type("_STFTShim", (), {})()
    shim.cpp_fresnel, shim.cpp_domain, shim.d_d = fres, domain, None
    shim._keepalive = (data, invC)
    return shim


def build(n_stft=16, nf=128, big_dt=21600.0, ind_lo=40, ind_hi=100,
          window_alpha=0.0, use_midpoint=False, freq_from_tdi_phase=True,
          n_side_bins=4):
    settings = STFTSettings(
        t0=10.0 * 86400.0, dt=big_dt, df=1.0 / big_dt, NT=n_stft, NF=nf,
        min_freq=ind_lo / big_dt, max_freq=ind_hi / big_dt, force_backend="cpu",
    )
    NT, NF = settings.NT, settings.NF_active
    rng = np.random.default_rng(SEED)
    # Deterministic non-trivial data + Hermitian-ish invC so every
    # add_ip_contrib term (cross-channel included) is exercised.
    data = np.ascontiguousarray(
        1e-20 * (rng.standard_normal((1, NCH, NT, NF))
                 + 1j * rng.standard_normal((1, NCH, NT, NF))))
    invC = np.zeros((1, NCH, NCH, NT, NF), np.complex128)
    for a in range(NCH):
        invC[0, a, a] = 1e40 * (1.0 + 0.1 * rng.random((NT, NF)))
        for b in range(a + 1, NCH):
            off = 1e39 * (rng.random((NT, NF)) - 0.5
                          + 1j * (rng.random((NT, NF)) - 0.5))
            invC[0, a, b] = off
            invC[0, b, a] = np.conj(off)
    invC = np.ascontiguousarray(invC)

    shim = _shim_group(settings, window_alpha, use_midpoint, data, invC)
    orbits = EqualArmlengthOrbits(force_backend="cpu")
    comp = STFTGBComputations(
        stft_comps=shim, T=NT * big_dt, t_ref=0.0, orbits=orbits,
        tdi_config="2nd generation", force_backend="cpu",
        n_side_bins=n_side_bins, window_factor=1.0,
        freq_from_tdi_phase=freq_from_tdi_phase,
    )
    return settings, comp


def make_params(settings, num=24):
    """Deterministic 9-param physical batch; two carriers pinned near the
    active-band edges so side-bin clamping paths run."""
    rng = np.random.default_rng(SEED + 1)
    df = settings.df
    lo, hi = settings.ind_min, settings.ind_max
    f0 = (lo + 6 + (hi - lo - 12) * rng.random(num)) * df
    f0[0] = (lo + 1) * df          # left edge: side bins clamp low
    f0[1] = (hi - 1) * df          # right edge: side bins clamp high
    amp = 10 ** rng.uniform(-21.5, -20.5, num)
    fdot = rng.uniform(-1e-15, 1e-15, num)
    phi0 = rng.uniform(0, 2 * np.pi, num)
    iota = rng.uniform(0.1, np.pi - 0.1, num)
    psi = rng.uniform(0, np.pi, num)
    lam = rng.uniform(0, 2 * np.pi, num)
    beta = rng.uniform(-1.2, 1.2, num)
    return np.stack([amp, f0, fdot, np.zeros(num), phi0, iota, psi, lam, beta],
                    axis=-1)


CONFIGS = {
    "rect": dict(window_alpha=0.0, use_midpoint=False, freq_from_tdi_phase=True),
    "tukey_mid": dict(window_alpha=0.3, use_midpoint=True, freq_from_tdi_phase=True),
    "astro": dict(window_alpha=0.0, use_midpoint=False, freq_from_tdi_phase=False),
}


def run_all(cfg_kwargs):
    settings, comp = build(**cfg_kwargs)
    p = make_params(settings)
    num = p.shape[0]
    p_swap = np.roll(p, 3, axis=0)          # deterministic partner set
    di0 = np.zeros(num, dtype=np.int32)

    out = {}
    ll = comp.get_ll_stft(p, data_index=di0, noise_index=di0)
    out["ll"] = np.asarray(ll)
    out["d_h"] = np.asarray(comp.d_h_out).copy()
    out["h_h"] = np.asarray(comp.h_h_out).copy()

    res = comp.get_swap_ll_stft(p, p_swap, data_index=di0, noise_index=di0)
    for name, arr in zip(
            ["sw_like_a", "sw_like_r", "sw_dha", "sw_dhr", "sw_aa", "sw_rr",
             "sw_ar"], res):
        out[name] = np.asarray(arr).copy()

    tpl = np.zeros((2, NCH, settings.NT, settings.NF_active), np.complex128)
    fill_idx = np.ascontiguousarray(np.arange(num, dtype=np.int32) % 2)
    factors = np.where(np.arange(num) % 2 == 0, 1.0, -0.5)
    comp.fill_global_stft(p, tpl, data_index=fill_idx,
                          factors=np.ascontiguousarray(factors))
    out["fill"] = tpl

    out["grad"] = np.asarray(comp.get_ll_grad_stft(
        p, data_index=di0, noise_index=di0)).copy()
    ga, gr = comp.get_swap_ll_grad_stft(p, p_swap, data_index=di0,
                                        noise_index=di0)
    out["gswap_a"] = np.asarray(ga).copy()
    out["gswap_r"] = np.asarray(gr).copy()

    n_f = 8  # fstat = (4 ll + 6 swap) block evals per source; keep it light
    N_re, M_re = comp.get_fstat_ll_stft(p[:n_f], data_index=di0[:n_f],
                                        noise_index=di0[:n_f])
    out["fstat_N"] = np.asarray(N_re).copy()
    out["fstat_M"] = np.asarray(M_re).copy()
    out["fstat_N_cmplx"] = np.asarray(comp.N_arr_cmplx).copy()
    out["fstat_M_cmplx"] = np.asarray(comp.M_mat_cmplx).copy()
    return out


def capture(path):
    blob = {}
    for cname, ckw in CONFIGS.items():
        res = run_all(ckw)
        for k, v in res.items():
            blob[f"{cname}/{k}"] = v
        print(f"[capture] {cname}: {len(res)} arrays "
              f"(|d_h|_0={abs(blob[f'{cname}/d_h'][0]):.6e})")
    np.savez(path, **blob)
    print(f"[capture] wrote {path} ({len(blob)} arrays)")


def compare(path):
    ref = np.load(path)
    n_bad = 0
    for cname, ckw in CONFIGS.items():
        res = run_all(ckw)
        for k, v in res.items():
            key = f"{cname}/{k}"
            r = ref[key]
            if np.array_equal(r, v):
                continue
            n_bad += 1
            d = np.abs(np.asarray(v) - r)
            with np.errstate(all="ignore"):
                rel = np.nanmax(d / np.maximum(np.abs(r), 1e-300))
            print(f"[DIFF] {key}: max|diff|={d.max():.3e}  max rel={rel:.3e}  "
                  f"n_diff={(d > 0).sum()}/{d.size}")
    if n_bad:
        print(f"[compare] FAILED: {n_bad} arrays differ")
        return 1
    print("[compare] BYTE-IDENTICAL: all arrays equal")
    return 0


def bench(reps=7):
    """CPU micro-bench (medians) on a heavier grid; run pre + post refactor."""
    for cname in ("rect", "tukey_mid"):
        ckw = dict(CONFIGS[cname], n_stft=64, n_side_bins=4)
        settings, comp = build(**ckw)
        p = np.repeat(make_params(settings), 6, axis=0)   # 144 sources
        num = p.shape[0]
        di = np.zeros(num, dtype=np.int32)
        p_swap = np.roll(p, 3, axis=0)
        tpl = np.zeros((2, NCH, settings.NT, settings.NF_active), np.complex128)
        fill_idx = np.ascontiguousarray(np.arange(num, dtype=np.int32) % 2)

        comp.get_ll_stft(p, data_index=di, noise_index=di)  # warm-up
        t_ll, t_sw, t_fill = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            comp.get_ll_stft(p, data_index=di, noise_index=di)
            t_ll.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            comp.get_swap_ll_stft(p, p_swap, data_index=di, noise_index=di)
            t_sw.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            comp.fill_global_stft(p, tpl, data_index=fill_idx)
            t_fill.append(time.perf_counter() - t0)
        print(f"[bench {cname}] num_bin={num} NT={settings.NT} "
              f"n_side={comp.n_side_bins}: "
              f"get_ll {np.median(t_ll)*1e3:8.2f} ms | "
              f"swap {np.median(t_sw)*1e3:8.2f} ms | "
              f"fill {np.median(t_fill)*1e3:8.2f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", metavar="NPZ")
    ap.add_argument("--compare", metavar="NPZ")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args(argv)
    rc = 0
    if args.capture:
        capture(args.capture)
    if args.compare:
        rc = compare(args.compare)
    if args.bench:
        bench()
    if not (args.capture or args.compare or args.bench):
        ap.error("pick at least one of --capture/--compare/--bench")
    return rc


if __name__ == "__main__":
    sys.exit(main())
