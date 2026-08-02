"""GB likelihood-engine speed shootout: chunked / v2 / v3 / v4-PCR / v4-banded.

Times the per-candidate scoring cost of every engine at a MATCHED
configuration (same grid, same reference stash, same fold), over a batch-size
scan, and reports per-candidate microseconds + speedup vs the chunked-het
baseline.  Accuracy is NOT the subject here -- that is settled by
gb_sighet_tier_assess.py / gb_sighet_v4_parity.py; this script answers "how
much faster", which can only be measured on the target hardware.

Engines
  chunked : GBWDMComputations (chunked heterodyne, the production baseline)
  v2      : sig-het v2 (reference stash + per-pixel ratio fold)
  v3      : node-ratio scorer (log-polar spline evaluated at the pixels)
  v4-pcr  : fixed-knot resample, knot->pixel via the cooperative spline solve
  v4-band : fixed-knot resample, knot->pixel via precomputed cardinal weights
            (no solve, no block sync, fewer shared arrays)

Run on the GPU box:
    USE_GPU=1 GPU_BACKEND=cuda12x python gb_sighet_speed_shootout.py
CPU smoke (verifies the harness before handoff; timings are not meaningful):
    SHOOT_NT=512 SHOOT_BATCHES=1,8 python gb_sighet_speed_shootout.py

Env: SHOOT_NT (12288 = 1 yr; 512 for the smoke), SHOOT_BATCHES ("1,8,64,256"),
     SHOOT_NR (64 fit nodes), SHOOT_K (128 knots), SHOOT_BAND (16 half-band),
     SHOOT_REPS (5 timed repetitions, min reported), BACKEND (cpu|gpu via
     USE_GPU/GPU_BACKEND), ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
BACKEND = os.environ.get("GPU_BACKEND", "cuda12x") if USE_GPU else "cpu"


def sync():
    if USE_GPU:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()


def build(nt, nr, knots, band):
    # GPU NOTE: gb_signal_het_make_reference_kernel asks for
    # (Nt + n_sparse_fd + nt_layer)*16 bytes of dynamic shared memory, so a
    # tall-and-thin grid (Nf=256, Nt=12288 -> 208 KB) blows an A100's 164 KB
    # cap. The production WDM grid is the opposite shape (Nf=1440, Nt=2160
    # -> ~51 KB); default to it on GPU, keep the thin grid for CPU smokes.
    dt = 10.0
    Nf = int(os.environ.get("SHOOT_NF", "1440" if USE_GPU else "256"))
    t0 = int(0.5 * YRSID_SI / dt) * dt
    edge = max(2, int(round(0.027 * nt)))
    tk = max(2, int(round(0.025 * nt)))
    shb = (nt + 512 + 512) * 16
    print(f"[grid] Nf={Nf} Nt={nt} | make_reference shared = "
          f"{shb/1024:.0f} KB ({'OK' if shb <= 163840 else 'TOO BIG for A100'})")
    orbits = ESAOrbits(force_backend=BACKEND)
    ws = WDMSettings(Nf, nt, dt, t0=t0, min_freq=1e-4, max_freq=2e-2,
                     min_time=edge * Nf * dt, max_time=(nt - edge) * Nf * dt,
                     force_backend=BACKEND)
    chunked = GBWDMComputations(
        ws, t_ref=t0, Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=48, N_cp_orbit=32, orbits=orbits,
        tdi_config="2nd generation", force_backend=BACKEND, d_d=0.0,
        tdi_type="XYZ", tukey_alpha=2.0 * tk / nt)
    chunked.convert_to_ra_dec = False
    engines = {}
    engines["v2"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=512,
        m_active_half_width=2)
    engines["v3"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=512,
        m_active_half_width=2, v3_n_nodes=nr)
    engines["v4-pcr"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=512,
        m_active_half_width=2, v3_n_nodes=nr, v4_knots=knots, v4_band=0)
    engines["v4-band"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=512,
        m_active_half_width=2, v3_n_nodes=nr, v4_knots=knots,
        v4_band=band)
    return ws, chunked, engines, Nf, dt, t0


def main():
    nt = int(os.environ.get("SHOOT_NT", "2160" if USE_GPU else "12288"))
    batches = [int(x) for x in
               os.environ.get("SHOOT_BATCHES", "1,8,64,256").split(",")]
    nr = int(os.environ.get("SHOOT_NR", "64"))
    knots = int(os.environ.get("SHOOT_K", "128"))
    band = int(os.environ.get("SHOOT_BAND", "16"))
    reps = int(os.environ.get("SHOOT_REPS", "5"))
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)

    ws, chunked, engines, Nf, dt, t0 = build(nt, nr, knots, band)
    print(f"[cfg] backend={BACKEND} Nt={nt} nr={nr} K={knots} band={band} "
          f"reps={reps} batches={batches}")

    # reference source + data stash (shared by every engine)
    rng = np.random.default_rng(19)
    ref = np.array([1e-22, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 0.4, 2.0, 0.3])
    ilo, ihi = ws.ind_min_f, ws.ind_max_f + 1
    href = np.zeros((3, Nf, nt))
    chunked.fill_global_wdm(ref[None, :], href, convert_to_ra_dec=False)
    h_act = np.ascontiguousarray(href[:, ilo:ihi, ws.active_slice_t])
    nfa, nta = h_act.shape[1], h_act.shape[2]
    invC = np.zeros((3, 3, nfa, nta))
    for c in range(3):
        invC[c, c] = 1.0
    holder = proto._FullGridWDMHolder(h_act, invC)

    results = {}
    for nb in batches:
        # candidate batch: small production-scale scatter about the reference
        cands = np.repeat(ref[None, :], nb, axis=0)
        cands[:, 0] *= np.exp(0.01 * rng.standard_normal(nb))
        cands[:, 1] += 1e-9 * rng.standard_normal(nb)
        cands[:, 5] += 0.01 * rng.standard_normal(nb)
        z = np.zeros(nb, dtype=np.int32)
        kw = dict(data_index=z, noise_index=z, N_vals=None,
                  waveform_kwargs={})

        # chunked baseline
        engc = WDMBandLikelihoodEngine(chunked, ws, nchannels=3,
                                       tdi_channel_setup="XYZ")
        engc.get_ll(holder, cands, phase_maximize=False, **kw)   # warm
        sync()
        tc = []
        for _ in range(reps):
            s = time.perf_counter()
            engc.get_ll(holder, cands, phase_maximize=False, **kw)
            sync()
            tc.append(time.perf_counter() - s)
        base = min(tc) / nb
        results.setdefault("chunked", {})[nb] = base

        for name, sh in engines.items():
            sh.clear_in_model()
            sh.setup_in_model(holder, ref[None, :], np.zeros(1, np.int32))
            eng = WDMBandLikelihoodEngine(sh, ws, nchannels=3,
                                          tdi_channel_setup="XYZ")
            try:
                eng.get_ll(holder, cands, phase_maximize=False, **kw)
                sync()
                ts = []
                for _ in range(reps):
                    s = time.perf_counter()
                    eng.get_ll(holder, cands, phase_maximize=False, **kw)
                    sync()
                    ts.append(time.perf_counter() - s)
                results.setdefault(name, {})[nb] = min(ts) / nb
            except Exception as e:                        # noqa: BLE001
                print(f"  [{name}] batch {nb} FAILED: {type(e).__name__}: {e}")
                results.setdefault(name, {})[nb] = float("nan")

    names = ["chunked", "v2", "v3", "v4-pcr", "v4-band"]
    print("\n[per-candidate microseconds]")
    print("batch    " + "".join(f"{n:>12s}" for n in names))
    for nb in batches:
        row = "".join(f"{results[n][nb]*1e6:12.1f}" for n in names)
        print(f"{nb:5d}    {row}")
    print("\n[speedup vs chunked]")
    print("batch    " + "".join(f"{n:>12s}" for n in names))
    for nb in batches:
        b = results["chunked"][nb]
        row = "".join(f"{b/results[n][nb]:11.2f}x" for n in names)
        print(f"{nb:5d}    {row}")
    np.savez(os.path.join(out_dir, "speed_shootout.npz"),
             **{f"{n}_{nb}": results[n][nb] for n in names for nb in batches},
             batches=np.array(batches), cfg=np.array([nt, nr, knots, band]))
    print(f"\n[out] {os.path.join(out_dir, 'speed_shootout.npz')}")


if __name__ == "__main__":
    main()
