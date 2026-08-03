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



def sighet_shared_bytes(n_nodes, n_knots, nchannels, m_half, N_sparse_t,
                        band_len, v4=True):
    """EXACT mirror of gb_sighet_v3/v4_shared_bytes in gb_tdi_on_the_fly.cu.

    Kept in lockstep with the C++ so the harness can size N_sparse_t to the
    device instead of guessing (N_PARAMS_MAX = 20 there).
    """
    M = 2 * m_half + 1
    b = 2 * 20 * 8                       # cand + ref params
    b += n_nodes * 8                     # t_nodes
    b += 2 * nchannels * n_nodes * 16    # tdi cand + ref
    b += 2 * n_nodes * 8                 # phiun c + r
    b += 2 * n_nodes * 8                 # amp_y + ph_y
    b += 2 * n_nodes * 8                 # flip + pjump
    b += n_nodes * 4 + n_nodes * 1       # count + fix_c
    b += 2 * nchannels * n_nodes * 8     # dlnA + dphi
    b += 6 * nchannels * n_nodes * 8     # spline coeffs
    n_fit_max = max(n_knots, n_nodes) if v4 else n_nodes
    b += n_fit_max * 8 + 8 * n_fit_max * 8       # B_b + pcr
    if v4:
        b += n_knots * 8                          # t_knots
        b += 2 * nchannels * n_knots * 8          # rk re/im
        if band_len <= 0:
            b += 6 * nchannels * n_knots * 8      # cR/cI
        b += nchannels * N_sparse_t * 16          # r_pix
    b += 2 * nchannels * M * N_sparse_t * 16      # r + dr
    return b + 64


def device_shared_limit(default=163840):
    """Max dynamic shared memory per block on device 0 (bytes)."""
    try:
        import cupy as cp
        return int(cp.cuda.Device(0).attributes[
            "MaxSharedMemoryPerBlockOptin"])
    except Exception:                                   # noqa: BLE001
        return default


def max_n_sparse_t(n_nodes, n_knots, nchannels, m_half, band_len, v4=True,
                   margin=0.92):
    """Largest N_sparse_t whose scorer shared footprint fits the device."""
    lim = device_shared_limit() * margin
    n = 8
    while sighet_shared_bytes(n_nodes, n_knots, nchannels, m_half, n + 8,
                              band_len, v4) < lim:
        n += 8
    return n


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
    # The v3/v4 SCORER kernels carry r + dr per (channel, active-m,
    # sparse-time) pixel: 2*3*M*N_sparse_t*16 bytes. N_sparse_t is fixed by
    # PHYSICS (Tobs / sparse interval), so 1-yr production sparse resolution
    # wants ~250 KB -- past an A100's 164 KB. Coarsen nt_layer on GPU so the
    # scorer fits (SHOOT_NSP_MAX / SHOOT_NTL override); v4 Phase-2's moment
    # contraction is what removes this ceiling for good.
    edge_n = edge
    nt_layer = 512
    if USE_GPU:
        nsp_max = int(os.environ.get(
            "SHOOT_NSP_MAX", str(max_n_sparse_t(nr, knots, 3, 2, 0, v4=True))))
        stride_need = max(1, -(-(nt - 2 * edge_n) // nsp_max))
        nt_layer = int(os.environ.get("SHOOT_NTL",
                                      str(max(1, nt // stride_need))))
    shb = (nt + 512 + nt_layer) * 16
    nsp_est = (nt - 2 * edge_n) // max(1, nt // max(1, nt_layer))
    lim = device_shared_limit() if USE_GPU else 163840
    b3 = sighet_shared_bytes(nr, knots, 3, 2, nsp_est, 0, v4=False)
    b4 = sighet_shared_bytes(nr, knots, 3, 2, nsp_est, 0, v4=True)
    b4b = sighet_shared_bytes(nr, knots, 3, 2, nsp_est, 16, v4=True)
    print(f"[grid] Nf={Nf} Nt={nt} nt_layer={nt_layer}")
    print(f"[shmem] device limit {lim/1024:.0f} KB | make_reference "
          f"{shb/1024:.0f} KB | N_sparse_t={nsp_est}: v3 {b3/1024:.0f} KB, "
          f"v4-pcr {b4/1024:.0f} KB, v4-band {b4b/1024:.0f} KB "
          f"({'OK' if max(b3, b4, b4b) <= lim else 'TOO BIG'})")
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
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=nt_layer,
        m_active_half_width=2)
    engines["v3"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=nt_layer,
        m_active_half_width=2, v3_n_nodes=nr)
    engines["v4-pcr"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=nt_layer,
        m_active_half_width=2, v3_n_nodes=nr, v4_knots=knots, v4_band=0)
    engines["v4-band"] = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=nt_layer,
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
