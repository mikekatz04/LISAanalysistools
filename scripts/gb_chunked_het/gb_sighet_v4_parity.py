"""Parity gate for sig-het V4 (fixed-knot ratio evaluation), 2026-08-02.

Three-way per-candidate delta comparison at the 1-yr tier-assess scaffold
(seed-19 refs, refs {0,2,5} by default, ~10 candidates per ref):

  * v3-compiled : gb_signal_het_v3_get_ll  (node-eval log-polar ratio fit,
                  spline evaluated DIRECTLY at the pixel times)
  * v4-compiled : gb_signal_het_v4_get_ll  (same fit, resampled onto K
                  FIXED uniform knots as linear complex values, pixel
                  evaluation through the fixed-knot not-a-knot spline)
  * v4-python   : the tier script's ``v4_delta`` path (proto.fit_ratio on
                  the slow series + scipy cardinal-spline resample + the
                  same mask/fd_dr/fold), evaluated at the KERNEL's pixel
                  convention (pix-off = 0; the tier script's calibrated
                  off is reported separately as a sensitivity check).

Decomposition (all three share the stash + fold):
  |v4C - v3C|  isolates the RESAMPLING stage (identical C++ fit);
  |v4C - v4P|  is dominated by the FIT implementation difference
               (C++ get_tdi_raw node evals + wdm_fit_cubic_spline vs
               Python slow-series ratio + scipy CubicSpline).

Deltas are RAW: delta = d_h - 0.5*h_h (shared reference stash, so no
reference-point subtraction is needed for parity).

GOLDEN mode (pre-refactor guard): GOLDEN=1 computes ONLY the v3-compiled
deltas with the currently installed .so and saves them to
ENV_OUT/v4_parity_v3_golden.npz. A later full run reloads the file and
reports max |v3_now - v3_golden| -- proves the v3 fit-stage refactor
(shared helper extraction) left v3 numerics untouched.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_v4_parity.py
Env: V4_NR (64 fit nodes), V4_K (128 knots), V4_REF_SEL ("0,2,5"),
     GOLDEN (0), ENV_OUT (./ratio_proto_out), BACKEND (cpu)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations


# ---------------------------------------------------------------------------
# Scaffold -- copied from gb_sighet_tier_assess.py so the two scripts are
# always on the same grid/config. Shared with gb_sighet_speed_shootout.py.
# ---------------------------------------------------------------------------

class XpGridWDMHolder:
    """xp-aware variant of proto._FullGridWDMHolder (device-resident slabs
    on CUDA so setup_in_model never round-trips through host)."""

    def __init__(self, xp, data_full, invC_full):
        self.linear_data_arr = [xp.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(invC_full).ravel()]

    def __len__(self):
        return 1


def make_refs(n_ref=8):
    """Seed-19 reference draws, verbatim from gb_sighet_tier_assess.py."""
    rngr = np.random.default_rng(19)
    refs = []
    for _ in range(n_ref):
        refs.append(np.array([
            10 ** rngr.uniform(-22.5, -21.0),
            rngr.uniform(1.5e-3, 1.5e-2),
            rngr.uniform(0.0, 3e-16),
            0.0,
            rngr.uniform(0, 2 * np.pi),
            np.arccos(rngr.uniform(-1, 1)),
            rngr.uniform(0, np.pi),
            rngr.uniform(0, 2 * np.pi),
            np.arcsin(rngr.uniform(-1, 1)),
        ]))
    return refs


def sighet_shared_bytes(n_nodes, n_knots, nchannels, m_half, N_sparse_t,
                        band_len, v4=True):
    """EXACT mirror of gb_sighet_v3/v4_shared_bytes (gb_tdi_on_the_fly.cu).

    Keep in lockstep with the C++ (N_PARAMS_MAX = 20 there) so the harness
    sizes N_sparse_t to the device instead of guessing.
    """
    M = 2 * m_half + 1
    b = 2 * 20 * 8
    b += n_nodes * 8
    b += 2 * nchannels * n_nodes * 16
    b += 2 * n_nodes * 8
    b += 2 * n_nodes * 8
    b += 2 * n_nodes * 8
    b += n_nodes * 4 + n_nodes * 1
    b += 2 * nchannels * n_nodes * 8
    b += 6 * nchannels * n_nodes * 8
    n_fit_max = max(n_knots, n_nodes) if v4 else n_nodes
    b += n_fit_max * 8 + 8 * n_fit_max * 8
    if v4:
        b += n_knots * 8
        b += 2 * nchannels * n_knots * 8
        if band_len <= 0:
            b += 6 * nchannels * n_knots * 8
        b += nchannels * N_sparse_t * 16
    b += 2 * nchannels * M * N_sparse_t * 16
    return b + 64


def device_shared_limit(default=163840):
    """Device-0 max dynamic shared memory per block (bytes)."""
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


def make_scaffold(backend="cpu", *, v3_n_nodes=64, v4_knots=128,
                  Nt=None, Nf=None, nt_layer=512, n_sparse_fd=512,
                  edge=None, tk=None):
    """The tier-assess scaffold; GRID IS CONFIGURABLE (V4_NT / V4_NF).

    GPU NOTE: gb_signal_het_make_reference_kernel asks for
    ``(Nt + n_sparse_fd + nt_layer) * 16`` bytes of dynamic shared memory,
    so the CPU-convenient tall-and-thin grid (Nf=256, Nt=12288 -> 208 KB)
    exceeds an A100's 164 KB cap. The PRODUCTION grid shape is the
    opposite (Nf=1440, Nt=2160 -> ~51 KB) and fits comfortably; default
    to that under a non-cpu backend.
    """
    dt = 10.0
    gpu = backend != "cpu"
    if Nt is None:
        Nt = int(os.environ.get("V4_NT", "2160" if gpu else "12288"))
    if Nf is None:
        Nf = int(os.environ.get("V4_NF", "1440" if gpu else "256"))
    if edge is None:
        edge = int(os.environ.get("V4_EDGE", str(max(2, Nt // 40))))
    if tk is None:
        tk = int(os.environ.get("V4_TK", str(max(2, int(0.025 * Nt)))))
    # The SCORER kernels (v3/v4) carry r + dr for every (channel, active-m,
    # sparse-time) pixel: 2*3*M*N_sparse_t*16 bytes, and N_sparse_t is set by
    # PHYSICS (Tobs / sparse-sampling interval), not by grid shape -- so a
    # 1-yr run with production sparse resolution wants ~250 KB and does not
    # fit an A100's 164 KB. Cap N_sparse_t on GPU by coarsening nt_layer
    # (V4_NSP_MAX / V4_NTL override). This is the constraint that the v4
    # Phase-2 moment contraction is designed to remove entirely.
    if gpu:
        # Size N_sparse_t to the DEVICE with the exact C++ formula, fitting
        # the heaviest variant (v4-PCR).
        nsp_max = int(os.environ.get(
            "V4_NSP_MAX",
            str(max_n_sparse_t(v3_n_nodes, v4_knots, 3, 2, 0, v4=True))))
        stride_need = max(1, -(-(Nt - 2 * edge) // nsp_max))
        nt_layer = int(os.environ.get("V4_NTL", str(max(1, Nt // stride_need))))
    shb = (Nt + n_sparse_fd + nt_layer) * 16
    nsp_est = (Nt - 2 * edge) // max(1, Nt // max(1, nt_layer))
    lim = device_shared_limit() if gpu else 163840
    b3 = sighet_shared_bytes(v3_n_nodes, v4_knots, 3, 2, nsp_est, 0, v4=False)
    b4 = sighet_shared_bytes(v3_n_nodes, v4_knots, 3, 2, nsp_est, 0, v4=True)
    b4b = sighet_shared_bytes(v3_n_nodes, v4_knots, 3, 2, nsp_est, 16, v4=True)
    print(f"[grid] Nf={Nf} Nt={Nt} nt_layer={nt_layer} edge={edge} taper={tk}")
    print(f"[shmem] device limit {lim/1024:.0f} KB | make_reference "
          f"{shb/1024:.0f} KB | N_sparse_t={nsp_est}: v3 {b3/1024:.0f} KB, "
          f"v4-pcr {b4/1024:.0f} KB, v4-band {b4b/1024:.0f} KB "
          f"({'OK' if max(b3, b4, b4b) <= lim else 'TOO BIG'})")
    t_start = int(0.5 * YRSID_SI / dt) * dt
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4,
                          max_freq=2e-2, min_time=edge * Nf * dt,
                          max_time=(Nt - edge) * Nf * dt,
                          force_backend=backend)
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start, Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend, d_d=0.0,
        tdi_type="XYZ", tukey_alpha=2.0 * tk / Nt)
    chunked.convert_to_ra_dec = False
    sighet = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=n_sparse_fd, n_cp_build=93, nt_layer=nt_layer,
        m_active_half_width=2, v3_n_nodes=v3_n_nodes, v4_knots=v4_knots)
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    xp = sighet.xp
    invC = xp.ascontiguousarray(
        xp.asarray(XYZ2SensitivityMatrix(wdm_set, model="scirdv1").invC,
                   dtype=xp.float64))
    return dict(chunked=chunked, sighet=sighet, wdm_set=wdm_set, invC=invC,
                ilo=int(ilo), ihi=int(ihi), Nf=Nf, Nt=Nt, dt=dt,
                t_start=t_start, xp=xp)


def setup_ref(sc, ref):
    """Inject the reference + build the in-model sig-het stash for it."""
    xp = sc["xp"]
    href = xp.zeros((3, sc["Nf"], sc["Nt"]))
    sc["chunked"].fill_global_wdm(xp.asarray(ref)[None, :], href,
                                  convert_to_ra_dec=False)
    h_act = href[:, sc["ilo"]:sc["ihi"], sc["wdm_set"].active_slice_t]
    holder = XpGridWDMHolder(xp, h_act, sc["invC"])
    sc["sighet"].clear_in_model()
    sc["sighet"].setup_in_model(holder, xp.asarray(ref)[None, :],
                                np.zeros(1, dtype=np.int32))
    return holder


DIRS = [("f0", 1), ("lnA", 0), ("iota", 5), ("psi", 6), ("lam", 7)]


def displace(ref, idx, s):
    p = ref.copy()
    if idx == 0:
        p[0] *= np.exp(s)
    else:
        p[idx] += s
    return p


def make_candidates(ref, Tobs):
    """10 deterministic trust-region candidates: the tier script's s0 per
    direction at scales {1x, 3x}."""
    drift_u = 1.0 / (2 * np.pi * Tobs)
    s0 = {"f0": 0.05 * drift_u, "lnA": 0.03, "iota": 0.03, "psi": 0.03,
          "lam": 0.01}
    cands = []
    for name, idx in DIRS:
        for scale in (1.0, 3.0):
            cands.append((f"{name}x{scale:g}",
                          displace(ref, idx, scale * s0[name])))
    return cands



def asnp(a):
    """Host numpy view of a numpy OR cupy array (GPU-safe).

    The compiled engines return device arrays under a CUDA backend, and
    ``np.asarray`` refuses to implicitly copy those -- everything that
    leaves the kernel for host-side arithmetic goes through here.
    """
    get = getattr(a, "get", None)
    return get() if callable(get) else np.asarray(a)


def compiled_delta(sighet, p, *, v3_n_nodes=0, v4_knots=0):
    """Raw delta d_h - 0.5*h_h through the compiled kernels, selected by
    temporarily setting the routing knobs (v4 wins when both are set)."""
    g = sighet._g
    saved = (g["v3_n_nodes"], g["v4_knots"])
    g["v3_n_nodes"], g["v4_knots"] = int(v3_n_nodes), int(v4_knots)
    try:
        sighet.get_ll(np.asarray(p)[None, :],
                      data_index=np.zeros(1, dtype=np.int32))
    finally:
        g["v3_n_nodes"], g["v4_knots"] = saved
    dh = float(asnp(sighet.last_d_h)[0])
    hh = float(asnp(sighet.last_h_h)[0])
    return dh - 0.5 * hh


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    golden_path = os.path.join(out_dir, "v4_parity_v3_golden.npz")
    golden_mode = int(os.environ.get("GOLDEN", "0")) == 1
    backend = os.environ.get("BACKEND", "cpu")
    nr = int(os.environ.get("V4_NR", "64"))
    K = int(os.environ.get("V4_K", "128"))
    ref_sel = [int(x) for x in
               os.environ.get("V4_REF_SEL", "0,2,5").split(",")]
    # The python reference arm runs on HOST arrays (proto.fold_py /
    # slow_series); under a CUDA backend the stash lives on the device, so
    # it is skipped by default there. v4-compiled vs v3-compiled is the
    # comparison that validates the CUDA kernel.
    skip_py = os.environ.get(
        "V4_SKIP_PY", "0" if backend == "cpu" else "1") == "1"
    if skip_py:
        print("[mode] python arm SKIPPED (compiled v3 vs compiled v4 only)")

    sc = make_scaffold(backend=backend, v3_n_nodes=nr, v4_knots=K)
    sighet = sc["sighet"]
    g = sighet._g
    N = g["n_sparse_fd"]
    tau_slow = np.arange(N) * (g["Tobs"] / N)
    refs = make_refs(8)

    from scipy.interpolate import CubicSpline as _CS
    # Fixed-knot grid + KERNEL-convention pixel times (pix-off = 0):
    # tau_b = (ind_min_t + n_sparse_local[b]) * Nf * dt, identical to the
    # compiled v3/v4 evaluation time t - t_start.
    if not skip_py:
        n_sl = asnp(sighet.n_sparse_local)
        n_global = g["ind_min_t"] + n_sl
        tau_pix0 = n_global.astype(float) * g["Nf"] * g["dt"]
        tau_k = np.linspace(0.0, g["Tobs"], K)
        Lm = np.ascontiguousarray(_CS(tau_k, np.eye(K), axis=0)(tau_pix0))

    has_v4 = hasattr(sighet.cpp, "gb_signal_het_v4_get_ll")
    if golden_mode:
        print("[golden] v3-compiled ONLY (pre-refactor .so); saving to",
              golden_path)
    elif not has_v4:
        raise RuntimeError("compiled gb_signal_het_v4_get_ll not found -- "
                           "rebuild the GBGPU wheel first (stale .so?).")

    c0_all_holder = {}

    def python_v4_delta(p, ref_ctx):
        """The tier script's v4_delta at pix-off = 0 (kernel convention)."""
        gen_l, s_ref, good_r, kf0_r, c0_all = ref_ctx
        m_act = proto.m_active_for(p[1], g)
        s_c, _, _ = proto.slow_series(gen_l, p, None, N, g, kf0_pin=kf0_r)
        rh, _ = proto.fit_ratio(s_c / s_ref, tau_slow, nr, "cubic",
                                good=good_r)
        r_nodes = rh(tau_k)                                  # (3, K)
        rh_pix = r_nodes @ Lm.T                              # (3, Nsp)
        c0r = c0_all[:, np.asarray(m_act) - g["ind_min_f"], :]
        mx = np.abs(c0r).max(axis=-1, keepdims=True)
        mask = np.abs(c0r) > np.maximum(1e-12 * mx, 1e-300)
        r_ii = np.where(mask, rh_pix[:, None, :], 0.0 + 0.0j)
        dr_ii = proto.fd_dr(r_ii, g["stride"])
        dh, hh = proto.fold_py(r_ii, dr_ii, m_act, sighet)
        return dh - 0.5 * hh

    rows = []           # (ref, cand-name, v3C, v4C, v4P)
    golden_rows = []
    for rrr in ref_sel:
        ref = refs[rrr]
        setup_ref(sc, ref)
        print(f"\n[ref{rrr:02d}] f0={ref[1]*1e3:.4f} mHz")

        if not golden_mode and not skip_py:
            gen_l = sighet._keep_alive["gb_gen"]
            s_ref, kf0_r, _ = proto.slow_series(gen_l, ref, None, N, g)
            good_r = proto.good_samples(s_ref)
            c0_all = asnp(sighet.c0_sparse_all)[0]
            ref_ctx = (gen_l, s_ref, good_r, kf0_r, c0_all)

        for name, p in make_candidates(ref, g["Tobs"]):
            d_v3c = compiled_delta(sighet, p, v3_n_nodes=nr)
            if golden_mode:
                golden_rows.append((rrr, name, d_v3c))
                print(f"    {name:10s} v3C={d_v3c:+.10e}")
                continue
            d_v4c = compiled_delta(sighet, p, v3_n_nodes=nr, v4_knots=K)
            if skip_py:
                d_v4p = np.nan
                print(f"    {name:10s} v3C={d_v3c:+.6e} v4C={d_v4c:+.6e} "
                      f"|v4C-v3C|={abs(d_v4c-d_v3c):.3e}")
            else:
                d_v4p = python_v4_delta(p, ref_ctx)
                print(f"    {name:10s} v3C={d_v3c:+.6e} v4C={d_v4c:+.6e} "
                      f"v4P={d_v4p:+.6e} |v4C-v3C|={abs(d_v4c-d_v3c):.3e} "
                      f"|v4C-v4P|={abs(d_v4c-d_v4p):.3e}")
            rows.append((rrr, name, d_v3c, d_v4c, d_v4p))

    if golden_mode:
        np.savez(golden_path,
                 ref=np.array([r[0] for r in golden_rows]),
                 name=np.array([r[1] for r in golden_rows]),
                 v3=np.array([r[2] for r in golden_rows]))
        print(f"\n[golden] saved {len(golden_rows)} v3-compiled deltas.")
        return

    v3c = np.array([r[2] for r in rows])
    v4c = np.array([r[3] for r in rows])
    v4p = np.array([r[4] for r in rows])
    print("\n[parity] over all (ref, candidate):")
    d43 = np.abs(v4c - v3c)
    scale = max(np.abs(v3c).max(), 1e-300)
    print(f"    max |v4C - v3C| (resampling stage only) = "
          f"{d43.max():.6e}   (rel to raw scale {scale:.3e}: "
          f"{d43.max()/scale:.2e})")
    if not np.all(np.isnan(v4p)):
        print(f"    max |v4C - v4P| (compiled vs python v4) = "
              f"{np.abs(v4c - v4p).max():.6e}")
        print(f"    max |v4P - v3C|                          = "
              f"{np.abs(v4p - v3c).max():.6e}")
    scale = np.abs(v4c).max()
    print(f"    (raw |delta| scale: max {scale:.3e})")

    if os.path.exists(golden_path):
        gold = np.load(golden_path)
        gmap = {(int(a), str(b)): float(c)
                for a, b, c in zip(gold["ref"], gold["name"], gold["v3"])}
        diffs = [abs(gmap[(r[0], r[1])] - r[2]) for r in rows
                 if (r[0], r[1]) in gmap]
        if diffs:
            print(f"    max |v3C now - v3C golden| (refactor guard) = "
                  f"{max(diffs):.6e}  over {len(diffs)} candidates")
    np.savez(os.path.join(out_dir, "v4_parity.npz"),
             ref=np.array([r[0] for r in rows]),
             name=np.array([r[1] for r in rows]),
             v3c=v3c, v4c=v4c, v4p=v4p)


if __name__ == "__main__":
    main()
