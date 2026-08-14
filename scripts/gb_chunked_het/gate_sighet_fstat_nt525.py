"""PREFLIGHT GATE 1 for submit_gf_23mo.sh -- sig-het F-stat at SIGHET_NT_LAYER=525.

ON-CLUSTER (GPU node) gate. Exact command (from the LISAanalysistools root,
in the run's environment, AFTER rebuilding the GBGPU wheel from GBGPU dev --
the fix lives in gb_tdi_on_the_fly.cu, so a stale .so re-creates the crash):

    BACKEND=cuda12x python scripts/gb_chunked_het/gate_sighet_fstat_nt525.py

(use BACKEND=cuda13x on the cuda13x env; add GATE_OUT=/path for the npz).

WHAT IT PROVES, in order:

  [A] CAPACITY -- builds sig-het F-stat references on one band at the TRUE
      23-month WDM grid shape (Nf=1440, Nt=16800, dt=2.5, Tobs=6.048e7 s)
      with nt_layer=525 (N_sparse_t=525) and runs an f0/sky sweep through
      ``gb_signal_het_fstat_get_ll``. This exercises BOTH capacity fixes:
        * gb_signal_het_make_reference at Nt=16800 needs 287 KB of shared
          memory in the historical all-shared carve -- over EVERY device --
          and now takes the global-memory twiddle-table fallback;
        * the fstat scorer's checked shared-memory opt-in
          (86 KB dynamic at N=525/mode 0; 136 KB at mode 1).
      Any failure is a named ValueError (kernel, bytes, device limit) or,
      if the 2026-08-12 nt_layers_1.log ``GPUassert: invalid argument
      gb_tdi_on_the_fly.cu:6747`` mechanism is something else entirely, the
      same GPUassert -- either way it fires HERE, not inside the 23-month
      submission.
  [B] MODE PARITY -- fstat_mode=0 (2 node stages + exact phi0 rotation,
      production) against fstat_mode=1 (4 independent stages) at 525:
      max rel |dF| <= 1e-6 (measured 6e-10 on the 90-d gate grid).
  [C] GRID CONSISTENCY at overlapping accuracy -- F at nt_layer=525 (32 h
      sparse spacing, the accuracy prescription) against nt_layer=420
      (40 h, adjacent to the validated ~35 h density): median rel |dF|
      <= 1e-2, max <= 3e-2 over the sweep. A capacity bug that corrupts
      (rather than kills) the big-N launch cannot pass this.
  [D] INFO ONLY -- F at the legacy-default landing spot nt_layer=60
      (280 h spacing at this grid; launches in the no-opt-in shared
      regime): numbers are printed for the record but NOT gated -- 60 is
      known-coarse at 23 months, which is exactly why 525 is prescribed.

PASS/FAIL is printed as the last line; exit code follows it.

Knobs (env): GATE_NT_LAYER (525), GATE_NT_LAYER_REF (420),
GATE_NT_LAYER_SMALL (60), GATE_NREFS (9), GATE_NSFD (1024),
GATE_OUT (./gate_sighet_fstat_out), BACKEND (cuda12x),
GATE_NT_GRID (16800; set 2160 for the 3-month grid).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_sighet_v4_parity import XpGridWDMHolder  # noqa: E402

from lisatools.detector import ESAOrbits                      # noqa: E402
from lisatools.domains import WDMSettings                     # noqa: E402
from lisatools.sensitivity import XYZ2SensitivityMatrix       # noqa: E402
from lisatools.utils.constants import YRSID_SI                # noqa: E402
from lisatools.sampling.fstat_gridfit import (                # noqa: E402
    sighet_fstat_ref_margin_hz,
)
from lisatools.sampling.fstat_proposal import compute_fstat   # noqa: E402
from gbgpu.gbcomps import GBWDMComputations                   # noqa: E402
from gbgpu.gbsignalhetcomputations import (                   # noqa: E402
    GBSignalHetComputations,
    _sighet_fstat_shared_bytes,
)


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def main():
    backend = os.environ.get("BACKEND", "cuda12x")
    NT = int(os.environ.get("GATE_NT_LAYER", "525"))
    NT_REF = int(os.environ.get("GATE_NT_LAYER_REF", "420"))
    NT_SMALL = int(os.environ.get("GATE_NT_LAYER_SMALL", "60"))
    n_refs = int(os.environ.get("GATE_NREFS", "9"))
    nsfd = int(os.environ.get("GATE_NSFD", "1024"))
    out_dir = os.environ.get("GATE_OUT", "./gate_sighet_fstat_out")
    os.makedirs(out_dir, exist_ok=True)

    # ---- the 23-month production grid SHAPE, narrow band ------------------
    # GATE_NT_GRID: WDM time-layer count of the target grid (16800 = the
    # 23-month/700-d shape; 2160 = the 3-month/90-d production shape).
    Nf, Nt, dt = 1440, int(os.environ.get("GATE_NT_GRID", "16800")), 2.5
    Tobs = Nf * Nt * dt                        # 6.048e7 s = 700 d
    layer_df = 1.0 / (2.0 * Nf * dt)
    edge = 20                                  # production 8-wavelet taper -> 20
    tukey_alpha = 2.0 * 8.0 / Nt
    t_start = int(0.5 * YRSID_SI / dt) * dt
    print(f"[gate] grid Nf={Nf} Nt={Nt} dt={dt} Tobs={Tobs:.4e} s "
          f"({Tobs/86400:.0f} d), band 5-7 mHz, backend={backend}")
    print(f"[gate] fstat scorer budget at N=525: mode0 "
          f"{_sighet_fstat_shared_bytes(64,128,3,2,525,32,2)} B, mode1 "
          f"{_sighet_fstat_shared_bytes(64,128,3,2,525,32,4)} B; "
          f"make_reference full carve {(Nt + nsfd + NT) * 16} B "
          f"(global-tw fallback expected on every device)")

    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=5e-3,
                          max_freq=7e-3, min_time=edge * Nf * dt,
                          max_time=(Nt - edge) * Nf * dt,
                          force_backend=backend)
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start, Nt_sub=256, n_pad=16, N_sparse=256,
        N_cp_sig=48, N_cp_orbit=32, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend, d_d=0.0,
        tdi_type="XYZ", tukey_alpha=tukey_alpha)
    chunked.convert_to_ra_dec = False

    ilo, ihi = int(wdm_set.ind_min_f), int(wdm_set.ind_max_f) + 1
    xp = chunked.xp
    invC = xp.ascontiguousarray(xp.asarray(
        XYZ2SensitivityMatrix(wdm_set, model="scirdv1").invC,
        dtype=xp.float64))

    # ---- one synthetic GB injected through the chunked engine -------------
    f0_t = (int(6e-3 / layer_df) + 0.37) * layer_df
    truth = np.array([1e-21, f0_t, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    href = xp.zeros((3, Nf, Nt))
    chunked.fill_global_wdm(xp.asarray(truth)[None, :], href,
                            convert_to_ra_dec=False)
    h_act = href[:, ilo:ihi, wdm_set.active_slice_t]
    holder = XpGridWDMHolder(xp, h_act, invC)
    del href

    # ---- reference comb + candidate sweep ---------------------------------
    margin = sighet_fstat_ref_margin_hz(128, Tobs)
    ref_f0 = f0_t + (np.arange(n_refs) - n_refs // 2) * margin
    refs = np.zeros((n_refs, 9))
    refs[:, 1] = ref_f0
    offs = np.array([0.0, -0.45, -0.25, 0.25, 0.45]) * margin
    skys = [(0.0, 0.0), (2.0, 0.5)]           # canonical + displaced
    rows = []
    for f0r in ref_f0:
        for off in offs:
            for lam, beta in skys:
                r = np.zeros(9)
                r[1] = f0r + off
                r[7] = lam
                r[8] = beta
                rows.append(r)
    cands = np.asarray(rows)
    anchor_mask = np.array(
        [(abs(r[1] - ref_f0).min() < 1e-15) and r[7] == 0.0 and r[8] == 0.0
         for r in rows])
    print(f"[gate] {n_refs} refs (spacing = margin {margin:.3e} Hz), "
          f"{len(cands)} candidate rows ({int(anchor_mask.sum())} anchors)")

    def run_arm(nt_layer, fstat_mode=0):
        sh = GBSignalHetComputations.for_band_engine(
            chunked, nt_layer=nt_layer, n_sparse_fd=nsfd,
            m_active_half_width=2, v3_n_nodes=64, v4_knots=128, v4_band=16)
        n_built = sh.setup_fstat_references(
            refs, holder, data_index=0,
            assert_max_df0=0.6 * margin)
        N_arr, M_up = sh.get_fstat_ll_wdm(xp.asarray(cands),
                                          fstat_mode=fstat_mode)
        F = _to_host(compute_fstat(N_arr, M_up))
        stash_ok = all(
            bool(np.isfinite(_to_host(sh._fstat[k]).view(np.float64)).all())
            for k in ("A0", "A1", "B0", "B1", "B0nc", "B1nc"))
        sh.clear_fstat_references()
        nsp = int(sh._g["N_sparse_t"])
        print(f"[arm] nt_layer={nt_layer} (N_sparse_t={nsp}, stride "
              f"{Nt // nt_layer} h) mode={fstat_mode}: {n_built} refs "
              f"built, F range [{F.min():.3e}, {F.max():.3e}], "
              f"stash finite={stash_ok}")
        return F, stash_ok

    def rel(a, b):
        scale = max(np.abs(b).max(), 1e-300)
        return np.abs(a - b) / scale

    results = {}
    fails = []

    # [A] capacity: the 525 arm must COMPLETE (any shared-memory failure
    # raises/asserts inside run_arm).
    F525, ok525 = run_arm(NT, 0)
    results["F525"] = F525
    if not ok525:
        fails.append("[A] non-finite fstat reference stash at nt_layer=525")

    # [B] mode 0 == mode 1 at 525.
    F525m1, _ = run_arm(NT, 1)
    results["F525_mode1"] = F525m1
    b = rel(F525m1, F525).max()
    print(f"[B] mode0 vs mode1 @ {NT}: max rel |dF| = {b:.3e} (tol 1e-6)")
    if b > 1e-6:
        fails.append(f"[B] mode parity {b:.3e} > 1e-6")

    # [C] 525 vs 420 (both at/near the validated ~35 h density).
    Fref, _ = run_arm(NT_REF, 0)
    results["Fref"] = Fref
    c = rel(F525, Fref)
    print(f"[C] {NT} vs {NT_REF}: median rel |dF| = {np.median(c):.3e} "
          f"(tol 1e-2), max = {c.max():.3e} (tol 3e-2); anchors max = "
          f"{c[anchor_mask].max():.3e}")
    if np.median(c) > 1e-2 or c.max() > 3e-2:
        fails.append(f"[C] grid consistency median {np.median(c):.3e} / "
                     f"max {c.max():.3e} over tolerance")

    # [D] info: legacy-default landing spot (280 h here -- known coarse).
    Fsm, _ = run_arm(NT_SMALL, 0)
    results["Fsmall"] = Fsm
    d = rel(F525, Fsm)
    print(f"[D] info: {NT} vs {NT_SMALL}: median rel |dF| = "
          f"{np.median(d):.3e}, max = {d.max():.3e} (NOT gated; 280 h "
          f"spacing is the coarseness 525 exists to fix)")

    np.savez(os.path.join(out_dir, "gate_sighet_fstat_nt525.npz"),
             cands=cands, refs=refs, anchor_mask=anchor_mask, **results)

    if fails:
        for f in fails:
            print("FAIL:", f)
        print("GATE: FAIL")
        return 1
    print("GATE: PASS -- submit_gf_23mo.sh PREFLIGHT GATE 1 satisfied "
          f"(SIGHET_NT_LAYER={NT} fstat fit runs on this device).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
