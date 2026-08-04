"""Correctness gate for sig-het V5, 2026-08-04.

V5 is v4-banded with the per-candidate fold scratch ELIMINATED rather than
relocated. ``r_sparse``/``dr_sparse`` are an M-fold replication of ``r_pix``
modulated by one bit per (row, pixel), and that bit -- the |c0| row floor --
depends only on the reference, so ``setup_in_model`` precomputes it and the
scorer rebuilds r/dr in registers. Nothing about the arithmetic changes: the
index mapping, the operands, the operation order and the block reduction are
v4's, so v5 should be **BIT-identical** to v4, not merely close.

That is the gate. A tolerance is still reported (and is what PASS/FAIL keys
on, so a future reordering does not silently fail the script), but any
non-zero delta at all is worth investigating -- it means an assumption in the
reconstruction is wrong, not that round-off accumulated differently.

Three arms per candidate, all through the COMPILED kernels (no hand-rolled
likelihood anywhere -- the ground truth is the existing validated engine):

  * v4-band  : ``gb_signal_het_v4_get_ll``            (the reference)
  * v5-flat  : ``gb_signal_het_v5_get_ll(v5_mode=2)`` (flat shared carve)
  * v5       : ``gb_signal_het_v5_get_ll(v5_mode=1)`` (phase-lifetime shared
               arena -- the production candidate, ~5 blocks/SM on an A100)

The 1-vs-2 arm matters beyond correctness: mode 1 overlays shared regions
whose lifetimes are disjoint, so if a phase boundary is wrong the two modes
disagree and this script says so. On CPU the arena is a plain host buffer and
both modes exercise the same offsets, so a CPU pass gates the RECONSTRUCTION
and the carve arithmetic, not the aliasing's dependence on __syncthreads().
Rerun with ``BACKEND=cuda12x`` to gate the real kernel.

Deltas are RAW ``d_h - 0.5*h_h`` against the shared reference stash, exactly
as ``gb_sighet_v4_parity.py`` reports them, so the two gates are directly
comparable.

Run:
    /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_v5_parity.py
Env:
    V4_NR (64 fit nodes), V4_K (128 knots), V5_BAND (16 half-band),
    V4_REF_SEL ("0,2"), BACKEND (cpu), ENV_OUT (./ratio_proto_out),
    V5_TOL (1e-10 relative), plus every V4_* grid knob
    gb_sighet_v4_parity.make_scaffold understands.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_v4_parity as par


def v5_delta(sighet, p, *, n_nodes, n_knots, v5_mode):
    """Raw ``d_h - 0.5*h_h`` through the compiled V5 kernel.

    Calls the binding directly rather than going through ``get_ll`` so the
    carve mode is explicit at the call site and the comparison cannot be
    perturbed by the engine's routing knobs.
    """
    xp = sighet.xp
    g = sighet._g
    if sighet._v4_band_arrays is None:
        sighet._v4_band_arrays = sighet._make_v4_band_arrays()
    x = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(p, dtype=float)))
    di = xp.zeros(1, dtype=xp.int32)
    d_h = xp.zeros(1, dtype=xp.float64)
    h_h = xp.zeros(1, dtype=xp.float64)
    sighet.cpp.gb_signal_het_v5_get_ll(
        sighet.tdi_wrap, d_h, h_h, sighet.c0_mask_all,
        sighet.A0_all, sighet.A1_all, sighet.B0_all, sighet.B1_all,
        sighet.B0nc_all, sighet.B1nc_all,
        sighet.n_sparse_local,
        sighet._v4_band_arrays[0], sighet._v4_band_arrays[1],
        sighet._v4_band_arrays[2],
        x, sighet.params_ref_all, di,
        1, int(sighet.params_ref_all.shape[0]),
        int(n_nodes), int(n_knots),
        9, 1, 2,
        g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
        g["nt_layer"], g["N_sparse_t"], g["stride"],
        g["ind_min_t"], g["ind_min_f"], g["m_half"],
        g["layer_df"], g["dt"], g["Tobs"], g["t0"],
        3, 0, 1,
        int(v5_mode))
    return (float(par.asnp(d_h)[0]) - 0.5 * float(par.asnp(h_h)[0]))


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    backend = os.environ.get("BACKEND", "cpu")
    nr = int(os.environ.get("V4_NR", "64"))
    K = int(os.environ.get("V4_K", "128"))
    band = int(os.environ.get("V5_BAND", "16"))
    tol = float(os.environ.get("V5_TOL", "1e-10"))
    ref_sel = [int(x) for x in
               os.environ.get("V4_REF_SEL", "0,2").split(",")]

    sc = par.make_scaffold(backend=backend, v3_n_nodes=nr, v4_knots=K)
    sighet = sc["sighet"]
    # The v4 scaffold builds the engine without a band; v5 IS the banded
    # path, so switch the knob and let the cardinal weights build lazily.
    sighet._g["v4_band"] = band
    sighet._v4_band_arrays = None

    for name in ("gb_signal_het_v4_get_ll", "gb_signal_het_v5_get_ll"):
        if not hasattr(sighet.cpp, name):
            raise RuntimeError(
                f"compiled {name} not found -- rebuild the GBGPU wheel "
                "(a stale .so silently hides new kernels).")

    refs = par.make_refs(8)
    Tobs = sighet._g["Tobs"]
    print(f"[cfg] backend={backend} n_r={nr} K={K} half-band={band} "
          f"N_sparse_t={sighet._g['N_sparse_t']} refs={ref_sel} tol={tol:g}")
    if backend == "cpu":
        print("[cfg] CPU: the shared arena is a host buffer, so this gate "
              "proves the RECONSTRUCTION and the carve arithmetic; rerun with "
              "BACKEND=cuda12x to gate the real kernel and "
              "GB_SIGHET_V5_VERBOSE=1 to read registers/thread + blocks/SM.")

    rows = []
    worst_ctl = 0.0
    worst_glb = 0.0
    n_exact = 0
    for ri in ref_sel:
        ref = refs[ri]
        par.setup_ref(sc, ref)
        print(f"\n[ref{ri:02d}]  f0={ref[1]:.6e}  A={ref[0]:.3e}")
        print(f"{'candidate':>12}  {'v4-band':>14}  {'v5-flat rel':>12}  "
              f"{'v5 rel':>12}  {'bit':>4}")
        for label, p in par.make_candidates(ref, Tobs):
            d4 = par.compiled_delta(sighet, p, v3_n_nodes=nr, v4_knots=K)
            dc = v5_delta(sighet, p, n_nodes=nr, n_knots=K, v5_mode=2)
            dg = v5_delta(sighet, p, n_nodes=nr, n_knots=K, v5_mode=1)
            scale = max(abs(d4), 1.0)
            rc = abs(dc - d4) / scale
            rg = abs(dg - d4) / scale
            worst_ctl = max(worst_ctl, rc)
            worst_glb = max(worst_glb, rg)
            exact = (dc == d4) and (dg == d4)
            n_exact += int(exact)
            rows.append((ri, label, d4, dc, dg, rc, rg))
            flag = "" if max(rc, rg) <= tol else "   <-- OVER"
            print(f"{label:>12}  {d4:14.6e}  {rc:12.3e}  {rg:12.3e}  "
                  f"{'==' if exact else '!=':>4}{flag}")

    print(f"\n[worst] v5-flat vs v4-band : {worst_ctl:.3e}")
    print(f"[worst] v5      vs v4-band : {worst_glb:.3e}")
    print(f"[worst] v5      vs v5-flat : "
          f"{max(abs(r[4] - r[3]) / max(abs(r[2]), 1.0) for r in rows):.3e}")
    # v5 changes no arithmetic, so this should be every row. A row that is
    # inside tolerance but not bit-identical means the reconstruction is
    # subtly different from what v4 materialised -- worth chasing even though
    # the gate passes.
    print(f"[bit  ] exactly equal to v4 : {n_exact}/{len(rows)} candidates"
          + ("" if n_exact == len(rows) else "   <-- expected all"))

    npz = os.path.join(out_dir, "v5_parity.npz")
    np.savez(npz,
             ref=np.array([r[0] for r in rows]),
             label=np.array([r[1] for r in rows]),
             v4=np.array([r[2] for r in rows]),
             v5_flat=np.array([r[3] for r in rows]),
             v5=np.array([r[4] for r in rows]),
             rel_flat=np.array([r[5] for r in rows]),
             rel_v5=np.array([r[6] for r in rows]),
             backend=backend, n_r=nr, K=K, band=band, tol=tol)
    print(f"[save] {npz}")

    ok = max(worst_ctl, worst_glb) <= tol
    print("\nGATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
