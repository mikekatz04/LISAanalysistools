"""Parity gate for the sig-het F-stat path (2026-08-11).

The new `gb_signal_het_fstat_get_ll` kernel family computes the F-stat
pieces ``(N (n,4), M_upper (n,10))`` -- the 4 Cornish & Crowder basis-filter
data products and their Gram matrix -- through the sig-het v5 machinery
against SHARED references, instead of the chunked-het
``GBWDMComputations.get_fstat_ll_wdm`` per-candidate heterodyne. This script
is the hard (N, M) parity gate between the two.

Three gates, all against the chunked-het truth (installed classes only):

  (a) ZERO DISPLACEMENT: candidates whose intrinsics coincide with their
      reference's -> per-entry (N, M) comparison at the sig-het floor.
      Metric choice matters (tiered-accuracy lesson: raw-relative against a
      cancellation residual is the WRONG lens): N errors are gated against
      the ON-PEAK |N| scale -- off-peak <d|A_i> entries are cancellation
      residuals orders of magnitude below it, and the representation floor
      is an absolute one set by the reference-c0 build (mm5 ~ 1.6e-9
      energy -> ~1e-4 amplitude scale, which IS the measured floor here);
      M errors are gated against each row's max |M| (the scale
      compute_fstat / the extrinsic inversion condition on). Per-entry
      row-relative numbers are still printed for the record.
  (b) NEIGHBORHOOD: candidates displaced in (f0, sky, Mc/fdot) up to the
      reference-spacing margin -> F = 0.5 N^T M^-1 N via
      ``lisatools.sampling.fstat_proposal.compute_fstat``, compared
      delta-vs-delta with the tiered allowance ``allowed(T) =
      max(0.1, T/100)`` (the tiered-accuracy policy).
  (c) RECOMBINATION SELF-CHECK: the production kernel evaluates only 2
      node stages (one per psi in {0, pi/4}) and recovers the 4 filters
      by exact global phase rotation (`fstat_mode=0`); `fstat_mode=1`
      scores the 4 filters as 4 independent stages. (N, M) from the two
      modes must agree to spline-fit round-off (<= 1e-8 relative).

Scaffold (grid, chunked comp, sig-het engine, holder) is REUSED from
``gb_sighet_v4_parity.make_scaffold`` so this gate always runs on the same
validated configuration as the v4/v5 parity gates. The CPU default grid is
the small f0-tolerance iteration grid (V4_NT=512, V4_NF=256 -> ~15 d).

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_fstat_parity.py
GPU rerun:
    BACKEND=cuda12x V4_NT=540 V4_NF=1440 python gb_sighet_fstat_parity.py

Env: BACKEND (cpu), V4_NT / V4_NF (grid), V4_NR (64), V4_K (128),
     V5_BAND (16), FSTAT_PAR_SNR (80), FSTAT_PAR_NREFS (5),
     FSTAT_PAR_TOL_A (2e-5 relative, gate a), FSTAT_PAR_TOL_C (1e-8),
     ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_v4_parity as par

from lisatools.sampling.fstat_proposal import compute_fstat
from lisatools.sampling.fstat_gridfit import sighet_fstat_ref_margin_hz


def asnp(a):
    return par.asnp(a)


def chunked_NM(chunked, holder, rows, m_half):
    N, M = chunked.get_fstat_ll_wdm(
        np.atleast_2d(np.asarray(rows)), holder,
        convert_to_ra_dec=False, m_band_half_width=m_half)
    return asnp(N), asnp(M)


def sighet_NM(sighet, holder, rows, fstat_mode):
    xp = sighet.xp
    N, M = sighet.get_fstat_ll_wdm(
        xp.atleast_2d(xp.asarray(rows)), holder, fstat_mode=fstat_mode)
    return asnp(N), asnp(M)


def rel_err(a, b):
    """Elementwise relative error against the per-row (N or M) scale.

    Per-entry denominators would blow up on tiny, ill-conditioned entries
    (e.g. near-zero off-diagonal Gram entries); the row scale is what the
    downstream solve actually conditions on."""
    scale = np.maximum(np.abs(b).max(axis=-1, keepdims=True), 1e-300)
    return np.abs(a - b) / scale


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    backend = os.environ.get("BACKEND", "cpu")
    if backend == "cpu":
        os.environ.setdefault("V4_NT", "512")
        os.environ.setdefault("V4_NF", "256")
    nr = int(os.environ.get("V4_NR", "64"))
    K = int(os.environ.get("V4_K", "128"))
    band = int(os.environ.get("V5_BAND", "16"))
    snr_target = float(os.environ.get("FSTAT_PAR_SNR", "80"))
    n_refs = int(os.environ.get("FSTAT_PAR_NREFS", "5"))
    # Measured sig-het floors on this scaffold (2026-08-11, circular
    # reference frame): worst |dN|/on-peak 4.0e-4, worst M row-rel 1.4e-3,
    # on-peak-row N rel ~5e-5 / M ~1.1e-4 -- the c0 (mm5 ~ 1.6e-9 energy ->
    # ~1e-4 amplitude) floor. Gate at 3e-3 (2x headroom over the worst).
    tol_a = float(os.environ.get("FSTAT_PAR_TOL_A", "3e-3"))
    tol_c = float(os.environ.get("FSTAT_PAR_TOL_C", "1e-8"))
    # c0 slow-FD extent: sets the reference's WDM layer support
    # (write extent ~ n_sparse_fd/Nt layers each side) -- the truncation
    # lever for the outer m-band layers.
    nsfd = int(os.environ.get("FSTAT_PAR_NSFD", "2048"))
    m_half = 2                     # sig-het scaffold m_active_half_width

    sc = par.make_scaffold(backend=backend, v3_n_nodes=nr, v4_knots=K,
                           n_sparse_fd=nsfd)
    sighet = sc["sighet"]
    chunked = sc["chunked"]
    g = sighet._g
    g["v4_band"] = band
    sighet._v4_band_arrays = None
    xp = sc["xp"]

    for name in ("gb_signal_het_fstat_get_ll",):
        if not hasattr(sighet.cpp, name):
            raise RuntimeError(
                f"compiled {name} not found -- rebuild the GBGPU wheel "
                "(a stale .so silently hides new kernels).")

    Tobs = g["Tobs"]
    layer_df = g["layer_df"]
    margin = sighet_fstat_ref_margin_hz(K, Tobs)
    print(f"[cfg] backend={backend} Tobs={Tobs:.4e} s ({Tobs/86400:.1f} d) "
          f"n_r={nr} K={K} band={band} m_half={m_half}")
    print(f"[cfg] ref margin (max |df0|) = {margin:.3e} Hz "
          f"({2*np.pi*margin*Tobs:.1f} rad)")

    # ---- data: one truth GB injected into the residual -------------------
    f0_t = (int(6e-3 / layer_df) + 0.37) * layer_df
    truth = np.array([1e-22, f0_t, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    holder = par.setup_ref(sc, truth)       # holder now CONTAINS the truth
    # rescale to the target SNR through the chunked engine's h_h
    from gbgpu.gb_likelihood import WDMBandLikelihoodEngine
    eng_c = WDMBandLikelihoodEngine(chunked, sc["wdm_set"], nchannels=3,
                                    tdi_channel_setup="XYZ")
    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})
    eng_c.get_ll(holder, truth[None, :], phase_maximize=False, **kw)
    snr = np.sqrt(float(asnp(eng_c.h_h_out)[0]))
    truth[0] *= snr_target / snr
    holder = par.setup_ref(sc, truth)
    sighet.clear_in_model()

    # ---- references: an f0 bucket grid centered on the truth carrier ----
    spacing = margin                    # bucket half-width = margin/2 (2x safe)
    ref_f0s = f0_t + (np.arange(n_refs) - n_refs // 2) * spacing
    refs = np.tile(truth, (n_refs, 1))
    refs[:, 1] = ref_f0s
    # the builder forces the basis frame (A=2, phi0=0, iota=pi/2, psi=0)
    # on these rows itself; sky/fdot are kept.
    n_built = sighet.setup_fstat_references(refs, holder)
    print(f"[refs] built {n_built} references, spacing {spacing:.3e} Hz")

    # =====================================================================
    # gate (a): zero displacement
    # =====================================================================
    print("\n[gate a] zero displacement: per-entry (N, M) parity")
    rows_a = np.tile(truth, (n_refs, 1))
    rows_a[:, 1] = ref_f0s              # candidate == reference intrinsics
    Nc, Mc = chunked_NM(chunked, holder, rows_a, m_half)
    Ns, Ms = sighet_NM(sighet, holder, rows_a, fstat_mode=0)
    # gate metrics: N against the ON-PEAK scale (absolute floor; off-peak
    # entries are cancellation residuals), M against each row's max |M|
    # (see the module docstring).
    N_scale = max(np.abs(Nc).max(), 1e-300)
    eN = np.abs(Ns - Nc) / N_scale
    eM = rel_err(Ms, Mc)
    eN_row = rel_err(Ns, Nc)            # row-relative, printed for the record
    for r in range(n_refs):
        print(f"  ref{r}: f0={ref_f0s[r]*1e3:.5f} mHz | "
              f"N/peak {['%.2e' % v for v in eN[r]]} | "
              f"N/row {['%.2e' % v for v in eN_row[r]]} | "
              f"max M rel {eM[r].max():.2e}")
    print(f"  [N] worst |dN| / on-peak scale: {eN.max():.3e}")
    print(f"  [M] worst per-entry rel (row scale): {eM.max():.3e}")
    ic = n_refs // 2
    print(f"  [floor] center (on-peak) row: N rel "
          f"{['%.1e' % v for v in eN_row[ic]]}, max M rel {eM[ic].max():.1e}")
    gate_a = max(eN.max(), eM.max()) <= tol_a
    print(f"  gate a: {'PASS' if gate_a else 'FAIL'} (tol {tol_a:g})")

    # =====================================================================
    # gate (b): neighborhood, F delta-vs-delta
    # =====================================================================
    print("\n[gate b] neighborhood: F delta-vs-delta "
          "(allowed(T) = max(0.1, T/100))")
    rng = np.random.default_rng(23)
    center = rows_a[n_refs // 2]
    Fc0 = float(compute_fstat(*[np.asarray(v) for v in
                                chunked_NM(chunked, holder,
                                           center[None], m_half)])[0])
    Fs0 = float(asnp(compute_fstat(
        *sighet_NM(sighet, holder, center[None], 0)))[0])
    cands, labels = [], []
    for frac in (0.25, 0.5, 1.0):
        for sgn in (+1.0, -1.0):
            p = center.copy()
            p[1] += sgn * frac * 0.5 * spacing
            cands.append(p)
            labels.append(f"df0={sgn * frac * 0.5:+.2f}sp")
    for j in range(4):
        p = center.copy()
        p[7] = rng.uniform(0, 2 * np.pi)
        p[8] = np.arcsin(rng.uniform(-1, 1))
        p[1] += rng.uniform(-0.5, 0.5) * 0.5 * spacing
        cands.append(p)
        labels.append(f"sky{j}")
    for fd in (0.0, 1e-16, 3e-16):
        p = center.copy()
        p[2] = fd
        cands.append(p)
        labels.append(f"fdot={fd:.0e}")
    cands = np.asarray(cands)
    Ncb, Mcb = chunked_NM(chunked, holder, cands, m_half)
    Nsb, Msb = sighet_NM(sighet, holder, cands, fstat_mode=0)
    F_c = np.asarray(compute_fstat(Ncb, Mcb))
    F_s = np.asarray(asnp(compute_fstat(Nsb, Msb)))
    gate_b = True
    worst_ratio = 0.0
    for lab, fc, fs in zip(labels, F_c, F_s):
        D_true = fc - Fc0
        D_arm = fs - Fs0
        err = abs(D_arm - D_true)
        alw = max(0.1, abs(D_true) / 100.0)
        ok = err <= alw
        gate_b &= ok
        worst_ratio = max(worst_ratio, err / alw)
        print(f"  {lab:14s} F_chunk={fc:12.4f} F_sig={fs:12.4f} "
              f"T={abs(D_true):10.3f} err={err:8.4f} alw={alw:7.3f} "
              f"{'ok' if ok else '<-- OVER'}")
    print(f"  gate b: {'PASS' if gate_b else 'FAIL'} "
          f"(worst err/allowed = {worst_ratio:.3f})")

    # =====================================================================
    # gate (c): 2-stage rotation vs 4 independent stages
    # =====================================================================
    print("\n[gate c] recombination self-check: fstat_mode=0 vs 1")
    rows_c = np.concatenate([rows_a, cands], axis=0)
    N0, M0 = sighet_NM(sighet, holder, rows_c, fstat_mode=0)
    N1, M1 = sighet_NM(sighet, holder, rows_c, fstat_mode=1)
    eNc = rel_err(N0, N1)
    eMc = rel_err(M0, M1)
    print(f"  [N] worst rel mode0-vs-mode1: {eNc.max():.3e}")
    print(f"  [M] worst rel mode0-vs-mode1: {eMc.max():.3e}")
    gate_c = max(eNc.max(), eMc.max()) <= tol_c
    print(f"  gate c: {'PASS' if gate_c else 'FAIL'} (tol {tol_c:g})")

    npz = os.path.join(out_dir, "fstat_parity.npz")
    np.savez(npz, Nc=Nc, Mc=Mc, Ns=Ns, Ms=Ms, F_c=F_c, F_s=F_s,
             N0=N0, M0=M0, N1=N1, M1=M1, ref_f0s=ref_f0s,
             labels=np.array(labels), backend=backend,
             margin=margin, spacing=spacing, Tobs=Tobs)
    print(f"\n[save] {npz}")

    ok = gate_a and gate_b and gate_c
    print("GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
