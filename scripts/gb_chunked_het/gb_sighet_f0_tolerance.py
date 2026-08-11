"""Delta-f0 displacement tolerance sweep for the sig-het likelihood (2026-08-11).

THE question this answers: how far in f0 can a candidate sit from the
heterodyne REFERENCE before each sig-het scorer path stops tracking the
chunked-het truth?  The validated in-model envelope is the trust region
``2*pi*|df0|*Tobs <= 0.5 rad``; the "sig-het F-stat with shared references"
design wants ONE reference to serve a whole f0-neighborhood, |df0| ~ 2e-6
to 1e-5 Hz -- hundreds to thousands of radians.  This script measures the
actual breakdown curve per scorer path.

Everything runs through INSTALLED classes (no hand-rolled likelihood):

  * truth        : ``GBWDMComputations`` chunked-het via
                   ``WDMBandLikelihoodEngine.get_ll`` (per-candidate
                   heterodyne -- does not degrade with df0)
  * sig-het arms : ``GBSignalHetComputations.get_ll`` with the scorer
                   selected by the class's own ``_g`` routing knobs
                   (the ``gb_sighet_v4_parity.compiled_delta`` idiom):
                     v2  = default in-kernel polyphase path
                     v3  = node-eval log-polar ratio fit (per-pixel spline
                           eval; the analytic de-rotation path)
                     v3n<NR2> = v3 with a larger fit-node count (lever test)
                     v5  = fixed-knot resample, banded, arena carve
                           (v4_knots=K, v5=1 -- bit-identical to v4)

Scaffold (grid, refs, holder, setup_in_model) is REUSED from
``gb_sighet_v4_parity.make_scaffold`` / ``setup_ref`` so this script is
always on the same validated configuration as the parity gates.

Metric (tiered-accuracy policy): DELTA-vs-DELTA.  For candidate p,
    D_true = chunk(p) - chunk(ref),   D_arm = arm(p) - arm(ref),
    err    = |D_arm - D_true|,        allowed(T) = max(0.1, T/100),
with T = |D_true|.  Raw deltas and the transfer coordinate
``2*pi*df0*Tobs`` (radians) are reported alongside.

Sanity anchors (abort if violated -- a broken harness must not produce a
curve):  (a) at the smallest df0 (deep inside the trust region) every arm's
err must be <= 0.1;  (b) at the trust-region edge (0.5 rad) every arm's
err must be <= max(0.5, 0.05*T).  Disable with F0TOL_NO_GATE=1 (debug only).

Arms:
  1. df0 sweep, log-spaced F0TOL_DF0_MIN..F0TOL_DF0_MAX (F0TOL_NDF0 pts)
     + the trust-region-edge point, + sign check at 3 magnitudes.
  2. sky displacement: F0TOL_NSKY full-sky draws at df0=0 and at the
     trust-region edge df0.
  3. combined worst case: antipodal sky + df0 in {2e-6, 1e-5, 1e-4} Hz.

Run (local CPU, ~15-day grid):
    OMP_NUM_THREADS=1 V4_NT=512 V4_NF=256 \
    /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_f0_tolerance.py
Run (local CPU, ~91-day grid):   V4_NT=3072 V4_NF=256 ...
Run (cluster GPU, 90-day production-shaped grid):
    BACKEND=cuda12x V4_NT=540 V4_NF=1440 python gb_sighet_f0_tolerance.py

Env: BACKEND (cpu), V4_NT / V4_NF (grid; cpu default 512 x 256 here),
     V4_NR (64 fit nodes), F0TOL_NR2 (256; 0 disables the v3-hi arm),
     V4_K (128 knots), V5_BAND (16), F0TOL_SNR (80),
     F0TOL_F0S ("4e-3,1e-2" reference carriers, Hz),
     F0TOL_NDF0 (14), F0TOL_DF0_MIN (1e-9), F0TOL_DF0_MAX (1e-4),
     F0TOL_NSKY (5), F0TOL_NSFD (512 slow-series samples -- the v2 f0
     envelope is ~0.8 * (NSFD/2)/Tobs), F0TOL_NO_GATE (0),
     ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_v4_parity as par

from gbgpu.gb_likelihood import WDMBandLikelihoodEngine


def sighet_raw(sighet, p, knobs):
    """Raw ``d_h - 0.5*h_h`` through the installed ``get_ll``, scorer path
    selected by temporarily setting the class's own routing knobs
    (v3_n_nodes, v4_knots, v5) -- the compiled_delta idiom extended to v5."""
    g = sighet._g
    saved = (g.get("v3_n_nodes", 0), g.get("v4_knots", 0), g.get("v5", 0))
    g["v3_n_nodes"], g["v4_knots"], g["v5"] = knobs
    try:
        sighet.get_ll(np.asarray(p)[None, :],
                      data_index=np.zeros(1, dtype=np.int32))
    finally:
        g["v3_n_nodes"], g["v4_knots"], g["v5"] = saved
    return (float(par.asnp(sighet.last_d_h)[0])
            - 0.5 * float(par.asnp(sighet.last_h_h)[0]))


def allowed_of(T):
    return max(0.1, abs(T) / 100.0)


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    backend = os.environ.get("BACKEND", "cpu")
    # CPU default grid here is the SMALL iteration grid; override with V4_NT
    # (make_scaffold reads V4_NT/V4_NF itself when Nt/Nf are None -> GPU
    # defaults under a CUDA backend).
    if backend == "cpu":
        os.environ.setdefault("V4_NT", "512")
        os.environ.setdefault("V4_NF", "256")
    nr = int(os.environ.get("V4_NR", "64"))
    nr2 = int(os.environ.get("F0TOL_NR2", "256"))
    K = int(os.environ.get("V4_K", "128"))
    band = int(os.environ.get("V5_BAND", "16"))
    snr_target = float(os.environ.get("F0TOL_SNR", "80"))
    f0_list = [float(x) for x in
               os.environ.get("F0TOL_F0S", "4e-3,1e-2").split(",")]
    n_df0 = int(os.environ.get("F0TOL_NDF0", "14"))
    df0_min = float(os.environ.get("F0TOL_DF0_MIN", "1e-9"))
    df0_max = float(os.environ.get("F0TOL_DF0_MAX", "1e-4"))
    n_sky = int(os.environ.get("F0TOL_NSKY", "5"))
    no_gate = os.environ.get("F0TOL_NO_GATE", "0") == "1"
    # v2's f0 envelope is the slow-series Nyquist: |df0| < (n_sparse_fd/2)/Tobs
    # (beat cycles across Tobs must be resolvable by the n_sparse_fd slow
    # samples). This is the v2 lever -- expose it for the GPU confirmation.
    nsfd = int(os.environ.get("F0TOL_NSFD", "512"))

    sc = par.make_scaffold(backend=backend, v3_n_nodes=nr, v4_knots=K,
                           n_sparse_fd=nsfd)
    sighet = sc["sighet"]
    chunked = sc["chunked"]
    g = sighet._g
    # v5 is the BANDED path; select the band once (cardinal weight arrays
    # rebuild lazily for it).
    g["v4_band"] = band
    sighet._v4_band_arrays = None
    for name in ("gb_signal_het_v3_get_ll", "gb_signal_het_v4_get_ll",
                 "gb_signal_het_v5_get_ll"):
        if not hasattr(sighet.cpp, name):
            raise RuntimeError(
                f"compiled {name} not found -- rebuild the GBGPU wheel "
                "(a stale .so silently hides new kernels).")

    eng_c = WDMBandLikelihoodEngine(chunked, sc["wdm_set"], nchannels=3,
                                    tdi_channel_setup="XYZ")
    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})
    Tobs = g["Tobs"]
    layer_df = g["layer_df"]
    df0_edge = 0.5 / (2.0 * np.pi * Tobs)       # trust-region edge (0.5 rad)

    paths = [("v2", (0, 0, 0)), ("v3", (nr, 0, 0))]
    if nr2 > 0:
        paths.append((f"v3n{nr2}", (nr2, 0, 0)))
    paths.append(("v5", (nr, K, 1)))
    pnames = [nm for nm, _ in paths]

    print(f"[cfg] backend={backend} Tobs={Tobs:.4e} s "
          f"({Tobs/86400:.1f} d) layer_df={layer_df:.3e} Hz "
          f"trust-edge df0={df0_edge:.3e} Hz")
    print(f"[cfg] paths={pnames} n_r={nr} K={K} band={band} "
          f"n_sparse_fd={nsfd} SNR target={snr_target}")
    print(f"[cfg] v2 slow-series Nyquist df0 = {0.5 * nsfd / Tobs:.3e} Hz "
          f"({np.pi * nsfd:.0f} rad); WDM pixel = {g['Nf'] * g['dt']:.0f} s "
          f"-> 0.4 rad/pixel at df0 = "
          f"{0.4 / (2 * np.pi * g['Nf'] * g['dt']):.3e} Hz")

    def chunk_raw(holder, p):
        eng_c.get_ll(holder, np.asarray(p)[None, :], phase_maximize=False,
                     **kw)
        return (float(par.asnp(eng_c.d_h_out)[0])
                - 0.5 * float(par.asnp(eng_c.h_h_out)[0]))

    # df0 grid: log sweep + the trust-edge point, sorted
    df0s = np.unique(np.concatenate([
        np.logspace(np.log10(df0_min), np.log10(df0_max), n_df0),
        [df0_edge]]))

    rows = []           # dicts; saved to npz at the end
    anchor_fail = []

    for ri, f0c in enumerate(f0_list):
        # carrier placed off layer-center (the inmodel-validate convention)
        f0 = (int(f0c / layer_df) + 0.37) * layer_df
        ref = np.array([1e-22, f0, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
        holder = par.setup_ref(sc, ref)
        eng_c.get_ll(holder, ref[None, :], phase_maximize=False, **kw)
        snr = np.sqrt(float(par.asnp(eng_c.h_h_out)[0]))
        ref[0] *= snr_target / snr
        holder = par.setup_ref(sc, ref)
        eng_c.get_ll(holder, ref[None, :], phase_maximize=False, **kw)
        snr = np.sqrt(float(par.asnp(eng_c.h_h_out)[0]))

        t0 = time.perf_counter()
        c_ref = chunk_raw(holder, ref)
        s_ref = {nm: sighet_raw(sighet, ref, kn) for nm, kn in paths}
        print(f"\n[ref{ri}] f0={ref[1]*1e3:.4f} mHz  SNR={snr:.1f}  "
              f"chunk(ref)={c_ref:+.4e}")
        print("[ref{}] ref-point offset vs chunk: ".format(ri)
              + "  ".join(f"{nm} {s_ref[nm]-c_ref:+.3e}" for nm in pnames))

        def measure(label, arm, p, df0_for_rad):
            D_true = chunk_raw(holder, p) - c_ref
            errs = {nm: abs((sighet_raw(sighet, p, kn) - s_ref[nm]) - D_true)
                    for nm, kn in paths}
            rad = 2.0 * np.pi * df0_for_rad * Tobs
            alw = allowed_of(D_true)
            rows.append(dict(ref=ri, arm=arm, label=label,
                             df0=df0_for_rad, rad=rad, D_true=D_true,
                             allowed=alw,
                             **{f"err_{nm}": errs[nm] for nm in pnames}))
            print(f"    {arm:5s} {label:16s} df0={df0_for_rad:+.3e} "
                  f"rad={rad:+9.3f} T={abs(D_true):11.3f} alw={alw:8.2f}  "
                  + " ".join(f"{nm}={errs[nm]:10.4f}" for nm in pnames))
            return D_true, errs

        # ---- arm 1: df0 sweep -------------------------------------------
        for df0 in df0s:
            p = ref.copy()
            p[1] += df0
            D_true, errs = measure(f"df0", "df0", p, df0)
            # anchors
            if df0 == df0s[0]:
                for nm in pnames:
                    if errs[nm] > 0.1:
                        anchor_fail.append(
                            f"ref{ri} smallest-df0 anchor: {nm} err "
                            f"{errs[nm]:.3f} > 0.1")
            if df0 == df0_edge:
                gate = max(0.5, 0.05 * abs(D_true))
                for nm in pnames:
                    if errs[nm] > gate:
                        anchor_fail.append(
                            f"ref{ri} trust-edge anchor: {nm} err "
                            f"{errs[nm]:.3f} > {gate:.3f}")
        if anchor_fail and not no_gate:
            print("\n[ANCHOR FAILURE] harness not trustworthy -- stopping "
                  "before producing a curve:")
            for a in anchor_fail:
                print("   ", a)
            sys.exit(2)

        # sign check at 3 magnitudes
        for df0 in (1e-8, 1e-6, 1e-5):
            if df0_min <= df0 <= df0_max:
                p = ref.copy()
                p[1] -= df0
                measure("neg", "df0-", p, -df0)

        # ---- arm 2: sky displacement ------------------------------------
        rng = np.random.default_rng(101 + ri)
        for j in range(n_sky):
            lam = rng.uniform(0, 2 * np.pi)
            beta = np.arcsin(rng.uniform(-1, 1))
            for tag, df0 in (("sky0", 0.0), ("skyE", df0_edge)):
                p = ref.copy()
                p[1] += df0
                p[7], p[8] = lam, beta
                measure(f"{tag}_{j}", "sky", p, df0)

        # ---- arm 3: combined worst case ---------------------------------
        lam_a = np.mod(ref[7] + np.pi, 2 * np.pi)
        beta_a = -ref[8]
        for df0 in (2e-6, 1e-5, 1e-4):
            if df0 > df0_max:
                continue
            p = ref.copy()
            p[1] += df0
            p[7], p[8] = lam_a, beta_a
            measure("antipode", "comb", p, df0)
        print(f"[ref{ri}] arm time {time.perf_counter()-t0:.1f} s")

    # ---- crossing summary ------------------------------------------------
    print("\n[summary] df0-arm crossing per path "
          "(first sweep point with err > allowed(T)):")
    T90 = 90 * 86400.0
    for nm in pnames:
        sel = sorted((r for r in rows if r["arm"] == "df0"),
                     key=lambda r: r["df0"])
        first_fail = None
        last_pass = None
        for r in sel:
            if r[f"err_{nm}"] <= r["allowed"]:
                if first_fail is None:
                    last_pass = r
            else:
                if first_fail is None:
                    first_fail = r
        if last_pass is None:
            print(f"    {nm:8s}: FAILS at every sweep point")
            continue
        lp_rad = last_pass["rad"]
        msg = (f"    {nm:8s}: last-pass df0={last_pass['df0']:.3e} Hz "
               f"({lp_rad:.1f} rad)")
        if first_fail is not None:
            msg += (f"  first-fail df0={first_fail['df0']:.3e} Hz "
                    f"({first_fail['rad']:.1f} rad, "
                    f"err={first_fail[f'err_{nm}']:.2f} > "
                    f"alw={first_fail['allowed']:.2f})")
        else:
            msg += "  (never fails in sweep range)"
        # transfer to 90 d + implied reference spacing (half-spacing = tol)
        df0_90 = lp_rad / (2 * np.pi * T90)
        spacing = 2.0 * df0_90
        n_ref = (2e-2 - 1e-4) / spacing if spacing > 0 else np.inf
        msg += (f"  | @90d tol={df0_90:.3e} Hz -> ref spacing "
                f"{spacing:.3e} Hz -> {n_ref:.3g} refs over 0.1-20 mHz")
        print(msg)

    npz = os.path.join(out_dir, "f0_tolerance.npz")
    np.savez(npz,
             arm=np.array([r["arm"] for r in rows]),
             label=np.array([r["label"] for r in rows]),
             ref=np.array([r["ref"] for r in rows]),
             df0=np.array([r["df0"] for r in rows]),
             rad=np.array([r["rad"] for r in rows]),
             D_true=np.array([r["D_true"] for r in rows]),
             allowed=np.array([r["allowed"] for r in rows]),
             **{f"err_{nm}": np.array([r[f"err_{nm}"] for r in rows])
                for nm in pnames},
             paths=np.array(pnames), Tobs=Tobs, backend=backend,
             n_r=nr, K=K, band=band, n_sparse_fd=nsfd,
             Nf=g["Nf"], dt=g["dt"])
    print(f"[save] {npz}")
    if anchor_fail:
        print("\n[WARN] anchors failed (gate disabled); curve untrusted.")


if __name__ == "__main__":
    main()
