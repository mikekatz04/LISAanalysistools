"""Verify the SPECIAL-frame EMRI template nulls the mojito EMRI #1 data INSIDE
the full_year_combined global-fit pipeline.

Setup: 1 EMRI (mojito id #1), no injected noise, factor=0 (the EMRI starting
coords == the injection sampling coords), 3-month Tobs (to save memory).

It drives the real pipeline exactly as run.py does up to its
"initial log likelihood" line --
    get_global_fit_settings() -> GlobalFit.load_info(priors)
    -> setup_acs(rebuild_residuals=True) -> acs.inner_product()/likelihood()
-- but stops before the sampler. With the SPECIAL EMRI frame wired into the
settings (ecliptic-polar sky + RAW file spin + is_ecliptic_latitude=False +
convert_to_ra_dec=True + ICRS orbits, t0=catalogue REF with the integer-sample
offset sliced), the template reproduces the mojito EMRI, so the residual inner
product <r|r> collapses to ~0 (<< the data <d|d>).
"""
import os
import threading
import time
import resource

# ---- env knobs consumed at settings-module import time (must precede import) ----
os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault(
    "MOJITO_DATA_PATH",
    "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
)
os.environ["EMRI_IDS"] = "1"   # single EMRI, catalogue id 1
os.environ["MBHB_IDS"] = ""    # no MBH
os.environ["SOBHB_IDS"] = ""   # no SOBBH
# 3-month observation window (YRSID_SI / 4) to keep the WDM grid / RAM small.
os.environ.setdefault("TOBS_TARGET", "7889537.440886401")
os.environ.setdefault("NWALKERS", "1")
os.environ.setdefault("NTEMPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import importlib.util
import numpy as np


def _memory_watchdog(limit_gb=24.0):
    while True:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        if rss > limit_gb:
            print(f"[watchdog] RSS {rss:.1f} GB > {limit_gb} GB; aborting", flush=True)
            os._exit(42)
        time.sleep(0.5)


def _load_settings_module():
    here = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(
        os.path.dirname(os.path.dirname(here)),  # .../LISAanalysistools
        "global_fit_input",
        "full_year_combined_global_fit_settings.py",
    )
    spec = importlib.util.spec_from_file_location("fy_settings", settings_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_priors(curr, gf):
    priors = {}
    for name in gf.engine_info.branch_names:
        si = curr.source_info.get(name)
        if si is None:
            continue
        p = si.get("priors") if isinstance(si, dict) else getattr(si, "priors", None)
        if p:
            priors.update(p)
    return priors


def main():
    threading.Thread(target=_memory_watchdog, daemon=True).start()

    fy = _load_settings_module()
    from mpi4py import MPI
    from eryn.state import BranchSupplemental
    from lisatools.globalfit.run import GlobalFit

    print(
        f"[cfg] DATA_PROCESSOR={fy.DATA_PROCESSOR} EMRI_IDS={fy.MOJITO_SOURCE_IDS['EMRI']} "
        f"DT={fy.DT} TOBS={fy.TOBS:.4e}s ({fy.TOBS/86400:.1f}d) "
        f"NF={fy.NF} NT={fy.NT} wavelet_dur={fy.WAVELET_DURATION:.0f}s",
        flush=True,
    )

    curr = fy.get_global_fit_settings()
    gi = curr.general_info
    ref = fy.MOJITO_REFERENCE_TIME
    off = gi.data_t0 - ref
    dt = gi.dt
    offset_int = int(round(off / dt))
    t0_shift = off - offset_int * dt
    print(
        f"[anchor] data_t0={gi.data_t0:.6f}  REF={ref:.6f}  off={off:.4f}s  "
        f"offset_int={offset_int}  t0_shift_to_data={t0_shift:+.5f}s  (|.|<dt={dt})",
        flush=True,
    )

    comm = MPI.COMM_WORLD
    bp = gi.main_file_path
    if os.path.exists(bp):
        os.remove(bp)  # fresh start -> load_info initializes from injection
        print(f"[init] removed stale backend {bp}", flush=True)

    gf = GlobalFit(curr, comm)
    priors = _build_priors(curr, gf)
    state = gf.load_info(priors)

    ntemps, nwalkers = gf.ntemps, gf.nwalkers
    state.supplemental = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nwalkers), (ntemps, 1))},
        base_shape=(ntemps, nwalkers), copy=True,
    )

    # --- single data AC (no template subtracted) ---
    acs_data = gf.setup_acs(state, rebuild_residuals=False)
    ac = acs_data.flatten()[0]
    d_d = float(np.asarray(ac.inner_product(complex=False)).real)

    # --- EMRI template h at the injection (factor=0) sampling coords ---
    emri_gen = curr.source_info["emri"].signal_gen
    coords_leaf = np.asarray(state.branches_coords["emri"][0, 0, 0], dtype=float)
    print(f"[template] injection sampling coords = {coords_leaf}", flush=True)
    _t0 = time.time()
    h = emri_gen(*coords_leaf)
    print(f"[template] built EMRI WDM template in {time.time() - _t0:.1f}s", flush=True)

    # All inner products feeding the likelihood:  <d|d>, <h|h>, <d|h>.
    opt, det = ac.template_snr(h)                  # opt=sqrt<h|h>, det=<d|h>/sqrt<h|h>
    h_h = float(opt) ** 2
    d_h = complex(ac.non_marg_d_h)                 # <d|h> (complex)
    r_r = d_d + h_h - 2.0 * d_h.real               # <(d-h)|(d-h)>
    logL_source = -0.5 * r_r                       # source term only
    noise_term = float(ac.likelihood(noise_only=True))
    logL_full = float(ac.template_likelihood(h, include_psd_info=True))

    print("\n===========  EMRI inner products @ injection (mojito id #1, global fit)  ===========", flush=True)
    print(f"  <d|d> (data . data)         = {d_d:.8e}     data SNR = {np.sqrt(d_d):.3f}", flush=True)
    print(f"  <h|h> (template . template) = {h_h:.8e}     opt SNR  = {opt:.3f}", flush=True)
    print(f"  <d|h> (data . template)     = {d_h.real:.8e}  (+{d_h.imag:.3e} i)  det SNR = {det:.3f}", flush=True)
    print(f"  <r|r> = <d|d>+<h|h>-2<d|h>  = {r_r:.8e}", flush=True)
    print(f"  overlap <d|h>/sqrt(<d|d><h|h>) = {d_h.real/np.sqrt(d_d*h_h):.8f}   "
          f"mismatch = {1 - d_h.real/np.sqrt(d_d*h_h):.3e}", flush=True)
    print("  ---- likelihood pieces ----", flush=True)
    print(f"  source term   -0.5<r|r>     = {logL_source:.8e}", flush=True)
    print(f"  noise term    -0.5 logdet   = {noise_term:.8e}", flush=True)
    print(f"  FULL logL @ injection       = {logL_full:.8e}", flush=True)
    print(f"\n  => <r|r>/<d|d> = {r_r/d_d:.3e}   (template nulls the data  <=>  -> 0)", flush=True)

    # --- cross-check: the ACTUAL global-fit residual path (build_template -> subtract) ---
    acs_resid = gf.setup_acs(state, rebuild_residuals=True)
    rr_pipe = float(np.ravel(np.asarray(acs_resid.inner_product(complex=False)).real)[0])
    ll_pipe = float(np.ravel(np.asarray(acs_resid.likelihood(complex=False)).real)[0])
    print("\n  ---- cross-check via the real residual pipeline (setup_acs rebuild_residuals=True) ----", flush=True)
    print(f"  pipeline <r|r>              = {rr_pipe:.8e}   (matches <r|r> above: "
          f"{np.isclose(rr_pipe, r_r, rtol=1e-5, atol=1.0)})", flush=True)
    print(f"  pipeline FULL logL          = {ll_pipe:.8e}   (matches FULL logL above: "
          f"{np.isclose(ll_pipe, logL_full, rtol=1e-5, atol=1.0)})", flush=True)


if __name__ == "__main__":
    main()
