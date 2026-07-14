"""Source-agnostic null-template check INSIDE the stock full_year_combined fit.

Drives the real pipeline (erebor.full_year_combined().build() ->
GlobalFit.load_info -> setup_acs -> acs.likelihood) for whichever single source
class is active (set MBHB_IDS / SOBHB_IDS / EMRI_IDS in the environment before
running), builds that branch's template at the EXACT catalogue injection
(factor=0) through the branch's engine-side ``signal_gen``
(``SourceSignalGen``: phentax MBH / TDI-on-the-fly SOBBH / legacy EMRI — the
same generators the production runs sample with), and reports every inner
product feeding the likelihood:

    <d|d>, <h|h>, <d|h>, <r|r>=<d|d>+<h|h>-2<d|h>, overlap, source term -0.5<r|r>,
    noise term, full logL  --  plus a setup_acs(rebuild_residuals=True) cross-check.

If the branch's wiring reproduces the mojito L1 source, <r|r>/<d|d> -> 0.

GB / VGB streams are handled by the sibling per-source fidelity scripts
(scripts/gb/gb_mojito_match.py, scripts/gb/vgb_mojito_match.py) — see the
run_mojito_null_checks.sh driver next to this file.
"""
import os
import threading
import time
import resource

os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault(
    "MOJITO_DATA_PATH",
    "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
)
os.environ.setdefault("NWALKERS", "1")
os.environ.setdefault("NTEMPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # MPI-only parallelism policy
os.environ.setdefault("MAKE_PLOTS", "0")

import numpy as np


def _watchdog(limit_gb=float(os.environ.get("NULL_CHECK_MEM_GB", "24"))):
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > limit_gb:
            print(f"[watchdog] RSS over {limit_gb} GB; aborting (set NULL_CHECK_MEM_GB to change)",
                  flush=True)
            os._exit(42)
        time.sleep(0.5)


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    from mpi4py import MPI
    from eryn.state import BranchSupplemental
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME
    from lisatools.globalfit.run import GlobalFit
    from lisatools.globalfit.stock import erebor

    # Stock variant build: source ids / window / grid all resolve from the
    # same env vars the production runs use (EMRI_IDS / MBHB_IDS / SOBHB_IDS,
    # CHOP_WINDOW, TOBS_TARGET, DT-family defaults).
    fit = erebor.full_year_combined(nwalkers=1, ntemps=1)
    gs = fit.general
    active = [k for k, v in gs.mojito_source_ids.items() if v]
    print(f"[cfg] data_mode={gs.data_mode} source_ids={gs.mojito_source_ids} "
          f"CHOP_WINDOW={gs.chop_window} DT={gs.dt} "
          f"tobs_target={gs.tobs_target:.4e}s ({gs.tobs_target/86400:.1f}d)", flush=True)
    print(f"[cfg] use_gpu={gs.use_gpu} gpu_backend={gs.gpu_backend} gpus={gs.gpus}", flush=True)

    curr = fit.build()
    gi = curr.general_info
    branch = [n for n in curr.source_info if n not in ("psd", "galfor")][0]
    setup = curr.source_info[branch]
    print(f"[branch] active class={active} -> branch={branch!r}  "
          f"signal_gen={type(setup.signal_gen).__name__}  "
          f"data_t0={gi.data_t0:.4f}  REF={MOJITO_REFERENCE_TIME:.4f}  "
          f"off={gi.data_t0 - MOJITO_REFERENCE_TIME:.3f}s  "
          f"TOBS={gi.Tobs:.4e}s ({gi.Tobs/86400:.1f}d)", flush=True)

    comm = MPI.COMM_WORLD
    bp = gi.main_file_path
    if os.path.exists(bp):
        os.remove(bp)
    gf = GlobalFit(curr, comm)

    priors = {}
    for name in gf.engine_info.branch_names:
        si = curr.source_info.get(name)
        if si is None:
            continue
        p = si.get("priors") if isinstance(si, dict) else getattr(si, "priors", None)
        if p:
            priors.update(p)

    state = gf.load_info(priors)
    ntemps, nwalkers = gf.ntemps, gf.nwalkers
    state.supplemental = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nwalkers), (ntemps, 1))},
        base_shape=(ntemps, nwalkers), copy=True,
    )

    # factor=0: force every walker/temp to the EXACT injection coords so the
    # null is clean (load_info uses factor=1e-5 for mbh/sobbh).
    inj = np.asarray(setup.injection, dtype=float)
    if inj.ndim == 1:
        inj = inj[None, :]
    state.branches_coords[branch][:] = inj[None, None]
    print(f"[inject] {branch} injection (sampling, factor=0) =\n{inj}", flush=True)

    # data-only AC
    acs_data = gf.setup_acs(state, rebuild_residuals=False)
    ac = acs_data.flatten()[0]
    d_d = float(np.ravel(np.asarray(ac.inner_product(complex=False)).real)[0])

    # template h at the injection coords (one leaf)
    _t0 = time.time()
    h = setup.signal_gen(*inj[0])
    print(f"[template] built {branch} template in {time.time() - _t0:.1f}s", flush=True)

    opt, det = ac.template_snr(h)
    h_h = float(opt) ** 2
    d_h = complex(ac.non_marg_d_h)
    r_r = d_d + h_h - 2.0 * d_h.real
    logL_source = -0.5 * r_r
    # source_only=False explicitly: full_year runs a fixed PSD
    # (likelihood_source_only=True), which otherwise collides with
    # noise_only on the container.
    noise_term = float(ac.likelihood(noise_only=True, source_only=False))
    logL_full = float(ac.template_likelihood(h, include_psd_info=True))

    print(f"\n===========  {branch.upper()} inner products @ injection (global fit)  ===========", flush=True)
    print(f"  <d|d> (data . data)         = {d_d:.8e}     data SNR = {np.sqrt(abs(d_d)):.3f}", flush=True)
    print(f"  <h|h> (template . template) = {h_h:.8e}     opt SNR  = {opt:.3f}", flush=True)
    print(f"  <d|h> (data . template)     = {d_h.real:.8e}  (+{d_h.imag:.3e} i)  det SNR = {det:.3f}", flush=True)
    print(f"  <r|r> = <d|d>+<h|h>-2<d|h>  = {r_r:.8e}", flush=True)
    print(f"  overlap <d|h>/sqrt(<d|d><h|h>) = {d_h.real/np.sqrt(d_d*h_h):.8f}   "
          f"mismatch = {1 - d_h.real/np.sqrt(d_d*h_h):.3e}", flush=True)
    print(f"  source term   -0.5<r|r>     = {logL_source:.8e}", flush=True)
    print(f"  noise term    -0.5 logdet   = {noise_term:.8e}", flush=True)
    print(f"  FULL logL @ injection       = {logL_full:.8e}", flush=True)
    print(f"\n  => <r|r>/<d|d> = {r_r/d_d:.3e}   (template nulls the data  <=>  -> 0)", flush=True)

    acs_resid = gf.setup_acs(state, rebuild_residuals=True)
    rr_pipe = float(np.ravel(np.asarray(acs_resid.inner_product(complex=False)).real)[0])
    ll_pipe = float(np.ravel(np.asarray(acs_resid.likelihood(complex=False)).real)[0])
    xcheck = bool(np.isclose(rr_pipe, r_r, rtol=1e-4, atol=1.0))
    print("\n  ---- cross-check via the real residual pipeline (setup_acs rebuild_residuals=True) ----", flush=True)
    print(f"  pipeline <r|r>              = {rr_pipe:.8e}   (matches: {xcheck})", flush=True)
    print(f"  pipeline FULL logL          = {ll_pipe:.8e}   (matches: "
          f"{np.isclose(ll_pipe, logL_full, rtol=1e-4, atol=1.0)})", flush=True)

    # Single machine-parseable line for the run_mojito_null_checks.sh driver.
    ov = d_h.real / np.sqrt(d_d * h_h)
    print(
        f"[RESULT] branch={branch} data_snr={np.sqrt(abs(d_d)):.4f} dd={d_d:.6e} "
        f"hh={h_h:.6e} dh={d_h.real:.6e} rr={r_r:.6e} overlap={ov:.8f} "
        f"mismatch={1 - ov:.3e} rr_over_dd={r_r / d_d:.3e} "
        f"source_logL={logL_source:.6e} xcheck={xcheck}",
        flush=True,
    )


if __name__ == "__main__":
    main()
