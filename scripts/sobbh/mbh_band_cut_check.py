"""MBH null-template residual vs a low-frequency cut, INSIDE the global fit.

Builds the MBH (id from MBHB_IDS) data + template at the exact catalogue
injection (factor=0) in the merger-centered window, then recomputes every inner
product KEEPING ONLY WDM frequency layers ABOVE a cut (f > f_cut). f_cut=0 is a
self-check: it must reproduce the full-band <r|r>. Tests whether the ~2e-3 MBH
residual is dominated by the loud sub-1mHz inspiral.
"""
import os
import threading
import time
import resource
import copy

os.environ.setdefault("DATA_PROCESSOR", "mojito")
os.environ.setdefault(
    "MOJITO_DATA_PATH",
    "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
)
os.environ.setdefault("MBHB_IDS", "1")
os.environ.setdefault("SOBHB_IDS", "")
os.environ.setdefault("EMRI_IDS", "")
os.environ.setdefault("CHOP_WINDOW", "1")
os.environ.setdefault("NWALKERS", "1")
os.environ.setdefault("NTEMPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import importlib.util
import numpy as np

CUTS = [0.0, 5e-4, 1e-3]


def _watchdog(limit_gb=28.0):
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > limit_gb:
            print(f"[watchdog] RSS over {limit_gb} GB; aborting", flush=True)
            os._exit(42)
        time.sleep(0.5)


def _load_settings():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        os.path.dirname(os.path.dirname(here)),
        "global_fit_input", "full_year_combined_global_fit_settings.py",
    )
    spec = importlib.util.spec_from_file_location("fy_settings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    fy = _load_settings()
    from mpi4py import MPI
    from eryn.state import BranchSupplemental
    from lisatools.globalfit.run import GlobalFit
    from lisatools.diagnostic import inner_product

    print(f"[cfg] MBHB_IDS={fy.MOJITO_SOURCE_IDS['MBHB']} CHOP_WINDOW={fy.CHOP_WINDOW} "
          f"DT={fy.DT} TOBS={fy.TOBS:.4e}s ({fy.TOBS/86400:.1f}d) NF={fy.NF} NT={fy.NT}", flush=True)

    curr = fy.get_global_fit_settings()
    gi = curr.general_info
    setup = curr.source_info["mbh"]
    comm = MPI.COMM_WORLD
    if os.path.exists(gi.main_file_path):
        os.remove(gi.main_file_path)
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
    inj = np.atleast_2d(np.asarray(setup.injection, dtype=float))
    state.branches_coords["mbh"][:] = inj[None, None]

    acs_data = gf.setup_acs(state, rebuild_residuals=False)
    ac = acs_data.flatten()[0]
    sens = ac.sens_mat

    _t0 = time.time()
    h = setup.signal_gen(*inj[0])
    print(f"[template] built MBH template in {time.time() - _t0:.1f}s", flush=True)

    base_d = ac._data
    settings = base_d.data_res_arr.settings
    Nf, Nt, layer_df = settings.Nf, settings.Nt, settings.layer_df
    ind_min_f, ind_max_f = settings._ind_min_f, settings._ind_max_f
    nch = ac.data.nchannels if hasattr(ac.data, "nchannels") else 3
    arr_shape = np.asarray(base_d.data_res_arr.arr).shape
    # The WDM array stores only the IN-BAND layers (both freq AND time are
    # band-restricted: freq ~ ind_max_f-ind_min_f, time ~ Nt - 2*time_margin),
    # so neither axis equals Nf or Nt. Identify the freq axis as the non-channel
    # axis whose size is closest to the in-band freq-layer count.
    expected_nf = ind_max_f - ind_min_f
    cand = [i for i in range(len(arr_shape)) if i != 0]  # skip channel axis 0
    freq_axis = min(cand, key=lambda i: abs(arr_shape[i] - expected_nf))
    n_layers = arr_shape[freq_axis]
    # array-layer j -> global WDM layer (ind_min_f + j) -> freq (ind_min_f+j)*layer_df
    global_layer = ind_min_f + np.arange(n_layers)
    freqs = global_layer * layer_df
    print(f"[wdm] arr_shape={arr_shape} nch={nch} Nf={Nf} Nt={Nt} freq_axis={freq_axis} "
          f"n_layers={n_layers} layer_df={layer_df:.6e}", flush=True)
    print(f"[wdm] in-band freqs span [{freqs[0]:.3e}, {freqs[-1]:.3e}] Hz "
          f"(ind_min_f={ind_min_f}, ind_max_f={ind_max_f})", flush=True)

    print("\n=========  MBH residual vs low-frequency cut (keep f > f_cut)  =========", flush=True)
    print(f"  {'f_cut[Hz]':>10}  {'m_cut':>6}  {'<d|d>':>12}  {'<h|h>':>12}  {'<d|h>':>12}  "
          f"{'<r|r>':>12}  {'overlap':>10}  {'mismatch':>10}  {'-0.5<r|r>':>12}", flush=True)
    for f_cut in CUTS:
        mask = freqs < f_cut  # zero layers BELOW the cut, keep above
        m_cut = int(mask.sum())
        d_cut = copy.deepcopy(base_d)
        h_cut = copy.deepcopy(h)
        sl = [slice(None)] * len(arr_shape)
        sl[freq_axis] = mask
        d_cut.data_res_arr.arr[tuple(sl)] = 0.0
        h_cut.arr[tuple(sl)] = 0.0
        dd = float(np.real(inner_product(d_cut, d_cut, psd=sens)))
        hh = float(np.real(inner_product(h_cut, h_cut, psd=sens)))
        dh = complex(inner_product(d_cut, h_cut, psd=sens, complex=True))
        rr = dd + hh - 2.0 * dh.real
        ov = dh.real / np.sqrt(dd * hh)
        print(f"  {f_cut:>10.1e}  {m_cut:>6d}  {dd:>12.5e}  {hh:>12.5e}  {dh.real:>12.5e}  "
              f"{rr:>12.5e}  {ov:>10.6f}  {1-ov:>10.3e}  {-0.5*rr:>12.5e}", flush=True)
        del d_cut, h_cut

    print("\n  (f_cut=0 row must match the full-band <d|d>=1.923e4, <r|r>=3.772e1 self-check)", flush=True)


if __name__ == "__main__":
    main()
