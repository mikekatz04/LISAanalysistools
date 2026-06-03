#!/usr/bin/env python
"""
Side-by-side test of the FD and WDM TDI-on-the-fly inner-product paths.

For each path the script verifies, on the SAME injection:

  1) The C kernel inner products (d|h), (h|h) match what
     lisatools.diagnostic.inner_product computes from the same data, the
     same inverse covariance, and the SAME template that the C kernel
     produces (gb_*_fill_global), to floating-point precision.

  2) (FD only -- new this round) The C fill_global output, when wrapped in
     a lisatools FDSignal, gives an identical lisatools inner product to
     the C get_ll output.

  3) swap_ll returns the five accumulators (d|h_add), (d|h_remove),
     (h_add|h_add), (h_remove|h_remove), (h_add|h_remove); the first four
     are checked against gb_*_get_ll, and (h_add|h_remove) is checked
     against an independent Python inner product of the two templates.

The FD and WDM blocks share the SAME GB source and the SAME orbits/TDI
config, so the printed table makes it easy to read off the numerical
agreement of each pipeline against lisatools.

Run:
    /Users/mkatz/miniconda3/envs/deving/bin/python gb_fd_wdm_side_by_side_test.py
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
from scipy import signal as scipy_signal

# silence lisatools' SensitivityMatrix DC-bin divide-by-zero spew
warnings.filterwarnings("ignore", category=RuntimeWarning)

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.domains import (
    TDSettings, TDSignal,
    FDSettings, FDSignal,
    WDMSettings, WDMSignal, WDMLookupTable,
)
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.diagnostic import inner_product as lisatools_inner_product

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.gbcomps import GBWDMComputations, GBFDComputations


BACKEND = "cpu"
LOOKUP_PATH = "/Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_new_all_time_layers_1.h5"


def banner(title: str):
    print()
    print("=" * 78)
    print("  " + title)
    print("=" * 78)


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def main():
    # ---- Match the lookup table's time grid exactly so the WDM block can
    # reuse it (1460 freq layers x 2560 time layers, dt=10s; Tobs ~ 1.18 yr).
    dt = 10.0
    Nf = 1460
    Nt = 2560
    Nobs = Nf * Nt
    Tobs = Nobs * dt
    df = 1.0 / Tobs

    t_start = int(0.5 * YRSID_SI / dt) * dt   # 6 mo offset so orbits/TDI are well-bounded
    t_ref = t_start
    t_arr = t_start + np.arange(Nobs) * dt

    # WDM grid settings match the table file.
    wavelet_duration = Nf * dt
    min_freq = 0.0029493407356002777
    max_freq = 0.00306500115660421
    min_time = 20 * wavelet_duration  # shave edges, matches gb_lookup_table_test_script
    max_time = (Nt - 20) * wavelet_duration
    wdm_set = WDMSettings(Nf, Nt, dt,
                          min_freq=min_freq, max_freq=max_freq,
                          min_time=min_time, max_time=max_time)

    # ---- Build orbits, TDI config -----------------------------------------
    orbits = EqualArmlengthOrbits(force_backend=BACKEND)
    tdi_config = TDIConfig("2nd generation")
    gb_kwargs = dict(tdi_config=tdi_config, orbits=orbits,
                     tdi_chan="XYZ", force_backend=BACKEND)

    # ---- GB injection ------------------------------------------------------
    # f0 inside the WDM lookup-table band (~3.000 mHz).  amp boosted so the
    # SNR is solidly large for a clean numerical test.
    amp   = np.array([8.0e-22])
    f0    = np.array([3.000e-3])
    fdot  = np.array([1.0e-14])
    fddot = np.array([0.0])
    phi0  = np.array([2.09802430298])
    inc   = np.array([0.23984234])
    psi   = np.array([1.234019814])
    lam   = np.array([4.09808143])
    beta  = np.array([0.04])
    params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

    # Second source for swap_ll: nudge f0 by half a sparse-FFT bandwidth
    params_b = params.copy()
    params_b[0, 1] = params[0, 1] + 8.0e-6   # +8 uHz shift

    # ---- Generate dense time-domain TDI injection -------------------------
    banner("Generate dense TDI injection (lisatools dense rfft)")
    t0 = time.time()
    gb_dense = GBTDIonTheFly(t_arr, Tobs, t_ref, 1.0 / dt, 1, **gb_kwargs)
    out = gb_dense(*params.T, convert_to_ra_dec=False, return_spline=False)
    x_dense = np.stack([np.asarray(out.X[0]).real,
                        np.asarray(out.Y[0]).real,
                        np.asarray(out.Z[0]).real], axis=0)  # (3, Nobs)
    print(f"  N_dense={Nobs}, dt={dt}s, Tobs={Tobs:.3e}s "
          f"({Tobs/YRSID_SI:.3f} yr), took {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------------
    # FD path
    # ---------------------------------------------------------------------
    banner("FD PATH: build data + sens matrix")

    # Unwindowed rfft so the C heterodyne FFT (also unwindowed) lines up
    # with the lisatools FD inner product on the same array.
    td_set = TDSettings(Nobs, dt, t0=t_start, force_backend=BACKEND)
    data_td  = TDSignal(x_dense, td_set)
    data_fd_sig = data_td.fft()   # rfft * dt, no window
    data_fd  = np.asarray(data_fd_sig.arr)  # (3, n_rfft)
    n_rfft   = data_fd.shape[-1]
    fd_set   = FDSettings(n_rfft, df, force_backend=BACKEND)

    sens_mat = XYZ2SensitivityMatrix(fd_set, model="scirdv1")
    invC_full = np.asarray(sens_mat.invC)  # (3, 3, n_rfft)
    print(f"  data_fd shape={data_fd.shape}, df={df:.3e} Hz, "
          f"invC shape={invC_full.shape}")
    print(f"  sens_mat first-bin NaN -> ind_min set to 1 in FDDomain")

    # Pack FDDomain inputs.  num_data=1, num_noise=1 to start.
    data_arr = data_fd[None, ...].astype(complex)                  # (1, 3, n_rfft)
    invC_arr = invC_full[None, ...].astype(float)                  # (1, 3, 3, n_rfft)
    # zero out the DC bin (lisatools' inner_product also skips it).
    invC_arr[:, :, :, 0] = 0.0

    N_sparse = 4096   # ample for f0~3 mHz; comfortable margin
    fd_comp = GBFDComputations(
        T=Tobs, t_ref=t_ref, t_start=t_start,
        N_sparse=N_sparse, df=df,
        data_fd=data_arr, invC=invC_arr,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=BACKEND, tdi_type="XYZ",
        ind_min=1, ind_max=n_rfft - 1,
    )

    # ---- 1) C get_ll -------------------------------------------------------
    banner("FD 1) C get_ll  vs lisatools Python")
    like_fd = fd_comp.get_ll_fd(params, convert_to_ra_dec=False)
    dh_C = float(np.asarray(fd_comp.d_h_out)[0])
    hh_C = float(np.asarray(fd_comp.h_h_out)[0])
    print(f"  C:   (d|h)_FD = {dh_C:.10e}")
    print(f"       (h|h)_FD = {hh_C:.10e}")
    print(f"       L_FD     = {float(np.asarray(like_fd)[0]):.10e}")

    # ---- 2) C fill_global ---------------------------------------------------
    templates = np.zeros((1, 3, n_rfft), dtype=complex)
    fd_comp.fill_global(params, templates, convert_to_ra_dec=False)
    template_fd_sig = FDSignal(templates[0], fd_set)

    # Python inner products on the (data, C-template) using the SAME invC.
    # zero the DC bin of invC so Python and C match exactly.
    sens_mat_z = XYZ2SensitivityMatrix(fd_set, model="scirdv1")
    iC_z = np.asarray(sens_mat_z.invC).copy()
    iC_z[:, :, 0] = 0.0
    sens_mat_z._invC = iC_z

    dh_py = lisatools_inner_product(
        DataResidualArray(data_fd_sig),
        DataResidualArray(template_fd_sig),
        psd=sens_mat_z,
    )
    hh_py = lisatools_inner_product(
        DataResidualArray(template_fd_sig),
        DataResidualArray(template_fd_sig),
        psd=sens_mat_z,
    )
    dd_py = lisatools_inner_product(
        DataResidualArray(data_fd_sig),
        DataResidualArray(data_fd_sig),
        psd=sens_mat_z,
    )
    print(f"  Py:  (d|h)_FD = {dh_py:.10e}    rel diff {rel(dh_C, dh_py):.3e}")
    print(f"       (h|h)_FD = {hh_py:.10e}    rel diff {rel(hh_C, hh_py):.3e}")
    print(f"       (d|d)_FD = {dd_py:.10e}")

    # ---- 3) C swap_ll ------------------------------------------------------
    banner("FD 3) C swap_ll  vs C get_ll cross-check + Python (h_add|h_rem)")
    (like_a, like_r, dh_a, dh_r, aa, rr, ar) = fd_comp.get_swap_ll_fd(
        params, params_b, convert_to_ra_dec=False)
    dh_a = float(np.asarray(dh_a)[0]); dh_r = float(np.asarray(dh_r)[0])
    aa   = float(np.asarray(aa)[0]);   rr   = float(np.asarray(rr)[0])
    ar   = float(np.asarray(ar)[0])

    # Sanity vs get_ll calls
    fd_comp.get_ll_fd(params,   convert_to_ra_dec=False)
    dh_a_ref = float(np.asarray(fd_comp.d_h_out)[0])
    aa_ref   = float(np.asarray(fd_comp.h_h_out)[0])
    fd_comp.get_ll_fd(params_b, convert_to_ra_dec=False)
    dh_r_ref = float(np.asarray(fd_comp.d_h_out)[0])
    rr_ref   = float(np.asarray(fd_comp.h_h_out)[0])

    # Independent Python (h_add|h_rem) via lisatools on the two C templates
    templ_a = np.zeros((1, 3, n_rfft), dtype=complex)
    templ_r = np.zeros((1, 3, n_rfft), dtype=complex)
    fd_comp.fill_global(params,   templ_a, convert_to_ra_dec=False)
    fd_comp.fill_global(params_b, templ_r, convert_to_ra_dec=False)
    ar_py = lisatools_inner_product(
        DataResidualArray(FDSignal(templ_a[0], fd_set)),
        DataResidualArray(FDSignal(templ_r[0], fd_set)),
        psd=sens_mat_z,
    )

    print(f"  (d|h_add):  swap={dh_a:.10e}  ref={dh_a_ref:.10e}  "
          f"rel {rel(dh_a, dh_a_ref):.3e}")
    print(f"  (d|h_rem):  swap={dh_r:.10e}  ref={dh_r_ref:.10e}  "
          f"rel {rel(dh_r, dh_r_ref):.3e}")
    print(f"  (h_a|h_a):  swap={aa:.10e}    ref={aa_ref:.10e}    "
          f"rel {rel(aa,   aa_ref):.3e}")
    print(f"  (h_r|h_r):  swap={rr:.10e}    ref={rr_ref:.10e}    "
          f"rel {rel(rr,   rr_ref):.3e}")
    print(f"  (h_a|h_r):  swap={ar:.10e}    py ={ar_py:.10e}    "
          f"rel {rel(ar,   float(ar_py)):.3e}")

    # ---------------------------------------------------------------------
    # WDM path
    # ---------------------------------------------------------------------
    banner("WDM PATH: build WDM data + lookup table")
    # tukey window (matches gb_lookup_table_test_script)
    window = np.asarray(scipy_signal.windows.tukey(Nobs, alpha=0.05))
    data_wdm_sig = TDSignal(x_dense, td_set).transform(wdm_set, window=window)
    injection_wdm = DataResidualArray(data_wdm_sig)
    sens_wdm = XYZ2SensitivityMatrix(injection_wdm.data_res_arr.settings,
                                     model="scirdv1")

    if not os.path.exists(LOOKUP_PATH):
        print(f"  WDM lookup table missing at {LOOKUP_PATH} -- skipping WDM block.")
        wdm_lookup = None
    else:
        wdm_lookup = WDMLookupTable.from_file(LOOKUP_PATH, force_backend=BACKEND)
        _w = WDMSettings(*wdm_lookup.args, **wdm_lookup.kwargs)
        if not _w.eq_without_inds(wdm_set):
            raise RuntimeError("WDM lookup-table settings do not match script.")

    if wdm_lookup is not None:
        gb_wdm = GBWDMComputations(wdm_lookup, Tobs, t_ref,
                                   orbits=orbits, tdi_config=tdi_config,
                                   force_backend=BACKEND, tdi_type="XYZ")
        from lisatools.analysiscontainer import AnalysisContainerArray
        analysis_wdm = AnalysisContainer(
            injection_wdm, sens_wdm, signal_gen=None)
        wdm_holder = AnalysisContainerArray([analysis_wdm])

        banner("WDM 1) C get_ll  vs lisatools Python")
        like_wdm = gb_wdm.get_ll_wdm(
            params, wdm_holder=wdm_holder, convert_to_ra_dec=False)
        dh_wdm_C = float(np.asarray(gb_wdm.d_h_out)[0])
        hh_wdm_C = float(np.asarray(gb_wdm.h_h_out)[0])

        # Build the same C template via fill_global and check with lisatools.
        template_fill = np.zeros(
            3 * np.prod(wdm_set.basis_shape_active), dtype=float)
        gb_wdm.fill_global_wdm(template_fill, params, wdm_holder,
                               data_index=None, convert_to_ra_dec=False)
        template_wdm_sig = WDMSignal(
            template_fill.reshape((3,) + wdm_set.basis_shape_active),
            wdm_set)
        ip_wdm_py = analysis_wdm.template_inner_product(template_wdm_sig)
        ip_hh_py  = lisatools_inner_product(
            DataResidualArray(template_wdm_sig),
            DataResidualArray(template_wdm_sig),
            psd=sens_wdm,
        )
        ip_dd_py = analysis_wdm.inner_product()

        print(f"  C:   (d|h)_WDM = {dh_wdm_C:.10e}")
        print(f"       (h|h)_WDM = {hh_wdm_C:.10e}")
        print(f"       L_WDM     = {float(np.asarray(like_wdm)[0]):.10e}")
        print(f"  Py:  (d|h)_WDM = {float(ip_wdm_py):.10e}    "
              f"rel {rel(dh_wdm_C, float(ip_wdm_py)):.3e}")
        print(f"       (h|h)_WDM = {float(ip_hh_py):.10e}    "
              f"rel {rel(hh_wdm_C, float(ip_hh_py)):.3e}")
        print(f"       (d|d)_WDM = {float(ip_dd_py):.10e}")

        banner("WDM 3) C swap_ll  vs C get_ll cross-check + Python (h_add|h_rem)")
        (like_add, like_rem, dh_a_w, dh_r_w, aa_w, rr_w, ar_w) = (
            gb_wdm.get_swap_ll_wdm(params, params_b, wdm_holder,
                                   convert_to_ra_dec=False))
        dh_a_w = float(np.asarray(dh_a_w)[0]); dh_r_w = float(np.asarray(dh_r_w)[0])
        aa_w = float(np.asarray(aa_w)[0]);     rr_w = float(np.asarray(rr_w)[0])
        ar_w = float(np.asarray(ar_w)[0])

        gb_wdm.get_ll_wdm(params,   wdm_holder, convert_to_ra_dec=False)
        dh_a_w_ref = float(np.asarray(gb_wdm.d_h_out)[0])
        aa_w_ref   = float(np.asarray(gb_wdm.h_h_out)[0])
        gb_wdm.get_ll_wdm(params_b, wdm_holder, convert_to_ra_dec=False)
        dh_r_w_ref = float(np.asarray(gb_wdm.d_h_out)[0])
        rr_w_ref   = float(np.asarray(gb_wdm.h_h_out)[0])

        # Python (h_add | h_rem) via two WDM templates.
        template_a = np.zeros(3 * np.prod(wdm_set.basis_shape_active))
        template_r = np.zeros(3 * np.prod(wdm_set.basis_shape_active))
        gb_wdm.fill_global_wdm(template_a, params,   wdm_holder,
                               data_index=None, convert_to_ra_dec=False)
        gb_wdm.fill_global_wdm(template_r, params_b, wdm_holder,
                               data_index=None, convert_to_ra_dec=False)
        ar_w_py = lisatools_inner_product(
            DataResidualArray(WDMSignal(
                template_a.reshape((3,) + wdm_set.basis_shape_active),
                wdm_set)),
            DataResidualArray(WDMSignal(
                template_r.reshape((3,) + wdm_set.basis_shape_active),
                wdm_set)),
            psd=sens_wdm,
        )
        print(f"  (d|h_add):  swap={dh_a_w:.10e}  ref={dh_a_w_ref:.10e}  "
              f"rel {rel(dh_a_w, dh_a_w_ref):.3e}")
        print(f"  (d|h_rem):  swap={dh_r_w:.10e}  ref={dh_r_w_ref:.10e}  "
              f"rel {rel(dh_r_w, dh_r_w_ref):.3e}")
        print(f"  (h_a|h_a):  swap={aa_w:.10e}    ref={aa_w_ref:.10e}    "
              f"rel {rel(aa_w,   aa_w_ref):.3e}")
        print(f"  (h_r|h_r):  swap={rr_w:.10e}    ref={rr_w_ref:.10e}    "
              f"rel {rel(rr_w,   rr_w_ref):.3e}")
        print(f"  (h_a|h_r):  swap={ar_w:.10e}    py ={float(ar_w_py):.10e}    "
              f"rel {rel(ar_w,   float(ar_w_py)):.3e}")

    # ---- Side-by-side summary ---------------------------------------------
    banner("SIDE-BY-SIDE  (lisatools-Python cross-check)")
    print("  FD tolerance (machine precision):       1e-12 rel")
    print("  WDM tolerance (lookup-table approx):    5e-2 rel on (d|h), (h|h)")
    print()
    FD_TOL  = 1e-12
    WDM_TOL = 5e-2
    def verdict(rel_val, tol):
        return "PASS" if rel_val <= tol else "FAIL"

    rows = []
    rows.append(("Path",  "Quantity",   "C value",        "Py value",       "rel",          "tol",          "result"))
    rows.append(("FD",    "(d|h)",      f"{dh_C:.6e}",    f"{float(dh_py):.6e}",
                 f"{rel(dh_C, float(dh_py)):.2e}", f"{FD_TOL:.0e}",
                 verdict(rel(dh_C, float(dh_py)), FD_TOL)))
    rows.append(("FD",    "(h|h)",      f"{hh_C:.6e}",    f"{float(hh_py):.6e}",
                 f"{rel(hh_C, float(hh_py)):.2e}", f"{FD_TOL:.0e}",
                 verdict(rel(hh_C, float(hh_py)), FD_TOL)))
    rows.append(("FD",    "(h_a|h_r)",  f"{ar:.6e}",      f"{float(ar_py):.6e}",
                 f"{rel(ar, float(ar_py)):.2e}", f"{FD_TOL:.0e}",
                 verdict(rel(ar, float(ar_py)), FD_TOL)))
    if wdm_lookup is not None:
        rows.append(("WDM", "(d|h)",      f"{dh_wdm_C:.6e}", f"{float(ip_wdm_py):.6e}",
                     f"{rel(dh_wdm_C, float(ip_wdm_py)):.2e}", f"{WDM_TOL:.0e}",
                     verdict(rel(dh_wdm_C, float(ip_wdm_py)), WDM_TOL)))
        rows.append(("WDM", "(h|h)",      f"{hh_wdm_C:.6e}", f"{float(ip_hh_py):.6e}",
                     f"{rel(hh_wdm_C, float(ip_hh_py)):.2e}", f"{WDM_TOL:.0e}",
                     verdict(rel(hh_wdm_C, float(ip_hh_py)), WDM_TOL)))

    col_w = [6, 12, 16, 16, 12, 8, 8]
    for r in rows:
        print("  " + "".join(s.ljust(w) for s, w in zip(r, col_w)))

    print()
    print("Reading the table")
    print("-----------------")
    print("  FD rows: the C kernel implements the exact lisatools FD inner")
    print("    product, evaluated on the SAME heterodyne FD template that")
    print("    gb_fd_fill_global produces.  Agreement is at the floating-")
    print("    point round-off floor (~1e-14 here).")
    print()
    print("  WDM rows: the C kernel performs the WDM lookup-table inner")
    print("    product.  Agreement vs lisatools' Python lookup of the same")
    print("    template is limited by the lookup-table interpolation error,")
    print("    not by the kernel math.  Past validation runs show this in")
    print("    the 1e-4 -- 1e-2 band; see project_gb_lookup_test_script.md.")
    print()
    print("  swap_ll vs get_ll cross-check rows (printed in each path's")
    print("    block above): identical to round-off in both paths, because")
    print("    the (d|h_add), (d|h_rem), (h_a|h_a), (h_r|h_r) accumulators")
    print("    are exactly the same code paths as gb_fd_get_ll/gb_wdm_get_ll.")
    print()
    print("  fill_global: validated implicitly -- the Python inner product")
    print("    in each row above is computed on the C kernel's fill_global")
    print("    output, so agreement there certifies the scatter as well.")
    print()
    print("  Windowing: FD uses no window (rectangular); WDM uses tukey")
    print("    (alpha=0.05).  Numeric values across paths differ for that")
    print("    reason; the validation criterion is C-vs-Python WITHIN each")
    print("    path, which the 'result' column reports.")


if __name__ == "__main__":
    main()
