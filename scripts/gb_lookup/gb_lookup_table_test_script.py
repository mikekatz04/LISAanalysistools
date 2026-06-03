#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
import matplotlib
# Use Agg by default so the verification run is non-interactive. If the user
# wants the heatmap window, they can override with MPLBACKEND=Qt5Agg etc.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    pass

from lisatools.detector import ESAOrbits, EqualArmlengthOrbits
from lisaconstants import ASTRONOMICAL_YEAR
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.response import icrs_to_ecliptic
from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.gbcomps import GBWDMComputations

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal, WDMSettings, WDMSignal, WDMLookupTable

from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

from eryn.moves import StretchMove
from eryn.ensemble import EnsembleSampler
from eryn.utils import PeriodicContainer

from eryn.state import State
from eryn.backends import HDFBackend
# credit Michael Katz and Alessandro Santini (with internal code contrubtions in docs)


    # HIGHLY RECOMMEND RUNNING THESE THINGS IN A SCRIPT IN THE TERMINAL, OTHERWISE BE CAREFUL TO RUN CELLS IN ORDER AS MUCH AS POSSIBLE

import time
class GBLookupWaveWrap:
    def __init__(self, t_arr, t_tdi_sparse, Tobs, t_ref, dt, num_bin, gb_tdi_kwargs, td_set, output_set, td_window):
        self.t_arr, self.t_tdi_sparse = t_arr, t_tdi_sparse
        self.Tobs, self.t_ref, self.dt, self.num_bin, self.gb_tdi_kwargs = Tobs, t_ref, dt, num_bin, gb_tdi_kwargs
        self.td_set, self.output_set = td_set, output_set
        assert isinstance(output_set, WDMSettings)
        self.td_window = td_window

        # TODO: maybe redo this?
        self.gb_gen = GBTDIonTheFly(
            self.t_tdi_sparse, self.Tobs, self.t_ref, self.dt, 1,
            **self.gb_tdi_kwargs
        )

    def __call__(self, *params):
        
        params = np.asarray([params])
        assert params.shape[-1] == 9
        # st = time.perf_counter()

        # print("check 1", time.perf_counter() - st)
        # st = time.perf_counter()
        wave_tmp = self.gb_gen(*params.T, convert_to_ra_dec=False, return_spline=True)

        t_arr = self.output_set.t_arr + self.t_ref
        f_deriv_tdi = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
        f_deriv_ref = wave_tmp.phase_ref_spl(t_arr[None, :], derivative=1)[0] / (2 * np.pi)
        f_deriv = f_deriv_ref + f_deriv_tdi

        fdot_deriv_tdi = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)), derivative=2)[0] / (2 * np.pi)
        fdot_deriv_ref = wave_tmp.phase_ref_spl(t_arr[None, :], derivative=2)[0]  / (2 * np.pi)
        fdot_deriv = fdot_deriv_ref + fdot_deriv_tdi
        tdi_amp = wave_tmp.tdi_amp_spl(np.tile(t_arr, (1, 3, 1)))[0]
        tdi_phase = wave_tmp.tdi_phase_spl(np.tile(t_arr, (1, 3, 1)))[0]
        ref_phase = wave_tmp.phase_ref_spl(t_arr[None, :])[0]
        
        # print("check 2", time.perf_counter() - st)
        # st = time.perf_counter()
        # breakpoint()
        # check_tdi_full = wave_tmp.eval_tdi(t_arr)
        # check_tdi_phase = -np.angle(check_tdi_full * np.exp(1j * ref_phase))

        # ref_phase = wave_tmp.phase_ref  # 2 * np.pi * int(f0[0] / wdm_settings.layer_df) * wdm_settings.layer_df * (output_deriv.t_arr - t_ref) 
        # tdi_phase = np.array([
        #     -np.angle(wave_tmp.X * np.exp(1j * ref_phase)),
        #     -np.angle(wave_tmp.Y * np.exp(1j * ref_phase)),
        #     -np.angle(wave_tmp.Z * np.exp(1j * ref_phase)),
        # ])

        # pi/2 PHASE SHIFT !!!!!!!!!!!!!!!!!!!!!!!!!
        # PHI_OFFSET env adds a constant (in rad) to the carrier phase passed
        # to the lookup — for diagnosing whether the residual at f_frac=0.5
        # is a constant phase-convention mismatch between build and eval.
        _phi_offset = float(os.environ.get("PHI_OFFSET", "0.0"))
        phi_t = ((tdi_phase + ref_phase) + np.pi / 2. + _phi_offset).flatten().copy()
        np.save("f_deriv", f_deriv)
        freq_t = f_deriv.flatten().copy()
        # FREEZE_FREQ=1 replaces the per-pixel Doppler-shifted instantaneous
        # frequency with the source's t=t_ref carrier frequency f0 (constant).
        # Diagnostic: if mm at f_frac=0.5 stays ~2e-4 with FREEZE_FREQ=1, the
        # residual is NOT from how Doppler-shifted f_t routes pixels to lookup
        # cells; if it gets worse, the lookup IS tracking Doppler properly.
        if os.environ.get("FREEZE_FREQ", "0") == "1":
            _f0_const = float(params[0, 1])  # f0 from input params
            freq_t = np.full_like(freq_t, _f0_const)
        # USE_FDOT_DERIV=1 passes the actual instantaneous fdot per pixel
        # through to the lookup. Default 0 keeps the previous fdot_t=0
        # behaviour so the fdot=0-only path stays exercised.
        # SLOPE_FDOT=1 (with WAVELET_EXTENT=K, default 3) replaces the
        # instantaneous fdot with a finite-difference slope over the
        # wavelet's effective time support:
        #   fdot_n = ( f(t_n + K*layer_dt) − f(t_n − K*layer_dt) ) / (2·K·layer_dt)
        # This better matches what the wavelet integrates (its support is
        # ~4-6 layer_dt, not a single pixel) and may capture frequency
        # drift over the wavelet window more accurately than the
        # instantaneous derivative.
        if os.environ.get("SLOPE_FDOT", "0") == "1":
            _ext = float(os.environ.get("WAVELET_EXTENT", "3"))
            _dt_step = _ext * self.output_set.layer_dt
            t_lo = t_arr - _dt_step
            t_hi = t_arr + _dt_step
            f_lo_tdi = wave_tmp.tdi_phase_spl(np.tile(t_lo, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
            f_lo_ref = wave_tmp.phase_ref_spl(t_lo[None, :], derivative=1)[0] / (2 * np.pi)
            f_hi_tdi = wave_tmp.tdi_phase_spl(np.tile(t_hi, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
            f_hi_ref = wave_tmp.phase_ref_spl(t_hi[None, :], derivative=1)[0] / (2 * np.pi)
            f_lo = f_lo_ref + f_lo_tdi
            f_hi = f_hi_ref + f_hi_tdi
            fdot_slope = (f_hi - f_lo) / (2.0 * _dt_step)
            # Print summary so the per-run mode is obvious.
            print(f"  [slope_fdot] extent={_ext} layer_dt ({_dt_step:.3e} s);"
                  f" max|fdot_slope|={np.abs(fdot_slope).max():.3e}"
                  f" vs max|fdot_inst|={np.abs(fdot_deriv).max():.3e}",
                  flush=True)
            fdot_t = fdot_slope.flatten().copy()
        elif os.environ.get("USE_FDOT_DERIV", "0") == "1":
            fdot_t = fdot_deriv.flatten().copy()
        else:
            fdot_t = np.full_like(freq_t, 0.0)
        amp_t = tdi_amp.flatten().copy()

        n_arr = np.tile(xp.arange(self.output_set.Nt)[self.output_set.active_slice_t], (3, 1))
        n_arr_in = n_arr.flatten().copy()
        n_min = self.output_set.ind_min_t
        m_min = self.output_set.ind_min_f

        _wdm_coeffs, _m_layers = wdm_lookup_table.get_wdm_coeffs(
            amp_t, phi_t, freq_t, fdot_t, n_arr_in,
            num_m_layers=int(os.environ.get("NUM_M_LAYERS", 2)),
        )
        wdm_coeffs = _wdm_coeffs.reshape(3, -1, _wdm_coeffs.shape[-1])
        m_layers = _m_layers.reshape(3, -1, _wdm_coeffs.shape[-1])
        n_layers = np.repeat(n_arr[:, :, None], m_layers.shape[-1], axis=-1)
        gb_fill_wave = xp.zeros((3, wdm_set.Nf_active, wdm_set.Nt_active))

        keep_m = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
        keep_n = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
        keep = keep_m & keep_n

        # print("check 3", time.perf_counter() - st)
        # st = time.perf_counter()
        channel_ind = np.repeat(np.arange(3)[:, None], m_layers.shape[-1] * m_layers.shape[-2], axis=-1).reshape(m_layers.shape)
        gb_fill_wave[channel_ind[keep], m_layers[keep] - m_min, n_layers[keep] - n_min] = wdm_coeffs[keep]
        # gb_fill_wave[:] = xp.roll(gb_fill_wave, 2, axis=-1)
        # print("check 4", time.perf_counter() - st)
        # st = time.perf_counter()
        gb_fill_wave_wdm = WDMSignal(gb_fill_wave, self.output_set)
        # print("check 5", time.perf_counter() - st)
        return gb_fill_wave_wdm


if __name__ == "__main__":
    backend = "cpu"

    xp = np if backend == "cpu" else cp

    orbits = ESAOrbits(force_backend=backend)
    # ORBIT_DT (s) or ORBIT_LINEAR=1 to configure orbits more densely.
    # Default base orbits use dt_base ~ 1.87 days with 5th-order interp.
    _orbit_dt = os.environ.get("ORBIT_DT", "")
    _orbit_linear = os.environ.get("ORBIT_LINEAR", "0") == "1"
    if _orbit_linear:
        orbits.configure(linear_interp_setup=True)
        print(f"[step] ORBITS: linear_interp_setup=True (dense linear interp)", flush=True)
    elif _orbit_dt.strip():
        orbits.configure(dt=float(_orbit_dt))
        print(f"[step] ORBITS: configured at dt={_orbit_dt}s with cubic spline", flush=True)
    else:
        print(f"[step] ORBITS: base (dt~1.87d, 5th-order interp)", flush=True)
    dt = 10.0  # mojito
    _Tobs = 1. * YRSID_SI
    # between half day and 3/4 day. Will be very close to half day
    # (Nf, Nt, wavelet_duration) = WDMSettings.adjust_to_even_bins(0.5 * 24 * 3600.0, 0.75 * 24 * 3600.0, dt, _Tobs)
    # NF, NT env overrides — change wavelet aspect at constant total signal
    # length (Nf*Nt) to test if dK/df numerical precision is the residual
    # source. Defaults preserve current 1460×2560 setup.
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))

    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig('2nd generation')  # mojito

    t_start = int(1 / 2 * YRSID_SI / dt) * dt  # 6 months
    t_arr = np.arange(Nobs) * dt + t_start
    data_inj = np.zeros((3, t_arr.shape[-1]))
    template = np.zeros((3, t_arr.shape[-1]))

    t_ref = t_start

    N_inj = 16384

    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_inj)
    gb_gen_inj = GBTDIonTheFly(
        t_tdi_inj, 
        Tobs,
        t_ref,
        1. / dt,
        1,
        **gb_tdi_kwargs
    )

    N = data_inj.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    # TUKEY_ALPHA env var sets a Tukey window on the injection TD->WDM
    # transform (and on the lookup build's td_window when build_kind allows
    # it). alpha=0.5 tapers 25% on each end. Default 0 (= unity window).
    _tukey_alpha = float(os.environ.get("TUKEY_ALPHA", "0.0"))
    if _tukey_alpha > 0:
        window = xp.asarray(signal.windows.tukey(N, alpha=_tukey_alpha))
    else:
        window = xp.ones(N)

    min_freq = 0.0001  # 0.0029493407356002777  # 255 * wdm_set.layer_df
    max_freq = 35.0e-3 # 0.00306500115660421  # 265 * wdm_set.layer_df
    fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    # shave edges?
    _EDGE_CUT = int(os.environ.get("EDGE_CUT", "20"))
    min_time = _EDGE_CUT * wavelet_duration
    max_time = (Nt - _EDGE_CUT) * wavelet_duration

    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq, min_time=min_time, max_time=max_time)
    
    # Plan A (n_ref_only) build path. WDM_BUILD_KIND env var picks the
    # build kind; FDOT_EPS controls the fdot grid density (0 → no fdot).
    BUILD_KIND     = os.environ.get("WDM_BUILD_KIND", "n_ref_only")
    EPS_FREQ       = float(os.environ.get("EPS_FREQ", 0.001))
    NUM_LAYERS_DIFF = int(os.environ.get("NUM_LAYERS_DIFF", 5))
    FDOT_EPS       = float(os.environ.get("FDOT_EPS", "0.0"))
    FDOT_MAX_FACTOR = float(os.environ.get("FDOT_MAX_FACTOR", "2.0"))
    # Auto-named store so fdot=0 and fdot grids don't clobber each other.
    _default_store = (
        f"wdm_lookup_{BUILD_KIND}"
        f"_efreq{EPS_FREQ:.3f}_nld{NUM_LAYERS_DIFF}"
        f"_fdot{FDOT_EPS:.2f}x{FDOT_MAX_FACTOR:.1f}.h5"
    )
    store_path = os.environ.get("WDM_STORE_PATH", _default_store)
    print(f"[step] build_kind={BUILD_KIND}  store_path={store_path}", flush=True)
    print(f"[step] EPS_FREQ={EPS_FREQ}  NUM_LAYERS_DIFF={NUM_LAYERS_DIFF}  "
          f"FDOT_EPS={FDOT_EPS}  FDOT_MAX_FACTOR={FDOT_MAX_FACTOR}", flush=True)

    ## lookup table setup
    if os.path.exists(store_path):
        wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=backend)
        _wdm_settings = WDMSettings(*wdm_lookup_table.args, **wdm_lookup_table.kwargs)
        if not _wdm_settings.eq_without_inds(wdm_set):
            raise ValueError("WDM Settings are not equivalent to lookup table. Either adjust to lookup table settings or regenerate the table.")

    else:
        # Plan A (n_ref_only) requires no windowing on the carrier (the build
        # asserts td_window=None). per_n accepts a non-trivial td_window —
        # if TUKEY_ALPHA > 0 we apply the same Tukey to the build's carrier
        # source so both the injection's WDM transform and the lookup table
        # build share the same windowing convention (suppresses FD leakage
        # that could differentially affect mid-band wavelet coefficients).
        if BUILD_KIND == "per_n" and _tukey_alpha > 0:
            td_window = window  # reuse the inj-side Tukey
            print(f"[step] using Tukey alpha={_tukey_alpha} for BOTH inj and per_n build td_window", flush=True)
        else:
            td_window = None
        m_ref = int(3e-3 / wdm_set.layer_df)
        norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(
            EPS_FREQ, wdm_set, m_ref=m_ref, num_layers_diff=NUM_LAYERS_DIFF,
        )

        if FDOT_EPS > 0.0:
            fdot_vals = WDMLookupTable.apply_eps_fdot(
                FDOT_EPS, wdm_set, fdot_max_factor=FDOT_MAX_FACTOR,
            )
        else:
            fdot_vals = np.array([0.0])
        print(f"[step] fdot_vals: n={len(fdot_vals)}  "
              f"range=[{float(np.min(fdot_vals)):.3e}, {float(np.max(fdot_vals)):.3e}]", flush=True)

        nchannel = 3
        # TIME_LAYERS — small Nt for the build (Plan A only reads one
        # pixel, so a much smaller Nt is equivalent and 10×+ faster).
        # Must have (TIME_LAYERS // 2) parity matching (wdm_set.Nt // 2).
        TIME_LAYERS_ENV = os.environ.get("TIME_LAYERS", "")
        if TIME_LAYERS_ENV.strip():
            _time_layers = int(TIME_LAYERS_ENV)
        else:
            _time_layers = None
        wdm_lookup_table = WDMLookupTable(
            wdm_set, nchannel,
            norm_freq_single_layer=norm_freq_single_layer,
            m_diffs=m_diffs,
            fdot_vals=fdot_vals,
            m_ref=m_ref,
            batch_size_gen=int(os.environ.get("BATCH_SIZE_GEN", "5")),
            td_window=td_window,
            store_path=store_path,
            verbose=True,
            build_kind=BUILD_KIND,
            time_layers=_time_layers,
        )

    # Quick sanity on the loaded table: under fdot=0 the sin branch at
    # (m_ref, n_ref) is rotation-only noise; once fdot != 0 it carries
    # real content. Print magnitudes so a fresh build is easy to eyeball.
    _sin_mag = float(np.max(np.abs(wdm_lookup_table.get(wdm_lookup_table.table_sin))))
    _cos_mag = float(np.max(np.abs(wdm_lookup_table.get(wdm_lookup_table.table_cos))))
    print(f"[diag] table_sin max|.| = {_sin_mag:.3e}  table_cos max|.| = {_cos_mag:.3e}  "
          f"build_kind={wdm_lookup_table.build_kind}  fdot_steps={len(wdm_lookup_table.fdot_vals)}", flush=True)

    # F_FRACS sweep: comma-separated f_frac values to test. Defaults span
    # on-grid (~0/1) and mid-grid (~0.5) per the Plan A validation request.
    # Set F_FRAC for a single value, F_FRACS for a list.
    if "F_FRAC" in os.environ:
        _f_frac_list = [float(os.environ["F_FRAC"])]
    else:
        _f_frac_list = [float(s) for s in os.environ.get("F_FRACS", "0.05,0.5,0.95").split(",")]

    # Shared one-shot construction (does not depend on f_frac).
    output_set = wdm_set
    if output_set != wdm_set:
        raise ValueError("This script requires WDM for the output_set.")
    sens_mat_proto = None  # build once injection settings are known

    N_sparse = int(os.environ.get("N_SPARSE", 4096))
    print(f"[step] N_sparse={N_sparse}  f_fracs={_f_frac_list}", flush=True)
    t_tdi_sparse = xp.linspace(t_arr[0], t_arr[-1], N_sparse)

    # C lookup now supports both per_n and n_ref_only via the LookupKind
    # enum (TDIonTheFly.hh) — default to running for either build kind.
    # RUN_C_LOOKUP=0 to force-skip.
    _run_c_lookup = os.environ.get("RUN_C_LOOKUP", "auto") != "0"
    gb_comps = None
    if _run_c_lookup:
        gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, t_ref, orbits=orbits, tdi_config=tdi_config, force_backend=backend)
    else:
        print(f"[step] skipping C lookup path (build_kind={wdm_lookup_table.build_kind}, "
              f"RUN_C_LOOKUP={os.environ.get('RUN_C_LOOKUP', 'auto')})", flush=True)

    _results = []
    # M_OFFSET shifts the source's m_floor by an integer relative to the
    # table's m_ref (e.g., M_OFFSET=1 puts the source one full layer above
    # m_ref while preserving f_frac). Use to probe whether per-layer
    # residual asymmetry depends on the parity/index of the source's m_floor.
    _m_offset = int(os.environ.get("M_OFFSET", "0"))
    for _ff_idx, _ff in enumerate(_f_frac_list):
        num_bin = 1
        amp = np.full(num_bin, 8.0e-22)
        f0 = np.full(num_bin, (wdm_lookup_table.m_ref + _m_offset + _ff) * wdm_set.layer_df)
        # SOURCE_FDOT (Hz/s). Default 1e-17 (essentially fdot=0). Set to
        # something in the lookup table's fdot range to exercise the
        # fdot interpolation path.
        fdot = np.full(num_bin, float(os.environ.get("SOURCE_FDOT", "1e-17")))
        fddot = np.full(num_bin, 0.0)
        phi0 = np.full(num_bin, 2.09802430298)
        inc = np.full(num_bin, 0.23984234)

        # NEED TO ADD FRAME TRANSFORM FOR PSI IF WORKING IN ECLIPTIC
        psi = np.full(num_bin, 1.234019814)
        lam = np.full(num_bin, 4.09808143)
        beta = np.full(num_bin, float(os.environ.get("BETA", "0.04")))
        params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam, beta]).T

        # Fresh injection buffer per iteration.
        _data_inj = xp.zeros_like(data_inj)
        inj_tmp = gb_gen_inj(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta,
                             convert_to_ra_dec=False, return_spline=True)
        _data_inj[:] = inj_tmp.eval_tdi(t_arr)

        # NOTE Plan A: keep the outer transform window as ones (line ~189)
        # so the injection has no global Tukey taper. min_time/max_time on
        # wdm_set already trims ~20 wavelet pixels off each edge.
        data_inj_all = TDSignal(_data_inj, settings=td_set).transform(output_set, window=window)
        injection = DataResidualArray(data_inj_all)
        if sens_mat_proto is None:
            sens_mat_proto = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")
        sens_mat = sens_mat_proto

        gb_gen_wrap = GBLookupWaveWrap(
            t_arr, t_tdi_sparse, Tobs, t_ref, dt,
            params.shape[0], gb_tdi_kwargs, td_set, output_set, window,
        )
        analysis = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)
        wdm_holder = AnalysisContainerArray([analysis])

        template_fill_wdm = None
        check_ll_2 = check_ip_2 = overlap = None
        if gb_comps is not None:
            template_fill = xp.zeros(3 * np.prod(wdm_set.basis_shape_active), dtype=float)
            gb_comps.fill_global_wdm(template_fill, params, wdm_holder, data_index=None, convert_to_ra_dec=False)
            template_fill_wdm = WDMSignal(template_fill.reshape((3,) + wdm_set.basis_shape_active), wdm_set)
            check_ll_2 = analysis.template_likelihood(template_fill_wdm)
            check_ip_2 = analysis.template_inner_product(template_fill_wdm)
            overlap = analysis.template_inner_product(template_fill_wdm, normalize=True)

        check_ip_d_d = analysis.inner_product()
        py_wdm_lookup = gb_gen_wrap(*params[0])
        tmp_val1 = analysis.template_inner_product(py_wdm_lookup)
        tmp_val2 = analysis.calculate_signal_inner_product(*params[0])
        tmp_val3 = analysis.calculate_signal_likelihood(*params[0], source_only=True)
        overlap_py = analysis.template_inner_product(py_wdm_lookup, normalize=True)

        _f_frac_meas = (f0[0] - (int(f0[0] / wdm_set.layer_df) * wdm_set.layer_df)) / wdm_set.layer_df
        _mm_c  = float(1.0 - overlap) if overlap is not None else float("nan")
        _mm_py = float(1.0 - overlap_py)

        # mm5 — mismatch restricted to a narrow ±5-layer band around f0
        # (matching gb_lookup_prior_draws.py's mismatch_5_layers convention:
        # min_freq = f0 - 3*layer_df, max_freq = f0 + 2*layer_df). This
        # focuses on the source's own neighborhood so the result is not
        # diluted by spurious template energy in far-away pixels — more
        # sensitive to lookup-table accuracy near the source.
        _new_wdm_set = WDMSettings(
            wdm_set.Nf, wdm_set.Nt, wdm_set.data_dt,
            min_time=wdm_set.min_time, max_time=wdm_set.max_time,
            min_freq=float(f0[0] - 3 * wdm_set.layer_df),
            max_freq=float(f0[0] + 2 * wdm_set.layer_df),
            force_backend=backend,
        )
        _m_lo = _new_wdm_set.ind_min_f - wdm_set.ind_min_f
        _m_hi = _new_wdm_set.ind_max_f - wdm_set.ind_min_f + 1
        _inj_here = DataResidualArray(WDMSignal(injection[:, _m_lo:_m_hi], _new_wdm_set))
        _sens_here = XYZ2SensitivityMatrix(_new_wdm_set, model="scirdv1")
        _ah5 = AnalysisContainer(_inj_here, _sens_here)
        _mm_py5_C = _mm_py5 = float("nan")
        if template_fill_wdm is not None:
            _tpl_here_c = DataResidualArray(WDMSignal(template_fill_wdm[:, _m_lo:_m_hi], _new_wdm_set))
            _mm_py5_C = float(1.0 - _ah5.template_inner_product(_tpl_here_c, normalize=True))
        _tpl_here_py = DataResidualArray(WDMSignal(py_wdm_lookup[:, _m_lo:_m_hi], _new_wdm_set))
        _mm_py5 = float(1.0 - _ah5.template_inner_product(_tpl_here_py, normalize=True))

        _results.append((_ff, _f_frac_meas, _mm_c, _mm_py, _mm_py5_C, _mm_py5))

        # SAVE_RESIDUAL=1 dumps the (injection - template) WDM array for
        # this f_frac iteration so it can be heatmapped later. Saves to
        # residual_ff{f_frac:.3f}_ch{ch}.npy for ch=0,1,2 plus a summary
        # of per-(m,n) absolute residual.
        if os.environ.get("SAVE_RESIDUAL", "0") == "1":
            _inj_arr = data_inj_all.arr   # (3, Nf_active, Nt_active)
            _tpl_arr = py_wdm_lookup.arr
            _resid = _inj_arr - _tpl_arr
            _ms = int(f0[0] / wdm_set.layer_df)
            _m_active = _ms - wdm_set.ind_min_f
            np.savez_compressed(
                f"residual_ff{_ff:.3f}.npz",
                inj=_inj_arr, tpl=_tpl_arr, resid=_resid,
                m_source_active=_m_active, m_source_abs=_ms,
                ind_min_f=wdm_set.ind_min_f, ind_min_t=wdm_set.ind_min_t,
                layer_df=wdm_set.layer_df, layer_dt=wdm_set.layer_dt,
                f0=float(f0[0]),
            )
            # Print per-layer L2 of residual to locate the bulk of the error.
            _r_per_m = np.sqrt((_resid ** 2).sum(axis=(0, 2)))  # (Nf_active,)
            _i_per_m = np.sqrt((_inj_arr ** 2).sum(axis=(0, 2)))
            print(f"  [resid] saved residual_ff{_ff:.3f}.npz; "
                  f"source at m_abs={_ms} (active {_m_active})")
            print(f"  [resid] per-layer  m_off   ||resid||   ||inj||   ratio")
            for _moff in [-3, -2, -1, 0, 1, 2, 3]:
                _mi = _m_active + _moff
                if 0 <= _mi < _r_per_m.size:
                    print(f"          {_moff:+5d}    {_r_per_m[_mi]:.3e}   "
                          f"{_i_per_m[_mi]:.3e}   "
                          f"{_r_per_m[_mi] / max(_i_per_m[_mi], 1e-30):.3e}")

        # Per-pixel diagnostic: pick 8 pixels in the dominant source layer and
        # compare injection vs Python lookup template side-by-side.
        if os.environ.get("DIAG_PIXEL", "0") == "1":
            _inj_arr = data_inj_all.arr  # (3, Nf_active, Nt_active)
            _tpl_arr = py_wdm_lookup.arr
            _ms = int(f0[0] / wdm_set.layer_df)
            _m_active = _ms - wdm_set.ind_min_f
            print(f"  [diag] source ms={_ms}, ind_min_f={wdm_set.ind_min_f}, "
                  f"layer in active grid = {_m_active}")
            for _ch in [0]:
                for _m_off in [-1, 0, 1]:
                    _m = _m_active + _m_off
                    if _m < 0 or _m >= _inj_arr.shape[1]:
                        continue
                    _inj_row = _inj_arr[_ch, _m, :]
                    _tpl_row = _tpl_arr[_ch, _m, :]
                    _rms_inj = float(np.sqrt(np.mean(_inj_row**2)))
                    _rms_tpl = float(np.sqrt(np.mean(_tpl_row**2)))
                    _ratio = _rms_tpl / _rms_inj if _rms_inj > 0 else float("nan")
                    _dot = float(np.sum(_inj_row * _tpl_row))
                    _norm = float(np.sqrt(np.sum(_inj_row**2) * np.sum(_tpl_row**2)) + 1e-300)
                    _cosine = _dot / _norm if _norm > 0 else float("nan")
                    print(f"  [diag] ch={_ch} layer_off={_m_off:+d} (m={_ms+_m_off})  "
                          f"rms_inj={_rms_inj:.3e} rms_tpl={_rms_tpl:.3e}  "
                          f"ratio={_ratio:.3f}  cosine={_cosine:+.4f}")
                    # show first 6 active pixel values
                    print(f"        inj first 6: {_inj_row[:6]}")
                    print(f"        tpl first 6: {_tpl_row[:6]}")

        print(f"\n=== f_frac sweep [{_ff_idx + 1}/{len(_f_frac_list)}]  "
              f"requested={_ff:.3f}  measured={_f_frac_meas:+.4f}  "
              f"f0={f0[0]*1e3:.5f} mHz ===")
        print(f"[result] base inner_product <d|d>           = {check_ip_d_d}")
        if gb_comps is not None:
            print(f"[result] template_inner_product (C lookup)  = {check_ip_2}")
        print(f"[result] template_inner_product (py lookup) = {tmp_val1}")
        print(f"[result] calculate_signal_inner_product     = {tmp_val2}")
        if gb_comps is not None:
            print(f"[result] template_likelihood (C lookup)     = {check_ll_2}")
        print(f"[result] template_likelihood (py lookup)    = {tmp_val3}")
        if gb_comps is not None:
            print(f"[result] Noise-weighted mismatch (C  lookup) = {_mm_c:.3e}")
        print(f"[result] Noise-weighted mismatch (py lookup) = {_mm_py:.3e}")
        if gb_comps is not None:
            print(f"[result] mm5 (±5-layer band, C  lookup)      = {_mm_py5_C:.3e}")
        print(f"[result] mm5 (±5-layer band, py lookup)      = {_mm_py5:.3e}")

        _nrows = 3 if template_fill_wdm is not None else 2
        fig, axes = plt.subplots(_nrows, 1, sharex=True, sharey=True, figsize=(11, 8))
        if _nrows == 2:
            ax1, ax3 = axes
            ax2 = None
        else:
            ax1, ax2, ax3 = axes
        data_inj_all.heatmap(fig=fig, ax=ax1, index=0, add_cax=True)
        if ax2 is not None:
            template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0)
        py_wdm_lookup.heatmap(fig=fig, ax=ax3, index=0)
        ax1.set_title(
            f"injection  f0={f0[0]*1e3:.5f} mHz  phi0={phi0[0]:.4f}  "
            f"f_frac={_f_frac_meas:+.3f}  (build_kind={wdm_lookup_table.build_kind})"
        )
        if ax2 is not None:
            ax2.set_title(f"C lookup template  (mm = {_mm_c:.3e})")
        ax3.set_title(f"py lookup template  (mm = {_mm_py:.3e})")
        ax1.set_ylim(f0[0] - 10 * wdm_set.layer_df, f0[0] + 10 * wdm_set.layer_df)
        plt.tight_layout()
        _out_path = f"gb_lookup_{wdm_lookup_table.build_kind}_ff{_f_frac_meas:+.3f}.png"
        plt.savefig(_out_path, dpi=120)
        if os.environ.get("SHOW_PLOTS", "0") == "1":
            plt.show()
        plt.close(fig)
        print(f"[plot] saved {_out_path}")

    print("\n[summary] f_frac sweep")
    print(f"  build_kind = {wdm_lookup_table.build_kind}")
    print(f"  fdot_steps = {len(wdm_lookup_table.fdot_vals)}")
    print("  requested  measured  mm_C        mm_py       mm5_C       mm5_py")
    for _ff, _ff_m, _mm_c, _mm_py, _mm_c5, _mm_py5 in _results:
        print(f"  {_ff:>9.3f}  {_ff_m:+.4f}   {_mm_c:.3e}   {_mm_py:.3e}   {_mm_c5:.3e}   {_mm_py5:.3e}")

    if os.environ.get("DEBUG_BREAK", "0") == "1":
        breakpoint()
    # ---- spline-path smoke check vs direct path -------------------------------
    # Set RUN_SPLINE_CHECK=1 to run the spline-flavored fill/get_ll/swap_ll/grad
    # kernels and report max-rel-diff vs the direct path on the same source.
    # Coarse density via env COARSE_PTS_PER_YEAR (default 256).
    if os.environ.get("RUN_SPLINE_CHECK", "0") == "1":
        coarse_pts_per_year = int(os.environ.get("COARSE_PTS_PER_YEAR", 256))
        print(f"\n[spline check] coarse_pts_per_year={coarse_pts_per_year}")

        # Use an in-band f0 so the kernels actually accumulate.
        m_lo_chk, m_hi_chk = wdm_set.ind_min_f, wdm_set.ind_max_f
        m_mid_chk = (m_lo_chk + m_hi_chk) // 2
        params_chk = params.copy()
        params_chk[:, 1] = m_mid_chk * wdm_set.layer_df

        def _rel_max(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            mask = np.abs(b) > 0
            if not np.any(mask):
                return 0.0
            return float(np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))

        tpl_d = xp.zeros(3 * np.prod(wdm_set.basis_shape_active), dtype=float)
        tpl_s = xp.zeros_like(tpl_d)
        gb_comps.fill_global_wdm(tpl_d, params_chk, wdm_holder, data_index=None, convert_to_ra_dec=False)
        gb_comps.fill_global_wdm(tpl_s, params_chk, wdm_holder, data_index=None, convert_to_ra_dec=False,
                                 use_spline=True, coarse_pts_per_year=coarse_pts_per_year)
        nz = np.abs(tpl_d) > 0
        if np.any(nz):
            rel = np.abs(tpl_s[nz] - tpl_d[nz]) / np.abs(tpl_d[nz])
            print(f"[spline check] fill_global: nz pixels={int(nz.sum())}, max rel={float(rel.max()):.3e}, median rel={float(np.median(rel)):.3e}")
        else:
            print("[spline check] fill_global: direct template all zero")

        _ = gb_comps.get_ll_wdm(params_chk, wdm_holder, convert_to_ra_dec=False)
        d_h_d, h_h_d = float(gb_comps.d_h_out[0]), float(gb_comps.h_h_out[0])
        _ = gb_comps.get_ll_wdm(params_chk, wdm_holder, convert_to_ra_dec=False,
                                use_spline=True, coarse_pts_per_year=coarse_pts_per_year)
        d_h_s, h_h_s = float(gb_comps.d_h_out[0]), float(gb_comps.h_h_out[0])
        print(f"[spline check] get_ll: d_h rel={abs(d_h_s - d_h_d)/max(abs(d_h_d),1e-300):.3e}, h_h rel={abs(h_h_s - h_h_d)/max(abs(h_h_d),1e-300):.3e}")

        pA = params_chk.copy()
        pB = params_chk.copy()
        m_off = min(m_mid_chk + 3, m_hi_chk)
        pB[:, 1] = m_off * wdm_set.layer_df
        out_d = gb_comps.get_swap_ll_wdm(pA, pB, wdm_holder, convert_to_ra_dec=False)
        out_s = gb_comps.get_swap_ll_wdm(pA, pB, wdm_holder, convert_to_ra_dec=False,
                                         use_spline=True, coarse_pts_per_year=coarse_pts_per_year)
        for nm, vd, vs in zip(
            ["like_add", "like_remove", "d_h_add", "d_h_remove", "add_add", "remove_remove", "add_remove"],
            out_d, out_s,
        ):
            vd = float(np.asarray(vd)[0]); vs = float(np.asarray(vs)[0])
            print(f"[spline check] swap_ll {nm:14s} rel={abs(vs - vd)/max(abs(vd),1e-300):.3e}")

        grad_d = np.asarray(gb_comps.get_ll_grad_wdm(params_chk, wdm_holder, convert_to_ra_dec=False))
        grad_s = np.asarray(gb_comps.get_ll_grad_wdm(params_chk, wdm_holder, convert_to_ra_dec=False,
                                                      use_spline=True, coarse_pts_per_year=coarse_pts_per_year))
        rels = np.abs(grad_s[0] - grad_d[0]) / np.maximum(np.abs(grad_d[0]), 1e-300)
        print(f"[spline check] get_ll_grad direct = {grad_d[0]}")
        print(f"[spline check] get_ll_grad spline = {grad_s[0]}")
        print(f"[spline check] get_ll_grad per-param rel = {rels}")
        print(f"[spline check] get_ll_grad max rel = {float(rels.max()):.3e}")

    # ---- swap-likelihood vs get-likelihood cross-check -------------------------
    # Placed before the Python-side gb_gen_wrap() call so it runs regardless of
    # whether the Python lookup path is healthy. Detailed description below in
    # the comment block before the configuration setup.
    _run_swap_ll_check = True
    if _run_swap_ll_check:
        # The injection f0 in `params` (18 mHz) sits far outside the WDM active
        # band [ind_min_f, ind_max_f]*layer_df, so both get_ll and swap_ll would
        # trivially return zero on it. Pick a test frequency in the middle of
        # the active band so the kernel actually accumulates contributions.
        layer_df = wdm_set.layer_df
        m_lo, m_hi = wdm_set.ind_min_f, wdm_set.ind_max_f
        m_mid_A = (m_lo + m_hi) // 2
        m_mid_B = m_mid_A + 3  # 3-layer offset; still inside the band if it fits
        if m_mid_B > m_hi:
            m_mid_B = m_hi
        f0_A = m_mid_A * layer_df
        f0_B = m_mid_B * layer_df

        params_A = params.copy()
        params_B = params.copy()
        params_A[:, 1] = f0_A
        params_B[:, 1] = f0_B
        print(f"[swap_ll test] active band layers=[{m_lo},{m_hi}], layer_df={layer_df:.6e}")
        print(f"[swap_ll test] sourceA f0={f0_A:.6e} (layer {m_mid_A}), sourceB f0={f0_B:.6e} (layer {m_mid_B})")

        _ = gb_comps.get_ll_wdm(params_A, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec=False)
        d_h_A_ref = float(gb_comps.d_h_out[0])
        h_h_A_ref = float(gb_comps.h_h_out[0])

        _ = gb_comps.get_ll_wdm(params_B, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec=False)
        d_h_B_ref = float(gb_comps.d_h_out[0])
        h_h_B_ref = float(gb_comps.h_h_out[0])

        like_A_ref = -0.5 * (gb_comps.d_d + h_h_A_ref - 2 * d_h_A_ref)
        like_B_ref = -0.5 * (gb_comps.d_d + h_h_B_ref - 2 * d_h_B_ref)

        params_add_in    = np.stack([params_A[0], params_A[0], params_B[0]], axis=0)
        params_remove_in = np.stack([params_A[0], params_B[0], params_A[0]], axis=0)

        (
            like_add,
            like_remove,
            d_h_add,
            d_h_remove,
            add_add,
            remove_remove,
            add_remove,
        ) = gb_comps.get_swap_ll_wdm(
            params_add_in, params_remove_in, wdm_holder,
            data_index=None, noise_index=None, convert_to_ra_dec=False,
        )

        def _xp_to_np(a):
            return a.get() if hasattr(a, "get") else np.asarray(a)

        d_h_add        = _xp_to_np(d_h_add)
        d_h_remove     = _xp_to_np(d_h_remove)
        add_add        = _xp_to_np(add_add)
        remove_remove  = _xp_to_np(remove_remove)
        add_remove     = _xp_to_np(add_remove)
        like_add       = _xp_to_np(like_add)
        like_remove    = _xp_to_np(like_remove)

        def _rel(a, b):
            denom = max(abs(b), 1e-300)
            return abs(a - b) / denom

        # bin 0: same source, all outputs == get_ll(sourceA)
        bin0_rel = max(
            _rel(d_h_add[0],       d_h_A_ref),
            _rel(d_h_remove[0],    d_h_A_ref),
            _rel(add_add[0],       h_h_A_ref),
            _rel(remove_remove[0], h_h_A_ref),
            _rel(add_remove[0],    h_h_A_ref),
            _rel(like_add[0],      like_A_ref),
            _rel(like_remove[0],   like_A_ref),
        )

        # bin 1: add=A, remove=B
        bin1_rel = max(
            _rel(d_h_add[1],       d_h_A_ref),
            _rel(d_h_remove[1],    d_h_B_ref),
            _rel(add_add[1],       h_h_A_ref),
            _rel(remove_remove[1], h_h_B_ref),
            _rel(like_add[1],      like_A_ref),
            _rel(like_remove[1],   like_B_ref),
        )

        # bin 2: add=B, remove=A. Must mirror bin 1 after swap.
        bin2_rel = max(
            _rel(d_h_add[2],       d_h_B_ref),
            _rel(d_h_remove[2],    d_h_A_ref),
            _rel(add_add[2],       h_h_B_ref),
            _rel(remove_remove[2], h_h_A_ref),
            _rel(like_add[2],      like_B_ref),
            _rel(like_remove[2],   like_A_ref),
        )

        # swap symmetry: cross term must be the same whichever side is "add"
        sym_rel = _rel(add_remove[1], add_remove[2])

        # Cauchy-Schwarz on each bin: |<h_a|h_r>|^2 <= <h_a|h_a> * <h_r|h_r>
        cs_eps = 1e-9
        cs_bin = []
        for k in range(3):
            lhs = add_remove[k] ** 2
            rhs = (1.0 + cs_eps) * add_add[k] * remove_remove[k]
            cs_bin.append((lhs, rhs, lhs <= rhs))

        print(f"==== swap_ll vs get_ll cross-check ====")
        print(f"  sourceA: d_h = {d_h_A_ref:+.8e}   h_h = {h_h_A_ref:+.8e}   like = {like_A_ref:+.8e}")
        print(f"  sourceB: d_h = {d_h_B_ref:+.8e}   h_h = {h_h_B_ref:+.8e}   like = {like_B_ref:+.8e}")
        print(f"  bin0 (A,A)  rel_max = {bin0_rel:.3e}")
        print(f"             d_h_add={d_h_add[0]:+.6e}  d_h_remove={d_h_remove[0]:+.6e}")
        print(f"             add_add={add_add[0]:+.6e}  remove_remove={remove_remove[0]:+.6e}  add_remove={add_remove[0]:+.6e}")
        print(f"  bin1 (A,B)  rel_max = {bin1_rel:.3e}")
        print(f"             d_h_add={d_h_add[1]:+.6e}  d_h_remove={d_h_remove[1]:+.6e}")
        print(f"             add_add={add_add[1]:+.6e}  remove_remove={remove_remove[1]:+.6e}  add_remove={add_remove[1]:+.6e}")
        print(f"  bin2 (B,A)  rel_max = {bin2_rel:.3e}")
        print(f"             d_h_add={d_h_add[2]:+.6e}  d_h_remove={d_h_remove[2]:+.6e}")
        print(f"             add_add={add_add[2]:+.6e}  remove_remove={remove_remove[2]:+.6e}  add_remove={add_remove[2]:+.6e}")
        print(f"  swap symmetry add_remove(A,B) vs add_remove(B,A)  rel = {sym_rel:.3e}")
        for k, (lhs, rhs, ok) in enumerate(cs_bin):
            status = "ok" if ok else "FAIL"
            print(f"  Cauchy-Schwarz bin{k}: |add_remove|^2={lhs:+.6e}  <h_a|h_a><h_r|h_r>={rhs:+.6e}  [{status}]")

        swap_tol = float(os.environ.get("SWAP_LL_TOL", 1e-10))
        sym_tol  = float(os.environ.get("SWAP_LL_SYM_TOL", 1e-9))
        worst = max(bin0_rel, bin1_rel, bin2_rel)
        cs_ok = all(c[2] for c in cs_bin)
        if worst > swap_tol:
            print(f"[FAIL] swap_ll <-> get_ll disagree above tol={swap_tol}; worst rel = {worst:.3e}")
        elif sym_rel > sym_tol:
            print(f"[FAIL] swap-symmetry violated: add_remove(A,B) != add_remove(B,A) (rel = {sym_rel:.3e}, tol = {sym_tol})")
        elif not cs_ok:
            print(f"[FAIL] Cauchy-Schwarz violated on at least one bin")
        else:
            print(f"[ok] swap_ll matches get_ll on each per-template piece (worst rel = {worst:.3e}, tol = {swap_tol}),")
            print(f"     cross term is symmetric under add↔remove (rel = {sym_rel:.3e}, tol = {sym_tol}),")
            print(f"     and obeys Cauchy-Schwarz on every bin.\n")


    # ---- chain-rule gradient cross-check --------------------------------------
    # Same setup as the swap_ll block above: source A and source B inside the
    # active WDM band.  We call the new C/CUDA gradient kernels
    #
    #     gb_comps.get_ll_grad_wdm(...)         (shape (num_bin, 9))
    #     gb_comps.get_swap_ll_grad_wdm(...)    (grad_add, grad_remove)
    #
    # which compute the per-binary parameter gradient of L = -1/2 (d_d + h_h - 2 d_h)
    # (and of the swap log-likelihood ratio) via per-pixel central differences on
    # top of the same fast_wdm_inner / wdm_lookup pipeline used by get_ll / swap_ll.
    #
    # As a reference we run a Python-side finite difference of the *same*
    # likelihood values (get_ll_wdm / get_swap_ll_wdm) with the *same* per-param
    # step sizes ``param_eps`` -- so the two methods are computing the same
    # mathematical object, just at different granularity (per-pixel vs per-run).
    # They must agree to round-off.
    _run_grad_check = bool(int(os.environ.get("RUN_GRAD_CHECK", "1")))
    if _run_grad_check:
        # Same per-parameter FD step that the C kernel will use by default
        # (see GBWDMComputations._DEFAULT_PARAM_EPS).  Keep them consistent so
        # the two FDs cancel exactly.
        param_eps_default = np.array(GBWDMComputations._DEFAULT_PARAM_EPS, dtype=np.float64)
        nparams = params_A.shape[1]
        assert param_eps_default.shape[0] == nparams

        def _xp_to_np_local(a):
            return a.get() if hasattr(a, "get") else np.asarray(a)

        # --- get_ll gradient: source A
        grad_C = gb_comps.get_ll_grad_wdm(
            params_A, wdm_holder,
            param_eps=param_eps_default,
            data_index=None, noise_index=None, convert_to_ra_dec=False,
        )
        grad_C = _xp_to_np_local(grad_C)[0]   # (nparams,)

        def _ll_at(p_arr):
            """Recompute L = -0.5 (d_d + h_h - 2 d_h) at the supplied params."""
            _ = gb_comps.get_ll_wdm(
                p_arr, wdm_holder,
                data_index=None, noise_index=None, convert_to_ra_dec=False,
            )
            d_h = float(_xp_to_np_local(gb_comps.d_h_out)[0])
            h_h = float(_xp_to_np_local(gb_comps.h_h_out)[0])
            return -0.5 * (gb_comps.d_d + h_h - 2 * d_h)

        grad_fd = np.zeros(nparams, dtype=np.float64)
        for k in range(nparams):
            eps_k = param_eps_default[k]
            if eps_k <= 0.0:
                continue
            pp = params_A.copy(); pp[:, k] = pp[:, k] + eps_k
            pm = params_A.copy(); pm[:, k] = pm[:, k] - eps_k
            grad_fd[k] = (_ll_at(pp) - _ll_at(pm)) / (2.0 * eps_k)

        param_names = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")
        print(f"==== get_ll gradient (C/CUDA chain rule) vs Python FD of get_ll ====")
        worst_grad = 0.0
        for k in range(nparams):
            denom = max(abs(grad_fd[k]), 1e-300)
            rel = abs(grad_C[k] - grad_fd[k]) / denom
            worst_grad = max(worst_grad, rel)
            print(f"  {param_names[k]:>5s}  C={grad_C[k]:+.8e}  FD={grad_fd[k]:+.8e}  rel={rel:.2e}")
        grad_tol = float(os.environ.get("GRAD_LL_TOL", 1e-8))
        status = "ok" if worst_grad < grad_tol else "FAIL"
        print(f"  worst rel = {worst_grad:.3e}   tol = {grad_tol}   [{status}]\n")

        # --- swap_ll gradient: (add, remove) on the three bins from above
        grad_add_C, grad_remove_C = gb_comps.get_swap_ll_grad_wdm(
            params_add_in, params_remove_in, wdm_holder,
            param_eps_add=param_eps_default,
            param_eps_remove=param_eps_default,
            data_index=None, noise_index=None, convert_to_ra_dec=False,
        )
        grad_add_C = _xp_to_np_local(grad_add_C)
        grad_remove_C = _xp_to_np_local(grad_remove_C)

        def _ll_diff_at(p_add, p_remove):
            """ll_diff = L(after swap) - L(before swap) at supplied params.

            Mirrors gb_wdm_swap_ll_kernel: returns -1/2 (-2 d_h_add + 2 d_h_remove
            - 2 add_remove + add_add + remove_remove). Same convention used by
            the C swap_ll_grad kernel."""
            (
                _like_add, _like_remove,
                d_h_add_v, d_h_remove_v,
                add_add_v, remove_remove_v, add_remove_v,
            ) = gb_comps.get_swap_ll_wdm(
                p_add, p_remove, wdm_holder,
                data_index=None, noise_index=None, convert_to_ra_dec=False,
            )
            d_h_a = _xp_to_np_local(d_h_add_v)
            d_h_r = _xp_to_np_local(d_h_remove_v)
            aa    = _xp_to_np_local(add_add_v)
            rr    = _xp_to_np_local(remove_remove_v)
            ar    = _xp_to_np_local(add_remove_v)
            return -0.5 * (-2.0 * d_h_a + 2.0 * d_h_r - 2.0 * ar + aa + rr)

        # FD w.r.t. add params
        grad_add_fd = np.zeros_like(grad_add_C)
        for k in range(nparams):
            eps_k = param_eps_default[k]
            if eps_k <= 0.0:
                continue
            pa_p = params_add_in.copy(); pa_p[:, k] = pa_p[:, k] + eps_k
            pa_m = params_add_in.copy(); pa_m[:, k] = pa_m[:, k] - eps_k
            ll_p = _ll_diff_at(pa_p, params_remove_in)
            ll_m = _ll_diff_at(pa_m, params_remove_in)
            grad_add_fd[:, k] = (ll_p - ll_m) / (2.0 * eps_k)

        # FD w.r.t. remove params
        grad_remove_fd = np.zeros_like(grad_remove_C)
        for k in range(nparams):
            eps_k = param_eps_default[k]
            if eps_k <= 0.0:
                continue
            pr_p = params_remove_in.copy(); pr_p[:, k] = pr_p[:, k] + eps_k
            pr_m = params_remove_in.copy(); pr_m[:, k] = pr_m[:, k] - eps_k
            ll_p = _ll_diff_at(params_add_in, pr_p)
            ll_m = _ll_diff_at(params_add_in, pr_m)
            grad_remove_fd[:, k] = (ll_p - ll_m) / (2.0 * eps_k)

        print(f"==== swap_ll gradient (C/CUDA chain rule) vs Python FD of swap_ll ====")
        worst_swap_grad = 0.0
        for which, gC, gFD in (("add", grad_add_C, grad_add_fd),
                                ("remove", grad_remove_C, grad_remove_fd)):
            print(f"  >>> theta_{which}")
            for bin_idx in range(gC.shape[0]):
                for k in range(nparams):
                    denom = max(abs(gFD[bin_idx, k]), 1e-300)
                    rel = abs(gC[bin_idx, k] - gFD[bin_idx, k]) / denom
                    worst_swap_grad = max(worst_swap_grad, rel)
                # only print the bin's worst-row to keep the log readable;
                # uncomment the per-k print above if you need full detail.
                row_rel = np.max(np.abs(gC[bin_idx] - gFD[bin_idx])
                                 / np.maximum(np.abs(gFD[bin_idx]), 1e-300))
                print(f"    bin{bin_idx}  worst rel = {row_rel:.3e}")
                # full per-parameter dump for inspection (concise)
                for k in range(nparams):
                    print(f"      {param_names[k]:>5s}  C={gC[bin_idx,k]:+.6e}  FD={gFD[bin_idx,k]:+.6e}")
        status = "ok" if worst_swap_grad < grad_tol else "FAIL"
        print(f"  swap-gradient worst rel = {worst_swap_grad:.3e}   tol = {grad_tol}   [{status}]\n")

    # The wrap's __call__ takes the 9 scalar params as *args (see GBLookupWaveWrap above).
    
    # Per-channel AET cross-check: convert XYZ → AET in WDM space and compare
    # each orthogonal AET channel separately (they decouple under AET1Sens).
    from lisatools.sensitivity import AET1SensitivityMatrix
    def xyz_to_aet(arr):
        X, Y, Z = arr[0], arr[1], arr[2]
        A = (Z - X) / np.sqrt(2.0)
        E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
        T = (X + Y + Z) / np.sqrt(3.0)
        return np.stack([A, E, T], axis=0)
    inj_aet = xyz_to_aet(injection.data_res_arr.arr)
    c_tpl_aet = xyz_to_aet(template_fill_wdm.arr)
    py_tpl_aet = xyz_to_aet(py_wdm_lookup.arr)
    sens_aet = AET1SensitivityMatrix(WDMSignal(inj_aet, wdm_set).settings)
    invC = sens_aet.invC
    prefactor = 4.0 * sens_aet.differential_component
    print(f"==== per-channel AET inner products ====")
    print(f"{'chan':6s} {'<d|d>':>12s} {'<d|h_C>':>12s} {'<d|h_py>':>12s} {'r_C':>9s} {'r_py':>9s}")
    for chan, name in enumerate("AET"):
        d_c = inj_aet[chan]; hC = c_tpl_aet[chan]; hPy = py_tpl_aet[chan]; ic = invC[chan]
        dd = (d_c*d_c*ic).sum()*prefactor
        dhC = (d_c*hC*ic).sum()*prefactor
        dhPy = (d_c*hPy*ic).sum()*prefactor
        print(f"  {name:4s} {dd:+12.4e} {dhC:+12.4e} {dhPy:+12.4e} "
              f"{(dhC/dd if dd!=0 else float('nan')):+9.4f} "
              f"{(dhPy/dd if dd!=0 else float('nan')):+9.4f}")
    print()

    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    injection.data_res_arr.heatmap(fig=fig, ax=ax1, index=0)
    template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True)
    py_wdm_lookup.heatmap(fig=fig, ax=ax3, index=0)
    ax1.set_title("injection")
    ax2.set_title("C lookup template")
    ax3.set_title("Python lookup template")
    plt.tight_layout()
    plt.savefig("gb_lookup_test_heatmap.png", dpi=120)
    plt.close()
    print("Saved heatmap to gb_lookup_test_heatmap.png")

    # Stop here unless the user opts into the MCMC stage. The MCMC below takes a
    # long time; for plain inner-product verification we don't need it.
    # if os.environ.get("RUN_MCMC", "0") != "1":
    #     print("Verification complete. Set RUN_MCMC=1 to also run the MCMC stage.")
    #     sys.exit(0)
    exit()
    analysis_mcmc = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)
    
    ntemps = 10
    nwalkers = 20


    # The order here defines full_basis — must never change
    full_basis = [
        "amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"
    ]

    # 12 sampled parameters — order matches priors_in keys
    sampled_basis = [
        "amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam", "sinbeta"
    ]

    key_map = {
        "cosinc": "inc",
        "sinbeta": "beta"
    }

    parameter_transforms = {
        'cosinc': np.arccos,
        'sinbeta': np.arcsin,
    }

    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,   # no transforms needed: M is linear, qS/qK are cosines
        fill_dict={
            'fddot0': 0.0,
        },
        key_map=key_map
    )

    priors = {"gb": ProbDistContainer({
        "amp": uniform_dist(1e-24, 1e-21),
        "f0": uniform_dist(1e-4, 30e-3),
        "fdot": uniform_dist(1e-19, 1e-12),      
        "phi0":     uniform_dist(0.0, 2*np.pi),    
        "cosinc":    uniform_dist(-1.0, 1.0),        
        "psi":     uniform_dist(0.0, np.pi),     
        "lam": uniform_dist(0.0, 2*np.pi),    
        "sinbeta":  uniform_dist(-1.0, 1.0),       
    })}


    factor_gen = 1e-2

    gen_dist = {"gb": ProbDistContainer({
        "amp": uniform_dist(amp[0] * (1.0 - factor_gen), amp[0] * (1.0 + factor_gen)),
        "f0": uniform_dist(f0[0] * (1.0 - 1e-8), f0[0] * (1.0 + 1e-8)),  # different value for frequency
        "fdot0":  uniform_dist(fdot[0] *  (1.0 - factor_gen), fdot[0] * (1.0 + factor_gen)),       # dimensionless spin
        "phi0":    uniform_dist(phi0[0] *  (1.0 - factor_gen), phi0[0] * (1.0 + factor_gen)),       # semi-latus rectum
        "cosinc":       uniform_dist(np.cos(inc[0]) *  (1.0 - factor_gen), np.cos(inc[0]) * (1.0 + factor_gen)),         # eccentricity
        "psi":     uniform_dist(psi[0] *  (1.0 - factor_gen), psi[0] * (1.0 + factor_gen)),        # luminosity distance [Gpc]
        "lam":    uniform_dist(lam[0] *  (1.0 - factor_gen), lam[0] * (1.0 + factor_gen)),        # cos(polar spin angle)
        "sinbeta":     uniform_dist(np.sin(beta[0]) *  (1.0 - factor_gen), np.sin(beta[0]) * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
    })}

    ndims = {"gb": len(sampled_basis)}

    periodic_container = PeriodicContainer({"gb": {"phi0": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}, key_order={"gb": sampled_basis})
    fp = f"test_gb_lookup_pe_new_new_1.h5"
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"gb": gen_dist["gb"].rvs(size=(ntemps, nwalkers, 1))})
        
    sampler = EnsembleSampler(
        nwalkers,
        ndims,                              # list of ndims, one per branch
        analysis_mcmc.eryn_likelihood_function,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(
            transform_fn=tc,
            source_only=True,
        ),
        moves=StretchMove(live_dangerously=True),  # allows for testing
        branch_names=["gb"],
        periodic=periodic_container,
        backend=fp
    )

    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(start_state.branches_coords)
        start_state.log_like = sampler.compute_log_like(start_state.branches_coords, logp=start_state.log_prior)[0]

    print("start log_like: ", start_state.log_like)
    
    inj_params = params[0, tc.test_inds].copy()
    inj_params[sampled_basis.index("cosinc")] = np.cos(inj_params[sampled_basis.index("cosinc")])
    inj_params[sampled_basis.index("sinbeta")] = np.sin(inj_params[sampled_basis.index("sinbeta")])
    tmp_state = State({"gb": np.tile(inj_params, (ntemps, nwalkers, 1, 1))})
    best_like = sampler.compute_log_like(tmp_state.branches_coords)[0]
    print("best_like: ", best_like)
    
    nsteps = 2000
    burn = 0
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=5, progress=True)


