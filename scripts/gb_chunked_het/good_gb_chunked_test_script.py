#!/usr/bin/env python
# coding: utf-8

# Drop-in counterpart to gb_lookup_table_test_script.py: all the WDM
# lookup-table machinery is replaced with the chunked-heterodyne template
# pipeline (GBWDMHeterodyne from gb_wdm_het) -- the exact same pipeline
# exercised in gb_chunked_prior_draws.py. Everything else (injection
# generation, settings, analysis containers, MCMC) is kept as close to
# the lookup test script as possible.

import os, sys
import numpy as np
import matplotlib
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

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal, WDMSettings, WDMSignal

from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

from eryn.moves import StretchMove
from eryn.moves.nuts import NUTSMove
from eryn.ensemble import EnsembleSampler
from eryn.utils import PeriodicContainer

from eryn.state import State
from eryn.backends import HDFBackend

# Chunked-heterodyne template + likelihood class, now living in
# ``fastlisaresponse.gbcomps`` (refactored away from the repo-root
# ``gb_wdm_het``: same C++/JAX chunked-het kernels, but constructor +
# methods follow the parallel-module convention --
# ``GBWDMComputations(force_backend=...)`` with ``fill_global_wdm``,
# ``get_ll_wdm``, ``get_swap_ll_wdm``, ``get_ll_grad_wdm``).
from fastlisaresponse.gbcomps import GBWDMComputations, GBFDComputations


import time
class GBChunkedWaveWrap:
    """Adapter: per-source (9,) param tuple -> ``WDMSignal`` on the
    active-band ``wdm_set``. Builds the template via
    :meth:`GBWDMComputations.fill_global_wdm` on the full-grid buffer
    and slices down to the active band for the validation plots /
    inner-product prints.
    """
    def __init__(self, gb_wdm_comp, wdm_set, Nf, Nt):
        self.gb_wdm_comp = gb_wdm_comp
        self.wdm_set = wdm_set
        self.Nf = Nf
        self.Nt = Nt

    def __call__(self, *params):
        params_arr = np.asarray(params, dtype=float).reshape(1, 9)
        template_full = np.zeros((3, self.Nf, self.Nt), dtype=float)
        self.gb_wdm_comp.fill_global_wdm(
            params_arr, template_full,
            convert_to_ra_dec=False, factors=None,
        )
        tpl_active = template_full[
            :, self.wdm_set.ind_min_f: self.wdm_set.ind_max_f + 1, :
        ]
        if self.wdm_set.Nt_active != self.wdm_set.Nt:
            tpl_active = tpl_active[:, :, self.wdm_set.active_slice_t]
        return WDMSignal(tpl_active, self.wdm_set)


class _FullGridWDMHolder:
    """Minimal duck-type for :meth:`GBWDMComputations.get_ll_wdm`'s
    ``wdm_holder`` argument.

    The chunked-het C++ kernel reads ``data`` and diagonal ``invC`` at
    layout ``(nchannels, Nf, Nt)`` (flat). The active-band
    :class:`AnalysisContainerArray` stores cross-channel invC at
    ``(nchannels, nchannels, Nf_active, Nt_active)`` -- a shape and
    band mismatch. So we wrap full-grid ``data_d`` + diagonal ``invC``
    arrays in a tiny shim that exposes the same
    ``linear_data_arr[0]`` / ``linear_psd_arr[0]`` / ``__len__``
    surface ``gbcomps.GBWDMComputations.get_ll_wdm`` reads.
    """
    def __init__(self, data_full, invC_diag_full):
        self.linear_data_arr = [np.ascontiguousarray(data_full).ravel()]
        self.linear_psd_arr  = [np.ascontiguousarray(invC_diag_full).ravel()]

    def __len__(self):
        return 1


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
    # NF, NT env overrides -- change wavelet aspect at constant total signal
    # length (Nf*Nt). Defaults preserve current 1460x2560 setup.
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
    # TUKEY_ALPHA env var sets an EXTRA Tukey window on the injection
    # TD->WDM transform. The chunked-het template builder already
    # applies its own per-chunk Tukey (``GBWDMComputations.resolved_tukey_alpha``,
    # set by ``recommended_tukey_alpha("heterodyne")`` -- 0.01 for
    # N_sparse>=512, 0.05 for N_sparse<512, both in the validated
    # 0.01-0.05 range). Default 0.0 here keeps the injection unwindowed
    # so the only Tukey in the pipeline is the template's intrinsic
    # one (matches the mm ~ 1e-9 validated regime); set TUKEY_ALPHA>0
    # only if you also know how to match it on the template side.
    _tukey_alpha = float(os.environ.get("TUKEY_ALPHA", "0.0"))
    if _tukey_alpha > 0:
        window = xp.asarray(signal.windows.tukey(N, alpha=_tukey_alpha))
    else:
        window = xp.ones(N)

    min_freq = 0.0001
    max_freq = 35.0e-3
    fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    # shave edges?
    _EDGE_CUT = int(os.environ.get("EDGE_CUT", "20"))
    min_time = _EDGE_CUT * wavelet_duration
    max_time = (Nt - _EDGE_CUT) * wavelet_duration

    wdm_set = WDMSettings(Nf, Nt, dt, t0=t_start,
                           min_freq=min_freq, max_freq=max_freq,
                           min_time=min_time, max_time=max_time)

    # --- chunked-heterodyne computations (template + likelihood) ----------
    # ``GBWDMComputations`` is the FastLISAResponseParallelModule whose
    # ``fill_global_wdm`` / ``get_ll_wdm`` / ``get_ll_grad_wdm`` route
    # through ``GBComputationGroupWrap.gb_wdm_het_*`` on the chosen
    # backend (C++ on cpu/cuda, pure-JAX on the ``jax`` backend).
    Nt_sub = int(os.environ.get("NT_SUB", 256))
    N_sparse = int(os.environ.get("N_SPARSE", 256))
    n_pad = int(os.environ.get("N_PAD", Nt_sub // 8))
    gb_wdm_comp = GBWDMComputations(
        wdm_set, t_ref=t_ref,
        Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
        N_cp_sig=int(os.environ.get("N_CP_SIG", 0)),
        N_cp_orbit=int(os.environ.get("N_CP_ORBIT", 0)),
        orbits=orbits,                                  # MUST match injection
        tdi_config="2nd generation",
        force_backend=backend,
        d_d=0.0,                                        # source-only return
        tdi_type="XYZ",
    )
    print(f"[step] chunked: n_chunks={gb_wdm_comp.n_chunks}, "
          f"T_chunk={gb_wdm_comp.T_chunk:.3e}s, "
          f"alpha={gb_wdm_comp.resolved_tukey_alpha} "
          f"(use_tukey={gb_wdm_comp.use_tukey})", flush=True)

    # F_FRACS sweep: comma-separated f_frac values to test. Defaults span
    # on-grid (~0/1) and mid-grid (~0.5). Set F_FRAC for a single value.
    if "F_FRAC" in os.environ:
        _f_frac_list = [float(os.environ["F_FRAC"])]
    else:
        _f_frac_list = [float(s) for s in os.environ.get("F_FRACS", "0.05,0.5,0.95").split(",")]

    # Shared one-shot construction (does not depend on f_frac).
    output_set = wdm_set
    if output_set != wdm_set:
        raise ValueError("This script requires WDM for the output_set.")
    sens_mat_proto = None  # build once injection settings are known

    print(f"[step] f_fracs={_f_frac_list}", flush=True)

    # Source's m-layer reference -- in-band layer near 3 mHz (the original
    # script's 18 mHz default is outside the WDM band).
    m_ref_source = int(3e-3 / wdm_set.layer_df)

    _results = []
    # M_OFFSET shifts the source's m_floor by an integer relative to m_ref_source.
    _m_offset = int(os.environ.get("M_OFFSET", "0"))
    for _ff_idx, _ff in enumerate(_f_frac_list):
        num_bin = 1
        amp = np.full(num_bin, 1.0e-22)
        f0 = np.full(num_bin, (m_ref_source + _m_offset + _ff) * wdm_set.layer_df)
        # SOURCE_FDOT (Hz/s). Default 1e-17 (essentially fdot=0).
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

        data_inj_all = TDSignal(_data_inj, settings=td_set).transform(output_set, window=window)
        injection = DataResidualArray(data_inj_all)
        if sens_mat_proto is None:
            sens_mat_proto = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")
        sens_mat = sens_mat_proto

        gb_gen_wrap = GBChunkedWaveWrap(gb_wdm_comp, wdm_set, Nf, Nt)
        analysis = AnalysisContainer(injection, sens_mat, signal_gen=gb_gen_wrap)
        wdm_holder = AnalysisContainerArray([analysis])

        # Build template via chunked-het. (No separate Python vs C path
        # like the lookup script; chunked-het is the single C++ template
        # builder.)
        template_fill_wdm = gb_gen_wrap(*params[0])
        check_ll = analysis.template_likelihood(template_fill_wdm)
        check_ip = analysis.template_inner_product(template_fill_wdm)
        overlap = analysis.template_inner_product(template_fill_wdm, normalize=True)

        check_ip_d_d = analysis.inner_product()
        tmp_val2 = analysis.calculate_signal_inner_product(*params[0])
        tmp_val3 = analysis.calculate_signal_likelihood(*params[0], source_only=True)

        _f_frac_meas = (f0[0] - (int(f0[0] / wdm_set.layer_df) * wdm_set.layer_df)) / wdm_set.layer_df
        _mm = float(1.0 - overlap)

        # mm5 -- mismatch restricted to a narrow +-5-layer band around f0.
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
        _tpl_here = DataResidualArray(WDMSignal(template_fill_wdm[:, _m_lo:_m_hi], _new_wdm_set))
        _mm5 = float(1.0 - _ah5.template_inner_product(_tpl_here, normalize=True))

        _results.append((_ff, _f_frac_meas, _mm, _mm5))

        # SAVE_RESIDUAL=1 dumps the (injection - template) WDM array.
        if os.environ.get("SAVE_RESIDUAL", "0") == "1":
            _inj_arr = data_inj_all.arr
            _tpl_arr = template_fill_wdm.arr
            _resid = _inj_arr - _tpl_arr
            _ms = int(f0[0] / wdm_set.layer_df)
            _m_active = _ms - wdm_set.ind_min_f
            np.savez_compressed(
                f"residual_chunked_ff{_ff:.3f}.npz",
                inj=_inj_arr, tpl=_tpl_arr, resid=_resid,
                m_source_active=_m_active, m_source_abs=_ms,
                ind_min_f=wdm_set.ind_min_f, ind_min_t=wdm_set.ind_min_t,
                layer_df=wdm_set.layer_df, layer_dt=wdm_set.layer_dt,
                f0=float(f0[0]),
            )
            _r_per_m = np.sqrt((_resid ** 2).sum(axis=(0, 2)))
            _i_per_m = np.sqrt((_inj_arr ** 2).sum(axis=(0, 2)))
            print(f"  [resid] saved residual_chunked_ff{_ff:.3f}.npz; "
                  f"source at m_abs={_ms} (active {_m_active})")
            print(f"  [resid] per-layer  m_off   ||resid||   ||inj||   ratio")
            for _moff in [-3, -2, -1, 0, 1, 2, 3]:
                _mi = _m_active + _moff
                if 0 <= _mi < _r_per_m.size:
                    print(f"          {_moff:+5d}    {_r_per_m[_mi]:.3e}   "
                          f"{_i_per_m[_mi]:.3e}   "
                          f"{_r_per_m[_mi] / max(_i_per_m[_mi], 1e-30):.3e}")

        print(f"\n=== f_frac sweep [{_ff_idx + 1}/{len(_f_frac_list)}]  "
              f"requested={_ff:.3f}  measured={_f_frac_meas:+.4f}  "
              f"f0={f0[0]*1e3:.5f} mHz ===")
        print(f"[result] base inner_product <d|d>             = {check_ip_d_d}")
        print(f"[result] template_inner_product (chunked-het) = {check_ip}")
        print(f"[result] calculate_signal_inner_product       = {tmp_val2}")
        print(f"[result] template_likelihood (chunked-het)    = {check_ll}")
        print(f"[result] calculate_signal_likelihood          = {tmp_val3}")
        print(f"[result] Noise-weighted mismatch (chunked-het) = {_mm:.3e}")
        print(f"[result] mm5 (+-5-layer band, chunked-het)     = {_mm5:.3e}")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11, 6))
        data_inj_all.heatmap(fig=fig, ax=ax1, index=0, add_cax=True)
        template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0)
        ax1.set_title(
            f"injection  f0={f0[0]*1e3:.5f} mHz  phi0={phi0[0]:.4f}  f_frac={_f_frac_meas:+.3f}"
        )
        ax2.set_title(f"chunked-het template  (mm = {_mm:.3e})")
        ax1.set_ylim(f0[0] - 10 * wdm_set.layer_df, f0[0] + 10 * wdm_set.layer_df)
        plt.tight_layout()
        _out_path = f"gb_chunked_ff{_f_frac_meas:+.3f}.png"
        plt.savefig(_out_path, dpi=120)
        if os.environ.get("SHOW_PLOTS", "0") == "1":
            plt.show()
        plt.close(fig)
        print(f"[plot] saved {_out_path}")

    print("\n[summary] f_frac sweep")
    print(f"  Nt_sub={Nt_sub}  N_sparse={N_sparse}  n_pad={n_pad}")
    print("  requested  measured  mm           mm5")
    for _ff, _ff_m, _mm, _mm5 in _results:
        print(f"  {_ff:>9.3f}  {_ff_m:+.4f}   {_mm:.3e}   {_mm5:.3e}")

    if os.environ.get("DEBUG_BREAK", "0") == "1":
        breakpoint()

    # Per-channel AET cross-check.
    from lisatools.sensitivity import AET1SensitivityMatrix
    def xyz_to_aet(arr):
        X, Y, Z = arr[0], arr[1], arr[2]
        A = (Z - X) / np.sqrt(2.0)
        E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
        T = (X + Y + Z) / np.sqrt(3.0)
        return np.stack([A, E, T], axis=0)
    inj_aet = xyz_to_aet(injection.data_res_arr.arr)
    tpl_aet = xyz_to_aet(template_fill_wdm.arr)
    sens_aet = AET1SensitivityMatrix(WDMSignal(inj_aet, wdm_set).settings)
    invC = sens_aet.invC
    prefactor = 4.0 * sens_aet.differential_component
    print(f"==== per-channel AET inner products ====")
    print(f"{'chan':6s} {'<d|d>':>12s} {'<d|h>':>12s} {'<d|h>/<d|d>':>14s}")
    for chan, name in enumerate("AET"):
        d_c = inj_aet[chan]; h = tpl_aet[chan]; ic = invC[chan]
        dd = (d_c * d_c * ic).sum() * prefactor
        dh = (d_c * h * ic).sum() * prefactor
        print(f"  {name:4s} {dd:+12.4e} {dh:+12.4e} "
              f"{(dh/dd if dd != 0 else float('nan')):+14.4f}")
    print()

    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    injection.data_res_arr.heatmap(fig=fig, ax=ax1, index=0)
    template_fill_wdm.heatmap(fig=fig, ax=ax2, index=0, add_cax=True)
    ax1.set_title("injection")
    ax2.set_title("chunked-het template")
    plt.tight_layout()
    plt.savefig("gb_chunked_test_heatmap.png", dpi=120)
    plt.close()
    print("Saved heatmap to gb_chunked_test_heatmap.png")

    # ---- Eryn MCMC (vectorized, C++ likelihood; WDM or FD) ----
    # DOMAIN selects the inner-product domain for the MCMC likelihood:
    #   wdm -> chunked-heterodyne (GBWDMHeterodyne.get_ll) -- default
    #   fd  -> FD heterodyne kernel (GBFDComputations.get_ll_fd)
    # Both expose a batched C++ "get_ll" that returns <d|h>, <h|h> for
    # a list of binaries in a single call -- ideal for Eryn vectorize=True.
    DOMAIN = os.environ.get("DOMAIN", "wdm").lower()
    if DOMAIN not in ("wdm", "fd"):
        raise ValueError(f"DOMAIN must be 'wdm' or 'fd', got {DOMAIN!r}")
    print(f"[mcmc] DOMAIN={DOMAIN}", flush=True)

    if DOMAIN == "wdm":
        # The injection lives on the WDM active band (Nf_active, Nt_active).
        # The chunked-het C++ kernel iterates m in [0, Nf) on the FULL WDM
        # grid, so we zero-fill the active-band injection (+ diag invC) onto
        # (3, Nf, Nt). Then we compute a one-time scale factor so that
        # chunked.get_ll's bare <d,d> sum matches lisatools' inner_product
        # (which multiplies by 4 * differential_component). See
        # gb_chunked_prior_draws.py lines ~400-450 for the same recipe.
        nch = 3
        inj_active_arr = np.asarray(injection.data_res_arr.arr)
        psd_active = np.asarray(sens_mat.sens_mat)
        if psd_active.ndim == 4:
            psd_diag = np.stack([psd_active[c, c] for c in range(nch)], axis=0)
        else:
            psd_diag = psd_active
        with np.errstate(divide="ignore", invalid="ignore"):
            invC_active = 1.0 / np.where(
                np.isfinite(psd_diag) & (psd_diag > 0),
                psd_diag, np.inf,
            )
        invC_active = np.where(np.isfinite(invC_active), invC_active, 0.0)

        data_d_full = np.zeros((nch, Nf, Nt), dtype=float)
        invC_full   = np.zeros_like(data_d_full)
        ilo = wdm_set.ind_min_f
        ihi = wdm_set.ind_max_f + 1
        if wdm_set.Nt_active == wdm_set.Nt:
            data_d_full[:, ilo:ihi, :] = inj_active_arr
            invC_full  [:, ilo:ihi, :] = invC_active
        else:
            tslice = wdm_set.active_slice_t
            data_d_full[:, ilo:ihi, tslice] = inj_active_arr
            invC_full  [:, ilo:ihi, tslice] = invC_active

        d_d_lt = float(np.real(analysis.inner_product()))
        print(f"[mcmc/wdm] d_d (lisatools) = {d_d_lt:.6e}", flush=True)
        
        # Layer-grouping is now the canonical narrow-band GB inner
        # product (matches mm5 / mm2 -- see ``gb_chunked_prior_draws.py``
        # validation, median mm5 ~1e-9). Default ON; env var still
        # lets you toggle off for the wide-band reference path used in
        # earlier validation runs.
        _use_layer_groups = os.environ.get("USE_LAYER_GROUPS", "1") == "1"
        _group_band_layers = int(os.environ.get("GROUP_BAND_LAYERS", 5))
        _margin_layers = int(os.environ.get("MARGIN_LAYERS", 0))
        if _use_layer_groups:
            print(f"[mcmc/wdm] use_layer_groups=True  band={_group_band_layers}  "
                  f"margin={_margin_layers}", flush=True)

        # Wrap the full-grid ``(3, Nf, Nt)`` data + diagonal invC into a
        # duck-type the new ``get_ll_wdm`` accepts (``len()`` +
        # ``linear_data_arr[0]`` + ``linear_psd_arr[0]``).
        wdm_holder_full = _FullGridWDMHolder(data_d_full, invC_full)

        def logl_vec(x, transform_fn=None, **_kw):
            """Vectorized chunked-het source-only log-likelihood for Eryn.

            x: ``(N, ndim_sampled)`` or ``(ndim_sampled,)`` array of
            sampled-basis params; transform_fn is the TransformContainer
            that maps sampled -> physical (9-D) basis. Returns shape ``(N,)``
            log-likelihood values.

            Dispatches through
            :meth:`GBWDMComputations.get_ll_wdm` -- a single batched
            ``gb_wdm_het_get_ll`` C++ kernel call across all walkers
            that returns ``-0.5 * (d_d + h_h - 2 d_h)``. With ``d_d=0``
            (set at construction) the return is the source-only piece;
            the global ``-0.5 <d|d>`` constant is added by the engine.
            """
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr[None, :]
            if transform_fn is not None:
                phys = transform_fn.both_transforms(x_arr.copy())  # (N, 9)
            else:
                phys = x_arr
            
            ll = gb_wdm_comp.get_ll_wdm(
                phys, wdm_holder_full,
                convert_to_ra_dec=False,
                use_layer_groups=_use_layer_groups,
                group_band_layers=_group_band_layers,
                margin_layers=_margin_layers,
            )
            if os.environ.get("DEBUG_LOGL", "0") == "1":
                breakpoint()
            return np.asarray(ll)

        # --------------------------------------------------------------
        # JAX-autograd gradient of the same chunked-het log-likelihood.
        # Plugged into eryn.moves.NUTSMove. The sampled->phys transform
        # (arccos for cosinc, arcsin for sinbeta; fddot filled to 0)
        # is reproduced in jax.numpy so jax.grad chain-rules through it.
        #
        # We pull chunk geometry, WDM window, and JAX-native orbits /
        # tdi_config from a sibling ``GBWDMComputations(force_backend=
        # 'jax', ...)`` instance, so all the host-side knobs come from
        # the same place as the C++ path. The standalone
        # ``gb_wdm_het_get_ll_jax`` kernel is used (the backend wrap
        # mutates numpy buffers as a side effect, which jax.grad can't
        # trace through).
        # --------------------------------------------------------------
        import jax
        import jax.numpy as _jax_xp                       # alias for clarity
        from fastlisaresponse.jax.wdm.heterodyne_kernels import (
            gb_wdm_het_get_ll_jax,
        )
        from fastlisaresponse.jax.sources.ucb import JaxUCBSource

        gb_wdm_comp_jax = GBWDMComputations(
            wdm_set, t_ref=t_ref,
            Nt_sub=Nt_sub, n_pad=n_pad, N_sparse=N_sparse,
            N_cp_sig=int(os.environ.get("N_CP_SIG", 0)),
            N_cp_orbit=int(os.environ.get("N_CP_ORBIT", 0)),
            orbits=orbits, tdi_config="2nd generation",
            force_backend="jax", d_d=0.0, tdi_type="XYZ",
        )
        # JAX-side static kernel args -- built once, reused per call.
        _jax_data_d = _jax_xp.asarray(data_d_full)
        _jax_invC   = _jax_xp.asarray(invC_full)
        _jax_chunk_t_starts = _jax_xp.asarray(gb_wdm_comp_jax.chunk_t_starts)
        _jax_chunk_keep_lo  = _jax_xp.asarray(gb_wdm_comp_jax.chunk_keep_lo)
        _jax_chunk_keep_hi  = _jax_xp.asarray(gb_wdm_comp_jax.chunk_keep_hi)
        _jax_chunk_n_lo     = _jax_xp.asarray(
            gb_wdm_comp_jax.chunk_n_global_offset
        )
        _jax_wdm_window     = _jax_xp.asarray(gb_wdm_comp_jax.wdm_window)
        _jax_orbits         = gb_wdm_comp_jax.cpp_orbits     # OrbitsWrapJAX
        _jax_tdi_config     = gb_wdm_comp_jax.cpp_tdi_config # TDIConfigWrapJAX
        _jax_source         = JaxUCBSource(t_ref=float(t_ref))

        def _sampled_to_phys_jax(x):
            """Replicate the TransformContainer (sampled -> phys 9-D).

            sampled order: amp, f0, fdot0, phi0, cosinc, psi, lam, sinbeta
            phys order:    amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta
            """
            xp = _jax_xp
            amp     = x[:, 0]
            f0      = x[:, 1]
            fdot0   = x[:, 2]
            phi0    = x[:, 3]
            cosinc  = x[:, 4]
            psi     = x[:, 5]
            lam     = x[:, 6]
            sinbeta = x[:, 7]
            inc     = xp.arccos(xp.clip(cosinc, -1.0 + 1e-12, 1.0 - 1e-12))
            beta    = xp.arcsin(xp.clip(sinbeta, -1.0 + 1e-12, 1.0 - 1e-12))
            fddot   = xp.zeros_like(amp)
            return xp.stack(
                [amp, f0, fdot0, fddot, phi0, inc, psi, lam, beta], axis=-1,
            )

        def _scalar_logl_sampled(x_arr):
            """Sum_i L_i for jax.grad; per-walker L_i depends only on
            x[i, :], so the gradient block-diagonalises and one grad
            call recovers (N, 8) per-walker gradients.
            """
            phys = _sampled_to_phys_jax(x_arr)
            d_h, h_h = gb_wdm_het_get_ll_jax(
                phys, _jax_data_d, _jax_invC,
                _jax_chunk_t_starts, _jax_chunk_keep_lo, _jax_chunk_keep_hi,
                _jax_chunk_n_lo,
                _jax_source, _jax_orbits, _jax_tdi_config,
                _jax_wdm_window,
                Nf=Nf, Nt=Nt, Nt_sub=Nt_sub, N_sparse=N_sparse,
                dt=dt, T_chunk=gb_wdm_comp_jax.T_chunk,
                tukey_alpha=gb_wdm_comp_jax.resolved_tukey_alpha,
            )
            return _jax_xp.sum(d_h - 0.5 * h_h)

        _grad_logl_sampled = jax.grad(_scalar_logl_sampled)

        def grad_logl_vec_jax(x, **_kw):
            """Untempered gradient of logl. NUTSMove applies beta itself."""
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr[None, :]
            g = np.asarray(_grad_logl_sampled(_jax_xp.asarray(x_arr)))
            return g

    else:  # DOMAIN == "fd"
        # FD path: FFT the TD injection (no window, dt-scaled rfft, same
        # convention as gb_fd_wdm_side_by_side_test.py:138-165 and matches
        # the C heterodyne FFT in GBFDComputations). XYZ2SensitivityMatrix
        # on the FDSettings yields the (3, 3, n_rfft) cross-channel invC
        # that the FD kernel consumes directly -- no diag approximation.
        td_inj_for_fd = TDSignal(_data_inj, td_set)
        fd_inj_sig = td_inj_for_fd.fft()                       # rfft * dt
        data_fd_arr = np.asarray(fd_inj_sig.arr)               # (3, n_rfft)
        n_rfft_fd = data_fd_arr.shape[-1]
        fd_set_mcmc = FDSettings(n_rfft_fd, df, force_backend=backend)
        sens_mat_fd = XYZ2SensitivityMatrix(fd_set_mcmc, model="scirdv1")
        invC_fd = np.asarray(sens_mat_fd.invC).copy()           # (3, 3, n_rfft)
        # Zero the DC bin (lisatools' inner_product also skips it; matches
        # the C FD kernel's ind_min=1 convention).
        invC_fd[:, :, 0] = 0.0

        data_arr_fd = data_fd_arr[None, ...].astype(complex)   # (1, 3, n_rfft)
        invC_arr_fd = invC_fd[None, ...].astype(float)         # (1, 3, 3, n_rfft)

        # N_sparse_fd: sparse-FFT length used by the FD heterodyne kernel
        # (must be power of 2). 4096 is the side-by-side test's default and
        # gives ample margin for f0 ~ a few mHz.
        N_sparse_fd = int(os.environ.get("FD_N_SPARSE", 4096))
        fd_comp = GBFDComputations(
            T=Tobs, t_ref=t_ref, t_start=t_start,
            N_sparse=N_sparse_fd, df=df,
            data_fd=data_arr_fd, invC=invC_arr_fd,
            orbits=orbits, tdi_config=tdi_config,
            force_backend=backend, tdi_type="XYZ",
            ind_min=1, ind_max=n_rfft_fd - 1,
            d_d=0.0,
        )
        print(f"[mcmc/fd] n_rfft={n_rfft_fd}  N_sparse_fd={N_sparse_fd}  "
              f"df={df:.3e} Hz", flush=True)

        def logl_vec(x, transform_fn=None, **_kw):
            """Vectorized FD-heterodyne source-only log-likelihood for Eryn.

            x: ``(N, ndim_sampled)`` or ``(ndim_sampled,)`` array of
            sampled-basis params; transform_fn is the TransformContainer
            that maps sampled -> physical (9-D) basis. Returns shape ``(N,)``
            log-likelihood values.

            Uses ``GBFDComputations.get_ll_fd`` (C++ FD kernel) which
            returns ``-0.5 * (d_d + h_h - 2 d_h)``. The instance is
            constructed with ``d_d=0`` so the kernel return is the
            source-only piece; lisatools / the engine adds the global
            ``-0.5 <d|d>`` constant separately when forming the total
            log-likelihood.
            """
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr[None, :]
            if transform_fn is not None:
                phys = transform_fn.both_transforms(x_arr.copy())  # (N, 9)
            else:
                phys = x_arr
            ll = np.asarray(fd_comp.get_ll_fd(phys, convert_to_ra_dec=False))
            _bad = ~np.isfinite(ll)
            if _bad.any():
                _idx = np.where(_bad)[0]
                print(f"[fd/NaN] {_bad.sum()}/{ll.size} non-finite ll values "
                      f"on this batch (logl_vec call)", flush=True)
                _full_basis = ["amp", "f0", "fdot0", "fddot0", "phi0", "inc",
                               "psi", "lam", "beta"]
                _samp_basis = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi",
                               "lam", "sinbeta"]
                for _k in _idx[:5]:
                    _ph = phys[_k]
                    _sm = x_arr[_k]
                    print(f"  [{_k}] ll={ll[_k]}", flush=True)
                    print(f"        sampled: " + ", ".join(
                        f"{_n}={_v:.6e}" for _n, _v in zip(_samp_basis, _sm)),
                          flush=True)
                    print(f"        phys   : " + ", ".join(
                        f"{_n}={_v:.6e}" for _n, _v in zip(_full_basis, _ph)),
                          flush=True)
                # Also report which physical params are out-of-domain.
                _amp  = phys[_idx, 0]
                _f0   = phys[_idx, 1]
                _fdot = phys[_idx, 2]
                _inc  = phys[_idx, 5]
                _beta = phys[_idx, 8]
                print(f"  range: amp [{_amp.min():.3e}, {_amp.max():.3e}]  "
                      f"f0 [{_f0.min():.3e}, {_f0.max():.3e}]  "
                      f"fdot [{_fdot.min():.3e}, {_fdot.max():.3e}]",
                      flush=True)
                print(f"         inc [{_inc.min():.4f}, {_inc.max():.4f}]  "
                      f"beta [{_beta.min():.4f}, {_beta.max():.4f}]",
                      flush=True)
                _nan_phys = ~np.isfinite(phys[_idx]).all(axis=1)
                print(f"  rows with NaN/Inf in phys after transform: "
                      f"{_nan_phys.sum()}/{len(_idx)}", flush=True)
                # Persist the first batch with offenders for offline analysis.
                if not hasattr(logl_vec, "_dumped"):
                    np.savez("fd_nan_debug.npz",
                             sampled=x_arr, phys=phys, ll=ll,
                             bad_idx=_idx)
                    print("  -> wrote fd_nan_debug.npz", flush=True)
                    logl_vec._dumped = True
            return ll

    ntemps = int(os.environ.get("NTEMPS", 10))
    nwalkers = int(os.environ.get("NWALKERS", 20))

    # The order here defines full_basis -- must never change
    full_basis = [
        "amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"
    ]

    # 8 sampled parameters -- order matches priors_in keys
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
        parameter_transforms=parameter_transforms,
        fill_dict={
            'fddot0': 0.0,
        },
        key_map=key_map
    )

    priors = {"gb": ProbDistContainer({
        "amp": uniform_dist(1e-24, 1e-21),
        "f0": uniform_dist(1e-4, 30e-3),
        "fdot": uniform_dist(1e-19, 1e-12),
        "phi0":    uniform_dist(0.0, 2*np.pi),
        "cosinc":  uniform_dist(-1.0, 1.0),
        "psi":     uniform_dist(0.0, np.pi),
        "lam":     uniform_dist(0.0, 2*np.pi),
        "sinbeta": uniform_dist(-1.0, 1.0),
    })}


    factor_gen = 1e-2

    gen_dist = {"gb": ProbDistContainer({
        "amp":     uniform_dist(amp[0] * (1.0 - factor_gen), amp[0] * (1.0 + factor_gen)),
        "f0":      uniform_dist(f0[0] * (1.0 - 1e-7), f0[0] * (1.0 + 1e-7)),
        "fdot0":   uniform_dist(fdot[0] * (1.0 - factor_gen), fdot[0] * (1.0 + factor_gen)),
        "phi0":    uniform_dist(phi0[0] * (1.0 - factor_gen), phi0[0] * (1.0 + factor_gen)),
        "cosinc":  uniform_dist(np.cos(inc[0]) * (1.0 - factor_gen), np.cos(inc[0]) * (1.0 + factor_gen)),
        "psi":     uniform_dist(psi[0] * (1.0 - factor_gen), psi[0] * (1.0 + factor_gen)),
        "lam":     uniform_dist(lam[0] * (1.0 - factor_gen), lam[0] * (1.0 + factor_gen)),
        "sinbeta": uniform_dist(np.sin(beta[0]) * (1.0 - factor_gen), np.sin(beta[0]) * (1.0 + factor_gen)),
    })}

    ndims = {"gb": len(sampled_basis)}

    periodic_container = PeriodicContainer({"gb": {"phi0": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}, key_order={"gb": sampled_basis})
    # Keep separate backend HDF files per domain so a wdm run and an fd run
    # don't clobber each other.
    _default_fp = f"test_new_2_gb_chunked_pe_{DOMAIN}.h5"
    fp = os.environ.get("MCMC_BACKEND_PATH", _default_fp)
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"gb": gen_dist["gb"].rvs(size=(ntemps, nwalkers, 1))})

    # Moves: 0.25 stretch + 0.75 NUTS (WDM only -- the FD path doesn't
    # have a JAX-autograd gradient wired in yet so it stays
    # stretch-only). NUTSMove takes the *untempered* likelihood
    # gradient (it applies the per-walker beta internally). ``scale``
    # gives a per-parameter natural step size so the diagonal mass
    # matrix matches the amp/f0/fdot/angle spread.
    if DOMAIN == "wdm":
        # Per-parameter natural step. Matches the in-band injection
        # scales; NUTSMove builds metric = diag(1 / scale**2).
        _nuts_scale = np.array([
            float(amp[0]) * 1e-3,    # amp
            float(f0[0]) * 1e-7,     # f0  (Hz)         -- ~1e-9 Hz step
            1.0e-15,                  # fdot (Hz/s)
            1.0e-2,                   # phi0
            1.0e-2,                   # cosinc
            1.0e-2,                   # psi
            1.0e-2,                   # lam
            1.0e-2,                   # sinbeta
        ], dtype=float)
        _nuts_step_size = float(os.environ.get("NUTS_STEP_SIZE", 0.1))
        _nuts_max_tree_depth = int(os.environ.get("NUTS_MAX_TREE_DEPTH", 5))
        _no_nuts = os.environ.get("NO_NUTS", "0") == "1"
        if _no_nuts:
            _moves = StretchMove(live_dangerously=True)
            print(f"[mcmc/wdm] moves: stretch only (NO_NUTS=1 -- skipping "
                  f"JAX-autograd NUTSMove)", flush=True)
        else:
            nuts_move = NUTSMove(
                grad_log_like_fn=grad_logl_vec_jax,
                ndim=len(sampled_basis),
                scale=_nuts_scale,
                step_size=_nuts_step_size,
                max_tree_depth=_nuts_max_tree_depth,
                adapt_step_size=False,
                live_dangerously=True,
            )
            _moves = [
                (StretchMove(live_dangerously=True), 0.25),
                (nuts_move,                          0.75),
            ]
            print(f"[mcmc/wdm] moves: stretch (0.25) + NUTS (0.75)  "
                  f"step_size={_nuts_step_size}  max_tree_depth={_nuts_max_tree_depth}",
                  flush=True)
    else:
        _moves = StretchMove(live_dangerously=True)
        print(f"[mcmc/fd] moves: stretch only (NUTS-FD gradient not wired yet)",
              flush=True)

    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        logl_vec,                                # vectorized C++ likelihood (WDM or FD)
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(
            transform_fn=tc,
        ),
        moves=_moves,
        branch_names=["gb"],
        periodic=periodic_container,
        backend=fp,
        vectorize=True,                          # batch all walkers per call
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

    if os.environ.get("DEBUG_BREAK", "0") == "1":
        breakpoint()
    # ---- Waveform / likelihood timing (matches the per-step batch shape
    #      Eryn uses internally: (ntemps * nwalkers, ndim) flattened).
    #      Set TIMER_REPEATS to control loop count; 0 disables.
    _timer_repeats = int(os.environ.get("TIMER_REPEATS", "10"))
    if _timer_repeats > 0:
        print(f"\n[timer] benchmarking ntemps*nwalkers={ntemps*nwalkers} batched "
              f"calls, {_timer_repeats} reps (warmup excluded)", flush=True)
        _bench_x = gen_dist["gb"].rvs(size=(ntemps * nwalkers,))           # (N, 8)
        # logl_vec
        _ = logl_vec(_bench_x, transform_fn=tc)          # warmup
        _t0 = time.perf_counter()
        for _ in range(_timer_repeats):
            _ll = logl_vec(_bench_x, transform_fn=tc)
        _t1 = time.perf_counter()
        _per_call_ll = (_t1 - _t0) / _timer_repeats
        print(f"[timer] logl_vec        : {_per_call_ll*1e3:8.2f} ms / call  "
              f"(N={_bench_x.shape[0]}, {_per_call_ll/_bench_x.shape[0]*1e6:6.1f} us / source)",
              flush=True)
        # grad_logl_vec_jax (WDM only, and only when NUTS is enabled --
        # NUTS is the only consumer; the autograd warmup OOMs at large
        # N=200 batches we use otherwise).
        if DOMAIN == "wdm" and not _no_nuts:
            _ = grad_logl_vec_jax(_bench_x)                                # JIT warmup
            _t0 = time.perf_counter()
            for _ in range(_timer_repeats):
                _g = grad_logl_vec_jax(_bench_x)
            _t1 = time.perf_counter()
            _per_call_grad = (_t1 - _t0) / _timer_repeats
            print(f"[timer] grad_logl (jax) : {_per_call_grad*1e3:8.2f} ms / call  "
                  f"(N={_bench_x.shape[0]}, {_per_call_grad/_bench_x.shape[0]*1e6:6.1f} us / source, "
                  f"{_per_call_grad/_per_call_ll:.1f}x logl)", flush=True)
        # Single-source slice (NUTS leapfrog step does N_walker forward
        # passes per tree depth; this is the kernel-launch-bound lower
        # limit for tiny batches).
        _bench_x1 = _bench_x[:1]
        _ = logl_vec(_bench_x1, transform_fn=tc)
        _t0 = time.perf_counter()
        for _ in range(_timer_repeats):
            _ll = logl_vec(_bench_x1, transform_fn=tc)
        _t1 = time.perf_counter()
        print(f"[timer] logl_vec  (N=1) : {(_t1-_t0)/_timer_repeats*1e3:8.2f} ms / call",
              flush=True)
        print(flush=True)

    nsteps = int(os.environ.get("NSTEPS", 2000))
    burn = int(os.environ.get("BURN", 0))
    thin_by = int(os.environ.get("THIN_BY", 5))
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=thin_by, progress=True)

    # ---- Posterior plot ----
    # Cold chain only (temperature index 0). Eryn stores chains as
    # (nsteps_kept, ntemps, nwalkers, nleaves_max, ndim).
    discard_frac = float(os.environ.get("DISCARD_FRAC", 0.25))
    n_stored = sampler.get_chain()["gb"].shape[0]
    discard = int(n_stored * discard_frac)
    chain = sampler.get_chain(discard=discard)["gb"]   # (ns, ntemps, nwalkers, nleaves, ndim)
    cold = chain[:, 0]                                  # (ns, nwalkers, nleaves, ndim)
    samples = cold.reshape(-1, cold.shape[-1])

    try:
        import corner
        fig = corner.corner(
            samples, labels=sampled_basis, truths=inj_params,
            show_titles=True, title_fmt=".4e", quantiles=[0.16, 0.5, 0.84],
        )
        fig.savefig(f"gb_chunked_posterior_{DOMAIN}.png", dpi=120)
        plt.close(fig)
        print(f"Saved posterior corner plot to gb_chunked_posterior_{DOMAIN}.png")
    except ImportError:
        nparams = samples.shape[-1]
        fig, axes = plt.subplots(nparams, nparams, figsize=(14, 14))
        for i in range(nparams):
            for j in range(nparams):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples[:, i], bins=40, color="steelblue", alpha=0.85)
                    ax.axvline(inj_params[i], color="red", lw=1)
                elif i > j:
                    ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.3, c="steelblue")
                    ax.axvline(inj_params[j], color="red", lw=0.8)
                    ax.axhline(inj_params[i], color="red", lw=0.8)
                else:
                    ax.axis("off")
                if i == nparams - 1:
                    ax.set_xlabel(sampled_basis[j])
                if j == 0 and i > 0:
                    ax.set_ylabel(sampled_basis[i])
        fig.suptitle("Posterior (chunked-het GB MCMC)", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"gb_chunked_posterior_{DOMAIN}.png", dpi=120)
        plt.close(fig)
        print(f"Saved posterior (matplotlib fallback) to gb_chunked_posterior_{DOMAIN}.png")
