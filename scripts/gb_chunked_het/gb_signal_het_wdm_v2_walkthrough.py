#!/usr/bin/env python
"""Explainer / walk-through for the v2 polyphase + de-rotated signal-het pipeline.

FOUR rows, one per log-likelihood TARGET (logL ~ 0, -5, -50, -500): the candidate
f0 is perturbed (binary-searched) so the DENSE lisatools logL lands on each
target. Each row:

  col 0:  candidate dense real-WDM template on the active m-band (m_floor +/-
          m_half). Row 0 (Df0=0) is the best-fit template.
  col 1:  v2 sparse complex-WDM Re(c1_sparse[m, n_sparse]) -- the polyphase output
          on the sparse-n grid (what the bin-fold consumes).
  col 2:  heterodyne ratio r(n) = c1_sparse / c0_sparse at the carrier layer:
          Re(r), Im(r), de-rotated r_demod. r ~flat at the peak, an increasingly
          wound phase ramp as the candidate walks off.
  col 3:  numerics -- the three logL's side by side: base lisatools (dense WDM),
          chunked-het get_ll, and signal-het REAL-projection get_ll
          (project_real=1); their differences vs base, mm5/mm2, params.

DATA / injection source:
  * default (synthetic): a fixed GB injection, ESAOrbits, full band -- data == ref.
  * GB_USE_MOJITO=1: the cached mojito GB data + the GB_RANK-th high-f catalogue
    source as the heterodyne reference, L1Orbits, narrow band (data != ref).

Run::  python gb_signal_het_wdm_v2_walkthrough.py            # synthetic
       GB_USE_MOJITO=1 GB_RANK=0 python gb_signal_het_wdm_v2_walkthrough.py
Env:   NT_LAYER (64), M_HALF (2), GB_NT, GB_RANK, OUT_PNG.
"""
from __future__ import annotations
import os, sys, gc, threading, time, resource
os.environ.setdefault("OMP_NUM_THREADS", "8")
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gbgpu  # noqa: F401  (registers the gbgpu backends)
from lisatools.detector import ESAOrbits, L1Orbits
from lisatools.domains import TDSignal
from lisatools.utils.constants import YRSID_SI
from lisatools.globalfit.preprocessing import find_file
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from gbgpu.gbcomps import GBWDMComputations

from gb_signal_het_wdm_v2_mm_sweep import build_v2_real_template, compute_mm
from gbsignalhetcomputations import GBSignalHetComputations, recommended_edge_cut

MEM_CAP_GB = float(os.environ.get("GB_MEM_CAP_GB", "10.0")); _IS_MAC = sys.platform == "darwin"


def _rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6


def _watchdog():
    while True:
        if _rss_gb() > MEM_CAP_GB:
            os._exit(42)
        time.sleep(0.3)

TARGETS = [0.0, -5.0, -50.0, -500.0]
REF = 97729089.327664
MOJ_PATH = "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
MOJ_CACHE = "/tmp/gb_mojito_data_365d.npz"


# ---------------------------------------------------------------------------
# Per-panel draw helpers. Each takes (ax, rd, sh) where ``rd`` is the per-row
# computed data and ``sh`` the shared constants, so the SAME code draws both the
# combined 4x4 grid and the individual standalone panels (PANEL_DIR export, used
# by the LaTeX explainer). Column order: 0 dense WDM, 1 sparse polyphase,
# 2 heterodyne ratio r(n), 3 numerics text.
# ---------------------------------------------------------------------------


def _draw_dense(ax, rd, sh):
    band = rd["c1_dense"][sh["CHAN"], sh["m_local"], :]
    vmax = float(np.abs(band).max()) or 1.0
    im = ax.imshow(band, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[0, sh["Nt_active"], sh["m_floor"] - sh["M_HALF"] - 0.5,
                           sh["m_floor"] + sh["M_HALF"] + 0.5])
    ax.set_xlabel("n (active time pixel)"); ax.set_ylabel("global m")
    ax.set_title(f"target logL={rd['tgt']:+.0f}: candidate dense real-WDM (chan 0)\n"
                 f"Df0={rd['df0']:.4f}*df  m_floor={sh['m_floor']}", fontsize=10)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _draw_sparse(ax, rd, sh):
    sr = rd["c1_sparse"][sh["CHAN"]].real; smax = float(np.abs(sr).max()) or 1.0
    im = ax.imshow(sr, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-smax, vmax=smax,
                   extent=[sh["n_sparse"][0], sh["n_sparse"][-1],
                           sh["m_active"][0] - 0.5, sh["m_active"][-1] + 0.5])
    ax.set_xlabel(f"n (sparse, stride={sh['stride']})"); ax.set_ylabel("global m (active)")
    ax.set_title(f"v2 polyphase: Re(c1_sparse)  shape={sr.shape}  Nt_layer={sh['Nt_layer']}", fontsize=10)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _draw_ratio(ax, rd, sh):
    n_sparse = sh["n_sparse"]
    ax.plot(n_sparse, rd["r"].real, "C0o-", label="Re(r)", ms=3, lw=0.8)
    ax.plot(n_sparse, rd["r"].imag, "C1o-", label="Im(r)", ms=3, lw=0.8)
    ax.plot(n_sparse, rd["r_demod"].real, "C2.--", label="Re(r_demod)", ms=2, lw=0.6, alpha=0.7)
    ax.plot(n_sparse, rd["r_demod"].imag, "C3.--", label="Im(r_demod)", ms=2, lw=0.6, alpha=0.7)
    ax.set_xlabel("n_sparse"); ax.set_ylabel("r = c1/c0")
    ax.set_title(f"heterodyne ratio at carrier m={sh['m_floor']}\nDf0={rd['Df0']*1e9:.3f} nHz", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)


def _draw_text(ax, rd, sh):
    ax.axis("off")
    text = (f"TARGET logL = {rd['tgt']:+.0f}    [{sh['src']}]\n\n"
            f"Df0 = {rd['df0']:.5f} * layer_df = {rd['Df0']*1e9:.4f} nHz\n\n"
            f"f0_ref = {sh['p_inj'][1]*1e3:.5f} mHz\n"
            f"snr    = {sh['snr']:.1f}   <d|d> = {sh['d_d']:.3e}\n"
            f"Nt={sh['Nt']} Nt_layer={sh['Nt_layer']} N_sparse_t={sh['N_sparse_t']}\n"
            f"m_half={sh['M_HALF']} (active layers={2*sh['M_HALF']+1})\n\n"
            f"  logL    (base lisatools) = {rd['ll_d']:>+13.5e}\n"
            f"  logL    (chunked het)    = {rd['ll_ch']:>+13.5e}\n"
            f"  logL    (signal het)     = {rd['ll_c']:>+13.5e}\n\n"
            f"  chunk  - base = {rd['ll_ch']-rd['ll_d']:>+11.3e}\n"
            f"  sighet - base = {rd['ll_c']-rd['ll_d']:>+11.3e}\n\n"
            f"  sighet <d|h> = {rd['d_h_c']:>+13.5e}\n"
            f"  sighet <h|h> = {rd['h_h_c']:>+13.5e}\n\n"
            f"  mm5 = {rd['mm5']:+.3e}\n  mm2 = {rd['mm2']:+.3e}\n")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=9)


_PANELS = [(_draw_dense, "dense", (6.4, 5.0)), (_draw_sparse, "sparse", (6.4, 5.0)),
           (_draw_ratio, "ratio", (6.4, 4.7)), (_draw_text, "numerics", (5.2, 5.0))]


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    backend = "cpu"; dt = 10.0; Nf = 1460
    USE_MOJITO = int(os.environ.get("GB_USE_MOJITO", "0"))
    Nt = int(os.environ.get("GB_NT", "512" if USE_MOJITO else "2560"))
    Nobs = Nf * Nt; Tobs = Nobs * dt
    Nt_layer = int(os.environ.get("NT_LAYER", "64")); M_HALF = int(os.environ.get("M_HALF", "2"))
    layer_df = 1.0 / (2.0 * Nf * dt); TUK = 0.05
    EC = recommended_edge_cut(Nt, TUK, "chunked")   # MATCHED edge-cut (real projection)

    # ---- mode-specific: orbits, t_ref/t0, band, injection params, data --------
    if USE_MOJITO:
        z = np.load(MOJ_CACHE)
        data_td = np.asarray(z["data_td"])[:3, :Nobs]; t0 = float(z["data_t0"]); t_ref = REF
        rank = int(os.environ.get("GB_RANK", "0"))
        A0p, f0, fdot, phi0, inc, psi, ra, dec = np.asarray(z["top_params"])[rank]
        p_inj = np.array([A0p, f0, fdot, 0.0, phi0, inc, psi, ra, dec])
        m_floor = int(f0 / layer_df); BAND = 15
        lo_f = (m_floor - BAND) * layer_df; hi_f = (m_floor + BAND + 1) * layer_df
        orbits = L1Orbits(find_file(os.path.join(MOJ_PATH, "data", "GB", "L1"), "GB", 0),
                          force_backend=backend, frame="icrs")
        pad = 1e5; lo = max(t0 - pad, float(orbits.sc_t0)); hi = min(t0 + Tobs + pad, float(orbits._sc_t_base[-1]))
        lt = np.asarray(orbits.ltt_t); mk = (lt >= lo) & (lt <= hi)
        orbits.ltt = np.asarray(orbits.ltt)[mk].copy(); orbits.ltt_t = lt[mk].copy()
        orbits.ltt_t0 = float(orbits.ltt_t[0]); del lt; gc.collect()
        orbits.configure(linear_interp_setup=True)
        src = f"mojito rank {rank}"
    else:
        orbits = ESAOrbits(force_backend=backend)
        p_inj = np.array([1e-21, 14.22e-3, 1e-16, 0.0, 1.4, np.pi / 3.0, 0.7, 2.1, 0.5])
        t0 = t_ref = int(0.5 * YRSID_SI / dt) * dt
        m_floor = int(p_inj[1] / layer_df); BAND = 15
        lo_f = (m_floor - BAND) * layer_df; hi_f = (m_floor + BAND + 1) * layer_df
        data_td = None
        src = "synthetic"

    tdi_config = "2nd generation"
    t_arr = np.arange(Nobs) * dt + t0

    # ---- synthetic injection data: one temporary generator, then discard ------
    if not USE_MOJITO:
        _cfg = TDIConfig(tdi_config, force_backend=backend)
        _gen = GBTDIonTheFly(np.linspace(t_arr[0], t_arr[-1], 16384), Tobs, t_ref, 1.0 / dt, 1,
                             tdi_config=_cfg, orbits=orbits, tdi_chan="XYZ", force_backend=backend)
        _sp = _gen(*[np.array([x]) for x in p_inj], convert_to_ra_dec=False, return_spline=True)
        data_td = np.asarray(_sp.eval_tdi(t_arr))[0]
        del _gen, _sp, _cfg; gc.collect()

    # ---- installed signal-het frontend: builds analysis/d_d/sparse_gen/c0/bin- -
    #      fold internally; get_ll(params) returns logL directly. Single source. -
    # SIGHET_REF_IMPL toggles the reference c0 build: "chunked" (default, the
    # chunked-het/polyphase WDM method, active band at full Nt; == dense to ~1e-14 --
    # test_sighet_ref_from_chunked.py) or "dense" (full-TD TDSignal.transform).
    sighet = GBSignalHetComputations(
        data_td, p_inj, Nf=Nf, Nt=Nt, dt=dt, t0=t0, t_ref=t_ref,
        orbits=orbits, tdi_config=tdi_config, min_freq=lo_f, max_freq=hi_f,
        edge_cut=EC, m_active_half_width=M_HALF, nt_layer=Nt_layer,
        tukey_alpha=TUK, force_backend=backend,
        reference_impl=os.environ.get("SIGHET_REF_IMPL", "chunked"))

    ka = sighet._keep_alive
    real_td_cb = ka["real_td_cb"]; window = ka["window"]; td_set = ka["td_set"]
    wsr = ka["wdm_set_real"]; wsc = ka["wdm_set_complex"]; sparse_gen = ka["sparse_gen"]
    analysis = sighet.analysis; d_d = sighet.d_d
    n_sparse = sighet.n_sparse_local; stride = sighet._g["stride"]
    ind_min_t = sighet._g["ind_min_t"]; ind_min_f = sighet._g["ind_min_f"]
    Nt_active = sighet._g["Nt_active"]; Nf_active = sighet._g["Nf_active"]
    c0_sparse = sighet.c0_sparse_all[0]
    snr = float(analysis.snr())

    # WDM data array (real) + full complex reference (for plots + chunked + mm) --
    data_wdm = np.asarray(TDSignal(data_td, settings=td_set).transform(wsr, window=window).arr)
    wdm_ref_complex = np.asarray(
        TDSignal(real_td_cb(p_inj), settings=td_set).transform(wsc, window=window).arr).copy()

    # ---- chunked-het get_ll (same real-WDM data/edge/band as the dense ref) ----
    invC_r = np.asarray(analysis.sens_mat.invC); invC_r = np.where(np.isfinite(invC_r), invC_r, 0.0)

    class _Holder:
        def __init__(s, data, iC):
            s.linear_data_arr = [np.ascontiguousarray(data).ravel()]
            s.linear_psd_arr = [np.ascontiguousarray(iC).ravel()]

        def __len__(s):
            return 1

    holder = _Holder(data_wdm, invC_r)
    # d_d baked in so get_ll_wdm returns the full logL (no manual -0.5*d_d here).
    comp = GBWDMComputations(wsr, t_ref=t_ref, Nt_sub=256, n_pad=256 // 8, N_sparse=256,
                             N_cp_sig=0, N_cp_orbit=0, orbits=orbits, tdi_config=tdi_config,
                             force_backend=backend, d_d=d_d, tdi_type="XYZ")

    def chunk_ll(p):
        return float(np.asarray(comp.get_ll_wdm(p.reshape(1, 9), holder, convert_to_ra_dec=False,
                                                use_layer_groups=True, group_band_layers=5,
                                                margin_layers=0))[0])

    def dense_logL(p):
        tpl = TDSignal(real_td_cb(p), settings=td_set).transform(wsr, window=window)
        return float(np.real(analysis.template_likelihood(tpl)))

    def find_df0(tgt):
        if tgt == 0.0: return 0.0
        lo_b, hi_b = 0.0, 0.5
        for _ in range(22):
            mid = 0.5 * (lo_b + hi_b)
            ll = dense_logL(p_inj + np.array([0, mid * layer_df, 0, 0, 0, 0, 0, 0, 0]))
            if ll < tgt: hi_b = mid
            else: lo_b = mid
        return 0.5 * (lo_b + hi_b)

    m_active = np.arange(m_floor - M_HALF, m_floor + M_HALF + 1); m_local = m_active - ind_min_f
    CHAN = 0; carrier_row = M_HALF
    print(f"[{src}] f0={p_inj[1]*1e3:.4f}mHz m_floor={m_floor} snr={snr:.1f} d_d={d_d:.3e} "
          f"Nt={Nt} EC={EC} band=[{lo_f*1e3:.3f},{hi_f*1e3:.3f}]mHz", flush=True)

    sh = dict(CHAN=CHAN, m_local=m_local, m_active=m_active, Nt_active=Nt_active,
              m_floor=m_floor, M_HALF=M_HALF, n_sparse=n_sparse, stride=stride,
              Nt_layer=Nt_layer, carrier_row=carrier_row, ind_min_t=ind_min_t, Nf=Nf,
              dt=dt, p_inj=p_inj, src=src, snr=snr, d_d=d_d, Nt=Nt,
              N_sparse_t=sparse_gen.N_sparse_t)

    # Optional per-panel export: PANEL_DIR=<dir> saves each subplot standalone
    # (panel_r{row}_c{col}_{name}.png) for individual use in the LaTeX explainer.
    panel_dir = os.environ.get("PANEL_DIR")
    panel_tag = os.environ.get("PANEL_TAG", f"rank{os.environ.get('GB_RANK', '0')}"
                               if USE_MOJITO else "synth")
    if panel_dir:
        os.makedirs(panel_dir, exist_ok=True)

    fig, axs = plt.subplots(len(TARGETS), 4, figsize=(22, 4.6 * len(TARGETS)))
    for row, tgt in enumerate(TARGETS):
        df0 = find_df0(tgt); p = p_inj.copy(); p[1] += df0 * layer_df
        td_c = real_td_cb(p); fd_rfft = np.fft.rfft(td_c * window, axis=-1)
        c1_sparse, _ = sparse_gen.sparse_from_rfft(fd_rfft, p)
        tpl_real = build_v2_real_template(sparse_gen, fd_rfft, p, p_inj, Nf_active,
                                          Nt_active, wdm_ref_complex, c0_sparse, dt, Nf, ind_min_t)
        c1_dense = np.asarray(TDSignal(td_c, settings=td_set).transform(wsr, window=window).arr)
        ll_d = dense_logL(p); ll_ch = chunk_ll(p)
        ll_c = float(np.asarray(sighet.get_ll(p))[0])
        d_h_c = float(sighet.last_d_h[0]); h_h_c = float(sighet.last_h_h[0])
        mm5 = compute_mm(data_wdm, tpl_real, wsr, float(p[1]), "mm5", backend)
        mm2 = compute_mm(data_wdm, tpl_real, wsr, float(p[1]), "mm2", backend)
        print(f"[row {row}] target={tgt:+.0f} Df0/df={df0:.4f}  dense={ll_d:+.4e} "
              f"chunk={ll_ch:+.4e} sighet={ll_c:+.4e}  "
              f"(ch-d)={ll_ch-ll_d:+.2e} (sh-d)={ll_c-ll_d:+.2e}  mm5={mm5:.2e}", flush=True)

        # heterodyne ratio at the carrier layer (+ de-rotated)
        c0c = c0_sparse[CHAN, m_local[carrier_row]]; c1c = c1_sparse[CHAN, carrier_row]
        mag = np.abs(c0c); mask = mag > 1e-12 * mag.max()
        r = np.where(mask, c1c / np.where(mask, c0c, 1.0), 0.0 + 0.0j)
        Df0 = p[1] - p_inj[1]; t_n = (ind_min_t + n_sparse) * Nf * dt
        r_demod = r * np.exp(-1j * (2.0 * np.pi * Df0 * t_n))

        rd = dict(tgt=tgt, df0=df0, p=p, Df0=Df0, c1_dense=c1_dense, c1_sparse=c1_sparse,
                  r=r, r_demod=r_demod, ll_d=ll_d, ll_ch=ll_ch, ll_c=ll_c,
                  d_h_c=d_h_c, h_h_c=h_h_c, mm5=mm5, mm2=mm2)

        for col, (drawfn, name, fsz) in enumerate(_PANELS):
            drawfn(axs[row, col], rd, sh)            # combined grid
            if panel_dir:                            # standalone panel
                pf, pa = plt.subplots(figsize=fsz)
                drawfn(pa, rd, sh)
                pf.tight_layout()
                pf.savefig(os.path.join(panel_dir, f"panel_{panel_tag}_r{row}_c{col}_{name}.png"),
                           dpi=130, bbox_inches="tight")
                plt.close(pf)

    fig.suptitle(f"Signal-het (v2 polyphase + de-rotation) explainer -- REAL-WDM get_ll  [{src}]\n"
                 f"candidate f0 walked to logL = {TARGETS}  (Nt_layer={Nt_layer}, m_half={M_HALF})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.environ.get("OUT_PNG", "sighet_explainer.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    if panel_dir:
        print(f"[panels] wrote {len(TARGETS)*len(_PANELS)} panels to {panel_dir} (tag={panel_tag})", flush=True)
    print(f"\n[plot] {out}\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
