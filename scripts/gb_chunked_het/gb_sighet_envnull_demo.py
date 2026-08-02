"""Demonstration: complex-WDM coefficients themselves null at envelope zeros.

Question under test (2026-08-01): "Im(WDM) should just be a 90-degree-rotated
copy of Re(WDM) (hence the factor of 2 in the complex inner product) -- so why
does edge-on inclination still break the ratio r = c1/c0?"

Answer this script makes visible: the quadrature (Re/Im) structure of the
complex WDM representation removes CARRIER zeros (cos passing through zero
every half cycle), not ENVELOPE zeros.  Because Im *is* the rotated copy of
Re, both quadratures are proportional to the same slowly-varying channel
amplitude -- when that amplitude crosses zero (edge-on source => linear
polarization => real envelope A+ F+_c(t) that changes sign as the antenna
pattern sweeps), Re and Im of the complex WDM coefficient vanish TOGETHER.

Source: the band-8 offender from the vgb accuracy test = ZTFJ0722 (eclipsing
DWD, iota = 89.66 deg) straight from the mojito VGB catalogue.

Panels (all from IMPLEMENTED machinery -- make_reference stash + the gated
prototype mirrors; nothing hand-rolled):
  P1  per-channel complex-envelope amplitude |P_c(t)|/max over the year
      (installed spline decomposition), edge-on vs the same source at
      iota = 0.3 -- min_env router metric annotated
  P2  |c0| on the carrier WDM layer from the EXACT make_reference stash
      (the array the v2 kernel divides by), per channel, log scale
  P3  zoom at the deepest null: Re(c0), Im(c0), |c0| -- both quadratures
      null together; the rotation cannot supply amplitude the channel
      does not carry
  P4  |r| = |c1/c0| for a small candidate displacement.  NOTE: a pure
      amplitude scaling is ratio-EXACT by linearity (c1 = (1+dlnA) c0
      identically), so the demo displaces psi -- for a linearly polarized
      source a polarization rotation SHIFTS the pattern-null times, and
      near c0's null c1 is no longer proportional => division blow-up at
      exactly the null pixels.  This is the production case: MCMC devs
      move all params, not amp alone.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_envnull_demo.py
Env: ENV_NT (default 12288 = 1 yr), ENV_NSFD (512), ENV_DLNA (2.3e-4),
     ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

# --- band-8 = ZTFJ0722, mojito vgb_cat_mojito_lite_processed.hdf5 ------------
AMP = 7.176054456355185e-23
F0 = 0.0014059273072807
FDOT = 2.841354105434733e-18
IOTA = 1.5648622073381158        # 89.66 deg -- eclipsing DWD
PSI = 1.2545873742863833
RA = 1.9301517831765533
DEC = -0.32565293401402245
PHI0 = 3.9162976324754704        # phi0 = +TrueAnomaly injection convention

# equatorial -> ecliptic (prototype pipeline runs convert_to_ra_dec=False)
EPS = math.radians(23.4392911)
BETA = math.asin(math.sin(DEC) * math.cos(EPS)
                 - math.cos(DEC) * math.sin(EPS) * math.sin(RA))
LAM = math.atan2(math.sin(RA) * math.cos(EPS)
                 + math.tan(DEC) * math.sin(EPS),
                 math.cos(RA)) % (2.0 * math.pi)

P_EDGE = np.array([AMP, F0, FDOT, 0.0, PHI0, IOTA, PSI, LAM, BETA])
P_FACE = P_EDGE.copy()
P_FACE[5] = 0.3
DLNA = float(os.environ.get("ENV_DLNA", "2.3e-4"))
DPSI = float(os.environ.get("ENV_DPSI", "1e-3"))


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)

    # ---- scaffold: identical to the ratio prototype (window policy) --------
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 256, int(os.environ.get("ENV_NT", "12288"))
    t_start = int(0.5 * YRSID_SI / dt) * dt
    edge = 40
    tk = 8
    alpha = 2.0 * tk / Nt
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=2e-2,
        min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
        force_backend=backend,
    )
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start,
        Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
        tukey_alpha=alpha,
    )
    chunked.convert_to_ra_dec = False
    sighet = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=int(os.environ.get("ENV_NSFD", "512")),
        n_cp_build=0, nt_layer=int(os.environ.get("ENV_NTL", "512")),
        m_active_half_width=2)
    g = sighet._g
    N = g["n_sparse_fd"]
    print(f"[grid] Nf={Nf} Nt={Nt} Tobs={g['Tobs']/86400:.1f} d "
          f"N={N} stride={g['stride']} N_sparse_t={g['N_sparse_t']}")
    print(f"[src ] ZTFJ0722 f0={F0*1e3:.6f} mHz iota={IOTA:.4f} rad "
          f"({math.degrees(IOTA):.2f} deg, |cos i|={abs(math.cos(IOTA)):.4f}) "
          f"lam={LAM:.4f} beta={BETA:.4f}")

    # zeros data holder -- setup_in_model only needs the shapes; c0 depends
    # on the reference params alone
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    nfa = ihi - ilo
    nta = np.zeros(Nt, bool)
    nta[wdm_set.active_slice_t] = True
    nta = int(nta.sum())
    h_act = np.zeros((3, nfa, nta))
    invC = np.zeros((3, 3, nfa, nta))
    for c in range(3):
        invC[c, c] = 1.0
    holder = proto._FullGridWDMHolder(h_act, invC)
    zeros = np.zeros(1, dtype=np.int32)
    sighet.setup_in_model(holder, P_EDGE[None, :], zeros)
    gen = sighet._keep_alive["gb_gen"]

    # ---- P1: complex-envelope amplitude (installed spline decomposition) ---
    s_edge, kf0, _ = proto.slow_series(gen, P_EDGE, None, N, g)
    s_face, _, _ = proto.slow_series(gen, P_FACE, None, N, g)
    tau_d = (np.arange(N) * g["Tobs"] / N) / 86400.0
    env_e = np.abs(s_edge)
    env_e /= env_e.max(axis=-1, keepdims=True)
    env_f = np.abs(s_face)
    env_f /= env_f.max(axis=-1, keepdims=True)
    min_env_edge = float(env_e.min(axis=-1).min())
    min_env_face = float(env_f.min(axis=-1).min())
    print(f"[env ] min_env edge-on: {env_e.min(axis=-1)} -> {min_env_edge:.2e}")
    print(f"[env ] min_env iota=0.3: {env_f.min(axis=-1)} -> {min_env_face:.2e}")

    # ---- P2/P3: the EXACT c0 stash the kernel divides by -------------------
    c0_all = np.asarray(sighet.c0_sparse_all)[0]        # (3, Nf_active, Nsp)
    m_act = proto.m_active_for(F0, g)
    ml = np.asarray(m_act) - g["ind_min_f"]
    rows0 = c0_all[:, ml, :]                            # (3, M, Nsp)
    ic = g["m_half"]                                    # carrier layer index
    row_c = rows0[:, ic, :]                             # (3, Nsp)
    n_sl = np.asarray(sighet.n_sparse_local)
    n_global = g["ind_min_t"] + int(n_sl[0]) + np.arange(g["N_sparse_t"]) * g["stride"]
    t_pix_d = n_global * Nf * dt / 86400.0
    row_n = np.abs(row_c) / np.abs(row_c).max(axis=-1, keepdims=True)
    cstar = int(np.unravel_index(np.argmin(row_n), row_n.shape)[0])
    i0 = int(np.argmin(row_n[cstar]))
    print(f"[c0  ] deepest carrier-row null: channel {'XYZ'[cstar]} "
          f"pixel {i0} (t={t_pix_d[i0]:.1f} d) |c0|/max={row_n[cstar, i0]:.2e}")
    print(f"[c0  ] per-channel row minima: {row_n.min(axis=-1)}")
    # envelope null nearest the c0 null (alignment check)
    j0 = int(np.argmin(np.abs(tau_d - t_pix_d[i0])))
    jn = int(np.argmin(env_e[cstar, max(0, j0 - 20):j0 + 20])) + max(0, j0 - 20)
    print(f"[algn] nearest envelope minimum: t={tau_d[jn]:.1f} d "
          f"(c0 null at {t_pix_d[i0]:.1f} d) env={env_e[cstar, jn]:.2e}")

    # ---- P4: division blow-up for a tiny candidate displacement ------------
    # pure-amp scaling is ratio-exact by linearity; displace psi so the
    # pattern-null times SHIFT (the production MCMC case)
    p_cand = P_EDGE.copy()
    p_cand[0] *= 1.0 + DLNA
    p_cand[6] += DPSI
    c1_sp, _ = proto.kernel_c1_full(sighet, p_cand)
    rows1 = c1_sp[:, ml, :]
    r, dr, mask = proto.ratio_dr(rows1, rows0, g["stride"])
    r_c = np.where(mask[:, ic, :], np.abs(r[:, ic, :]), np.nan)
    print(f"[r   ] dlnA={DLNA:.1e} dpsi={DPSI:.1e} candidate: "
          f"max|r|={np.nanmax(r_c):.3e} median|r|={np.nanmedian(r_c):.6f} "
          "on carrier row (~1 everywhere if no nulls)")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5))
    colors = dict(zip("XYZ", ("C0", "C1", "C2")))
    for c, ch in enumerate("XYZ"):
        ax[0, 0].semilogy(tau_d, env_e[c], color=colors[ch], lw=0.8, label=ch)
    ax[0, 0].semilogy(tau_d, env_f.min(axis=0), color="0.55", lw=1.2, ls="--",
                      label=r"same src, $\iota=0.3$ (min ch)")
    ax[0, 0].axhline(0.1, color="r", ls=":", lw=1)
    ax[0, 0].axhline(0.05, color="r", ls="-.", lw=1)
    ax[0, 0].text(0.01, 0.11, "v2 route threshold", color="r", fontsize=8,
                  transform=ax[0, 0].get_yaxis_transform())
    ax[0, 0].set_title(
        f"P1  envelope $|P_c(t)|$/max  (min_env {min_env_edge:.1e} "
        f"vs {min_env_face:.2f} at $\\iota$=0.3)", fontsize=10)
    ax[0, 0].set_xlabel("time [d]")
    ax[0, 0].legend(fontsize=8, ncol=4)

    for c, ch in enumerate("XYZ"):
        ax[0, 1].semilogy(t_pix_d, row_n[c], color=colors[ch], lw=0.8, label=ch)
    ax[0, 1].axhline(1e-12, color="k", ls=":", lw=1, label="kernel row floor")
    ax[0, 1].set_title("P2  $|c_0|$ carrier layer (make_reference stash)",
                       fontsize=10)
    ax[0, 1].set_xlabel("time [d]")
    ax[0, 1].legend(fontsize=8, ncol=4)

    w = 30
    sl = slice(max(0, i0 - w), min(len(t_pix_d), i0 + w))
    scale = np.abs(row_c[cstar]).max()
    ax[1, 0].plot(t_pix_d[sl], row_c[cstar, sl].real / scale, "C0.-", lw=0.8,
                  ms=3, label="Re$(c_0)$")
    ax[1, 0].plot(t_pix_d[sl], row_c[cstar, sl].imag / scale, "C3.-", lw=0.8,
                  ms=3, label="Im$(c_0)$")
    ax[1, 0].plot(t_pix_d[sl], np.abs(row_c[cstar, sl]) / scale, "k-", lw=1.5,
                  label="$|c_0|$")
    ax[1, 0].axhline(0, color="0.7", lw=0.5)
    ax[1, 0].set_title(f"P3  {'XYZ'[cstar]}-channel zoom: Re/Im quadratures "
                       "vanish together", fontsize=10)
    ax[1, 0].set_xlabel("time [d]")
    ax[1, 0].legend(fontsize=8)

    for c, ch in enumerate("XYZ"):
        ax[1, 1].semilogy(t_pix_d, r_c[c], color=colors[ch], lw=0.8, label=ch)
    ax[1, 1].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[1, 1].set_title(
        f"P4  $|r|=|c_1/c_0|$, dlnA={DLNA:.0e} d$\\psi$={DPSI:.0e} "
        f"(max {np.nanmax(r_c):.2f})", fontsize=10)
    ax[1, 1].set_xlabel("time [d]")
    ax[1, 1].legend(fontsize=8, ncol=4)

    fig.suptitle(
        "Band-8 VGB (ZTFJ0722, $\\iota$=89.66$^\\circ$, eclipsing): complex-WDM "
        "quadrature structure removes carrier zeros, not envelope zeros",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fp = os.path.join(out_dir, "envnull_band8.png")
    fig.savefig(fp, dpi=140)
    print(f"[out ] {fp}")
    np.savez(os.path.join(out_dir, "envnull_band8.npz"),
             tau_d=tau_d, env_edge=env_e, env_face=env_f,
             t_pix_d=t_pix_d, row_c=row_c, r_abs=r_c,
             cstar=cstar, i0=i0, params_edge=P_EDGE)


if __name__ == "__main__":
    main()
