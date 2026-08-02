"""Census: what fraction of sources pass the min_env router?

Monte-Carlo an isotropic prior population (cos iota uniform, sky isotropic,
psi uniform) through the installed spline-envelope build and measure min_env
(min over the year of per-channel complex-envelope amplitude / channel max --
validated in gb_sighet_envnull_demo.py to equal the kernel-facing |c0| row
modulus).  Also computes min_env EXACTLY for every VGB in the mojito VGB
catalogue (the eclipse-discovery-biased set).

Thresholds (many-ref test, 1yr stride24): v2 division needs min_env > ~0.1;
rung-i needs > ~0.05; < ~0.01 is hard-fail for both.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_minenv_census.py
Env: MC_NDRAWS (400), ENV_NT (12288), ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import sys
import time

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

EPS_OBL = math.radians(23.4392911)


def icrs_to_ecl(ra, dec):
    beta = math.asin(math.sin(dec) * math.cos(EPS_OBL)
                     - math.cos(dec) * math.sin(EPS_OBL) * math.sin(ra))
    lam = math.atan2(math.sin(ra) * math.cos(EPS_OBL)
                     + math.tan(dec) * math.sin(EPS_OBL),
                     math.cos(ra)) % (2.0 * math.pi)
    return lam, beta


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    ndraw = int(os.environ.get("MC_NDRAWS", "400"))

    # ---- scaffold (identical to gb_sighet_envnull_demo) --------------------
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 256, int(os.environ.get("ENV_NT", "12288"))
    t_start = int(0.5 * YRSID_SI / dt) * dt
    edge, tk = 40, 8
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(Nf, Nt, dt, t0=t_start, min_freq=1e-4,
                          max_freq=2e-2, min_time=edge * Nf * dt,
                          max_time=(Nt - edge) * Nf * dt,
                          force_backend=backend)
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start, Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits,
        tdi_config="2nd generation", force_backend=backend, d_d=0.0,
        tdi_type="XYZ", tukey_alpha=2.0 * tk / Nt)
    chunked.convert_to_ra_dec = False
    sighet = GBSignalHetComputations.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=0, nt_layer=64,
        m_active_half_width=2)
    g = sighet._g
    N = g["n_sparse_fd"]
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    nfa = ihi - ilo
    nta = np.zeros(Nt, bool)
    nta[wdm_set.active_slice_t] = True
    nta = int(nta.sum())
    invC = np.zeros((3, 3, nfa, nta))
    for c in range(3):
        invC[c, c] = 1.0
    holder = proto._FullGridWDMHolder(np.zeros((3, nfa, nta)), invC)
    zeros = np.zeros(1, dtype=np.int32)
    dummy = np.array([1e-22, 3e-3, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.3])
    sighet.setup_in_model(holder, dummy[None, :], zeros)
    gen = sighet._keep_alive["gb_gen"]

    def min_env_of(p9):
        s, _, _ = proto.slow_series(gen, p9, None, N, g)
        a = np.abs(s)
        return float((a / a.max(axis=-1, keepdims=True)).min())

    # ---- isotropic Monte Carlo --------------------------------------------
    rng = np.random.default_rng(11)
    cosi = rng.uniform(-1.0, 1.0, ndraw)
    f0s = 10.0 ** rng.uniform(math.log10(7e-4), math.log10(8e-3), ndraw)
    psis = rng.uniform(0.0, math.pi, ndraw)
    lams = rng.uniform(0.0, 2.0 * math.pi, ndraw)
    betas = np.arcsin(rng.uniform(-1.0, 1.0, ndraw))
    phi0s = rng.uniform(0.0, 2.0 * math.pi, ndraw)

    t0 = time.time()
    me = np.empty(ndraw)
    for i in range(ndraw):
        p9 = np.array([1e-22, f0s[i], 0.0, 0.0, phi0s[i],
                       math.acos(cosi[i]), psis[i], lams[i], betas[i]])
        me[i] = min_env_of(p9)
        if (i + 1) % 100 == 0:
            print(f"  [mc] {i+1}/{ndraw} ({time.time()-t0:.0f}s)")

    for thr, tag in [(0.1, "v2 route (<0.1)"), (0.05, "rung-i route (<0.05)"),
                     (0.02, "marginal (<0.02)"), (0.01, "hard (<0.01)")]:
        frac = float((me < thr).mean())
        print(f"[mc  ] {tag:22s}: {frac*100:5.1f}%  ({int((me<thr).sum())}/{ndraw})")
    # inclination-only gate misclassification
    lo_i = np.abs(cosi) < 0.1
    print(f"[mc  ] |cos i|<0.1 fraction: {lo_i.mean()*100:.1f}%; "
          f"fails (<0.1 min_env) at |cos i|>=0.1: "
          f"{float(((me < 0.1) & ~lo_i).mean())*100:.1f}% of population")

    # ---- every VGB in the catalogue ---------------------------------------
    import h5py
    cat = h5py.File(os.path.expanduser(
        "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/catalogues/"
        "vgb_cat_mojito_lite_processed.hdf5"), "r")["Binaries"]
    vf0 = np.asarray(cat["GW22FrequencySSBFrame"])
    vinc = np.asarray(cat["InclinationAngle"])
    vpsi = np.asarray(cat["PolarisationAngle"])
    vra = np.asarray(cat["RightAscension"])
    vdec = np.asarray(cat["Declination"])
    vamp = np.asarray(cat["Amplitude"])
    vphi = np.asarray(cat["TrueAnomaly"])
    vid = [x.decode() if isinstance(x, bytes) else str(x)
           for x in np.asarray(cat["ID"])]
    nv = len(vf0)
    vme = np.empty(nv)
    for i in range(nv):
        lam, beta = icrs_to_ecl(float(vra[i]), float(vdec[i]))
        p9 = np.array([vamp[i], vf0[i], 0.0, 0.0, vphi[i], vinc[i],
                       vpsi[i], lam, beta])
        vme[i] = min_env_of(p9)
    order = np.argsort(vme)
    print(f"[vgb ] {nv} catalogue VGBs; min_env<0.1: {(vme<0.1).sum()}, "
          f"<0.05: {(vme<0.05).sum()}, <0.01: {(vme<0.01).sum()}")
    for i in order:
        flag = ("HARD" if vme[i] < 0.01 else
                "v2+rung-i" if vme[i] < 0.05 else
                "v2-only" if vme[i] < 0.1 else "clean")
        print(f"    {vid[i]:12s} f0={vf0[i]*1e3:7.4f} mHz "
              f"|cos i|={abs(math.cos(vinc[i])):.3f} min_env={vme[i]:.4f} "
              f"[{flag}]")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    xs = np.sort(me)
    ax[0].semilogx(xs, np.arange(1, ndraw + 1) / ndraw, "C0-", lw=1.5)
    for thr, c, lb in [(0.1, "r", "v2 route"), (0.05, "m", "rung-i route"),
                       (0.01, "k", "hard")]:
        ax[0].axvline(thr, color=c, ls=":", lw=1)
        ax[0].text(thr, 1.02, f"{lb}\n{(me<thr).mean()*100:.1f}%",
                   color=c, fontsize=8, ha="center")
    ax[0].set_xlabel("min_env")
    ax[0].set_ylabel("CDF (isotropic population)")
    ax[0].set_title(f"min_env CDF, {ndraw} isotropic draws, 1 yr", fontsize=10)

    ax[1].semilogy(np.abs(cosi), me, "C0.", ms=4, alpha=0.6,
                   label="isotropic MC")
    ax[1].semilogy(np.abs(np.cos(vinc)), vme, "r*", ms=11, mec="k", mew=0.4,
                   label="catalogue VGBs")
    i8 = vid.index("ZTFJ0722")
    ax[1].annotate("ZTFJ0722 (band 8)",
                   (abs(math.cos(vinc[i8])), vme[i8]),
                   textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax[1].axhline(0.1, color="r", ls=":", lw=1)
    ax[1].axhline(0.05, color="m", ls=":", lw=1)
    ax[1].set_xlabel(r"$|\cos\iota|$")
    ax[1].set_ylabel("min_env")
    ax[1].set_title("min_env vs inclination: iota drives it, sky spreads it",
                    fontsize=10)
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fp = os.path.join(out_dir, "minenv_census.png")
    fig.savefig(fp, dpi=140)
    print(f"[out ] {fp}")
    np.savez(os.path.join(out_dir, "minenv_census.npz"),
             me=me, cosi=cosi, f0s=f0s, psis=psis, lams=lams, betas=betas,
             vgb_me=vme, vgb_ids=np.array(vid), vgb_inc=vinc, vgb_f0=vf0)


if __name__ == "__main__":
    main()
