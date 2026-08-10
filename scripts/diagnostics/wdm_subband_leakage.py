#!/usr/bin/env python
"""WDM sub-band leakage & residual-bookkeeping investigation.

Question (verbatim framing): two GB sources that do NOT overlap in frequency
DO overlap in WDM (their time-frequency supports intersect). Two claims must
not be conflated:

  (a) The inner product is basis-independent, so ``<hA|hB> ~ 0`` must hold in
      WDM too. If it fails numerically, the WDM inner product is wrong.
  (b) The residual bookkeeping is NOT basis-independent. If band A subtracts
      its template from pixels band B is concurrently reading, B's ``dll`` is
      computed against a residual that moved underneath it.

The algebra that links them (derived, then verified numerically in section 3):
with ``ll = -1/2 <r|r>``, band B proposing ``r -> r - dh_B`` gets

    dll_B(r) = <r|dh_B> - 1/2 <dh_B|dh_B>

so if band A concurrently changed the residual by ``dh_A`` (``r_fresh =
r_stale - dh_A``) then

    dll_B(r_fresh) - dll_B(r_stale) = -<dh_A | dh_B>          (*)

**The bookkeeping error is exactly the cross inner product of the two
changes.** It is therefore governed by the same suppression as (a), and the
go/no-go question reduces to measuring ``|<dh_A|dh_B>|`` against the
accept/reject noise floor (a dll of order 1).

Sections
--------
1. OVERLAP    normalized ``|<hA|hB>|`` in FD and WDM vs frequency separation,
              swept from sub-bin out past one WDM layer. Confirms (a) and
              exposes the suppression law.
2. ENERGY     fractional WDM energy outside +-k layers of a source's own
              layer, vs Tukey alpha. Quantifies the pixel overlap that (b)
              worries about.
3. BOOKKEEP   the bookkeeping error itself: ``|<dh_A|dh_B>|`` for an in-model
              step and for the RJ worst case (A added/removed whole), vs
              separation. Verified against (*) by an independent direct
              ``dll`` difference. THIS is the go/no-go number.
4. TOBS       section 3 vs Tobs (3 mo / 1 yr / 2 yr).

Run (CPU, single process; pin the thread pools per the laptop budget):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python wdm_subband_leakage.py --sections 1,2,3,4

Env / flags
-----------
    --sections     comma list of section numbers to run (default 1,2,3)
    --out          output directory (default ./wdm_subband_leakage_out)
    N_SPARSE       sparse control points for GBTDIonTheFly (default 8192)
    MID_FREQ_MHZ   carrier of source A in mHz (default 4.0)
    N_DF           separation samples in the sweeps (default 40)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# GBTDIonTheFly declares _BACKEND_PREFIX = "gbgpu"; importing gbgpu is what
# registers that namespace with the backend manager (without it the
# constructor raises "'gbgpu_cpu' is not a valid backend name").
import gbgpu  # noqa: F401
from gbgpu.utils.utility import get_N

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.datacontainer import DataResidualArray
from lisatools.diagnostic import inner_product
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, WDMSettings

# Stock WDM grid knobs (erebor GeneralSettings): the wavelet pixel duration is
# pinned to [3600, 4400] s INDEPENDENT of Tobs, so layer_df = 1/(2*Nf*dt) is
# Tobs-independent -- only Nt grows. Section 4 depends on this.
WAVELET_DURATION = 3600.0
DT = 10.0

# GB base parameters (same source as the zero-separation test, so the two
# scripts are directly comparable).
BASE = dict(
    amp=8.0e-22, fdot=1.0e-17, fddot=0.0, phi0=2.09802430298,
    inc=0.23984234, psi=1.234019814, lam=4.09808143, beta=0.04,
)


def _f(x):
    """Plain float from a numpy/cupy scalar or 1-element array."""
    if hasattr(x, "get"):
        x = x.get()
    a = np.asarray(x).reshape(-1)
    assert a.size == 1, f"expected scalar, got shape {a.shape}"
    return float(a[0])


class Setup:
    """One (Tobs, grid) configuration with everything needed to make signals."""

    def __init__(self, Tobs_yr: float, mid_freq: float, n_sparse: int,
                 tukey_alpha: float | None = None, edge_layers_t: int = 8):
        self.Tobs_yr = Tobs_yr
        self.dt = DT
        self.Nf = int(round(WAVELET_DURATION / DT))
        assert self.Nf % 2 == 0, "Nf must be even"
        self.layer_dt = WAVELET_DURATION
        self.layer_df = 1.0 / (2.0 * self.Nf * self.dt)

        Nt = int(np.floor(Tobs_yr * YRSID_SI / self.layer_dt))
        if Nt % 2:
            Nt -= 1
        self.Nt = Nt
        self.N = self.Nf * self.Nt
        self.Tobs = self.N * self.dt
        self.df_fd = 1.0 / self.Tobs
        self.mid_freq = mid_freq

        # Active band: keep well clear of DC/Nyquist and of the time edges.
        self.edge_layers_t = edge_layers_t
        # 8 layers clear of DC/Nyquist -- the same margin the production run
        # uses for neighbour subtraction (GB_SUBTRACT_BUFFER_LAYERS=8).
        edge_layers_f = int(os.environ.get("EDGE_LAYERS_F", 8))
        self.min_freq = edge_layers_f * self.layer_df
        self.max_freq = (self.Nf - edge_layers_f) * self.layer_df
        assert self.min_freq < mid_freq < self.max_freq, (
            f"mid_freq={mid_freq:.3e} outside active band "
            f"[{self.min_freq:.3e}, {self.max_freq:.3e}]"
        )
        min_time = edge_layers_t * self.layer_dt
        max_time = (self.Nt - edge_layers_t) * self.layer_dt

        self.td_set = TDSettings(self.N, self.dt, force_backend="cpu")
        self.wdm_set = WDMSettings(
            self.Nf, self.Nt, self.dt,
            min_freq=self.min_freq, max_freq=self.max_freq,
            min_time=min_time, max_time=max_time, force_backend="cpu",
        )
        freqs = np.fft.rfftfreq(self.N, self.dt)
        self.fd_set = FDSettings(
            len(freqs), freqs[1] - freqs[0],
            min_freq=self.min_freq, max_freq=self.max_freq, force_backend="cpu",
        )

        # Tukey taper covering edge_layers_t wavelet layers per side (the
        # stock WINDOW_TUKEY_ALPHA default is 0.05; this parameterization lets
        # section 2 sweep it while keeping the layer interpretation).
        self.tukey_alpha = (
            2.0 * edge_layers_t / self.Nt if tukey_alpha is None else tukey_alpha
        )
        self.window = sp_signal.windows.tukey(self.N, alpha=self.tukey_alpha)

        self.sens_fd = XYZ2SensitivityMatrix(self.fd_set, model="scirdv1")
        self.sens_wdm = XYZ2SensitivityMatrix(self.wdm_set, model="scirdv1")

        # TD-accurate generator (no lookup tables): generate in TD, then
        # transform to FD and WDM so both bases see the SAME waveform.
        t_start = 0.0
        self.t_arr = np.arange(self.N) * self.dt + t_start
        t_sparse = np.linspace(self.t_arr[0], self.t_arr[-1], n_sparse)
        self.gb_gen = GBTDIonTheFly(
            t_sparse, self.Tobs, t_start, 1.0 / self.dt, 1,
            tdi_config=TDIConfig("2nd generation"),
            orbits=ESAOrbits(force_backend="cpu"),
            tdi_chan="XYZ", force_backend="cpu",
        )
        self._cache: dict[tuple, tuple] = {}
        # Amplitude is calibrated to a realistic GB SNR (see calibrate_amp) so
        # the section-3 y-axis reads directly as a dlnL. The waveform is
        # linear in amp, so this is an exact rescale, not a regeneration.
        self.amp = BASE["amp"]
        self.snr = None

    def calibrate_amp(self, target_snr: float) -> float:
        """Rescale the source amplitude to hit ``target_snr`` in WDM."""
        _, wdm = self.sigs(self.mid_freq)
        snr0 = np.sqrt(ip(wdm, wdm, self.sens_wdm))
        self.amp = BASE["amp"] * target_snr / snr0
        self._cache.clear()
        _, wdm = self.sigs(self.mid_freq)
        self.snr = np.sqrt(ip(wdm, wdm, self.sens_wdm))
        return self.snr

    def describe(self) -> str:
        s = "" if self.snr is None else f"  SNR={self.snr:.1f}"
        return (
            f"Tobs={self.Tobs / YRSID_SI:.4f} yr  Nf={self.Nf} Nt={self.Nt} "
            f"N={self.N}  layer_df={self.layer_df:.4e} Hz  "
            f"1/Tobs={self.df_fd:.4e} Hz  layer_df*Tobs={self.layer_df * self.Tobs:.1f} "
            f"bins  tukey_alpha={self.tukey_alpha:.4f}{s}"
        )

    def td(self, f0: float, **over) -> np.ndarray:
        """Time-domain XYZ for a GB at f0 (BASE params, optional overrides)."""
        p = dict(BASE)
        p["amp"] = self.amp
        p.update(over)
        params = np.array([[p["amp"], f0, p["fdot"], p["fddot"], p["phi0"],
                            p["inc"], p["psi"], p["lam"], p["beta"]]])
        out = self.gb_gen(*params.T, convert_to_ra_dec=False, return_spline=True)
        td = np.asarray(out.eval_tdi(self.t_arr))
        if td.ndim == 3 and td.shape[0] == 1:
            td = td[0]
        return td

    def sigs(self, f0: float, **over):
        """(FDSignal, WDMSignal) for a GB at f0. Cached."""
        key = (round(f0, 18),) + tuple(sorted(over.items()))
        if key not in self._cache:
            ts = TDSignal(self.td(f0, **over), settings=self.td_set)
            self._cache[key] = (
                ts.transform(self.fd_set, window=self.window),
                ts.transform(self.wdm_set, window=self.window),
            )
        return self._cache[key]

    def wdm_of_td(self, td: np.ndarray):
        return TDSignal(td, settings=self.td_set).transform(
            self.wdm_set, window=self.window)


def ip(a, b, sens, normalize=False):
    """Noise-weighted inner product through the installed diagnostic path."""
    return _f(inner_product(DataResidualArray(a), DataResidualArray(b),
                            psd=sens, normalize=normalize))


def arr(sig) -> np.ndarray:
    """Underlying coefficient array of a lisatools Signal."""
    return np.asarray(sig.arr)


def diff(sig1, sig0, settings):
    """``sig1 - sig0`` as a Signal in the same domain (the basis is linear)."""
    return settings.associated_class(arr(sig1) - arr(sig0), settings)


# ---------------------------------------------------------------------------
# Section 1 -- overlap vs separation, FD vs WDM
# ---------------------------------------------------------------------------
def section_overlap(su: Setup, n_df: int, out: str) -> dict:
    print("\n== 1. OVERLAP: |<hA|hB>| normalized, FD vs WDM ==")
    print(f"   {su.describe()}")

    fd_A, wdm_A = su.sigs(su.mid_freq)
    snr_fd = np.sqrt(ip(fd_A, fd_A, su.sens_fd))
    snr_wdm = np.sqrt(ip(wdm_A, wdm_A, su.sens_wdm))
    print(f"   SNR_A: FD={snr_fd:.4f}  WDM={snr_wdm:.4f}  "
          f"ratio={snr_wdm / snr_fd:.6f}")

    # Separations from sub-bin out past one WDM layer, in units of 1/Tobs.
    lo, hi = 0.25, 3.0 * su.layer_df * su.Tobs
    seps_bins = np.unique(np.round(np.geomspace(lo, hi, n_df), 6))
    ov_fd = np.zeros(len(seps_bins))
    ov_wdm = np.zeros(len(seps_bins))

    t0 = time.time()
    for i, sb in enumerate(seps_bins):
        f_B = su.mid_freq + sb * su.df_fd
        fd_B, wdm_B = su.sigs(f_B)
        ov_fd[i] = abs(ip(fd_A, fd_B, su.sens_fd, normalize=True))
        ov_wdm[i] = abs(ip(wdm_A, wdm_B, su.sens_wdm, normalize=True))
        if i % 8 == 0:
            print(f"   [{i + 1:3d}/{len(seps_bins)}] df={sb:9.3f}/Tobs "
                  f"({sb / (su.layer_df * su.Tobs):7.4f} layer)  "
                  f"FD={ov_fd[i]:.3e}  WDM={ov_wdm[i]:.3e}")
    print(f"   swept {len(seps_bins)} separations in {time.time() - t0:.1f}s")

    d = dict(seps_bins=seps_bins.tolist(), ov_fd=ov_fd.tolist(),
             ov_wdm=ov_wdm.tolist(),
             layer_bins=float(su.layer_df * su.Tobs),
             snr_fd=float(snr_fd), snr_wdm=float(snr_wdm), desc=su.describe())
    plot_overlap(d, out)
    return d


def plot_overlap(d: dict, out: str) -> None:
    """Redraw section 1 from its saved data (also used by ``--replot``)."""
    seps_bins = np.asarray(d["seps_bins"])
    ov_fd = np.asarray(d["ov_fd"])
    ov_wdm = np.asarray(d["ov_wdm"])
    layer_bins = d["layer_bins"]
    lo = seps_bins[0]
    # Envelope: the overlap oscillates (sinc-like fringes), so the running max
    # from the right is the honest "worst case at or beyond this separation".
    env_fd = np.maximum.accumulate(ov_fd[::-1])[::-1]
    env_wdm = np.maximum.accumulate(ov_wdm[::-1])[::-1]

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.loglog(seps_bins, np.maximum(ov_fd, 1e-16), "o-", ms=3.5, lw=1.2,
              alpha=0.55, color="tab:blue",
              label=r"FD  $|\langle h_A|h_B\rangle|$")
    ax.loglog(seps_bins, np.maximum(ov_wdm, 1e-16), "s-", ms=3.5, lw=1.2,
              alpha=0.55, color="tab:orange",
              label=r"WDM $|\langle h_A|h_B\rangle|$")
    ax.loglog(seps_bins, env_fd, "-", lw=2.2, color="tab:blue",
              label="FD envelope (worst case beyond)")
    ax.loglog(seps_bins, env_wdm, "--", lw=2.2, color="tab:orange",
              label="WDM envelope")
    ax.axvline(layer_bins, color="k", ls=":", lw=1.5)
    ax.annotate("one WDM layer", xy=(layer_bins, 3e-2),
                xytext=(layer_bins * 0.35, 3e-2), rotation=90,
                fontsize=9, va="center", ha="center")
    ax.axvspan(lo, layer_bins, color="0.5", alpha=0.08)
    ax.set_xlabel(r"frequency separation $\Delta f$  [$1/T_{\rm obs}$ bins]")
    ax.set_ylabel(r"normalized $|\langle h_A|h_B\rangle|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    sec = ax.secondary_xaxis(
        "top", functions=(lambda b: b / layer_bins, lambda l: l * layer_bins))
    sec.set_xlabel(r"$\Delta f$  [WDM layers]")
    fig.suptitle(
        "Two sources are already orthogonal well INSIDE one WDM layer.\n"
        "WDM tracks FD below ~50 bins; above it WDM holds a floor "
        r"$\sim\!10^{-3}$ that FD does not.", fontsize=11.5)
    fig.text(0.5, 0.005, d.get("desc", ""), ha="center", fontsize=7.5,
             color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    p = os.path.join(out, "s1_overlap_fd_vs_wdm.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"   -> {p}")

    # Ratio panel: how closely does WDM reproduce FD?
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    good = ov_fd > 1e-12
    ax.semilogx(seps_bins[good], ov_wdm[good] / ov_fd[good], "o-", ms=4)
    ax.axhline(1.0, color="k", ls=":")
    ax.axvline(layer_bins, color="k", ls=":", lw=1.4)
    ax.set_xlabel(r"$\Delta f$  [$1/T_{\rm obs}$ bins]")
    ax.set_ylabel("WDM / FD overlap")
    ax.set_title("Basis agreement of the cross inner product", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(out, "s1_overlap_ratio.png")
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    print(f"   -> {p2}")


# ---------------------------------------------------------------------------
# Section 2 -- WDM energy outside the source's own layer, vs Tukey alpha
# ---------------------------------------------------------------------------
def section_energy(su_kwargs: dict, out: str) -> dict:
    print("\n== 2. ENERGY: WDM power outside the source's own layer ==")
    alphas = [0.0, 0.01, 0.05, 0.1, 0.2]
    ks = np.arange(0, 9)
    curves = {}
    for a in alphas:
        su = Setup(tukey_alpha=(a if a > 0 else 1e-12), **su_kwargs)
        _, wdm_A = su.sigs(su.mid_freq)
        # (nchan, Nf_active, Nt_active) -> power per layer, summed over
        # channels and time pixels. Unweighted power: this is the pixel-
        # support question, not a likelihood.
        pw = np.abs(arr(wdm_A)) ** 2
        while pw.ndim > 3:
            pw = pw[0]
        per_layer = pw.sum(axis=(0, -1)) if pw.ndim == 3 else pw.sum(axis=-1)
        m0 = int(np.argmax(per_layer))
        tot = per_layer.sum()
        frac = np.array([
            1.0 - per_layer[max(0, m0 - k): m0 + k + 1].sum() / tot for k in ks
        ])
        curves[a] = frac
        print(f"   tukey_alpha={a:5.3f}  peak layer={m0}  "
              + "  ".join(f"k={k}:{frac[k]:.2e}" for k in (0, 1, 2, 4)))

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for a, frac in curves.items():
        ax.semilogy(ks, np.maximum(frac, 1e-20), "o-", ms=5,
                    label=rf"Tukey $\alpha$={a:g}")
    ax.set_xlabel(r"half-width $k$  [WDM layers about the source's own layer]")
    ax.set_ylabel("fraction of WDM power outside $\\pm k$ layers")
    ax.set_title("WDM frequency localization of a GB vs the Tukey taper",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out, "s2_energy_outside_layers.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"   -> {p}")
    return {str(a): v.tolist() for a, v in curves.items()}


# ---------------------------------------------------------------------------
# Section 3 -- the bookkeeping error
# ---------------------------------------------------------------------------
def _step(su: Setup, f0: float, snr: float, kind: str):
    """A realistic accepted-proposal change dh for a source at f0.

    ``in_model``: a 1-sigma-ish MCMC step (df0 ~ 1/(snr*Tobs), dphi ~ 1/snr,
    dA/A ~ 1/snr).  ``rj``: the whole template (birth/death worst case).
    """
    fd0, wdm0 = su.sigs(f0)
    if kind == "rj":
        return fd0, wdm0
    df0 = 1.0 / (snr * su.Tobs)
    over = dict(phi0=BASE["phi0"] + 1.0 / snr, amp=su.amp * (1.0 + 1.0 / snr))
    fd1, wdm1 = su.sigs(f0 + df0, **over)
    return diff(fd1, fd0, su.fd_set), diff(wdm1, wdm0, su.wdm_set)


def section_bookkeeping(su: Setup, n_df: int, out: str, tag: str = "") -> dict:
    print(f"\n== 3. BOOKKEEPING error |<dh_A|dh_B>| {tag} ==")
    print(f"   {su.describe()}")
    fd_A, wdm_A = su.sigs(su.mid_freq)
    snr = np.sqrt(ip(wdm_A, wdm_A, su.sens_wdm))
    print(f"   SNR_A (WDM) = {snr:.3f}")

    layer_bins = su.layer_df * su.Tobs
    lo, hi = 1.0, 3.0 * layer_bins
    seps = np.unique(np.round(np.geomspace(lo, hi, n_df), 6))

    res = {k: np.zeros(len(seps)) for k in ("inmodel", "rj")}
    meas = {k: np.zeros(len(seps)) for k in ("inmodel", "rj")}

    def ll(hA, hB, d):
        """-1/2 <r|r> for the residual r = d - hA - hB (WDM)."""
        r = su.wdm_set.associated_class(arr(d) - arr(hA) - arr(hB), su.wdm_set)
        return -0.5 * ip(r, r, su.sens_wdm)

    zero = su.wdm_set.associated_class(np.zeros_like(arr(wdm_A)), su.wdm_set)

    for i, sb in enumerate(seps):
        f_B = su.mid_freq + sb * su.df_fd
        # Band B's accepted in-model step: h_B_old -> h_B_new.
        _, wB_old = su.sigs(f_B)
        dfB = 1.0 / (snr * su.Tobs)
        _, wB_new = su.sigs(f_B + dfB,
                            phi0=BASE["phi0"] + 1.0 / snr,
                            amp=su.amp * (1.0 + 1.0 / snr))
        dwB = diff(wB_new, wB_old, su.wdm_set)

        for kind in ("inmodel", "rj"):
            if kind == "rj":
                # RJ worst case: band A's source is born (or dies) whole.
                wA_old, wA_new = zero, wdm_A
            else:
                wA_old = wdm_A
                _, wA_new = su.sigs(su.mid_freq + 1.0 / (snr * su.Tobs),
                                    phi0=BASE["phi0"] + 1.0 / snr,
                                    amp=su.amp * (1.0 + 1.0 / snr))
            dwA = diff(wA_new, wA_old, su.wdm_set)

            # Predicted by (*).
            res[kind][i] = abs(ip(dwA, dwB, su.sens_wdm))

            # Measured end-to-end: band B's dll evaluated on the STALE
            # residual (still holding A's old template) vs on the FRESH one.
            # Truth data = both sources at their new values.
            d = su.wdm_set.associated_class(
                arr(wA_new) + arr(wB_new), su.wdm_set)
            dll_stale = ll(wA_old, wB_new, d) - ll(wA_old, wB_old, d)
            dll_fresh = ll(wA_new, wB_new, d) - ll(wA_new, wB_old, d)
            meas[kind][i] = abs(dll_fresh - dll_stale)

        if i % 8 == 0:
            print(f"   [{i + 1:3d}/{len(seps)}] df={sb:9.3f}/Tobs "
                  f"({sb / layer_bins:7.4f} layer)  "
                  f"in-model={res['inmodel'][i]:.3e}  RJ={res['rj'][i]:.3e}")

    # Relative agreement between the algebra (*) and the measured dll gap.
    rel = {}
    for k in ("inmodel", "rj"):
        den = np.maximum(np.abs(res[k]), 1e-300)
        rel[k] = float(np.max(np.abs(meas[k] - res[k]) / den))
    # The measured side differences two ll values of order <r|r> ~ 1e2-1e3 to
    # extract a gap that can be ~1e-6 of them, so the achievable agreement is
    # catastrophic-cancellation limited, not algebra limited. 1e-4 relative is
    # a comfortable pass; anything above that is a real disagreement.
    print(f"   identity (*) max rel. mismatch: in-model={rel['inmodel']:.2e}  "
          f"RJ={rel['rj']:.2e}  "
          f"({'OK' if max(rel.values()) < 1e-4 else 'MISMATCH'})")
    vmax = max(rel.values())

    d = dict(seps=seps.tolist(), inmodel=res["inmodel"].tolist(),
             rj=res["rj"].tolist(),
             inmodel_measured=meas["inmodel"].tolist(),
             rj_measured=meas["rj"].tolist(),
             layer_bins=float(layer_bins), snr=float(snr),
             mid_freq=float(su.mid_freq), Tobs=float(su.Tobs),
             identity_rel_mismatch=vmax, desc=su.describe(), tag=tag)
    plot_bookkeeping(d, out)
    return d


def plot_bookkeeping(d: dict, out: str) -> None:
    """Redraw section 3/4 from its saved data (also used by ``--replot``)."""
    seps = np.asarray(d["seps"])
    res = {k: np.asarray(d[k]) for k in ("inmodel", "rj")}
    layer_bins = d["layer_bins"]
    snr = d["snr"]
    tag = d.get("tag", "")
    env = {k: np.maximum.accumulate(res[k][::-1])[::-1] for k in res}

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    ax.loglog(seps, np.maximum(res["rj"], 1e-18), "s-", ms=3.5, lw=1.1,
              alpha=0.5, color="tab:red")
    ax.loglog(seps, env["rj"], "-", lw=2.3, color="tab:red",
              label=r"RJ worst case  $|\langle h_A|\delta h_B\rangle|$ (envelope)")
    ax.loglog(seps, np.maximum(res["inmodel"], 1e-18), "o-", ms=3.5, lw=1.1,
              alpha=0.5, color="tab:blue")
    ax.loglog(seps, env["inmodel"], "-", lw=2.3, color="tab:blue",
              label=r"in-model  $|\langle \delta h_A|\delta h_B\rangle|$ (envelope)")

    ax.axhspan(1.0, 1e6, color="tab:red", alpha=0.06)
    ax.axhline(1.0, color="k", lw=1.3)
    ax.text(seps[0] * 1.1, 1.35,
            r"$|\Delta\ln L|=1$: above here the error can flip an accept/reject",
            fontsize=8.5)

    # Operating points, in 1/Tobs bins.
    mid_f = d.get("mid_freq", 4.0e-3)
    # Tobs is recoverable from the stored layer geometry when absent.
    tobs = d.get("Tobs") or (layer_bins / (1.0 / (2.0 * 360 * DT)))
    min_band = 2.0 * int(np.atleast_1d(
        get_N(1e-30, mid_f, tobs, oversample=1)).ravel()[0])
    marks = [
        (min_band, "tab:green", r"$2\,\mathrm{get\_N}$ (min band)"),
        (layer_bins, "k", "1 layer"),
        (2 * layer_bins, "0.45", "2 layers = today's\nparity separation"),
    ]
    for x, c, lab in marks:
        if seps[0] <= x <= seps[-1]:
            ax.axvline(x, color=c, ls=":", lw=1.6)
            # Axes-fraction y so the label always lands inside the panel.
            ax.annotate(lab, xy=(x, 0.02), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=8.5, color=c,
                        va="bottom", ha="right")
    ax.set_xlabel(r"band separation $\Delta f$  [$1/T_{\rm obs}$ bins]")
    ax.set_ylabel(r"induced bookkeeping error  $|\Delta\ln L|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    sec = ax.secondary_xaxis(
        "top", functions=(lambda b: b / layer_bins, lambda l: l * layer_bins))
    sec.set_xlabel(r"$\Delta f$  [WDM layers]")
    fig.suptitle(
        "GO/NO-GO: error a concurrent accepted change in band A induces\n"
        "in band B's $\\Delta\\ln L$", fontsize=12)
    ax.text(0.985, 0.97,
            f"error $\\propto \\rho_A$ (here $\\rho_A$={snr:.0f});\n"
            f"a $\\rho$=100 neighbour scales these by {100 / snr:.0f}x",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(fc="w", ec="0.7", alpha=0.85, boxstyle="round,pad=0.35"))
    fig.text(0.5, 0.005, d.get("desc", ""), ha="center", fontsize=7.5,
             color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    p = os.path.join(out, f"s3_bookkeeping_error{tag}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"   -> {p}")


# ---------------------------------------------------------------------------
# Section 5 -- how conservative is the shared-sky assumption?
# ---------------------------------------------------------------------------
def section_sky(su: Setup, out: str, n_draw: int = 24) -> dict:
    """Sections 1/3 give A and B the SAME sky/inc/psi, which maximizes their
    correlation. Real neighbours are independent on the sky. Redo the RJ
    bookkeeping error at a few separations with B drawn isotropically."""
    print("\n== 5. SKY: shared-sky (worst case) vs independent-sky neighbours ==")
    snr = su.snr if su.snr is not None else np.sqrt(
        ip(*(su.sigs(su.mid_freq)[1],) * 2, su.sens_wdm))
    _, wdm_A = su.sigs(su.mid_freq)
    layer_bins = su.layer_df * su.Tobs
    targets = [128.0, 256.0, 1024.0, layer_bins]
    rng = np.random.default_rng(20260806)

    rows = {}
    for sb in targets:
        f_B = su.mid_freq + sb * su.df_fd
        # shared sky (the section-3 configuration)
        _, wB0 = su.sigs(f_B)
        _, wB1 = su.sigs(f_B + 1.0 / (snr * su.Tobs),
                         phi0=BASE["phi0"] + 1.0 / snr,
                         amp=su.amp * (1.0 + 1.0 / snr))
        shared = abs(ip(wdm_A, diff(wB1, wB0, su.wdm_set), su.sens_wdm))

        vals = []
        for _ in range(n_draw):
            over = dict(
                lam=float(rng.uniform(0, 2 * np.pi)),
                beta=float(np.arcsin(rng.uniform(-1, 1))),
                psi=float(rng.uniform(0, np.pi)),
                inc=float(np.arccos(rng.uniform(-1, 1))),
                phi0=float(rng.uniform(0, 2 * np.pi)),
            )
            _, b0 = su.sigs(f_B, **over)
            o1 = dict(over)
            o1["phi0"] = over["phi0"] + 1.0 / snr
            o1["amp"] = su.amp * (1.0 + 1.0 / snr)
            _, b1 = su.sigs(f_B + 1.0 / (snr * su.Tobs), **o1)
            vals.append(abs(ip(wdm_A, diff(b1, b0, su.wdm_set), su.sens_wdm)))
        vals = np.array(vals)
        rows[sb] = dict(shared=float(shared), draws=vals.tolist())
        print(f"   sep={sb:8.1f} bins ({sb / layer_bins:6.3f} layer)  "
              f"shared-sky={shared:.3e}  random-sky: "
              f"median={np.median(vals):.3e} p90={np.percentile(vals, 90):.3e} "
              f"max={vals.max():.3e}")

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    xs = list(rows)
    ax.boxplot([rows[s]["draws"] for s in xs], positions=range(len(xs)),
               widths=0.5, showfliers=True)
    ax.plot(range(len(xs)), [rows[s]["shared"] for s in xs], "r*", ms=15,
            label="shared sky (sections 1/3 configuration)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{s:.0f}\n({s / layer_bins:.2f} lyr)" for s in xs])
    ax.axhline(1.0, color="k", lw=1.2)
    ax.set_xlabel(r"band separation  [$1/T_{\rm obs}$ bins]")
    ax.set_ylabel(r"RJ bookkeeping error $|\Delta\ln L|$")
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.suptitle("Shared sky is the conservative choice: independent-sky\n"
                 "neighbours sit at or below it", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "s5_sky_dependence.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"   -> {p}")
    return {str(k): v for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3")
    ap.add_argument("--out", default="wdm_subband_leakage_out")
    ap.add_argument("--replot", default=None,
                    help="redraw the figures from a saved results.json "
                         "(no recomputation) and exit")
    args = ap.parse_args()
    want = {s.strip() for s in args.sections.split(",") if s.strip()}
    os.makedirs(args.out, exist_ok=True)

    if args.replot:
        with open(args.replot) as fh:
            saved = json.load(fh)
        if "overlap" in saved:
            plot_overlap(saved["overlap"], args.out)
        for k, v in saved.items():
            if k.startswith("bookkeeping"):
                plot_bookkeeping(v, args.out)
        print(f"replotted into {args.out}")
        return

    n_sparse = int(os.environ.get("N_SPARSE", 8192))
    mid = float(os.environ.get("MID_FREQ_MHZ", "4.0")) * 1e-3
    n_df = int(os.environ.get("N_DF", 40))
    base_kw = dict(mid_freq=mid, n_sparse=n_sparse)
    target_snr = float(os.environ.get("TARGET_SNR", "20"))

    results: dict = {}
    su3 = None
    if want & {"1", "3", "4", "5"}:
        su3 = Setup(Tobs_yr=0.25, **base_kw)
        print(f"[calib] SNR -> {su3.calibrate_amp(target_snr):.3f} "
              f"(amp={su3.amp:.4e})")

    if "1" in want:
        results["overlap"] = section_overlap(su3, n_df, args.out)
    if "2" in want:
        results["energy"] = section_energy(dict(Tobs_yr=0.25, **base_kw), args.out)
    if "3" in want:
        results["bookkeeping_3mo"] = section_bookkeeping(
            su3, n_df, args.out, tag="_3mo")
    if "4" in want:
        # Cost scales with N = Nf*Nt, i.e. linearly in Tobs; 2 yr is ~8x the
        # 3-month case per separation. TOBS_LIST trims it.
        yrs = [float(v) for v in
               os.environ.get("TOBS_LIST", "1.0,2.0").split(",") if v.strip()]
        for yr, tg in ((y, f"_{y:g}yr") for y in yrs):
            su = Setup(Tobs_yr=yr, **base_kw)
            # Same source, longer baseline -> SNR grows as sqrt(Tobs); keep the
            # PHYSICAL source fixed by reusing the 3-month amplitude.
            su.amp = su3.amp if su3 is not None else su.amp
            su._cache.clear()
            _, w = su.sigs(su.mid_freq)
            su.snr = np.sqrt(ip(w, w, su.sens_wdm))
            results[f"bookkeeping{tg}"] = section_bookkeeping(
                su, max(20, n_df // 2), args.out, tag=tg)

    if "5" in want:
        results["sky"] = section_sky(su3, args.out)

    p = os.path.join(args.out, "results.json")
    with open(p, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
