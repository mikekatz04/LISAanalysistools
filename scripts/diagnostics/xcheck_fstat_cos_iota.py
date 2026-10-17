"""Cross-check of the F-stat maximiser -> template cos(iota) convention.

Noise-free single-source residuals (SNR 10) at cos iota +0.6, +0.3, -0.3, -0.6,
fixed other parameters, separate frequencies (one residual each), on a grid
with the production layer geometry (Nf*dt = 3600 s, the GB lookup table's).
Two F-stat paths: (a) the sig-het shared-reference F-stat
(build_sighet_call_fstat on the production-configured sig-het comp, lookup
reference), (b) the exact chunked F-stat (get_fstat_ll_wdm). Both maximisers
through fstat_maximized_extrinsics. The pinned template (sampling phi0 =
phi0_max, cos_iota = +/-cos(iota_max), psi = psi_max, dist from A_max at the
truth intrinsics) is scored with: the chunked comp; the sig-het wrapper's
get_ll_wdm with no in-model reference (= the RJ birth path, delegates to the
chunked comp); the sig-het in-model kernel (reference at the truth); the GB
lookup comp.
"""
import os
import sys

os.environ.setdefault("SIGHET_REF_BUILD", "lookup")
os.environ.setdefault("SIGHET_ANCHOR_ENGINE", "lookup")
os.environ.setdefault("GB_SIGHET_INMODEL_WINDOWED", "1")

import numpy as np  # noqa: E402

from lisatools.detector import ESAOrbits  # noqa: E402
from lisatools.domains import WDMSettings  # noqa: E402
from lisatools.utils.constants import YRSID_SI  # noqa: E402

sys.path.insert(0, "/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools/tests")
from _para_search_world import covariance  # noqa: E402

from gbgpu.gbcomps import GBWDMComputations  # noqa: E402
from lisatools.globalfit.stock.erebor.gb import build_sighet_engine, sighet_engine_kwargs  # noqa: E402
from lisatools.globalfit.stock.erebor.transforms import (  # noqa: E402
    gb_amp_from_dist, make_gb_transform_container)
from lisatools.globalfit.stock.erebor.variants.gb_no_fg import GBNoFgGBSettings  # noqa: E402
from lisatools.sampling.fstat_gridfit import build_sighet_call_fstat  # noqa: E402
from lisatools.sampling.fstat_proposal import fstat_maximized_extrinsics  # noqa: E402

NF, DT, NT, EDGE = 180, 20.0, 512, 40


class _H:
    def __init__(self, d, iC):
        self.linear_data_arr = [np.ascontiguousarray(d[None]).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(iC[None]).ravel()]

    def __len__(self):
        return 1


t0 = int(0.5 * YRSID_SI / DT) * DT
orbits = ESAOrbits(force_backend="cpu")
T = NF * NT * DT
margin = 64 * NF * DT
orbits._configure(t_arr=np.arange(-margin, T + margin + 1000.0, 1000.0) + t0)
wdm = WDMSettings(NF, NT, DT, t0=t0, min_freq=1e-4, max_freq=2.5e-2,
                  min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                  force_backend="cpu")
info = GBNoFgGBSettings()
print("sig-het engine kwargs:", {k: v for k, v in sighet_engine_kwargs(info).items()})
ch = GBWDMComputations(wdm, t_ref=t0, Nt_sub=int(info.nt_sub), n_pad=int(info.n_pad),
                       N_sparse=int(info.n_sparse), N_cp_sig=int(info.n_cp_sig),
                       N_cp_orbit=int(info.n_cp_orbit), orbits=orbits,
                       tdi_config="2nd generation", force_backend="cpu", d_d=0.0,
                       tdi_type="XYZ")
ch.convert_to_ra_dec = False
sig = build_sighet_engine(info, ch, branch="gb")
lookup = None
try:
    from gbgpu.gblookupcomputations import GBLookupWDMComputations
    from lisatools.wdm_lookup_store import resolve_gb_lookup_table

    lookup = GBLookupWDMComputations(
        wdm, t_ref=t0, Nt_sub=int(info.nt_sub), n_pad=int(info.n_pad),
        N_sparse=int(info.n_sparse), N_cp_sig=int(info.n_cp_sig),
        N_cp_orbit=int(info.n_cp_orbit), orbits=orbits, tdi_config="2nd generation",
        force_backend="cpu", tdi_type="XYZ",
        lookup_table=resolve_gb_lookup_table(None, None), lookup_n_nodes=-1,
        lookup_k_coarse=True, lookup_fill=False)
    lookup.convert_to_ra_dec = False
except Exception as exc:  # noqa: BLE001
    print("lookup comp unavailable:", type(exc).__name__, exc)
nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
nta = int(wdm.Nt_active)
C = covariance(wdm)
iC = np.broadcast_to(np.linalg.inv(C).transpose(1, 2, 0)[:, :, :, None],
                     (3, 3, nfa, nta)).copy()
tf = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                 use_distance=True, mc_lims=(0.001, 1.0))
tobs = float(wdm.Tobs)
print(f"grid Nf {NF} dt {DT} Nt {NT}: Tobs {tobs:.4e} s, layer_df {wdm.layer_df:.4e}, "
      f"active {nfa} x {nta}")
z1 = np.zeros(1, dtype=np.int32)


def inject(f0, cos_i, phi0=1.0, psi=0.7, alpha=2.0, sdelta=0.3, mc=0.3, snr=10.0):
    x = np.array([[1.0, f0 * 1e3, mc, phi0, cos_i, psi, alpha, sdelta, 0.0]])
    p = np.asarray(tf.both_transforms(x))
    h = np.zeros((1, 3, nfa, nta))
    ch.fill_global_wdm(p, h.reshape(-1), data_index=z1, factors=np.ones(1))
    s = snr / np.sqrt(np.einsum("cft,cdft,dft->", h[0], iC, h[0]))
    x[0, 0] /= s
    return x, np.asarray(tf.both_transforms(x)), _H(h[0] * s, iC)


def lnl_scorers(P, hold, p_truth):
    out = {}
    ch.get_ll_wdm(P, hold, data_index=np.zeros(len(P), np.int32),
                  noise_index=np.zeros(len(P), np.int32))
    out["chunked"] = ch.d_h_out.real - 0.5 * ch.h_h_out.real
    sig.get_ll_wdm(P, hold, data_index=np.zeros(len(P), np.int32),
                   noise_index=np.zeros(len(P), np.int32))
    out["sighet_rj"] = sig.d_h_out.real - 0.5 * sig.h_h_out.real
    try:
        sig.setup_in_model(hold, p_truth, np.zeros(1, np.int32))
        sig.get_ll_wdm(P, hold, data_index=np.zeros(len(P), np.int32),
                       noise_index=np.zeros(len(P), np.int32))
        out["sighet_inmodel"] = np.asarray(sig.d_h_out).real - 0.5 * np.asarray(sig.h_h_out).real
    except Exception as exc:  # noqa: BLE001
        print("  sig-het in-model scorer failed:", type(exc).__name__, exc)
    finally:
        try:
            sig.clear_in_model()
        except Exception:  # noqa: BLE001
            pass
    if lookup is not None:
        lookup.get_ll_wdm(P, hold, data_index=np.zeros(len(P), np.int32),
                          noise_index=np.zeros(len(P), np.int32))
        out["lookup"] = np.asarray(lookup.d_h_out).real - 0.5 * np.asarray(lookup.h_h_out).real
    return out


def pinned(x, A, phi0_max, iota_max, psi_max):
    """Sampling rows: [pin(+cos), -cos, -phi0 & +cos, -phi0 & -cos]."""
    rows = []
    for sp, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r = x[0].copy()
        r[0] = gb_amp_from_dist(r[1] * 1e-3, r[2], 1.0) / A
        r[3] = (sp * phi0_max) % (2 * np.pi)
        r[4] = sc * np.cos(iota_max)
        r[5] = psi_max % np.pi
        rows.append(r)
    return np.asarray(tf.both_transforms(np.array(rows)))


def run(cases, label):
    print(f"\n=== {label} ===")
    hdr = (f"{'cos_i':>6} {'path':>8} {'F':>7} {'iota_max':>8} {'pi-iota':>7} "
           f"{'phi0_max':>8} {'psi_max':>7} | scorer: lnL(+cos) lnL(-cos) "
           f"[-phi0: +cos -cos] | truth lnL")
    print(hdr)
    for kk, (f0, cos_i, kw) in enumerate(cases):
        x, p_truth, hold = inject(f0, cos_i, **kw)
        truth = lnl_scorers(p_truth, hold, p_truth)
        paths = {}
        Nc, Mc = ch.get_fstat_ll_wdm(p_truth, hold, data_index=z1, noise_index=z1)
        paths["chunked"] = (Nc, Mc)
        call = build_sighet_call_fstat(sig, hold, xp=np, Tobs=tobs,
                                       f0_lims_hz=(f0 - 2e-6, f0 + 2e-6),
                                       data_index=0, noise_index=0)
        paths["sighet"] = call(p_truth)
        sig.clear_fstat_references()
        for path, (N, M) in paths.items():
            A, ph, io, ps, F = (float(np.asarray(v)[0]) for v in
                                fstat_maximized_extrinsics(np.asarray(N), np.asarray(M)))
            P = pinned(x, A, ph, io, ps)
            sc = lnl_scorers(P, hold, p_truth)
            first = True
            for name, v in sc.items():
                pre = (f"{cos_i:+6.2f} {path:>8} {F:7.2f} {io:8.4f} {np.pi - io:7.4f} "
                       f"{ph:8.4f} {ps:7.4f}" if first else " " * 52)
                print(f"{pre} | {name:>14}: {v[0]:8.2f} {v[1]:8.2f} "
                      f"[{v[2]:8.2f} {v[3]:8.2f}] | {truth[name][0]:7.2f}")
                first = False
        tr = x[0]
        print(f"{'':>6} truth sampling: phi0 {tr[3]:.4f} cos_iota {tr[4]:+.3f} "
              f"(iota {np.arccos(tr[4]):.4f}) psi {tr[5]:.4f}; physical phi0 "
              f"{p_truth[0, 4] % (2 * np.pi):.4f} iota {p_truth[0, 5]:.4f}")


fixed = dict(phi0=1.0, psi=0.7, alpha=2.0, sdelta=0.3, mc=0.3)
run([(3.0e-3, +0.6, fixed), (3.2e-3, +0.3, fixed), (3.4e-3, -0.3, fixed),
     (3.6e-3, -0.6, fixed)], "requested: cos iota +0.6/+0.3/-0.3/-0.6, other params fixed")
rng = np.random.default_rng(4)
extra = []
for k, c in enumerate((+0.6, -0.6, +0.6, -0.6)):
    extra.append((2.2e-3 + 0.35e-3 * k, c,
                  dict(phi0=rng.uniform(0, 2 * np.pi), psi=rng.uniform(0, np.pi),
                       alpha=rng.uniform(0, 2 * np.pi), sdelta=rng.uniform(-0.9, 0.9),
                       mc=0.3)))
run(extra, "robustness: random phi0 / psi / sky, cos iota +-0.6")
