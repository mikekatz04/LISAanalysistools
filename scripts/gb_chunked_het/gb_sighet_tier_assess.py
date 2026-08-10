"""Tiered accuracy assessment for sig-het v2 / rung-i (USER spec 2026-08-02).

Requirement: for candidates whose TRUE lnL sits within T of the maximum
(reference), the sig-het error dLL = |(arm delta) - (dense delta)| must obey
    T <= 50    ->  dLL << 1 (~0.01-0.1)
    T ~  100   ->  dLL <~ 1
    T ~ 1000   ->  dLL ~ 10-20   (i.e. relative 1e-2 to 1e-3)
with everything beyond T ~ 1000 gateable (-inf prior / trust region).

Deltas are taken FROM THE REFERENCE for both the arm and dense
(delta-vs-delta), which is the sampling-relevant comparison and removes any
constant reference-point offset.

For each ref (seed-19 many-ref draws: spans clean -> catastrophic incl the
ref02 builder-slip case) and each direction (f0, lnA, iota, psi, lam), the
displacement is scaled to hit each tier: probe at s0, then
s = s0*sqrt(T/|dLL0|) with one refinement.

Run: /Users/mkatz/miniconda3/envs/deving/bin/python gb_sighet_tier_assess.py
Env: TIER_NREF (8), TIER_LIST ("1,10,50,100,1000"), TIER_NR (64),
     ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_ratio_build_prototype as proto

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    n_ref = int(os.environ.get("TIER_NREF", "8"))
    tiers = [float(x) for x in
             os.environ.get("TIER_LIST", "1,10,50,100,1000").split(",")]
    nr = int(os.environ.get("TIER_NR", "64"))

    backend = "cpu"
    dt = 10.0
    # Grid knobs (defaults = the original ~1 yr configuration). TIER_NT is what
    # sets T_obs = Nf*Nt*dt, so the n_r-vs-T_obs question is swept with it:
    # Nt=12288 -> 1.0 yr, Nt=3072 -> 0.25 yr.
    Nf = int(os.environ.get("TIER_NF", "256"))
    Nt = int(os.environ.get("TIER_NT", "12288"))
    t_start = int(0.5 * YRSID_SI / dt) * dt
    # Edge crop and Tukey taper are fractions of the time axis, so they scale
    # with Nt; the literals are the 1-yr values they came from.
    _sc = Nt / 12288.0
    edge = int(os.environ.get("TIER_EDGE", str(max(4, round(330 * _sc)))))
    tk = int(os.environ.get("TIER_TK", str(max(4, round(307 * _sc)))))
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
        chunked, n_sparse_fd=512, n_cp_build=93,
        nt_layer=int(os.environ.get("TIER_NT_LAYER", "512")),
        m_active_half_width=2)
    g = sighet._g
    N = g["n_sparse_fd"]
    eng_s = WDMBandLikelihoodEngine(sighet, wdm_set, nchannels=3,
                                    tdi_channel_setup="XYZ")
    gen = None

    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    invC = np.ascontiguousarray(
        np.asarray(XYZ2SensitivityMatrix(wdm_set, model="scirdv1").invC),
        dtype=np.float64)
    zeros = np.zeros(1, dtype=np.int32)
    kw = dict(data_index=zeros, noise_index=zeros, N_vals=None,
              waveform_kwargs={})
    tau_slow = np.arange(N) * (g["Tobs"] / N)
    drift_u = 1.0 / (2 * np.pi * g["Tobs"])

    rngr = np.random.default_rng(19)
    refs = []
    for _ in range(n_ref):
        refs.append(np.array([
            10 ** rngr.uniform(-22.5, -21.0),
            rngr.uniform(1.5e-3, 1.5e-2),
            rngr.uniform(0.0, 3e-16),
            0.0,
            rngr.uniform(0, 2 * np.pi),
            np.arccos(rngr.uniform(-1, 1)),
            rngr.uniform(0, np.pi),
            rngr.uniform(0, 2 * np.pi),
            np.arcsin(rngr.uniform(-1, 1)),
        ]))

    DIRS = [("f0", 1, 0.05 * drift_u), ("lnA", 0, 0.03),
            ("iota", 5, 0.03), ("psi", 6, 0.03), ("lam", 7, 0.01)]

    def displace(ref, idx, s):
        p = ref.copy()
        if idx == 0:
            p[0] *= np.exp(s)
        else:
            p[idx] += s
        return p

    ref_sel = os.environ.get("TIER_REF_SEL", "")
    if ref_sel:
        keep = {int(x) for x in ref_sel.split(",")}
        print(f"[sel] running refs {sorted(keep)} only")
    else:
        keep = set(range(len(refs)))

    rows = []
    for rrr, ref in enumerate(refs):
        if rrr not in keep:
            continue
        href = np.zeros((3, Nf, Nt))
        chunked.fill_global_wdm(ref[None, :], href, convert_to_ra_dec=False)
        h_act = np.ascontiguousarray(
            href[:, ilo:ihi, wdm_set.active_slice_t])
        holder = proto._FullGridWDMHolder(h_act, invC)

        def dense(p):
            ht = np.zeros((3, Nf, Nt))
            chunked.fill_global_wdm(p[None, :], ht, convert_to_ra_dec=False)
            ht_a = ht[:, ilo:ihi, wdm_set.active_slice_t]
            dh = float(np.einsum("cmn,cdmn,dmn->", h_act, invC, ht_a))
            hh = float(np.einsum("cmn,cdmn,dmn->", ht_a, invC, ht_a))
            return dh - 0.5 * hh

        sighet.clear_in_model()
        sighet.setup_in_model(holder, ref[None, :], zeros)
        if gen is None:
            gen_l = sighet._keep_alive["gb_gen"]
        gen_l = sighet._keep_alive["gb_gen"]
        spl_r = proto.build_spl(gen_l, ref)
        s_ref, kf0_r, _ = proto.slow_series(gen_l, ref, None, N, g,
                                            spl=spl_r)
        good_r = proto.good_samples(s_ref)
        a_norm = np.abs(s_ref)
        min_env = float((a_norm / a_norm.max(axis=1, keepdims=True)).min())
        c0_all = np.asarray(sighet.c0_sparse_all)[0]

        # ---- rung-ii pixel-time calibration (prototype pix-off cal, ported
        # per the USER care protocol: never trust an uncalibrated pixel map)
        n_sl = np.asarray(sighet.n_sparse_local)
        n_global = (g["ind_min_t"] + int(n_sl[0])
                    + np.arange(g["N_sparse_t"]) * g["stride"])
        window_full = np.asarray(sighet.window_full, dtype=np.float64)
        p_cal = displace(ref, 1, 0.1 * drift_u)
        m_cal = proto.m_active_for(p_cal[1], g)
        s_cal, _, _ = proto.slow_series(gen_l, p_cal, None, N, g,
                                        kf0_pin=kf0_r)
        X_cal = proto.X_lin_from_slow(s_cal, g, N)
        c1_cal = proto.polyphase_py(X_cal, kf0_r, m_cal, g, window_full,
                                    n_sl)
        c0_cal = c0_all[:, np.asarray(m_cal) - g["ind_min_f"], :]
        r_ex, _, mask_cal = proto.ratio_dr(c1_cal, c0_cal, g["stride"], 0.0)
        rh_cal, _ = proto.fit_ratio(s_cal / s_ref, tau_slow, nr, "cubic",
                                    good=good_r)
        # calibration row = the row actually CARRYING the carrier power
        # (layer fraction ~1 puts it a row off the nominal center; a
        # center-row choice there has an empty mask -> crash)
        imc = int(np.argmax(np.abs(c0_cal).max(axis=(0, 2))))
        best_off, best_err = 0.0, np.inf
        mm = mask_cal[:, imc, :]
        if not mm.any():
            print("    [pix-off] SKIPPED (no valid pixels on max row); off=0")
        else:
            for off in (-1.0, -0.5, 0.0, 0.5, 1.0):
                tp = (n_global + off) * 256 * 10.0
                rh_v = rh_cal(tp)
                err = np.abs(rh_v - r_ex[:, imc, :])[mm].max()
                if err < best_err:
                    best_off, best_err = off, err
        tau_pix = (n_global + best_off) * 256 * 10.0
        print(f"    [pix-off] off={best_off:+.1f} err={best_err:.3e}")

        NR_II = [int(x) for x in
                 os.environ.get("TIER_NR_II", "16,32,64").split(",")]

        def rungii_delta(p, nr_ii):
            m_act = proto.m_active_for(p[1], g)
            s_c, _, _ = proto.slow_series(gen_l, p, None, N, g,
                                          kf0_pin=kf0_r)
            rh, _ = proto.fit_ratio(s_c / s_ref, tau_slow, nr_ii, "cubic",
                                    good=good_r)
            c0r = c0_all[:, np.asarray(m_act) - g["ind_min_f"], :]
            rh_v = rh(tau_pix)
            mx = np.abs(c0r).max(axis=-1, keepdims=True)
            mask = np.abs(c0r) > np.maximum(1e-12 * mx, 1e-300)
            r_ii = np.where(mask, rh_v[:, None, :], 0.0 + 0.0j)
            dr_ii = proto.fd_dr(r_ii, g["stride"])
            dh, hh = proto.fold_py(r_ii, dr_ii, m_act, sighet)
            return dh - 0.5 * hh

        # ---- sig-het v4: fixed-knot cardinal resampling + (optionally)
        # the moment-contraction algebra. Column computes r at pixels via
        # L @ r_nodes (identical fold path to rung-ii otherwise), so
        # v4_err = rung-ii_err + resampling(K). GATE V4-ALG (first ref):
        # the precomputed moment contraction must equal the explicit fold
        # to machine precision -- proves the analytic identity.
        from scipy.interpolate import CubicSpline as _CS
        V4_K = [int(x) for x in
                os.environ.get("TIER_V4_K", "64,128,256").split(",")]
        L_v4 = {}
        for K in V4_K:
            tau_k = np.linspace(0.0, g["Tobs"], K)
            Lm = _CS(tau_k, np.eye(K), axis=0)(tau_pix)     # (Nsp, K)
            L_v4[K] = (tau_k, np.ascontiguousarray(Lm))

        def v4_delta(p, K):
            m_act = proto.m_active_for(p[1], g)
            s_c, _, _ = proto.slow_series(gen_l, p, None, N, g,
                                          kf0_pin=kf0_r)
            rh, _ = proto.fit_ratio(s_c / s_ref, tau_slow, nr, "cubic",
                                    good=good_r)
            tau_k, Lm = L_v4[K]
            r_nodes = rh(tau_k)                              # (3, K)
            rh_pix = r_nodes @ Lm.T                          # (3, Nsp)
            c0r = c0_all[:, np.asarray(m_act) - g["ind_min_f"], :]
            mx = np.abs(c0r).max(axis=-1, keepdims=True)
            mask = np.abs(c0r) > np.maximum(1e-12 * mx, 1e-300)
            r_ii = np.where(mask, rh_pix[:, None, :], 0.0 + 0.0j)
            dr_ii = proto.fd_dr(r_ii, g["stride"])
            dh, hh = proto.fold_py(r_ii, dr_ii, m_act, sighet)
            return dh - 0.5 * hh

        if rrr == min(keep) and V4_K:
            # GATE V4-ALG: moment contraction == explicit fold (d_h only
            # needs A0/A1; h_h needs B0/B1(+nc); dr enters via the same
            # fd_dr applied to the cardinal columns -> fold the FD into L').
            K0 = V4_K[0]
            tau_k, Lm = L_v4[K0]
            m_act = proto.m_active_for(ref[1], g)
            ml_g = np.asarray(m_act) - g["ind_min_f"]
            c0r = c0_all[:, ml_g, :]
            mx = np.abs(c0r).max(axis=-1, keepdims=True)
            mask = np.abs(c0r) > np.maximum(1e-12 * mx, 1e-300)
            Ld = proto.fd_dr(np.broadcast_to(
                Lm.T[None, :, :], (1, K0, Lm.shape[0])).copy(),
                g["stride"])[0]                              # (K, Nsp) FD of columns
            A0 = np.asarray(sighet.A0_all)[0][:, ml_g, :]
            A1 = np.asarray(sighet.A1_all)[0][:, ml_g, :]
            r_test = (np.random.default_rng(3).standard_normal((3, K0))
                      + 1j * np.random.default_rng(4).standard_normal((3, K0)))
            rp = r_test @ Lm.T
            rmask = np.where(mask, rp[:, None, :], 0.0 + 0.0j)
            drmask = proto.fd_dr(rmask, g["stride"])
            dh_f, hh_f = proto.fold_py(rmask, drmask, m_act, sighet)
            # contraction: alpha_{c,k} with mask folded into per-(c,m)
            # effective L (mask varies by (c,m) row)
            dh_c = 0.0 + 0.0j
            for c in range(3):
                for im in range(len(ml_g)):
                    Lcm = np.where(mask[c, im, :][None, :].T, Lm, 0.0)
                    Lcmd = proto.fd_dr(
                        Lcm.T[None, :, :].copy(), g["stride"])[0]
                    al = A0[c, im, :] @ Lcm + A1[c, im, :] @ Lcmd.T
                    dh_c += al @ r_test[c]
            rel = abs(0.5 * dh_c.real - dh_f) / max(abs(dh_f), 1e-300)
            print(f"    [V4-ALG] d_h contraction vs fold rel diff = {rel:.3e}")

        def v2_delta(p):
            eng_s.get_ll(holder, p[None, :], phase_maximize=False, **kw)
            return float(eng_s.d_h_out[0] - 0.5 * eng_s.h_h_out[0])

        def rungi_delta(p):
            m_act = proto.m_active_for(p[1], g)
            s_c, _, _ = proto.slow_series(gen_l, p, None, N, g,
                                          kf0_pin=kf0_r)
            rh, _ = proto.fit_ratio(s_c / s_ref, tau_slow, nr, "cubic",
                                    good=good_r)
            X_u = proto.X_lin_from_slow(rh(tau_slow) * s_ref, g, N)
            c1_u = proto.polyphase_py(
                X_u, kf0_r, m_act, g,
                np.asarray(sighet.window_full, dtype=np.float64),
                np.asarray(sighet.n_sparse_local))
            c0r = c0_all[:, np.asarray(m_act) - g["ind_min_f"], :]
            r, dr, _ = proto.ratio_dr(c1_u, c0r, g["stride"], g["max_r"])
            dh, hh = proto.fold_py(r, dr, m_act, sighet)
            return dh - 0.5 * hh

        d0 = dense(ref)
        v0 = v2_delta(ref)
        i0 = rungi_delta(ref)
        ii0 = {nr_ii: rungii_delta(ref, nr_ii) for nr_ii in NR_II}
        v40 = {K: v4_delta(ref, K) for K in V4_K}
        print(f"\n[ref{rrr:02d}] f0={ref[1]*1e3:.4f} mHz min_env={min_env:.4f}"
              f"  ref-point offset: v2 {v0 - d0:+.2f} rung-i {i0 - d0:+.2f} "
              + " ".join(f"ii{k} {v - d0:+.3f}" for k, v in ii0.items()))

        for dname, idx, s0 in DIRS:
            s = s0
            dl = dense(displace(ref, idx, s)) - d0
            tries = 0
            while abs(dl) < 0.05 and tries < 3:
                s *= 10.0
                dl = dense(displace(ref, idx, s)) - d0
                tries += 1
            if abs(dl) < 0.05:
                continue
            for T in tiers:
                sc = s * np.sqrt(T / abs(dl))
                p = displace(ref, idx, sc)
                dT = dense(p) - d0
                if abs(dT) > 0.05:      # one refinement toward the tier
                    sc *= np.sqrt(T / abs(dT))
                    p = displace(ref, idx, sc)
                    dT = dense(p) - d0
                ev2 = abs((v2_delta(p) - v0) - dT)
                eri = abs((rungi_delta(p) - i0) - dT)
                eii = [abs((rungii_delta(p, nr_ii) - ii0[nr_ii]) - dT)
                       for nr_ii in NR_II]
                ev4 = [abs((v4_delta(p, K) - v40[K]) - dT) for K in V4_K]
                rows.append((rrr, min_env, dname, T, dT, ev2, eri,
                             *eii, *ev4))
                print(f"    {dname:4s} T={T:6.0f} trueDLL={dT:9.2f} "
                      f"dLL_v2={ev2:9.4f} dLL_rung-i={eri:9.4f} "
                      + " ".join(f"ii{nr_ii}={e:9.4f}"
                                 for nr_ii, e in zip(NR_II, eii))
                      + " " + " ".join(f"v4K{K}={e:9.4f}"
                                       for K, e in zip(V4_K, ev4)))

    ncols = len(rows[0])
    arr = np.array([(r[0], r[1], r[3], r[4], r[5], r[6], *r[7:ncols])
                    for r in rows])
    np.savez(os.path.join(out_dir, "tier_assess.npz"),
             rows=arr, dirs=np.array([r[2] for r in rows]))
    nr_ii_list = [int(x) for x in
                  os.environ.get("TIER_NR_II", "16,32,64").split(",")]
    v4_k_list = [int(x) for x in
                 os.environ.get("TIER_V4_K", "64,128,256").split(",")]
    cols = ([("v2", 4), ("rung-i", 5)]
            + [(f"ii{nr_ii}", 6 + k)
               for k, nr_ii in enumerate(nr_ii_list)]
            + [(f"v4K{K}", 6 + len(nr_ii_list) + k)
               for k, K in enumerate(v4_k_list)])
    print("\n[spec] per-tier worst / median over all (ref, dir):")
    for T in tiers:
        m = arr[:, 2] == T
        if not m.any():
            continue
        for lbl, col in cols:
            v = arr[m, col]
            allowed = max(0.1, T / 100.0)
            print(f"    T={T:6.0f} {lbl:6s}: median {np.median(v):9.4f} "
                  f"worst {v.max():9.4f}  vs allowed ~{allowed:.1f} "
                  f"-> pass {(v <= allowed).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
