"""ONE figure that proves the GB likelihood engines on BOTH axes at once.

Measures accuracy and speed for chunked / v2 / v3 / v4-PCR / v4-banded in a
single run, on whatever backend it is given, and renders a six-panel figure:

  A  per-candidate cost vs Tobs          (does cost scale with observation?)
  B  speedup vs chunked                  (the operational number)
  C  tiered accuracy: |dLL| vs displacement, against the allowed(T) band
  D  the money plot: accuracy vs speed   (which engine dominates?)
  E  batch scaling                       (where does throughput saturate?)
  F  shared-memory footprint + occupancy (what limits the grid?)

ACCURACY REFERENCE = the chunked engine.  It was validated against full
time-domain truth (plain h+/hx at every dt through ResponseWrapper/TDI-2) at
3e-9 relative, so it is the trustworthy stand-in and avoids a dense einsum
per candidate.  Errors are DELTA-vs-DELTA from the reference point -- the
sampling-relevant quantity -- per the tiered accuracy spec:

    allowed(T) = max(0.1, T/100)   for a candidate whose true |dlnL| is T
    (T > 1000 is gated by the trust region, not required to be accurate)

Run (GPU):
    USE_GPU=1 GPU_BACKEND=cuda12x python gb_sighet_proof_figure.py
Run (CPU smoke, small):
    PROOF_NT_LIST=1024 PROOF_BATCHES=8 PROOF_NREF=1 python gb_sighet_proof_figure.py

Env: PROOF_NT_LIST ("2160,4320,8640" GPU / "1024" CPU), PROOF_BATCHES
     ("256,1024,4096" GPU / "8" CPU), PROOF_SPEED_NT (Tobs used for panels
     D/E, default the first of PROOF_NT_LIST), PROOF_NREF (4), PROOF_TIERS
     ("1,10,50,100,1000"), PROOF_NR (64), PROOF_K (128), PROOF_BAND (16),
     PROOF_REPS (5), ENV_OUT (./ratio_proto_out)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_sighet_speed_shootout as shoot          # scaffold + shmem formulas

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import WDMBandLikelihoodEngine

USE_GPU = shoot.USE_GPU
BACKEND = shoot.BACKEND

# Categorical hues in FIXED order, CVD-validated (adjacent-pair dE >= 8 under
# deuteranopia and protanopia, normal-vision dE >= 15).  Colour follows the
# engine, never its rank.
ENGINES = ["chunked", "v2", "v3", "v4-pcr", "v4-band", "v4-tuned"]
CLR = {"chunked": "#666666", "v2": "#E69F00", "v3": "#0072B2",
       "v4-pcr": "#009E73", "v4-band": "#D55E00", "v4-tuned": "#D55E00"}
# v4-tuned IS v4-banded with a T_obs-matched node count: same entity, so the
# same hue, distinguished by line style (never by a new colour).
LS = {n: "-" for n in ("chunked", "v2", "v3", "v4-pcr", "v4-band")}
LS["v4-tuned"] = "--"
MRK = {"chunked": "o", "v2": "s", "v3": "^", "v4-pcr": "D",
       "v4-band": "v", "v4-tuned": "*"}
INK, INK2, GRID = "#1a1a1a", "#555555", "#d8d8d8"


def asnp(a):
    g = getattr(a, "get", None)
    return g() if callable(g) else np.asarray(a)


def build_all(nt, nr, knots, band):
    """Scaffold + every engine at one Tobs (grid sized to the device).

    Adds "v4-tuned": v4-banded with the node count matched to the baseline
    by nr_law, so the figure shows BOTH the fixed-node line (flat cost, the
    scaling claim) and the tuned line (what the baseline actually needs).
    """
    ws, chunked, engines, Nf, dt, t0 = shoot.build(nt, nr, knots, band)
    from gbgpu.gbsignalhetcomputations import GBSignalHetComputations as _G
    ntl = engines["v3"]._g["nt_layer"]
    nr_t = nr_law(nt, Nf, dt, base=nr)
    engines["v4-tuned"] = _G.for_band_engine(
        chunked, n_sparse_fd=512, n_cp_build=93, nt_layer=ntl,
        m_active_half_width=2, v3_n_nodes=nr_t, v4_knots=knots,
        v4_band=band)
    print(f"[tuned] Nt={nt}: n_r {nr} -> {nr_t} "
          f"({nt*Nf*dt/86400/365.25:.2f} yr)")
    return ws, chunked, engines, Nf, dt, t0



def nr_law(nt, Nf=1440, dt=10.0, base=64):
    """Fit nodes predicted for this baseline.

    The ratio is structured by the ANNUAL modulation, so the node count a
    spline needs scales as T_obs / 1 yr (polynomial phase costs none).
    ``base`` is the value calibrated at 1 yr.  Used for the "v4-tuned"
    line: same engine as v4-banded, node count matched to the baseline
    instead of pinned at 64 -- which is where short observations get a
    near-linear speedup, since raw node evals dominate the cost.
    """
    return int(np.clip(round(base * nt * Nf * dt / 86400.0 / 365.25), 4, 256))


def make_refs(n):
    rng = np.random.default_rng(19)
    out = []
    for _ in range(n):
        out.append(np.array([
            10 ** rng.uniform(-22.5, -21.0), rng.uniform(1.5e-3, 1.5e-2),
            rng.uniform(0.0, 3e-16), 0.0, rng.uniform(0, 2 * np.pi),
            np.arccos(rng.uniform(-1, 1)), rng.uniform(0, np.pi),
            rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))]))
    return out


def main():
    out_dir = os.environ.get("ENV_OUT", "./ratio_proto_out")
    os.makedirs(out_dir, exist_ok=True)
    nt_list = [int(x) for x in os.environ.get(
        "PROOF_NT_LIST",
        "180,1080,2160,4320,8640" if USE_GPU else "1024").split(",")]
    batches = [int(x) for x in os.environ.get(
        "PROOF_BATCHES",
        "16,64,256,1024,4096,16384" if USE_GPU else "8").split(",")]
    speed_nt = int(os.environ.get("PROOF_SPEED_NT", str(nt_list[0])))
    n_ref = int(os.environ.get("PROOF_NREF", "4"))
    tiers = [float(x) for x in
             os.environ.get("PROOF_TIERS", "1,10,50,100,1000").split(",")]
    nr = int(os.environ.get("PROOF_NR", "64"))
    knots = int(os.environ.get("PROOF_K", "128"))
    band = int(os.environ.get("PROOF_BAND", "16"))
    reps = int(os.environ.get("PROOF_REPS", "5"))

    # ================= SPEED: per-candidate cost vs Tobs x batch ============
    speed = {}          # (nt, batch, engine) -> us/candidate
    shmem, occ, meta = {}, {}, {}
    for nt in nt_list:
        ws, chunked, engs, Nf, dt, t0 = build_all(nt, nr, knots, band)
        xp = chunked.xp
        ilo, ihi = ws.ind_min_f, ws.ind_max_f + 1
        ref = np.array([1e-22, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 0.4, 2.0, 0.3])
        href = xp.zeros((3, Nf, nt))
        chunked.fill_global_wdm(xp.asarray(ref)[None, :], href,
                                convert_to_ra_dec=False)
        h_act = xp.ascontiguousarray(href[:, ilo:ihi, ws.active_slice_t])
        invC = xp.zeros((3, 3) + h_act.shape[1:])
        for c in range(3):
            invC[c, c] = 1.0
        holder = shoot.XpGridWDMHolder(xp, h_act, invC)
        meta[nt] = dict(Tobs_d=nt * Nf * dt / 86400.0,
                        nsp=int(chunked_nsp(engs)), Nf=Nf)
        rng = np.random.default_rng(7)
        for nb in batches:
            cands = np.repeat(ref[None, :], nb, axis=0)
            cands[:, 0] *= np.exp(0.01 * rng.standard_normal(nb))
            cands[:, 5] += 0.01 * rng.standard_normal(nb)
            cands = xp.asarray(cands)
            z = np.zeros(nb, dtype=np.int32)
            kw = dict(data_index=z, noise_index=z, N_vals=None,
                      waveform_kwargs={})
            engc = WDMBandLikelihoodEngine(chunked, ws, nchannels=3,
                                           tdi_channel_setup="XYZ")
            speed[(nt, nb, "chunked")] = timeit(engc, holder, cands, kw, reps)
            for name, sh in engs.items():
                sh.clear_in_model()
                sh.setup_in_model(holder, xp.asarray(ref)[None, :],
                                  np.zeros(1, np.int32))
                e = WDMBandLikelihoodEngine(sh, ws, nchannels=3,
                                            tdi_channel_setup="XYZ")
                try:
                    speed[(nt, nb, name)] = timeit(e, holder, cands, kw, reps)
                except Exception as exc:                     # noqa: BLE001
                    print(f"  [{name}] nt={nt} nb={nb} FAILED: {exc}")
                    speed[(nt, nb, name)] = np.nan
            print(f"[speed] Nt={nt} batch={nb}: " + " ".join(
                f"{n}={speed[(nt, nb, n)]:.1f}us" for n in ENGINES))
        # footprints at this grid
        g0 = list(engs.values())[0]._g
        nsp, ntl = g0["N_sparse_t"], g0["nt_layer"]
        shmem[nt] = {
            "v2": shoot.v2_shared_bytes(3, 2, ntl, nsp, 512),
            "v3": shoot.sighet_shared_bytes(nr, knots, 3, 2, nsp, 0, v4=False),
            "v4-pcr": shoot.sighet_shared_bytes(nr, knots, 3, 2, nsp, 0,
                                                v4=True),
            "v4-band": shoot.sighet_shared_bytes(nr, knots, 3, 2, nsp, 16,
                                                 v4=True)}
        lim = shoot.device_shared_limit()
        occ[nt] = {k: int(lim // v) for k, v in shmem[nt].items()}

    # ================= ACCURACY: tiered spec vs the chunked reference ======
    # Measured at EVERY Tobs: the shared-memory cap holds N_sparse_t roughly
    # fixed, so the sparse-sampling interval grows with Tobs and the fold's
    # interpolation error grows with it -- the accuracy cost of a longer
    # baseline is exactly what this sweep exposes.
    err_nt = {}
    for acc_nt in nt_list:
        ws, chunked, engs, Nf, dt, t0 = build_all(acc_nt, nr, knots, band)
        err_nt[acc_nt] = accuracy_at(acc_nt, ws, chunked, engs, Nf, dt,
                                     tiers, n_ref)
        print(f"[acc] Nt={acc_nt} done: " + " ".join(
            f"{n}={np.median(err_nt[acc_nt][n][tiers[-2]]):.2e}"
            for n in ENGINES[1:]))
    err = err_nt[speed_nt]

    # ---- persist BOTH axes, not just the figure ------------------------
    # speed_rows : (nt, batch, engine_index, us_per_candidate)
    # err_rows   : (nt, engine_index, tier, error)  -- every measurement,
    #              so the accuracy claims are reproducible from the file
    #              alone and can be re-binned without re-running.
    speed_rows = np.array([[nt, nb, ENGINES.index(n), speed[(nt, nb, n)]]
                           for nt in nt_list for nb in batches
                           for n in ENGINES], dtype=float)
    err_rows = np.array(
        [[nt, ENGINES.index(n), T, e]
         for nt in nt_list for n in ENGINES[1:] for T in tiers
         for e in err_nt[nt][n][T]], dtype=float) if err_nt else np.zeros((0, 4))
    np.savez(os.path.join(out_dir, "proof_figure.npz"),
             speed=speed_rows, err=err_rows, engines=np.array(ENGINES),
             tiers=np.array(tiers), nt_list=np.array(nt_list),
             batches=np.array(batches))

    # ---- printed accuracy table (the log carries the numbers too) -------
    print("\n[ACCURACY] worst / median |dlnL| vs the chunked reference"
          "   (allowed = max(0.1, T/100))")
    hdr = "  Tobs[yr] engine     " + "".join(f"{('T=' + str(int(T))):>18s}"
                                             for T in tiers)
    print(hdr)
    for nt in nt_list:
        for n in ENGINES[1:]:
            cells = ""
            for T in tiers:
                v = err_nt[nt][n][T]
                if v:
                    ok = "OK " if max(v) <= max(0.1, T / 100.0) else "OVER"
                    cells += f"{max(v):8.2e}/{np.median(v):7.1e}{ok:>3s}"
                else:
                    cells += f"{'--':>18s}"
            print(f"  {meta[nt]['Tobs_d']/365.25:7.2f}  {n:10s}{cells}")
    print("\n[SPEED] us/candidate at batch " + str(batches[-1]))
    print("  Tobs[yr] " + "".join(f"{n:>12s}" for n in ENGINES))
    for nt in nt_list:
        print(f"  {meta[nt]['Tobs_d']/365.25:7.2f} "
              + "".join(f"{speed[(nt, batches[-1], n)]:12.1f}"
                        for n in ENGINES))
    make_figure(out_dir, nt_list, batches, speed_nt, tiers, speed, err,
                err_nt, shmem, occ, meta, nr, knots, band)



def accuracy_at(nt, ws, chunked, engs, Nf, dt, tiers, n_ref):
    """Tiered |dlnL| errors vs the chunked reference at one Tobs."""
    xp = chunked.xp
    ilo, ihi = ws.ind_min_f, ws.ind_max_f + 1
    err = {n: {T: [] for T in tiers} for n in ENGINES[1:]}
    drift = 1.0 / (2 * np.pi * (nt * Nf * dt))
    DIRS = [("f0", 1, 0.05 * drift), ("lnA", 0, 0.03), ("iota", 5, 0.03),
            ("psi", 6, 0.03), ("lam", 7, 0.01)]
    for ref in make_refs(n_ref):
        href = xp.zeros((3, Nf, nt))
        chunked.fill_global_wdm(xp.asarray(ref)[None, :], href,
                                convert_to_ra_dec=False)
        h_act = xp.ascontiguousarray(href[:, ilo:ihi, ws.active_slice_t])
        invC = xp.zeros((3, 3) + h_act.shape[1:])
        for c in range(3):
            invC[c, c] = 1.0
        holder = shoot.XpGridWDMHolder(xp, h_act, invC)
        engc = WDMBandLikelihoodEngine(chunked, ws, nchannels=3,
                                       tdi_channel_setup="XYZ")
        eng_of = {}
        for name, sh in engs.items():
            sh.clear_in_model()
            sh.setup_in_model(holder, xp.asarray(ref)[None, :],
                              np.zeros(1, np.int32))
            eng_of[name] = WDMBandLikelihoodEngine(sh, ws, nchannels=3,
                                                   tdi_channel_setup="XYZ")

        def delta(engine, p):
            z = np.zeros(1, dtype=np.int32)
            engine.get_ll(holder, xp.asarray(p)[None, :], phase_maximize=False,
                          data_index=z, noise_index=z, N_vals=None,
                          waveform_kwargs={})
            return (float(asnp(engine.d_h_out)[0])
                    - 0.5 * float(asnp(engine.h_h_out)[0]))

        d0 = {"chunked": delta(engc, ref)}
        for name in eng_of:
            d0[name] = delta(eng_of[name], ref)
        for _dn, idx, s0 in DIRS:
            s = s0
            dl = delta(engc, disp(ref, idx, s)) - d0["chunked"]
            tries = 0
            while abs(dl) < 0.05 and tries < 3:
                s *= 10.0
                dl = delta(engc, disp(ref, idx, s)) - d0["chunked"]
                tries += 1
            if abs(dl) < 0.05:
                continue
            for T in tiers:
                sc = s * np.sqrt(T / abs(dl))
                p = disp(ref, idx, sc)
                dT = delta(engc, p) - d0["chunked"]
                if abs(dT) > 0.05:
                    sc *= np.sqrt(T / abs(dT))
                    p = disp(ref, idx, sc)
                    dT = delta(engc, p) - d0["chunked"]
                for name in eng_of:
                    err[name][T].append(abs((delta(eng_of[name], p)
                                             - d0[name]) - dT))
    return err


def chunked_nsp(engs):
    return list(engs.values())[0]._g["N_sparse_t"]


def disp(ref, idx, s):
    p = ref.copy()
    if idx == 0:
        p[0] *= np.exp(s)
    else:
        p[idx] += s
    return p


def timeit(engine, holder, cands, kw, reps):
    engine.get_ll(holder, cands, phase_maximize=False, **kw)
    shoot.sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        engine.get_ll(holder, cands, phase_maximize=False, **kw)
        shoot.sync()
        ts.append(time.perf_counter() - t)
    return min(ts) / cands.shape[0] * 1e6


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def make_figure(out_dir, nt_list, batches, speed_nt, tiers, speed, err,
                err_nt, shmem, occ, meta, nr, knots, band):
    nb_ref = batches[-1]
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26,
                          left=0.055, right=0.985, top=0.885, bottom=0.075)
    tobs = [meta[nt]["Tobs_d"] / 365.25 for nt in nt_list]

    # --- A: cost vs Tobs --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    for n in ENGINES:
        y = [speed[(nt, nb_ref, n)] for nt in nt_list]
        ax.plot(tobs, y, marker=MRK[n], color=CLR[n], lw=2, ms=7,
                ls=LS.get(n, "-"), label=n)
        ax.annotate(f"{y[-1]:.0f}", (tobs[-1], y[-1]), color=INK,
                    fontsize=8, xytext=(6, 0), textcoords="offset points",
                    va="center")
    ax.set_yscale("log")
    ax.set_xlabel("observation time [yr]")
    ax.set_ylabel("per-candidate cost [$\\mu$s]")
    ax.set_title("A · cost vs observation time", color=INK, fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2)

    # --- B: speedup vs chunked -------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    for n in ENGINES[1:]:
        y = [speed[(nt, nb_ref, "chunked")] / speed[(nt, nb_ref, n)]
             for nt in nt_list]
        ax.plot(tobs, y, marker=MRK[n], color=CLR[n], lw=2, ms=7,
                ls=LS.get(n, "-"), label=n)
        ax.annotate(f"{y[-1]:.0f}$\\times$", (tobs[-1], y[-1]), color=INK,
                    fontsize=8, xytext=(6, 0), textcoords="offset points",
                    va="center")
    ax.axhline(1.0, color=CLR["chunked"], lw=1.5, ls="--")
    ax.set_xlabel("observation time [yr]")
    ax.set_ylabel("speedup vs chunked-het")
    ax.set_title("B · the heterodyne payoff grows with $T_{obs}$", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    # --- C: tiered accuracy ----------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); style(ax)
    allowed = [max(0.1, T / 100.0) for T in tiers]
    ax.fill_between(tiers, 1e-8, allowed, color="#2f7d4f", alpha=0.10,
                    zorder=0)
    ax.plot(tiers, allowed, color="#2f7d4f", lw=1.6, ls="--",
            label="allowed(T)")
    for n in ENGINES[1:]:
        med = [np.median(err[n][T]) if err[n][T] else np.nan for T in tiers]
        wor = [np.max(err[n][T]) if err[n][T] else np.nan for T in tiers]
        ax.plot(tiers, med, marker=MRK[n], color=CLR[n], lw=2, ms=6,
                ls=LS.get(n, "-"), label=n)
        ax.plot(tiers, wor, color=CLR[n], lw=1, ls=":", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true $|\\Delta\\ln L|$ from reference,  T")
    ax.set_ylabel("likelihood error  $|\\delta\\ln L|$")
    ax.set_title(f"C · tiered accuracy @ {meta[speed_nt]['Tobs_d']/365.25:.2f} yr "
                 "(solid median, dotted worst)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    # --- D: accuracy-vs-cost trajectory across Tobs ----------------------
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    Tm = min(tiers, key=lambda t: abs(t - 100.0))
    for n in ENGINES[1:]:
        xs = [speed[(nt, nb_ref, n)] for nt in nt_list]
        ys = [np.max(err_nt[nt][n][Tm]) if err_nt[nt][n][Tm] else np.nan
              for nt in nt_list]
        ax.plot(xs, ys, color=CLR[n], lw=1.6, alpha=0.75, zorder=2)
        ax.scatter(xs, ys, s=[26 + 20 * k for k in range(len(xs))],
                   color=CLR[n], marker=MRK[n], edgecolor="white",
                   linewidth=1.2, zorder=3, label=n)
    ax.axhline(max(0.1, Tm / 100.0), color="#2f7d4f", lw=1.6, ls="--")
    ax.annotate(f"allowed at T={Tm:.0f}", (ax.get_xlim()[0],
                                           max(0.1, Tm / 100.0)),
                color="#2f7d4f", fontsize=8, xytext=(4, 4),
                textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("per-candidate cost [$\\mu$s]  (lower better)")
    ax.set_ylabel(f"worst error at T={Tm:.0f}  (lower better)")
    ax.set_title("D · accuracy vs cost, marker size = $T_{obs}$",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    # --- E: throughput saturation (fraction of asymptotic rate) ----------
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    for n in ENGINES:
        y = np.array([speed[(speed_nt, nb, n)] for nb in batches], float)
        eff = np.nanmin(y) / y                      # 1.0 = saturated
        ax.plot(batches, 100 * eff, marker=MRK[n], color=CLR[n], lw=2,
                ms=7, ls=LS.get(n, "-"), label=n)
        sat = next((b for b, e in zip(batches, eff) if e >= 0.95), None)
        if sat is not None:
            ax.axvline(sat, color=CLR[n], lw=0.8, ls=":", alpha=0.5)
    ax.axhline(95, color=INK2, lw=1.0, ls="--")
    ax.annotate("95% of peak", (batches[0], 95), color=INK2, fontsize=8,
                xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("candidates per call")
    ax.set_ylabel("% of peak throughput")
    ax.set_ylim(0, 108)
    ax.set_title("E · GPU saturation (dotted = 95% point)", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2,
              loc="lower right")

    # --- F: shared memory + occupancy ------------------------------------
    ax = fig.add_subplot(gs[1, 2]); style(ax)
    nt0 = nt_list[0]
    names = [n for n in ENGINES[1:]]
    vals = [shmem[nt0][n] / 1024 for n in names]
    bars = ax.bar(range(len(names)), vals, color=[CLR[n] for n in names],
                  width=0.62)
    lim = shoot.device_shared_limit() / 1024
    ax.axhline(lim, color="#b3261e", lw=1.6, ls="--")
    ax.annotate(f"device limit {lim:.0f} KB", (len(names) - 0.5, lim),
                color="#b3261e", fontsize=8, ha="right",
                xytext=(0, 4), textcoords="offset points")
    for b, n in zip(bars, names):
        ax.annotate(f"{occ[nt0][n]} blk/SM", (b.get_x() + b.get_width() / 2,
                                              b.get_height()),
                    color=INK, fontsize=8, ha="center",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, color=INK2)
    ax.set_ylabel("scorer shared memory [KB]")
    ax.set_title("F · what limits the grid (and occupancy)", color=INK,
                 fontsize=10, loc="left")

    nsp = meta[nt0]["nsp"]
    fig.suptitle(
        "GB likelihood engines: accuracy and speed in one measurement",
        fontsize=14, color=INK, x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.925,
             f"{BACKEND} · WDM Nf={meta[nt0]['Nf']}, dt=10 s · "
             f"{nr} fit nodes, K={knots} knots, band={band} · "
             f"N_sparse_t={nsp} · batch={nb_ref} · "
             f"accuracy reference = chunked-het (validated to 3e-9 vs "
             f"full time-domain truth)",
             fontsize=9, color=INK2, ha="left")
    fp = os.path.join(out_dir, "gb_engine_proof.png")
    fig.savefig(fp, dpi=160, facecolor="white")
    print(f"\n[out] {fp}")


if __name__ == "__main__":
    main()
