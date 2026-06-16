"""Mask-aligned comparison of the TWO phentax methods that feed the two MBH
response paths, with the residual resolved in frequency and time:

  A feed (pyResponse):  compute_polarizations_at_once   -> h+, hx  (modes summed)
  B feed (on-the-fly):  compute_strain_components_amp_phase -> per-mode amp/phase

Both are generated on the SAME uniform grid (coarse_grain=False, delta_t=DT),
psi=0 (so compute_polarizations applies no rotation), then masked to the common
valid region.  Complex strains:
  H_A = h+ - i*hx            (phentax: h+=Re(sum sc), hx=-Im(sum sc))
  H_B = sum_modes amp*exp(i*phase)
If H_A == H_B the two phentax methods agree (-> the on-the-fly discrepancy is in
the response STRUCTURE: per-mode-project-then-sum).  If they differ, the bug is
in the phentax feed itself.

Outputs: overall mismatch, the noise-unweighted mismatch vs F_MIN (cumulative in
frequency), residual |H_A-aH_B| vs time, and a plot.  At natural inc + edge-on.
No orbit / response -> fast.
"""
import os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import phentax.waveform as pw
from lisatools.globalfit.recipe_steps import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container

MBHB_ID = int(os.environ.get("MBHB_ID", "0"))
DATA_CACHE = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"
HMS = (21, 33, 44); TOL = 1e-12; DT = 10.0; DUR = 5 * 86400.0


def banner(s): print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78, flush=True)


def feeds(g, m1, m2, s1z, s2z, dist, phi_ref, inc):
    """Return (t_valid, H_A, H_B) on the common masked uniform grid, psi=0."""
    tA, mA, hpA, hcA = g.compute_polarizations_at_once(
        m1, m2, s1z, s2z, dist, phi_ref, inc, 0.0, delta_t=DT, t_min=-DUR, t_ref=0.0)
    tB, mB, amp, ph = g.compute_strain_components_amp_phase(
        m1, m2, s1z, s2z, dist, phi_ref, inc, 0.0, delta_t=DT, t_min=-DUR, t_ref=0.0)
    tA = np.asarray(tA)[0]; mA = np.asarray(mA)[0].astype(bool)
    mB = np.asarray(mB)[0].astype(bool)
    hpA = np.asarray(hpA)[0]; hcA = np.asarray(hcA)[0]
    sc = np.asarray(amp)[0] * np.exp(1j * np.asarray(ph)[0])   # (nmodes, ntimes)
    H_full = sc.sum(axis=0)                                    # (ntimes,)
    mask = mA & mB                                             # common valid region
    H_A = (hpA - 1j * hcA)[mask]                               # h+ - i hx
    H_B = H_full[mask]
    return tA[mask], H_A, H_B


def main():
    z = np.load(DATA_CACHE, allow_pickle=True); cat = z["cat"].item()
    wf = np.asarray(make_mbh_transform_container().both_transforms(
        np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    m1, m2, s1z, s2z, dist, phi_ref = wf[0], wf[1], wf[2], wf[3], wf[4], wf[5]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for col, (lbl, inc) in enumerate([("natural inc=0.42", float(wf[6])), ("edge-on inc=pi/2", np.pi / 2)]):
        g = pw.IMRPhenomTHM(T=DUR, higher_modes=list(HMS), include_negative_modes=True,
                            t_low_fit=True, coarse_grain=False, atol=TOL, rtol=TOL)
        t, H_A, H_B = feeds(g, m1, m2, s1z, s2z, dist, phi_ref, inc)
        n = len(H_A)
        # best-fit complex scale alpha (removes overall amp+phase) and residual
        alpha = np.vdot(H_B, H_A) / np.vdot(H_B, H_B)
        resid = H_A - alpha * H_B
        # overall (white) mismatch
        mm = 1 - abs(np.vdot(H_A, H_B)) / np.sqrt(np.vdot(H_A, H_A).real * np.vdot(H_B, H_B).real)
        banner(f"{lbl}: phentax feed A vs B  (n={n})")
        print(f"  best-fit scale alpha = {alpha:.6f}  |alpha|={abs(alpha):.6f}", flush=True)
        print(f"  WHITE mismatch(H_A, H_B)          = {mm:.4e}", flush=True)
        print(f"  ||H_A - alpha*H_B|| / ||H_A||     = {np.linalg.norm(resid)/np.linalg.norm(H_A):.4e}", flush=True)

        # frequency-domain residual (uniform grid -> rfft)
        from scipy.signal.windows import tukey
        w = tukey(n, 0.1)
        f = np.fft.rfftfreq(n, DT)
        FA = np.fft.rfft(H_A * w); FB = np.fft.rfft(alpha * H_B * w)
        sel = f > 0
        # cumulative white mismatch as a function of low-frequency cutoff
        num = np.cumsum((np.conj(FA) * FB).real[::-1])[::-1]
        dA = np.cumsum((np.abs(FA) ** 2)[::-1])[::-1]
        dB = np.cumsum((np.abs(FB) ** 2)[::-1])[::-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            mm_cum = 1 - num / np.sqrt(dA * dB)
        print(f"  freq where cumulative-from-f mm crosses 1e-3: "
              f"{f[sel][np.argmax(mm_cum[sel] < 1e-3)] if np.any(mm_cum[sel]<1e-3) else float('nan'):.2e} Hz", flush=True)

        # ---- plots ----
        ax = axes[0, col]
        ax.loglog(f[sel], np.abs(FA[sel]), label="|H_A(f)| (compute_polarizations)")
        ax.loglog(f[sel], np.abs(FB[sel]), "--", label="|alpha H_B(f)| (strain_components)", alpha=.8)
        ax.loglog(f[sel], np.abs((FA - FB)[sel]), ":", color="crimson", label="|H_A - alpha H_B|")
        ax.set_title(f"{lbl}: FD strain residual"); ax.set_xlabel("f [Hz]"); ax.legend(fontsize=8)
        axb = axes[1, col]
        td = (t - t[-1]) / 3600.0   # hours from merger
        axb.plot(td, np.abs(H_A), label="|H_A|", lw=1.0)
        axb.plot(td, np.abs(alpha * H_B), "--", label="|alpha H_B|", lw=1.0, alpha=.8)
        axb.plot(td, np.abs(resid), ":", color="crimson", label="|H_A - alpha H_B|")
        axb.set_yscale("log"); axb.set_title(f"{lbl}: TD strain residual")
        axb.set_xlabel("hours from merger"); axb.legend(fontsize=8)

    fig.suptitle("phentax feed comparison: compute_polarizations vs strain_components (mask-aligned, psi=0)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"/tmp/mbh_phentax_feed_compare_id{MBHB_ID}.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\nDONE.  plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
