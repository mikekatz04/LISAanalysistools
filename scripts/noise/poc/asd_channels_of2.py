#!/usr/bin/env python
"""PROOF OF CONCEPT — the ASD figure, with the O(f^2) verdict and the T target.

Restyles ``asd_channels.py``'s cached curves (no brick re-pour) and adds the two
things the first version lacked:

1. **The missing component**, ``|data - model|``, in every panel. In X/Y/Z/A/E it
   is |noise| about zero -- the model is right there. In T it is the entire
   21% deficit, and it IS the foreground T response any fix has to produce, so
   plotting it turns "the model is wrong" into a target with a shape.

2. **What that shape has to be.** Three reference curves in the T panel, each
   the model's own foreground A-channel power rescaled and normalized to the
   missing component at 1 mHz:
       flat        a frequency-INDEPENDENT fractional leak, the signature of a
                   geometric (unequal-arm) break of the Sagnac null;
       (f/f*)^2    a finite-frequency leak at the order Mathur & Cornish compute;
       (f/f*)^4    the equal-arm null's own scaling (Hartwig et al. eq 35b,
                   C^TT = (16/3) sin^2(pi f L) sin^2(2 pi f L) C^zz).
   Whichever the missing component follows says which mechanism is responsible.

On the O(f^2) term itself: ``of2_null_test.py`` shows the leading-order null is
an exact identity, ``sum_n XY_lm = -1/2 sum_n XX_lm`` for EVERY sky multipole
(1.5e-15 over 437 phase points), holding only after the threefold channel sum --
not pointwise, which is why the per-pair correlations are still anisotropic. For
the T power to start at (fL)^4 as the equal-arm null requires, the O(f^2) term
must satisfy the same identity, i.e. contribute exactly nothing to T. So the
O(f^2) curve is drawn where it belongs: identically zero, same as the leading
order. Confirming it needs the appendix's long cross modes -- see that script.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

CHANNELS = ("X", "Y", "Z", "A", "E", "T")
FSTAR = 299792458.0 / (2.0 * np.pi * 2.5e9)  # 19.09 mHz

C_DATA = "#0b0b0b"
C_TOTAL = "#eb6834"
C_INST = "#2a78d6"
C_FG = "#1a9e77"
C_MISS = "#b5179e"
C_INK = "0.35"


def asd(power):
    return np.sqrt(np.clip(2.0 * power, 1e-300, None))


def visible(power, ref, floor=1e-8):
    out = asd(power)
    return np.where(np.asarray(power) > floor * float(np.max(ref)), out, np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cache", help="npz written by asd_channels.py --cache")
    p.add_argument("--band-lo", type=float, default=5e-4)
    p.add_argument("--band-hi", type=float, default=3e-3)
    p.add_argument("--norm-freq", type=float, default=1e-3,
                   help="frequency at which the T reference shapes are matched to "
                   "the missing component (default 1 mHz)")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    z = np.load(args.cache)
    f = z["f"]
    curves = {c: {k: z[f"{c}_{k}"] for k in ("data", "total", "inst", "fg")}
              for c in CHANNELS}

    band = (f >= args.band_lo) & (f <= args.band_hi)
    print(f"{args.cache}: {len(f)} layers, {f[0]:.3e}-{f[-1]:.3e} Hz")
    print(f"\n  {'chan':>5s} {'data/model':>11s} {'missing/model':>14s} "
          f"{'missing/fg_A':>13s}")
    fg_A = curves["A"]["fg"]
    for c in CHANNELS:
        cur = curves[c]
        miss = cur["data"] - cur["total"]
        ratio = np.mean((cur["data"] / cur["total"])[band])
        mrel = np.mean((miss / cur["total"])[band])
        mfg = np.mean((miss / fg_A)[band])
        print(f"  {c:>5s} {ratio:11.4f} {mrel:+14.4f} {mfg:+13.5f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), sharex=True)
    fmhz = f * 1e3

    for ax, c in zip(axes.ravel(), CHANNELS):
        cur = curves[c]
        miss = cur["data"] - cur["total"]
        ax.axvspan(args.band_lo * 1e3, args.band_hi * 1e3, color="0.85", alpha=0.35,
                   lw=0, zorder=0)
        ax.plot(fmhz, asd(cur["data"]), color=C_DATA, lw=1.6, label="data", zorder=6)
        ax.plot(fmhz, asd(cur["total"]), color=C_TOTAL, lw=1.4, label="model total")
        ax.plot(fmhz, visible(cur["inst"], cur["total"]), color=C_INST, lw=1.1, ls="--",
                label="instrument")

        if np.max(cur["fg"]) > 1e-12 * np.max(cur["total"]):
            ax.plot(fmhz, visible(cur["fg"], cur["total"]), color=C_FG, lw=1.6, ls="--",
                    label="foreground")
        else:
            ax.plot([], [], color=C_FG, lw=1.6, ls="--",
                    label=r"foreground $\equiv 0$  (all orders)")

        # |data - model|, not the positive part. Masking the negatives broke the
        # line into fragments wherever the residual crossed zero, which in the
        # five well-fitted channels is constantly -- and a dashed gap reads as
        # missing data rather than as "the residual changed sign". The absolute
        # value is continuous; on a log axis its zero crossings show as notches,
        # which is the honest picture of a residual consistent with noise.
        ax.plot(fmhz, visible(np.abs(miss), cur["total"], 1e-6),
                color=C_MISS, lw=2.0, label="|data $-$ model|", zorder=5)

        if c == "T":
            ref = np.interp(args.norm_freq, f, np.abs(miss))
            for power, style, lab in ((0, ":", "flat leak (unequal arms)"),
                                      (2, "-.", r"$(f/f_*)^2$  [O($f^2$) term]"),
                                      (4, (0, (5, 2)), r"$(f/f_*)^4$  [equal-arm null]")):
                shape = fg_A * (f / FSTAR) ** power
                shape = shape * ref / np.interp(args.norm_freq, f, shape)
                ax.plot(fmhz, asd(shape), color=C_INK, lw=1.0, ls=style, label=lab)
            ax.annotate("equal-arm model puts NO power here;\n"
                        "this one is the unequal-arm response",
                        xy=(0.97, 0.04), xycoords="axes fraction",
                        fontsize=9, color=C_MISS, va="bottom", ha="right", weight="bold")

        lo = min(np.nanmin(asd(cur["data"])), np.nanmin(asd(cur["total"])))
        hi = max(np.nanmax(asd(cur["data"])), np.nanmax(asd(cur["total"])))
        ax.set_ylim(0.35 * lo, 2.5 * hi)
        ax.set(xscale="log", yscale="log", title=f"channel {c}")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5, loc="upper left")

    for ax in axes[1]:
        ax.set_xlabel("frequency [mHz]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"ASD [Hz$^{-1/2}$]")

    fig.suptitle("foreground ASD over the data — top: TDI X/Y/Z, bottom: A/E/T.  "
                 "Magenta = |data − model|, i.e. what the model still fails to explain.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
