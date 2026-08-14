#!/usr/bin/env python
"""Whiten a noise run's own data at parameter sets you name, and compare them.

    python scripts/noise/whitening_test.py noise-galfor-pe2/noise_foreground_try3_testing.h5 \
        --unequal-arm --wdm-psd-method layer_calibrated --modulation scripts/noise/modulation_multi.dat \
        --params "notebook:amp=1.18955159143e-44,fk=2.10304500452e-3,alpha=3.39975538551,f_1=2.46392506051e-3,f_2=0.989314627841e-3"

``ppc_noise.py`` overlays draws from the posterior a run FOUND. This asks the
prior question: does a parameter set from somewhere else -- a Fourier-domain
least-squares fit, an injection, a neighbouring run's median -- whiten this
run's data better than the run's own posterior does? If it does, the run has a
sampling or prior problem rather than a model problem, and the two are worth
separating before touching the model.

The data, the grid and the noise model are the run's own (same builder as
``ppc_noise``, same flags), so the only thing that varies between the sets
being compared is the parameter vector. Three statistics, all of ``w**2 / C``
over the run's active pixels:

    chi2/pix   the mean. 1 under a correct model, with sd sqrt(2/Npix) -- the
               single number for "is the overall level right".
    ratio      per-layer time-average of w**2 over the same of C, against the
               estimator's own 1-sigma band. This is the panel that localizes a
               shape error in frequency; an overall normalization error moves
               every layer together instead.
    whitened   the histogram of w/sqrt(C) against N(0,1). It is the weakest of
               the three -- it marginalizes over frequency, so a 20% variance
               error confined to a few layers barely widens it -- and it is
               here precisely because it is the test that says "fine" when the
               ratio panel says otherwise.

``--posterior-median`` (default on when the file carries a chain) adds the run's
own posterior median as a set, which is what everything else is compared to.
Unnamed psd values fall back to the injection ``[Soms_d, Sa_a] = [1.5e-11,
3e-15]``, so a galfor-only set is meaningful on its own.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corner_noise  # noqa: E402
import ppc_noise  # noqa: E402
import run_noise_only  # noqa: E402

# The order ``HyperbolicTangentGalacticForeground.specific_Sh_function`` takes,
# NOT the notebook's (amp, alpha, fk, ...) argument order -- map by name.
GALFOR_BASIS = ("amp", "fk", "alpha", "f_1", "f_2")
PSD_BASIS = ("Soms_d", "Sa_a")
PSD_INJECTION = (1.5e-11, 3e-15)

C_SETS = ["#2a78d6", "#eb6834", "#4a3aa7", "#1a9e77"]  # validated categorical slots
C_INK = "0.25"


def parse_param_set(spec):
    """``"name:k=v,k=v"`` -> ``(name, {k: float(v)})``."""
    if ":" not in spec:
        raise SystemExit(f"--params needs a NAME: prefix; got {spec!r}")
    name, _, body = spec.partition(":")
    out = {}
    for item in body.split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in GALFOR_BASIS + PSD_BASIS:
            raise SystemExit(
                f"unknown parameter {key!r}; expected one of "
                f"{', '.join(GALFOR_BASIS + PSD_BASIS)}"
            )
        out[key] = float(value)
    return name.strip(), out


def physical_vectors(values, mode):
    """``(psd_physical, galfor_physical_or_None)`` from a name->value dict."""
    psd = np.array([values.get(k, d) for k, d in zip(PSD_BASIS, PSD_INJECTION)], dtype=float)
    if mode == "instrument":
        return psd, None
    missing = [k for k in GALFOR_BASIS if k not in values]
    if missing:
        raise SystemExit(
            f"a foreground run needs all five galfor parameters; missing {', '.join(missing)}"
        )
    return psd, np.array([values[k] for k in GALFOR_BASIS], dtype=float)


def full_log_like(sens, data_arr):
    """Gaussian log-likelihood with the FULL channel covariance, up to a constant.

    ``chi2/pix`` and the ratio panel both read only ``C[i, i]`` -- they are
    per-channel statistics. The likelihood the run actually samples uses the
    whole 3x3 pixel covariance, cross-spectra included, and the unequal-arm
    model's off-diagonals are not a free ride: a parameter set can whiten every
    diagonal and still be beaten on the off-diagonals. So the comparison needs
    this number too, or "whitens better" is only a claim about the diagonal.

    ``-0.5 * (w^T C^-1 w + ln det C)``, dropping the parameter-independent
    ``-0.5 N ln(2 pi)``; only differences between sets are meaningful.
    """
    from lisatools.utils.utility import asnumpy

    C = np.asarray(asnumpy(sens.sens_mat))  # (nch, nch, Nf, Nt)
    nch = C.shape[0]
    good = np.isfinite(C).all(axis=(0, 1)) & (np.real(C[0, 0]) > 0)
    Cm = np.moveaxis(C[:, :, good], (0, 1), (-2, -1))  # (Npix, nch, nch)
    w = np.moveaxis(np.asarray(asnumpy(data_arr))[:, good], 0, -1)  # (Npix, nch)
    _, logabsdet = np.linalg.slogdet(Cm)
    # numpy's batched solve wants a trailing column, not a vector: (P,n,n) x
    # (P,n,1). Passing (P,n) is read as a single (n,n) rhs and fails on P != n.
    solved = np.linalg.solve(Cm, w[..., None])[..., 0]
    quad = np.real(np.einsum("pi,pi->p", np.conj(w), solved))

    # The same likelihood with the cross-spectra DELETED. Comparing the two
    # splits any preference into "fits the per-channel PSDs" and "fits the
    # channel correlations": a set that wins on diag and loses on full is being
    # rejected by the off-diagonals alone, which no per-channel statistic sees.
    diag = np.real(np.einsum("pii->pi", Cm))
    quad_d = np.einsum("pi,pi->p", w**2, 1.0 / diag)
    logdet_d = np.log(diag).sum(axis=-1)

    return (
        dict(
            full=float(-0.5 * (quad.sum() + logabsdet.sum())),
            diag=float(-0.5 * (quad_d.sum() + logdet_d.sum())),
            quad=float(quad.sum()),
            logdet=float(logabsdet.sum()),
        ),
        int(good.sum()),
    )


def reduce_one(general_info, backend, name, psd_params, galfor_params, channels, smooth):
    """``w**2`` vs ``C`` reductions for ONE parameter set, per channel."""
    from lisatools.utils.utility import asnumpy

    kwargs = {} if galfor_params is None else dict(galfor_params=galfor_params)
    sens = backend(f"whiten_{name}", psd_params, **kwargs)
    data_arr = asnumpy(general_info.input_data_residual_array.data_res_arr.arr)
    ll, npix_full = full_log_like(sens, data_arr)

    out = {"_ll": ll, "_ll_npix": npix_full}
    for ch in channels:
        C = ppc_noise.covariance_channel(sens, ch)
        keep = np.isfinite(C).all(axis=1) & (C > 0).all(axis=1)
        w, C = data_arr[ch][keep], C[keep]
        layer = ppc_noise.freq_stat(C)
        whitened = (w / np.sqrt(C)).ravel()
        chi2 = float(np.mean(w**2 / C))
        out[ch] = dict(
            keep=keep,
            npix=w.size,
            chi2=chi2,
            chi2_z=(chi2 - 1.0) / np.sqrt(2.0 / w.size),
            model_freq=layer,
            model_time=ppc_noise.time_stat(C, layer, smooth),
            sigma_freq=ppc_noise.freq_sigma(C),
            sigma_time=ppc_noise.time_sigma(C, layer, smooth),
            data_freq=ppc_noise.freq_stat(w**2),
            data_time=ppc_noise.time_stat(w**2, layer, smooth),
            whitened_sd=float(np.std(whitened)),
            whitened_hist=np.histogram(whitened, bins=np.linspace(-6, 6, 121), density=True),
        )
    return out


def plot(results, channels, f_arr, t_arr, title, path):
    """Ratio / time / whitened-histogram rows, one column per channel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    names = list(results)
    ncol = len(channels)
    fig, axes = plt.subplots(3, ncol, figsize=(5.6 * ncol, 11), squeeze=False)

    for j, ch in enumerate(channels):
        keep = results[names[0]][ch]["keep"]
        f = np.asarray(f_arr)[keep] * 1e3
        t = np.asarray(t_arr) / 86400.0

        ax = axes[0][j]
        for k, name in enumerate(names):
            o = results[name][ch]
            ax.plot(f, o["data_freq"] / o["model_freq"], color=C_SETS[k % len(C_SETS)], lw=1.2,
                    label=f"{name}  (chi2/pix {o['chi2']:.4f}, z {o['chi2_z']:+.1f})")
            ax.fill_between(f, 1 - 1.645 * o["sigma_freq"], 1 + 1.645 * o["sigma_freq"],
                            color=C_SETS[k % len(C_SETS)], alpha=0.12, lw=0)
        ax.axhline(1.0, color=C_INK, ls="--", lw=0.9)
        ax.set(xscale="log", xlabel="frequency [mHz]", ylabel="data / model, per layer",
               title=f"channel {'XYZ'[ch]}")
        # Decade ticks with 2/5 subdivisions and NO minor labels: the default
        # log formatter writes both and they collide at this figure width.
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        span = max(
            np.nanmax(np.abs(results[n][ch]["data_freq"] / results[n][ch]["model_freq"] - 1.0))
            for n in names
        )
        ax.set_ylim(1 - min(span, 0.35) * 1.15, 1 + min(span, 0.35) * 1.15)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

        ax = axes[1][j]
        for k, name in enumerate(names):
            o = results[name][ch]
            ax.plot(t, o["data_time"] / o["model_time"], color=C_SETS[k % len(C_SETS)], lw=0.6)
        ax.axhline(1.0, color=C_INK, ls="--", lw=0.9)
        ax.set(xlabel="time from data start [days]", ylabel="broadband power / model")
        ax.grid(alpha=0.25)

        ax = axes[2][j]
        grid = np.linspace(-6, 6, 400)
        for k, name in enumerate(names):
            counts, edges = results[name][ch]["whitened_hist"]
            centers = 0.5 * (edges[1:] + edges[:-1])
            ax.plot(centers, counts, color=C_SETS[k % len(C_SETS)], lw=1.2,
                    label=f"{name}  (sd {results[name][ch]['whitened_sd']:.4f})")
        ax.plot(grid, np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi), color=C_INK, ls="--", lw=1.0,
                label="N(0,1)")
        ax.set(yscale="log", ylim=(1e-5, 1.0), xlabel=r"$w/\sqrt{C}$", ylabel="density")
        ax.legend(fontsize=8, loc="lower center")
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", help="run HDF5 -- supplies the grid, the data and the mode")
    p.add_argument("--params", action="append", default=[], metavar="NAME:k=v,...",
                   help="a named parameter set (repeatable). Keys: "
                   "amp, fk, alpha, f_1, f_2, Soms_d, Sa_a")
    p.add_argument("--discard", type=int, default=0, help="burn-in dropped for the median set")
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--no-posterior-median", dest="posterior_median", action="store_false",
                   help="omit the run's own posterior median from the comparison")
    p.add_argument("--channel", default="0",
                   help="TDI channel index, a comma-separated list, or 'all' (default 0 = X)")
    p.add_argument("--time-smooth", type=int, default=24,
                   help="running-mean width in wavelet columns for the time panel (default 24)")
    p.add_argument("--noise-file", default=run_noise_only.NOISE_FILE)
    p.add_argument("--galfor-file", default=run_noise_only.GALFOR_FILE)
    p.add_argument("--modulation", nargs="?", const=run_noise_only.MODULATION_FILE, default=None,
                   metavar="PATH", help="pass exactly what the run was given")
    p.add_argument("--unequal-arm", action="store_true", help="pass it iff the run did")
    p.add_argument("--wdm-psd-method", choices=("fold", "layer_constant", "layer_calibrated"),
                   default="fold", help="pass what the run used (see ppc_noise.py)")
    p.add_argument("--two-years", action="store_true",
                   help="force the full-brick grid; normally inferred from the stored Nt")
    p.add_argument("--grid", default="auto", choices=("auto", "lite", "full"))
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--scratch-dir", default="./gf_output_ppc/")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("-o", "--out", help="output png (default: <file>_whiten.png)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    branches = ppc_noise.chain_branches(args.file)
    if "psd" not in branches:
        raise SystemExit(f"no psd branch in {args.file}; found {branches}")
    mode = "foreground" if "galfor" in branches else "instrument"
    branches = ["psd"] + (["galfor"] if mode == "foreground" else [])

    inferred_full, inferred_two_years = ppc_noise.resolve_grid(
        args.file, args.grid, args.noise_file
    )
    args.full = inferred_full if args.grid == "auto" else args.grid == "full"
    args.two_years = args.two_years or inferred_two_years

    read_kwargs = dict(discard=args.discard, thin=args.thin)
    flats = {b: corner_noise.load_samples(args.file, b, mask=False, **read_kwargs)
             for b in branches}
    log_sampling = {b: corner_noise.resolve_basis(flats[b]) == "log" for b in branches}

    sets = [parse_param_set(s) for s in args.params]
    if not sets and not args.posterior_median:
        raise SystemExit("nothing to compare: pass --params or keep the posterior median.")

    print(f"{args.file}")
    print(f"  mode            {mode}  (from the chain's branches)")
    print(f"  grid            {'two-year brick' if args.two_years else ('full' if args.full else 'lite')}"
          f" (nt={ppc_noise.stored_nt(args.file)})")
    print(f"  instrument      {'unequal-arm' if args.unequal_arm else 'equal-arm'}"
          + (f", wdm_psd_method={args.wdm_psd_method}" if args.unequal_arm else ""))
    print(f"  galfor mod.     {args.modulation or 'stationary'}")
    print("\nbuilding the run's data + noise model (loads and re-pours the bricks)...", flush=True)

    general_info, source_info = ppc_noise.build_general_and_sources(args, mode, log_sampling)
    settings = general_info.domain_settings
    backend = general_info.sensitivity_backend
    transforms = {b: getattr(s, "transform", None) for b, s in source_info.items()}

    nch = general_info.input_data_residual_array.data_res_arr.arr.shape[0]
    channels = list(range(nch)) if args.channel == "all" else [
        int(c) for c in args.channel.split(",") if c.strip()
    ]
    for ch in channels:
        if not 0 <= ch < nch:
            raise SystemExit(f"channel {ch} out of range; the data has {nch}")

    ordered = []
    if args.posterior_median:
        med = {b: np.median(flats[b], axis=0) for b in branches}
        psd_med = ppc_noise.to_physical(transforms["psd"], med["psd"])
        gal_med = (None if mode == "instrument"
                   else ppc_noise.to_physical(transforms["galfor"], med["galfor"]))
        ordered.append(("run posterior median", psd_med, gal_med))
    for name, values in sets:
        psd_p, gal_p = physical_vectors(values, mode)
        ordered.append((name, psd_p, gal_p))

    print(f"\ngrid {settings.Nf_active} layers x {settings.Nt_active} columns, "
          f"{settings.f_arr[0]:.2e}-{settings.f_arr[-1]:.2e} Hz, {settings.Tobs / 86400:.2f} d")
    print("\nparameter sets")
    for name, psd_p, gal_p in ordered:
        print(f"  {name}")
        print(f"    psd    " + "  ".join(f"{k}={v:.6g}" for k, v in zip(PSD_BASIS, psd_p)))
        if gal_p is not None:
            print(f"    galfor " + "  ".join(f"{k}={v:.6g}" for k, v in zip(GALFOR_BASIS, gal_p)))

    results = {}
    for name, psd_p, gal_p in ordered:
        print(f"\nwhitening at {name!r}...", flush=True)
        results[name] = reduce_one(
            general_info, backend, name.replace(" ", "_"), psd_p, gal_p,
            channels, args.time_smooth,
        )

    print(f"\n{'set':24s} {'ch':>3s} {'chi2/pix':>10s} {'z':>8s} {'freq 90%':>9s} "
          f"{'time 90%':>9s} {'sd(w/sqrtC)':>12s}")
    for name in results:
        for ch in channels:
            o = results[name][ch]
            cov_f = ppc_noise.within(o["data_freq"] / o["model_freq"], 1.0, o["sigma_freq"])
            cov_t = ppc_noise.within(o["data_time"], o["model_time"], o["sigma_time"])
            print(f"  {name[:22]:22s} {'XYZ'[ch]:>3s} {o['chi2']:10.4f} {o['chi2_z']:+8.1f} "
                  f"{100 * cov_f:8.1f}% {100 * cov_t:8.1f}% {o['whitened_sd']:12.4f}")
    ref_name = next(iter(results))
    ref_ll = results[ref_name]["_ll"]
    print(f"\nlog-likelihood over all {results[ref_name]['_ll_npix']} pixels "
          "(constant dropped); delta vs the first set, positive = preferred")
    print(f"  {'set':32s} {'full 3x3':>16s} {'diagonal only':>16s} "
          f"{'d(full)':>12s} {'d(diag)':>12s}")
    for name in results:
        ll = results[name]["_ll"]
        print(f"  {name[:32]:32s} {ll['full']:16.1f} {ll['diag']:16.1f} "
              f"{ll['full'] - ref_ll['full']:+12.1f} {ll['diag'] - ref_ll['diag']:+12.1f}")
    print("  a set that wins on 'diagonal only' but loses on 'full 3x3' is being "
          "rejected by the CROSS-spectra")

    npix = results[next(iter(results))][channels[0]]["npix"]
    print(f"  chi2/pix over {npix} active pixels (1 +/- {np.sqrt(2 / npix):.4f}); "
          "sd(w/sqrtC) is 1 under a correct model; 90% expected in each coverage column")

    out_path = args.out or f"{os.path.splitext(args.file)[0]}_whiten.png"
    plot(results, channels, settings.f_arr, settings.t_arr,
         f"{os.path.basename(args.file)} — whitening at named parameter sets", out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
