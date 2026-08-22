"""Build ``gb_truth_3to21.npz`` -- the frozen recovery denominator.

The status page (``gf_monitor_gen.py``) quotes every completeness / purity /
per-source recovery number against ONE frozen set: the catalogue galactic
binaries detectable (optimal SNR > 7) over 3-21.94 mHz under the v3 run's own
fitted noise at iteration 78. The original build script lived in a session
scratchpad and was lost -- which is why this one is IN the repo. Committed
2026-08-19; validated by reproducing the original count (812 detectable).

DETECTABILITY IS PER-TOBS. The observation time sets the FD bin width, the
waveform duration and hence the optimal SNR itself, so a truth set is only
valid for the run it was built against: at 1 year the same catalogue source
integrates 4x longer and the detectable count grows. ``Tobs`` is therefore
read from the STORE's own domain settings (``global_fit/domain_settings/args``
-> ``attrs["0"] * attrs["1"] * attrs["2"]``, the same expression the monitor
uses), with the 3-month value as the fallback and ``--tobs`` as an explicit
override, and the value used is stamped into both output npz files. The
monitor refuses a truth set whose stamped ``tobs`` does not match the run.

Nothing here is re-derived: the noise, waveform and SNR route are copied from
the monitor's own recovery section (GBGPU ``run_wave`` on CPU, lisatools
``A2TDISens``/``E2TDISens`` fed the run's sampled instrument + foreground) --
the same path the run's likelihood uses.

Shape of the output (the monitor's ``TRU`` contract):
    f0    (N,)   Hz
    amp   (N,)   GW amplitude
    snr   (N,)   optimal SNR under the frozen noise
    det   (N,)   bool, snr > 7
    phys  (N,9)  GBGPU physical basis [amp, f0, fdot, fddot, phi0, iota,
                 psi, lam(icrs), beta(icrs)]
plus provenance attrs (store path, iteration, band, psd/galfor params).

The amplitude PREFILTER: running waveforms for every one of the 15.5M
catalogue rows is pointless when SNR/amp at fixed f is bounded. A unit-SNR
kappa curve (best-case orientation, coarse f grid, buffered 2x) cuts the
waveform set to the plausibly-detectable tail; the same curve is written to
``kappa_grid.npz`` for the monitor's population panels.

Usage::

    OMP_NUM_THREADS=1 python scripts/diagnostics/build_truth.py \
        STORE.h5 [--iteration 78] [--out gb_truth_3to21.npz] [--tobs SEC]

Runs on CPU in a few minutes. Keep the thread pins: laptop policy.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import h5py

FLO, FHI = 3e-3, 21.94e-3
SNR_DET = 7.0
TOBS_3MO = 7776000.0   # the original 3-month set; also the fallback Tobs
DT = 2.5
NW_3MO = 128      # FD points per waveform at 3 months (the monitor's own NW_)

MOJITO_CAT = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/catalogues/"
    "wdwd_cat_mojito_lite_processed.hdf5")


def store_tobs(store, fallback=TOBS_3MO):
    """Observation time of the run, from the store's own domain settings.

    ``global_fit/domain_settings/args`` carries the WDM/FD grid arguments as
    attrs "0"/"1"/"2" = (Nt, Nf, dt) -- their product is Tobs. This is exactly
    how ``gf_monitor_gen.py`` derives ``SCI_TOBS``, so a truth set built here
    and a page rendered there agree by construction. Falls back to the 3-month
    value when the store predates the group.
    """
    try:
        with h5py.File(store, "r") as f:
            a = dict(f["global_fit/domain_settings/args"].attrs)
        return float(a["0"]) * float(a["1"]) * float(a["2"])
    except Exception as e:
        print(f"WARNING: no domain settings in {store} ({type(e).__name__}); "
              f"falling back to Tobs = {fallback}")
        return float(fallback)


def nw_for(tobs):
    """FD points per waveform at this Tobs.

    ``N`` has to span the Doppler-modulated bandwidth of the source, which is
    a fixed number of HERTZ (a few tens of 1/yr sidebands); the FD bin is
    1/Tobs, so the bandwidth in BINS grows in proportion to Tobs. Hold the
    3-month value at exactly 128 (reproducing the frozen 812-source set) and
    scale up in powers of two from there -- 512 at one year. Over-sizing N is
    safe (the extra bins are ~zero and the slow part is merely sampled more
    finely); under-sizing truncates the waveform and silently loses SNR.
    """
    n = NW_3MO * max(float(tobs) / TOBS_3MO, 1.0)
    return int(min(2 ** int(np.ceil(np.log2(n))), 2048))


def fitted_noise(store, it):
    """Cold-chain median psd + galfor params at ``it`` (the monitor's own
    ``np.median(psd_cold[-1], axis=0)`` evaluated at a chosen row)."""
    with h5py.File(store, "r") as f:
        g = f["global_fit"]
        psd = np.median(np.asarray(g["chain"]["psd"][it, 0, 0, :, 0, :]),
                        axis=0)
        gal = np.median(np.asarray(g["chain"]["galfor"][it, 0, 0, :, 0, :]),
                        axis=0)
    return psd, gal


def sens_grids(psd_p, gal_p, df):
    from lisatools import detector as lisa_models
    from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens
    from lisatools.stochastic import (
        HyperbolicTangentGalacticForeground as HTGF)
    lm = lisa_models.LISAModel(psd_p[0] ** 2, psd_p[1] ** 2,
                               lisa_models.DefaultOrbits(), "sampled")
    nk = dict(model=lm, stochastic_params=tuple(gal_p),
              stochastic_function=HTGF)
    ng = int(2.35e-2 / df) + 2
    fg = np.maximum(np.arange(ng) * df, df)
    sa = np.asarray(get_sensitivity(fg, sens_fn=A2TDISens, **nk), float)
    se = np.asarray(get_sensitivity(fg, sens_fn=E2TDISens, **nk), float)
    return sa, se


def catalogue_phys(t_ref):
    """(N,9) GBGPU physical rows for every catalogue GB in [FLO, FHI].

    Column conventions copied from the run's own catalogue path
    (``gb_catalogue_to_sampling_basis`` handles the epoch shift and ICRS
    frame; ``both_transforms`` of the 8-col basis is what the sampler feeds
    GBGPU) -- here we only need the physical 9-vector, which
    ``gb_catalogue_to_sampling_basis`` exposes directly through the run's
    transform container."""
    from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
    with h5py.File(MOJITO_CAT, "r") as f:
        b = f["Binaries"]
        f0 = np.asarray(b["GW22FrequencySSBFrame"][:], float)
        sel = (f0 >= FLO) & (f0 <= FHI)
        entry = {k: np.asarray(b[k][sel]) for k in (
            "Amplitude", "GW22FrequencySSBFrame",
            "GW22FrequencyDerivativeSourceFrame", "InclinationAngle",
            "PolarisationAngle", "RightAscension", "Declination",
            "TrueAnomaly", "TimeReferenceSSBFrame", "ChirpMassSSBFrame",
            "LuminosityDistance", "Eccentricity")}
    # 8-col sampling basis [lnA, f0 mHz, fdot, phi0, cos_i, psi, lam, sinb]
    rows = np.atleast_2d(gb_catalogue_to_sampling_basis(entry))
    phys = np.column_stack([
        np.exp(rows[:, 0]), rows[:, 1] * 1e-3, rows[:, 2],
        np.zeros(len(rows)), rows[:, 3], np.arccos(rows[:, 4]),
        rows[:, 5], rows[:, 6], np.arcsin(np.clip(rows[:, 7], -1, 1)),
    ])
    return phys


def opt_snr(phys, sa, se, gbw, df, tobs, nw):
    """Optimal SNR of each row: 4 df sum(|A|^2/SA + |E|^2/SE), sqrt.

    ``tobs`` is the run's observation time -- it enters BOTH the waveform
    (``T=`` in ``run_wave``, which sets how long the source integrates) and
    the inner-product measure ``df = 1/tobs``; ``sa``/``se`` must already be
    sampled on that same ``df`` grid.
    """
    out = np.zeros(len(phys))
    B = 20000
    for lo in range(0, len(phys), B):
        p = phys[lo:lo + B]
        gbw.run_wave(*[np.ascontiguousarray(p[:, k]) for k in range(9)],
                     N=nw, T=tobs, dt=DT, tdi2=True, tdi_channel_setup="AE")
        A = np.asarray(gbw.A); E = np.asarray(gbw.E)
        s = np.asarray(gbw.start_inds).astype(int)
        for i in range(len(p)):
            if s[i] < 0 or s[i] + nw > sa.size:
                continue
            _sa = sa[s[i]:s[i] + nw]; _se = se[s[i]:s[i] + nw]
            out[lo + i] = np.sqrt(
                4 * df * (np.abs(A[i]) ** 2 / _sa
                          + np.abs(E[i]) ** 2 / _se).real.sum())
        print(f"  snr: {min(lo + B, len(phys))}/{len(phys)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store")
    ap.add_argument("--iteration", type=int, default=78)
    ap.add_argument("--out", default="gb_truth_3to21.npz")
    ap.add_argument("--kappa-out", default="kappa_grid.npz")
    ap.add_argument("--tobs", type=float, default=None,
                    help="observation time [s]; default: read from the "
                         "store's global_fit/domain_settings/args")
    ap.add_argument("--nw", type=int, default=None,
                    help="FD points per waveform; default: 128 scaled by "
                         "Tobs/3 months, rounded up to a power of two")
    a = ap.parse_args(argv)

    tobs = float(a.tobs) if a.tobs else store_tobs(a.store)
    nw = int(a.nw) if a.nw else nw_for(tobs)
    df = 1.0 / tobs
    print(f"Tobs = {tobs:.1f} s ({tobs / 86400.0:.1f} d), df = {df:.4g} Hz, "
          f"N per waveform = {nw}")
    psd_p, gal_p = fitted_noise(a.store, a.iteration)
    print(f"noise @ it {a.iteration}: psd={psd_p} galfor={gal_p}")
    sa, se = sens_grids(psd_p, gal_p, df)

    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import (
        GB_MOJITO_T_REF)
    orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))

    phys = catalogue_phys(GB_MOJITO_T_REF)
    print(f"catalogue rows in [{FLO}, {FHI}] Hz: {len(phys)}")

    # ---- kappa prefilter -------------------------------------------------
    # Best-case-orientation unit-SNR curve on a coarse grid: face-on,
    # psi=phi0=0, one sky draw per node maximised over 8 sky positions.
    fgrid = np.geomspace(FLO, FHI, 48)
    kap = np.zeros(fgrid.size)
    A0 = 1e-22
    for isky in range(8):
        rng = np.random.default_rng(isky)
        lam = rng.uniform(0, 2 * np.pi); beta = np.arcsin(rng.uniform(-1, 1))
        proto = np.column_stack([
            np.full(fgrid.size, A0), fgrid, np.zeros(fgrid.size),
            np.zeros(fgrid.size), np.zeros(fgrid.size),
            np.zeros(fgrid.size), np.zeros(fgrid.size),
            np.full(fgrid.size, lam), np.full(fgrid.size, beta)])
        kap = np.maximum(kap, opt_snr(proto, sa, se, gbw, df, tobs, nw) / A0)
    keep = phys[:, 0] * np.interp(phys[:, 1], fgrid, kap) > 0.5 * SNR_DET
    print(f"kappa prefilter keeps {keep.sum()} / {len(phys)}")
    # keys are the monitor's contract: kappa_grid.npz carries fgrid/fit. The
    # curve is a unit-SNR ceiling under THIS Tobs, so stamp it too.
    np.savez(a.kappa_out, fgrid=fgrid, fit=kap, tobs=np.array(tobs))

    snr = np.zeros(len(phys))
    snr[keep] = opt_snr(phys[keep], sa, se, gbw, df, tobs, nw)
    det = snr > SNR_DET
    print(f"DETECTABLE (SNR > {SNR_DET}): {det.sum()}")

    np.savez_compressed(
        a.out, f0=phys[:, 1], amp=phys[:, 0], snr=snr, det=det, phys=phys,
        store=np.array(a.store), iteration=np.array(a.iteration),
        psd_params=psd_p, galfor_params=gal_p,
        band=np.array([FLO, FHI]), tobs=np.array(tobs), nw=np.array(nw))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
