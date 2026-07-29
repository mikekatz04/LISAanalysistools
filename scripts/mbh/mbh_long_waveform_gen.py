"""Generate + store LONG (1.5 yr) TD XYZ templates for the local run MBHBs.

Mirrors the stock legacy path (``get_mbh_phenom_wave_gen`` /
``mbh_mojito_match_debug.py``): catalogue row -> ``mbh_catalogue_to_
sampling_basis`` -> ``make_mbh_transform_container().both_transforms`` ->
``PhenomTHMTDIWaveform.compute_tdi_channels`` -> ``place_td_signal_on_grid``
on the REF-anchored 1.5-yr grid. Differences from the run defaults, both
deliberate for this band study: ``freq_max=0.15`` (the run's generator is
itself capped at 25 mHz — that cap is part of what we are measuring) and
``waveform_duration = full span`` (not the 1-month window).

Default rows: 0, 16, 18 (local L1 files; mergers at 0.822 / 0.305 / 0.252 yr,
all inside the window). MBHB catalogue IDs are 0-based (ID == row).

Env: MBH_ROWS (default "0,16,18"), T_YRS (1.5), DT (2.5), MBH_GEN_MAX_FREQ
(0.15), MBH_DURATION_YRS ('' -> full span).
"""

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import gc
import logging
import time

import h5py
import numpy as np

from lisatools.detector import L1Orbits
from lisatools.domains import TDSettings, place_td_signal_on_grid
from lisatools.globalfit.preprocessing import find_file
from lisatools.globalfit.recipe import (
    MOJITO_REFERENCE_TIME,
    mbh_catalogue_to_sampling_basis,
)
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.utils.constants import YRSID_SI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mbh_long_gen")

MOJITO_ROOT = os.path.expanduser(
    os.environ.get(
        "MOJITO_DATA_PATH", "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
    )
)
CAT_PATH = os.path.join(
    MOJITO_ROOT, "catalogues", "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5"
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "gf_output", "mbh_long_waveforms",
)


def load_catalogue_row(row: int) -> dict:
    with h5py.File(CAT_PATH, "r") as f:
        grp = f["Binaries"]
        return {k: float(grp[k][row]) for k in grp.keys()}


def build_orbits(t_lo: float, t_hi: float):
    """L1 orbits with ltt arrays sliced to the window + coarse position grid
    (the mbh_mojito_match_debug RAM guard — pyResponseTDI deepcopies x2)."""
    noise_file = find_file(
        os.path.join(MOJITO_ROOT, "data", "INSTRUMENT", "L1"), "NOISE", 0
    )
    orb = L1Orbits(noise_file, force_backend="cpu", frame="icrs")
    pad = 1.0e5
    ltt_t = np.asarray(orb.ltt_t)
    m = (ltt_t >= max(t_lo - pad, ltt_t[0])) & (ltt_t <= min(t_hi + pad, ltt_t[-1]))
    orb.ltt = np.asarray(orb.ltt)[m].copy()
    orb.ltt_t = ltt_t[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    del ltt_t
    gc.collect()
    orb.configure(linear_interp_setup=True, dt=300.0)
    return orb


def main():
    rows = [int(r) for r in os.environ.get("MBH_ROWS", "0,16,18").split(",")]
    T_yrs = float(os.environ.get("T_YRS", "1.5"))
    dt = float(os.environ.get("DT", "2.5"))
    gen_max_freq = float(os.environ.get("MBH_GEN_MAX_FREQ", "0.15"))
    dur_env = os.environ.get("MBH_DURATION_YRS", "").strip()

    os.makedirs(OUT_DIR, exist_ok=True)
    Nt = int(round(T_yrs * YRSID_SI / dt))
    span = Nt * dt
    t0 = MOJITO_REFERENCE_TIME
    duration = float(dur_env) * YRSID_SI if dur_env else span

    grid = TDSettings(N=Nt, dt=dt, t0=t0, force_backend="cpu")
    log.info("orbits (window-sliced) ...")
    orbits = build_orbits(t0, t0 + span)

    wave_gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(
            higher_modes=[21, 33, 44],
            include_negative_modes=True,
            t_low_fit=True,
            coarse_grain=False,
            atol=1e-12,
            rtol=1e-12,
        ),
        Tobs=duration,
        start_freq=7e-5,
        use_reference_time=True,
        waveform_t0=t0,
        data_td_settings=grid,
        tdi_generation="2nd generation",
        tdi_channels="XYZ",
        sampling_frequency=1.0 / dt,
        orbits=orbits,
        order=30,
        tukey_alpha=0.0,
        stft_dt=None,
        freq_min=1e-4,
        freq_max=gen_max_freq,
        fft_batch_size=2,
        buffer_time=15_000.0,
        output_domain_settings=None,  # TD output path
        force_backend="cpu",
    )
    del orbits
    gc.collect()

    tc = make_mbh_transform_container()

    for row in rows:
        entry = load_catalogue_row(row)
        samp = mbh_catalogue_to_sampling_basis(entry)
        params_in = tc.both_transforms(np.asarray(samp, dtype=float))
        log.info(
            "row %d (ID %d): Mtot=%.3e q=%.2f t_merge=%.3f yr SNR_est=%.0f | "
            "generating ...",
            row, int(entry["ID"]), entry["TotalMassSSBFrame"],
            entry["MassRatio"],
            entry["TimeCoalescencePhenomTPHMSSBFrame"] / YRSID_SI,
            entry["EstimatedSNR"],
        )
        tic = time.time()
        times, channels = wave_gen.compute_tdi_channels(*params_in)
        arr = np.asarray(
            place_td_signal_on_grid(
                np.atleast_2d(channels)[:3], grid, times=times
            ).arr
        )[:, :Nt]
        toc = time.time() - tic
        out = os.path.join(OUT_DIR, f"mbh_row{row}_id{int(entry['ID'])}.npz")
        np.savez(
            out,
            xyz=arr, dt=dt, t_start=t0, T_yrs=T_yrs, params=samp,
            row=row, cat_id=int(entry["ID"]),
            t_plunge_ssb=entry["TimeCoalescencePhenomTPHMSSBFrame"],
            estimated_snr_catalogue=entry["EstimatedSNR"],
        )
        nz = np.flatnonzero(np.abs(arr[0]) > 0)
        log.info(
            "row %d done in %.1f s -> %s | nonzero span [%.3f, %.3f] yr, "
            "max|X|=%.3e",
            row, toc, out,
            nz[0] * dt / YRSID_SI if len(nz) else np.nan,
            nz[-1] * dt / YRSID_SI if len(nz) else np.nan,
            np.abs(arr[0]).max(),
        )
        del times, channels, arr
        gc.collect()


if __name__ == "__main__":
    main()
