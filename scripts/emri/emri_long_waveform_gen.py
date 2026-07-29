"""Generate + store LONG (1.5 yr) TD XYZ templates for the two run EMRIs.

Drives the exact stock template path the global fit uses — catalogue row ->
``emri_catalogue_to_waveform_basis`` -> ``get_emri_response_wrapper``
(SPECIAL frame, L1 ICRS orbits, TDI-2 XYZ, order 40, REF-anchored) — for
catalogue ROWS 0 and 1 (catalogue ``ID`` fields 1 and 2; ``EMRI_IDS`` is
row-based). T = 1.5 yr covers both plunges (row 0: t_c ~ 0.95 yr, row 1:
t_c ~ 1.42 yr); ``remove_garbage="zero"`` zero-pads after plunge.

Output: ``gf_output/emri_long_waveforms/emri_row{r}_id{ID}.npz`` with the
three TD channels, grid metadata, and the 14-param waveform-basis row.

Env knobs: EMRI_ROWS (default "0,1"), T_YRS (default 1.5), DT (default 2.5).
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import logging
import time

import h5py
import numpy as np

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import find_file
from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.emri import emri_catalogue_to_waveform_basis
from lisatools.sources.emri.response import get_emri_response_wrapper
from lisatools.utils.constants import YRSID_SI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("emri_long_gen")

MOJITO_ROOT = os.path.expanduser(
    os.environ.get(
        "MOJITO_DATA_PATH", "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
    )
)
CAT_PATH = os.path.join(
    MOJITO_ROOT, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5"
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "gf_output", "emri_long_waveforms",
)


def load_catalogue_row(row: int) -> dict:
    with h5py.File(CAT_PATH, "r") as f:
        grp = f["Binaries"]
        return {k: grp[k][row] for k in grp.keys()}


def main():
    rows = [int(r) for r in os.environ.get("EMRI_ROWS", "0,1").split(",")]
    T_yrs = float(os.environ.get("T_YRS", "1.5"))
    dt = float(os.environ.get("DT", "2.5"))

    os.makedirs(OUT_DIR, exist_ok=True)

    # Same REF-anchored grid arithmetic as get_emri_wave_wrap with
    # data_t0 == REF (window_start_offset = 0): no offset, no sub-sample shift.
    out_N = int(round(T_yrs * YRSID_SI / dt))
    resp_Tobs = out_N * dt

    noise_file = find_file(
        os.path.join(MOJITO_ROOT, "data", "INSTRUMENT", "L1"), "NOISE", 0
    )
    log.info("orbits from %s", noise_file)
    orbits = L1Orbits(noise_file, force_backend="cpu", frame="icrs")

    wave_gen = get_emri_response_wrapper(
        Tobs=resp_Tobs,
        dt=dt,
        t_start=MOJITO_REFERENCE_TIME,
        t0_shift_to_data=0.0,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan="XYZ",
        role="template",
        order=40,
        orbits=orbits,
        force_backend="cpu",
    )

    for row in rows:
        entry = load_catalogue_row(row)
        cat_id = int(entry["ID"])
        params = emri_catalogue_to_waveform_basis(entry)
        log.info(
            "row %d (catalogue ID %d): M=%.4g mu=%.3f a=%.4f p0=%.3f e0=%.4f "
            "xI0=%+.0f dist=%.3f Gpc | generating T=%.2f yr, N=%d ...",
            row, cat_id, *params[:5], params[5], params[6], T_yrs, out_N,
        )
        tic = time.time()
        xyz = wave_gen(*params)
        toc = time.time() - tic
        arr = np.asarray(xyz)[:, :out_N]
        out = os.path.join(OUT_DIR, f"emri_row{row}_id{cat_id}.npz")
        np.savez(
            out,
            xyz=arr,
            dt=dt,
            t_start=MOJITO_REFERENCE_TIME,
            T_yrs=T_yrs,
            params=params,
            row=row,
            cat_id=cat_id,
            t_plunge_ssb=float(entry["TimeCoalescenceSSBFrame"]),
            estimated_snr_catalogue=float(entry["EstimatedSNR"]),
        )
        nz = np.flatnonzero(np.abs(arr[0]) > 0)
        log.info(
            "row %d done in %.1f s -> %s | shape %s, nonzero span "
            "[%.3f, %.3f] yr, max|X|=%.3e",
            row, toc, out, arr.shape,
            nz[0] * dt / YRSID_SI if len(nz) else np.nan,
            nz[-1] * dt / YRSID_SI if len(nz) else np.nan,
            np.abs(arr[0]).max(),
        )


if __name__ == "__main__":
    main()
