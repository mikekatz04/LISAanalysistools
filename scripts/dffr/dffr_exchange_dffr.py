"""dffr side of the exchange: the standalone package's DFFR outputs on the inputs LAT captured.

Runs in the environment of the dffr repository, not in the lisatools one:

    DFFR_REPOSITORY/.venv/bin/python dffr_exchange_dffr.py --inputs LAT.npz --output DFFR.npz

Reads the inputs that ``dffr_exchange_lat.py --capture`` stored (the kinematics table, the grid,
the settings, the sources, the data and the inverse covariance) and computes with the compiled
backend of standalone dffr the outputs LAT stored: the segment table, the compact templates and
their band starts, the fill_global grid, and (d|h) and (h|h).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import dffr
import numpy as np
from dffr.detector import TABLE_FIELDS


def build_response(inputs, device: str):
    """The grid and the standalone response of the inputs LAT stored, on the CPU or a GPU."""
    description = json.loads(str(inputs["description"]))
    grid = dffr.STFTGrid(
        t0=description["t0"],
        segment_duration=description["segment_duration"],
        num_time_segments=description["num_segments"],
        first_bin=description["first_bin"],
        num_bins=description["num_bins"],
    )
    fields = {name: inputs[f"kinematics__{name}"] for name, _ in TABLE_FIELDS}
    kinematics = dffr.OrbitKinematics.from_arrays(grid, fields, force_backend=device)
    response = dffr.DirectFastFresnelResponse(
        kinematics,
        t_ref=description["t_ref"],
        window_taper=description["window_taper"],
        num_side_bins=description["num_side_bins"],
        fourier_origin_shift=True,
        linear_envelope=description["linear_envelope"],
        first_order_delays=description["first_order_delays"],
        chirp_rate_floor=description["chirp_rate_floor"],
    )
    return grid, response


def dffr_commit() -> str:
    """Commit of the dffr clone that the running environment imports."""
    repository = Path(dffr.__file__).resolve().parents[2]
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", required=True, metavar="LAT_NPZ")
    parser.add_argument("--output", required=True, metavar="DFFR_NPZ")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args(argv)

    inputs = np.load(args.inputs)
    grid, response = build_response(inputs, args.device)
    xp = response.xp
    params = xp.asarray(inputs["params"])

    outputs = {}
    for name, value in response.segment_table(params).items():
        outputs[f"segment_table__{name}"] = value
    template, start = response.generate_batch(params)
    outputs["compact_template"] = template
    outputs["compact_start"] = start
    grid_template = xp.zeros((1, 3, grid.num_time_segments, grid.num_bins), dtype=xp.complex128)
    response.fill_global(params, grid_template)
    outputs["fill_global"] = grid_template
    stft_data = dffr.STFTData(
        grid, xp.asarray(inputs["data"]), xp.asarray(inputs["inverse_covariance"])
    )
    response.get_ll(params, stft_data)
    outputs["d_h"] = response.d_h
    outputs["h_h"] = response.h_h

    commit = dffr_commit()
    host = {
        name: (value.get() if hasattr(value, "get") else np.asarray(value))
        for name, value in outputs.items()
    }
    np.savez(
        args.output,
        metadata=json.dumps(dict(dffr_commit=commit, device=args.device)),
        **{f"dffr__{name}": value for name, value in host.items()},
    )
    print(f"[exchange] wrote {args.output} (dffr {commit[:7]})")


if __name__ == "__main__":
    main()
