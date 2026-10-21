"""dffr side of the timing comparison: standalone dffr on the inputs dffr_timing_lat.py stored.

Runs in the environment of the dffr repository, not in the lisatools one:

    DFFR_PYTHON dffr_timing_dffr.py --time INPUTS.npz --output THEIRS.npz --device gpu

Times ``fill_global`` and ``get_ll`` of the compiled backend on the first B sources for every
batch size B, with the protocol of ``stft_benchmarks.time_call``: one warm-up call, then
``--repeats`` calls, each closed by a device synchronisation. ``get_ll`` computes (d|d) on every
call; that step (``dffr.dffr._data_inner_products``) is timed on its own too. Keeps (d|h) and
(h|h) of the largest batch, and with ``--keep-fill`` its fill_global grid.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import dffr
import numpy as np
from dffr.dffr import _data_inner_products
from dffr_exchange_dffr import build_response, dffr_commit

BATCH_SIZES = (1, 10, 100, 1000, 10000)


def time_call(call, synchronize, repeats):
    """The protocol of ``stft_benchmarks.time_call``."""
    call()
    synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        synchronize()
        times.append(time.perf_counter() - start)
    return times


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--time", required=True, metavar="INPUTS_NPZ")
    parser.add_argument("--output", required=True, metavar="THEIRS_NPZ")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--keep-fill", action="store_true", help="store the fill_global grid")
    args = parser.parse_args(argv)

    inputs = np.load(args.time)
    description = json.loads(str(inputs["description"]))
    grid, response = build_response(inputs, args.device)
    xp = response.xp

    def synchronize():
        if hasattr(xp, "cuda"):
            xp.cuda.runtime.deviceSynchronize()

    params_all = xp.asarray(np.ascontiguousarray(inputs["params"]))
    stft_data = dffr.STFTData(
        grid, xp.asarray(inputs["data"]), xp.asarray(inputs["inverse_covariance"])
    )
    template = xp.zeros((1, 3, grid.num_time_segments, grid.num_bins), dtype=xp.complex128)

    times = {}
    for batch in args.batch_sizes:
        params = xp.ascontiguousarray(params_all[:batch])
        indices = xp.zeros(batch, dtype=xp.int32)
        calls = {
            "fill_global": lambda: response.fill_global(params, template),
            "get_ll": lambda: response.get_ll(params, stft_data),
            "d_d": lambda: _data_inner_products(stft_data, indices, indices),
        }
        for method, call in calls.items():
            times[f"{method}/{batch}"] = np.asarray(time_call(call, synchronize, args.repeats))
            print(
                f"[timing] dffr {method:12s} batch {batch:6d}: "
                f"{1e6 * np.median(times[f'{method}/{batch}']) / batch:10.3f} us per source",
                flush=True,
            )

    outputs = {}
    if args.keep_fill:
        template[...] = 0.0
        response.fill_global(params, template)
        outputs["fill_global"] = template
    response.get_ll(params, stft_data)
    outputs["d_h"] = response.d_h
    outputs["h_h"] = response.h_h
    host = {
        name: (value.get() if hasattr(value, "get") else np.asarray(value))
        for name, value in outputs.items()
    }

    repository = Path(dffr.__file__).resolve().parents[2]
    tracked_changes = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    versions = {}
    for package in ("numpy", "cupy-cuda13x", "cupy-cuda12x", "gpubackendtools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = dict(
        dffr_commit=dffr_commit(),
        dffr_tracked_changes=[line for line in tracked_changes if line],
        backend=response.backend_name,
        versions=versions,
        device=args.device,
        description=description,
        repeats=args.repeats,
    )
    np.savez(
        args.output,
        metadata=json.dumps(metadata),
        **{f"time__{key}": value for key, value in times.items()},
        **{f"output__{key}": value for key, value in host.items()},
    )
    print(
        f"[timing] wrote {args.output} ({response.backend_name}, dffr {metadata['dffr_commit'][:7]})"
    )


if __name__ == "__main__":
    main()
