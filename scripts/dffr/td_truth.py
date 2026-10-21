"""Time-domain TDI 2.0 truth of the validation sources, through the LISA Consortium chain.

Runs in the environment of the dffr repository, not in the lisatools one:

    DFFR_REPOSITORY/.venv/bin/python td_truth.py --sources SOURCES.json --output DIR
        [--chain gws|instrument|both] [--days N] [--names NAME ...]

Reads the sources and the grid that ``stft_accuracy.py --write-sources`` stores. For each source,
``dffr.comparison.TimeDomainTruth`` runs lisagwresponse on the strain at the exact retarded times
and pytdi's X2, Y2 and Z2 on the six link responses (``gws``), or lisainstrument with every noise
and clock deviation off in between (``instrument``), on the ESA orbits that lisaanalysistools
ships. The output holds X2, Y2 and Z2 in fractional frequency at the grid's samples, with times
counted from the start of the orbit file, as LAT's orbit classes count them. ``both`` runs the
two chains, keeps the first as the truth and records how far the second lies from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import dffr
import numpy as np
from dffr.comparison import TimeDomainTruth

#: Simulated before and after the data [s], a whole number of samples: covers the TDI 2.0 delays
#: and the interpolation stencils, as in the standalone package's truth runs.
MARGIN = 1000.0


def series_on_grid(truth, params, t_ref, start, num_samples, sample_interval, chain):
    """X2, Y2, Z2 (3, num_samples) at start + i * sample_interval, absolute TCB times."""
    end = start + num_samples * sample_interval
    series, times = truth.time_series(params, t_ref, start, end, chain=chain, tdi_generation=2)
    first = int(round((start - times[0]) / sample_interval))
    if abs(times[first] - start) > 1e-6 * sample_interval:
        raise RuntimeError("the truth's samples do not fall on the grid")
    return series[:, first : first + num_samples]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chain", choices=("gws", "instrument", "both"), default="gws")
    parser.add_argument("--days", type=float, default=None, help="shorter than the grid's span")
    parser.add_argument("--names", nargs="*", default=None)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    grid = json.loads(arguments.sources.read_text())
    sample_interval = grid["sample_interval"]
    num_samples = grid["num_samples"]
    if arguments.days is not None:
        num_samples = int(round(arguments.days * 86400.0 / sample_interval))
    orbits = dffr.OrbitsWrapper()
    orbits_md5 = hashlib.md5(Path(orbits.path).read_bytes()).hexdigest()
    if orbits_md5 != grid["orbits_md5"]:
        raise SystemExit("the dffr environment ships another ESA orbit file than LAT")
    # * The file's t0 plus whole-second offsets is exact, so both sides sample the same times.
    start = orbits.time_origin + grid["data_start"]
    t_ref = orbits.time_origin + grid["t_ref"]
    chains = ("gws", "instrument") if arguments.chain == "both" else (arguments.chain,)
    names = arguments.names or list(grid["sources"])

    for name in names:
        params = np.asarray(grid["sources"][name], dtype=np.float64)
        record = dict(
            source=name,
            params=params.tolist(),
            data_start=grid["data_start"],
            t_ref=grid["t_ref"],
            num_samples=num_samples,
            sample_interval=sample_interval,
            orbits=orbits.path,
            orbits_md5=orbits_md5,
            dffr_version=dffr.__version__,
            chains={},
        )
        results = {}
        for chain in chains:
            started = time.time()
            with tempfile.TemporaryDirectory(prefix="td_truth_") as directory:
                truth = TimeDomainTruth(
                    orbits,
                    sample_interval=sample_interval,
                    margin=MARGIN,
                    work_directory=directory,
                )
                results[chain] = series_on_grid(
                    truth, params, t_ref, start, num_samples, sample_interval, chain
                )
            record["chains"][chain] = dict(seconds=round(time.time() - started, 1))
            print(f"[truth] {name} {chain}: {record['chains'][chain]['seconds']} s", flush=True)
        truth_chain = chains[0]
        if len(chains) == 2:
            reference, other = results[chains[0]], results[chains[1]]
            peaks = np.abs(reference).max(axis=-1)
            record["chains"][chains[1]]["difference"] = (
                np.abs(other - reference).max(axis=-1) / peaks
            ).tolist()
            record["chains"][chains[1]]["scale"] = (
                np.einsum("ct,ct->c", other, reference)
                / np.einsum("ct,ct->c", reference, reference)
            ).tolist()
            print(
                f"[truth] {name} {chains[1]} against {chains[0]}: largest difference per channel "
                + " ".join(f"{value:.2e}" for value in record["chains"][chains[1]]["difference"]),
                flush=True,
            )
        record["truth_chain"] = truth_chain
        np.savez(arguments.output / f"{name}.npz", series=results[truth_chain])
        (arguments.output / f"{name}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
