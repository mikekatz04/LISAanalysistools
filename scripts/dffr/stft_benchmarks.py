"""Wall-time benchmarks of the seven STFT GB methods of ``gbgpu.gbcomps.STFTGBComputations``.

Times ``get_ll_stft``, ``get_swap_ll_stft``, ``get_fstat_ll_stft``, ``fill_global_stft``,
``get_ll_grad_stft``, ``get_swap_ll_grad_stft`` and ``information_matrix`` against batch size on
a production fixture of ``stft_fixtures.py``. Each point is one warm-up call followed by
``--repeats`` timed calls; on the GPU the device is synchronised before each clock reading. The
times include the Python work of each method, which the global fit pays as well. ``--response``
selects the response; DFFR has no ``information_matrix`` yet, so it is left out.

Usage:
    python stft_benchmarks.py --device cpu --mojito-l1 FILE.h5 --output OUT.json
    python stft_benchmarks.py --device gpu --mojito-l1 FILE.h5 --num-segments 365 \\
        --batch-sizes 1 10 100 1000 10000 --output OUT.json

Results and the commands of each run are recorded in ``docs/response/dffr/tests.md``.
"""

import argparse
import json
import sys
import time

import numpy as np

import stft_fixtures as fixtures
from stft_golden_outputs import swap_partners

METHODS = (
    "get_ll_stft",
    "get_swap_ll_stft",
    "get_fstat_ll_stft",
    "fill_global_stft",
    "get_ll_grad_stft",
    "get_swap_ll_grad_stft",
    "information_matrix",
)


def method_calls(fixture, batch_size):
    """One argument-free call per method on the first ``batch_size`` sources of the fixture."""
    comp = fixture.comp
    xp = comp.xp
    params = fixture.params[:batch_size]
    cells = xp.asarray(fixture.cells[:batch_size])
    starts = fixture.window_starts
    params_partner = params[swap_partners(fixture.cells[:batch_size])]
    templates = xp.zeros((fixture.num_templates,) + fixture.template_shape, dtype=xp.complex128)
    likelihood_indices = dict(data_index=cells, noise_index=cells, start_freq_inds=starts)
    return {
        "get_ll_stft": lambda: comp.get_ll_stft(params, **likelihood_indices),
        "get_swap_ll_stft": lambda: comp.get_swap_ll_stft(
            params, params_partner, **likelihood_indices
        ),
        "get_fstat_ll_stft": lambda: comp.get_fstat_ll_stft(params, **likelihood_indices),
        "fill_global_stft": lambda: comp.fill_global_stft(
            params, templates, data_index=cells, start_freq_inds=starts
        ),
        "get_ll_grad_stft": lambda: comp.get_ll_grad_stft(params, **likelihood_indices),
        "get_swap_ll_grad_stft": lambda: comp.get_swap_ll_grad_stft(
            params, params_partner, **likelihood_indices
        ),
        "information_matrix": lambda: comp.information_matrix(
            params, noise_index=cells, start_freq_inds=starts
        ),
    }


def time_call(call, backend, repeats):
    call()
    fixtures.synchronize(backend)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        fixtures.synchronize(backend)
        times.append(time.perf_counter() - start)
    return times


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        "--backend",
        dest="backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu or gpu (--backend is the older name)",
    )
    parser.add_argument("--mojito-l1", default=None, help="Mojito L1 .h5 file (l1 fixtures)")
    parser.add_argument(
        "--config", choices=list(fixtures.PRODUCTION_CONFIGS), default="l1_tdi2_envelope"
    )
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--without-swap",
        action="store_true",
        help="leave out get_swap_ll_stft and get_swap_ll_grad_stft (CPU runs from step 2 on)",
    )
    parser.add_argument("--response", choices=fixtures.RESPONSES, default=fixtures.RESPONSES[0])
    parser.add_argument("--output", required=True, metavar="OUT_JSON")
    args = parser.parse_args(argv)
    if args.without_swap:
        args.methods = [method for method in args.methods if "swap" not in method]
    if args.response == "dffr":
        # * The DFFR response has no Fisher matrix yet.
        args.methods = [method for method in args.methods if method != "information_matrix"]

    fixture = fixtures.build_production_fixture(
        args.config,
        args.backend,
        mojito_l1=args.mojito_l1,
        num_segments=args.num_segments,
        num_sources=max(args.batch_sizes),
        response=args.response,
    )
    results = []
    for batch_size in args.batch_sizes:
        calls = method_calls(fixture, batch_size)
        for method in args.methods:
            times = time_call(calls[method], args.backend, args.repeats)
            median = float(np.median(times))
            results.append(
                dict(
                    method=method,
                    batch_size=batch_size,
                    times_s=times,
                    median_s=median,
                    min_s=float(np.min(times)),
                    median_per_source_s=median / batch_size,
                )
            )
            print(
                f"[bench] {method:22s} batch {batch_size:6d}: median {median * 1e3:10.3f} ms, "
                f"min {min(times) * 1e3:10.3f} ms",
                flush=True,
            )

    record = dict(
        metadata=fixtures.run_metadata(args.backend, args.mojito_l1),
        config=args.config,
        response=args.response,
        fixture=fixture.description,
        repeats=args.repeats,
        results=results,
    )
    with open(args.output, "w") as output:
        json.dump(record, output, indent=1)
    print(f"[bench] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
