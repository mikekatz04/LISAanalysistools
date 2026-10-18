"""Registers, stack and spills of the STFT kernels, from ptxas, for two versions of the kernels.

    python kernel_resources.py --build-dir <GBGPU build tree> \\
        --reference-header <older lat_stft_kernels.hh> --output-dir <folder> [--arch 80]

GBGPU's ``gb_tdi_on_the_fly.cu`` instantiates every STFT kernel of LAT's ``lat_stft_kernels.hh``.
The script compiles it twice with the command that the build tree records (``ninja -t commands``),
for one GPU architecture and with ptxas statistics: once as the tree builds it, and once with the
reference header first on the include path. It then prints, per kernel, the registers, the stack
frame, the spill stores and loads and the bytes of the kernel arguments of both versions. Only
nvcc is needed, not a GPU. The registers set how many blocks of 128 threads fit on one SM.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET = "src/gbgpu/cutils/CMakeFiles/gbgpu_gpu_cgbgpu.dir/gb_tdi_on_the_fly.cu.o"
FIELDS = ("registers", "stack", "spill_stores", "spill_loads", "argument_bytes")


def compile_command(build_dir, ninja):
    """The recorded nvcc command of the target, as a list of arguments."""
    listing = subprocess.run(
        [ninja, "-t", "commands", TARGET], cwd=build_dir, check=True, capture_output=True, text=True
    )
    return shlex.split(listing.stdout.strip().splitlines()[-1])


def ptxas_command(arguments, output, arch, include_dir=None):
    """The compile command for one architecture, with statistics and without dependency files."""
    command, skip_next = [], False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument in ("-MT", "-MF", "-o"):
            skip_next = True
            continue
        if argument == "-MD" or argument.startswith("--generate-code="):
            continue
        command.append(argument)
    if include_dir is not None:
        command.insert(1, f"-I{include_dir}")
    return command + [
        f"--generate-code=arch=compute_{arch},code=[sm_{arch}]",
        "-Xptxas",
        "-v",
        "-o",
        str(output),
    ]


def run_ptxas(command, build_dir, log_path):
    result = subprocess.run(command, cwd=build_dir, capture_output=True, text=True)
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        sys.exit(f"compilation failed, see {log_path}")
    return log_path.read_text()


def demangled_kernel(name):
    plain = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
    return re.sub(r"<.*", "", plain.replace("void ", ""))


def kernel_stats(log, arch):
    """{kernel name: {field: value}} for the STFT kernels of a ptxas log."""
    stats, current = {}, None
    for line in log.splitlines():
        entry = re.search(rf"Compiling entry function '(\w+)' for 'sm_{arch}'", line)
        if entry:
            current = demangled_kernel(entry.group(1))
            stats[current] = {}
            continue
        if current is None:
            continue
        frame = re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line
        )
        if frame:
            stats[current].update(
                stack=int(frame.group(1)),
                spill_stores=int(frame.group(2)),
                spill_loads=int(frame.group(3)),
            )
        used = re.search(r"Used (\d+) registers", line)
        if used:
            stats[current]["registers"] = int(used.group(1))
            # ? sm_90 does not report the argument bank; the column then shows "-".
            arguments = re.search(r"(\d+) bytes cmem\[0\]", line)
            if arguments:
                stats[current]["argument_bytes"] = int(arguments.group(1))
    return {name: value for name, value in stats.items() if name.startswith("stft_")}


def blocks_per_sm(registers, threads=128, registers_per_sm=65536, granularity=8):
    """Blocks of `threads` that the registers allow on one SM (64 K registers on sm_80 and sm_90,
    allocated per warp in steps of 8 per thread)."""
    rounded = -(-registers // granularity) * granularity
    return registers_per_sm // (rounded * threads)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-dir", required=True, help="GBGPU build tree (holds build.ninja)")
    parser.add_argument("--reference-header", required=True, help="older lat_stft_kernels.hh")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="80", help="GPU architecture: 80 (A100) or 90 (H100)")
    parser.add_argument(
        "--ninja", default=None, help="ninja executable (default: on PATH or next to python)"
    )
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ninja = args.ninja or shutil.which("ninja") or str(Path(sys.executable).with_name("ninja"))
    arguments = compile_command(build_dir, ninja)

    with tempfile.TemporaryDirectory() as include_dir:
        # * Only the kernel header is replaced; every other header comes from the build tree.
        shutil.copy(args.reference_header, Path(include_dir) / "lat_stft_kernels.hh")
        reference_log = run_ptxas(
            ptxas_command(arguments, output_dir / "reference.o", args.arch, include_dir),
            build_dir,
            output_dir / f"ptxas_reference_sm{args.arch}.log",
        )
    candidate_log = run_ptxas(
        ptxas_command(arguments, output_dir / "candidate.o", args.arch),
        build_dir,
        output_dir / f"ptxas_candidate_sm{args.arch}.log",
    )
    reference = kernel_stats(reference_log, args.arch)
    candidate = kernel_stats(candidate_log, args.arch)

    print(f"ptxas, sm_{args.arch}: reference / candidate per kernel")
    print(
        f"{'kernel':32s} "
        + " ".join(f"{field:>17s}" for field in FIELDS)
        + f" {'blocks per SM':>15s}"
    )
    for name in sorted(set(reference) | set(candidate)):
        cells = [
            f"{reference.get(name, {}).get(field, '-')!s:>7s} / {candidate.get(name, {}).get(field, '-')!s:<7s}"
            for field in FIELDS
        ]
        blocks = [
            str(blocks_per_sm(stats[name]["registers"])) if name in stats else "-"
            for stats in (reference, candidate)
        ]
        print(
            f"{name:32s} "
            + " ".join(f"{cell:>17s}" for cell in cells)
            + f" {' / '.join(blocks):>15s}"
        )


if __name__ == "__main__":
    main()
