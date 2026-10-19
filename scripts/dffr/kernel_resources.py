"""Registers, stack and shared memory of the STFT and DFFR kernels as GBGPU's GPU module links them.

    python kernel_resources.py --build-dir <GBGPU build tree> --output-dir <folder> [--arch 80] \\
        [--reference-build-dir <older build tree> | --reference-header <older lat_stft_kernels.hh>]

The GPU module compiles its CUDA files with relocatable device code and joins them in one
device-link step. A kernel that calls a function of another file gets its register count at that
step: the largest count among the kernel and the functions it calls. ptxas, run on one file,
reports the kernel's own registers only. The script compiles every CUDA file of the module for one
architecture with the commands the build tree records (``ninja -t commands``), links them as the
build does, and reads each kernel's registers, stack frame and static shared memory from the
linked code (``cuobjdump -res-usage``). The kernel's own registers and spills come from ptxas.

The reference is another build tree (the frozen environment of an earlier step), or the same tree
with an older ``lat_stft_kernels.hh`` first on the include path. Only nvcc is needed, not a GPU.

With ``--device-lto`` the candidate is compiled to LTO IR and linked with device link-time
optimisation (``-dlto``), which lets nvlink inline calls between files. With the same tree as the
reference build tree, the report then compares the module without and with it. ptxas runs at the
link only, so the column of each kernel's own registers stays empty for the candidate. For -dlto,
nvlink opens bin/../nvvm/lib64/libnvvm.so next to itself, which the pip wheels of the CUDA toolkit
do not hold (they keep libnvvm in lib/). ``--toolkit`` then names a copy of the toolkit that holds
that path; its bin/nvcc replaces the recorded one (docs/response/dffr/tests.md, step 4).

Blocks of 128 threads per SM follow from the registers (64 K per SM, allocated per warp in steps
of 8 per thread) and from the shared memory. The shared memory of a block is the static part, the
dynamic part of the tile's templates and 1 kB the system keeps per block. The SM holds 164 kB on
the A100 (sm_80) and 228 kB on the H100 (sm_90). The dynamic part of the DFFR tile is the size of
its template arena. A small host program built with the flags of gb_dffr.cu prints it, and a
second one the dynamic shared memory of the DFFR Fisher matrix, which depends on the parameters
it differences: the kernel table shows the 8 of the global fit, a list both 8 and 9.

Two properties of the sm_90 code: its static shared memory already holds the 1 kB the system
keeps (cuobjdump reports ptxas's figure plus 1024 bytes), and it reports registers for kernels
only, so the list of called functions stays empty there.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEVICE_LINK_TARGET = "src/gbgpu/cutils/CMakeFiles/gbgpu_gpu_cgbgpu.dir/cmake_device_link.o"
DFFR_OBJECT_SUFFIX = "/gb_dffr.cu.o"
KERNEL_NAME = re.compile(r"^(stft|dffr)_\w+_kernel$")
THREADS_PER_BLOCK = 128
REGISTERS_PER_SM = 65536
#: Shared memory per SM at the largest carveout [bytes].
SHARED_PER_SM = {"80": 164 * 1024, "90": 228 * 1024}
#: Shared memory the system keeps per block, where the static shared memory does not hold it.
SHARED_RESERVED_PER_BLOCK = {"80": 1024, "90": 0}
#: Templates each consumer holds at once; every other tile kernel holds one.
TEMPLATES_PER_KERNEL = {
    "stft_swap_ll_kernel": 2,
    "stft_swap_ll_grad_kernel": 2,
    "stft_get_fstat_ll_kernel": 2,
}
#: Functions that are not kernels are listed from this many registers on.
CALLEE_REGISTER_FLOOR = 129

DFFR_TEMPLATE_PROBE = """#include <cstdio>
#include "gb_dffr_source.hh"
#include "lat_dffr_kernels.hh"
int main() {
  std::printf("16 %zu\\n", DFFRTile<GBDFFRSource, 16>::table_bytes());
  std::printf("32 %zu\\n", DFFRTile<GBDFFRSource, 32>::table_bytes());
  return 0;
}
"""
# * The dynamic shared memory of the DFFR Fisher matrix depends on the parameters it differences:
# * the 8 of the global fit (fddot filled) and all 9, at 4 and 8 rows per pass.
DFFR_FISHER_PROBE = """#include <cstdio>
#include "gb_dffr_source.hh"
#include "lat_dffr_kernels.hh"
int main() {
  const int all_params[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  const int test_inds[8] = {0, 1, 2, 4, 5, 6, 7, 8};
  const double steps[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const DFFRFisherLayout nine = dffr_information_layout<GBDFFRSource>(all_params, steps, 9, 9, false);
  const DFFRFisherLayout eight = dffr_information_layout<GBDFFRSource>(test_inds, steps, 9, 8, false);
  std::printf("8 4 %zu\\n", dffr_information_bytes<GBDFFRSource, 4>(eight));
  std::printf("8 8 %zu\\n", dffr_information_bytes<GBDFFRSource, 8>(eight));
  std::printf("9 4 %zu\\n", dffr_information_bytes<GBDFFRSource, 4>(nine));
  std::printf("9 8 %zu\\n", dffr_information_bytes<GBDFFRSource, 8>(nine));
  return 0;
}
"""
FISHER_KERNEL = "dffr_information_matrix_kernel"
#: Parameters of the Fisher figure in the kernel table: the 8 of the global fit.
FISHER_TABLE_PARAMETERS = 8


def recorded_command(build_dir, ninja, target, toolkit=None):
    """The last command ninja records for a target, as a list of arguments, with the nvcc of
    `toolkit` if given."""
    listing = subprocess.run(
        [ninja, "-t", "commands", target], cwd=build_dir, check=True, capture_output=True, text=True
    )
    command = shlex.split(listing.stdout.strip().splitlines()[-1])
    if toolkit is not None and command[0].endswith("/nvcc"):
        command[0] = str(Path(toolkit) / "bin" / "nvcc")
    return command


def single_architecture(arguments, arch, code=None):
    """The arguments without dependency files and output, generating `code` (default: the SASS of
    the architecture) for one architecture."""
    kept, skip_next = [], False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument in ("-MT", "-MF", "-o"):
            skip_next = True
            continue
        if argument == "-MD" or argument.startswith("--generate-code="):
            continue
        kept.append(argument)
    return kept + [f"--generate-code=arch=compute_{arch},code=[{code or f'sm_{arch}'}]"]


def run_logged(command, build_dir, log_path):
    result = subprocess.run(command, cwd=build_dir, capture_output=True, text=True)
    log_path.write_text(shlex.join(command) + "\n" + result.stdout + result.stderr)
    if result.returncode != 0:
        sys.exit(f"command failed, see {log_path}")
    return result.stdout + result.stderr


def link_module(
    build_dir, ninja, arch, work_dir, include_dir, jobs, device_lto=False, toolkit=None
):
    """Compile every CUDA file of the device link for one architecture and link them.

    Returns the linked object, the ptxas output of all files and the recorded object names.
    """
    link = recorded_command(build_dir, ninja, DEVICE_LINK_TARGET, toolkit)
    link_output = link[link.index("-o") + 1]
    objects = [
        argument for argument in link if argument.endswith(".cu.o") and argument != link_output
    ]

    def compile_one(numbered):
        number, recorded_object = numbered
        command = single_architecture(
            recorded_command(build_dir, ninja, recorded_object, toolkit),
            arch,
            f"lto_{arch}" if device_lto else None,
        )
        if include_dir is not None:
            command.insert(1, f"-I{include_dir}")
        target = work_dir / f"{number:02d}_{Path(recorded_object).name}"
        log = run_logged(
            command + ["-Xptxas", "-v", "-o", str(target)], build_dir, target.with_suffix(".log")
        )
        return str(target), log

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        compiled = list(pool.map(compile_one, enumerate(objects)))
    replacement = dict(zip(objects, (target for target, _ in compiled)))

    command = []
    for argument in single_architecture(link, arch):
        # ? Objects of host code only carry no device code; the link leaves them out.
        if argument.endswith(".o"):
            if argument in replacement:
                command.append(replacement[argument])
            continue
        command.append(argument)
    if device_lto:
        command.append("-dlto")
    linked = work_dir / "device_link.o"
    run_logged(command + ["-o", str(linked)], build_dir, work_dir / "device_link.log")
    return linked, "\n".join(log for _, log in compiled), objects


def run_probe(build_dir, ninja, arch, work_dir, objects, toolkit, name, text, required=True):
    """Lines a host program prints, built with the flags of gb_dffr.cu; None for a module without
    gb_dffr.cu, or when an optional probe does not compile against this tree."""
    dffr_objects = [entry for entry in objects if entry.endswith(DFFR_OBJECT_SUFFIX)]
    if not dffr_objects:
        return None
    command = single_architecture(
        recorded_command(build_dir, ninja, dffr_objects[0], toolkit), arch
    )
    command = [
        argument
        for argument in command
        if argument not in ("-c", "-rdc=true") and not argument.endswith("gb_dffr.cu")
    ]
    source = work_dir / f"{name}.cu"
    source.write_text(text)
    program = work_dir / name
    full_command = command + [str(source), "-o", str(program)]
    if required:
        run_logged(full_command, build_dir, work_dir / f"{name}.log")
    else:
        # ? A tree from before the DFFR Fisher has no dffr_information_bytes to call.
        result = subprocess.run(full_command, cwd=build_dir, capture_output=True, text=True)
        (work_dir / f"{name}.log").write_text(
            shlex.join(full_command) + "\n" + result.stdout + result.stderr
        )
        if result.returncode != 0:
            return None
    return subprocess.run([str(program)], check=True, capture_output=True, text=True).stdout


def dffr_template_bytes(build_dir, ninja, arch, work_dir, objects, toolkit=None):
    """{rows: bytes of one DFFR template}, or {} for a module without gb_dffr.cu."""
    printed = run_probe(
        build_dir,
        ninja,
        arch,
        work_dir,
        objects,
        toolkit,
        "dffr_template_bytes",
        DFFR_TEMPLATE_PROBE,
    )
    if printed is None:
        return {}
    return {int(rows): int(size) for rows, size in (line.split() for line in printed.splitlines())}


def dffr_fisher_bytes(build_dir, ninja, arch, work_dir, objects, toolkit=None):
    """{(parameters, rows): dynamic shared bytes of the DFFR Fisher}, or {} without it."""
    printed = run_probe(
        build_dir,
        ninja,
        arch,
        work_dir,
        objects,
        toolkit,
        "dffr_fisher_bytes",
        DFFR_FISHER_PROBE,
        required=False,
    )
    if printed is None:
        return {}
    return {
        (int(parameters), int(rows)): int(size)
        for parameters, rows, size in (line.split() for line in printed.splitlines())
    }


def demangle(names):
    plain = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return plain.stdout.splitlines()


def label(plain):
    """Kernel name with its DFFR variant, e.g. 'stft_get_ll_kernel (DFFR, 32 rows)'."""
    base = re.sub(r"[<(].*", "", plain.removeprefix("void ")).strip()
    rows = re.search(r"DFFRTile<\w+, (\d+)>", plain) or re.search(
        r"^void dffr_\w+<\w+, (\d+)>", plain
    )
    if rows is None:
        return base
    return f"{base} (DFFR, {rows.group(1)} rows)"


def linked_resources(cuobjdump, linked):
    """{mangled name: {registers, stack, static_shared}} of every function in the linked code."""
    listing = subprocess.run(
        [cuobjdump, "-res-usage", str(linked)], check=True, capture_output=True, text=True
    ).stdout
    resources = {}
    for name, usage in re.findall(r"Function (\S+):\n\s+(REG:.*)", listing):
        fields = dict(re.findall(r"(\w+):(\d+)", usage))
        resources[name] = dict(
            registers=int(fields["REG"]),
            stack=int(fields["STACK"]),
            static_shared=int(fields["SHARED"]),
        )
    return resources


def own_resources(ptxas_log, arch):
    """{mangled kernel name: {own_registers, spill_stores, spill_loads}} from ptxas."""
    stats, current = {}, None
    for line in ptxas_log.splitlines():
        entry = re.search(rf"Compiling entry function '(\w+)' for 'sm_{arch}'", line)
        if entry:
            current = entry.group(1)
            stats[current] = {}
            continue
        if current is None:
            continue
        frame = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if frame:
            stats[current].update(spill_stores=int(frame.group(1)), spill_loads=int(frame.group(2)))
        used = re.search(r"Used (\d+) registers", line)
        if used:
            stats[current]["own_registers"] = int(used.group(1))
            current = None
    return stats


def blocks_per_sm(registers, shared_bytes, arch, granularity=8):
    """Blocks of THREADS_PER_BLOCK that fit on one SM: (by registers, by shared memory)."""
    rounded = -(-registers // granularity) * granularity
    by_registers = REGISTERS_PER_SM // (rounded * THREADS_PER_BLOCK)
    by_shared = SHARED_PER_SM[arch] // (shared_bytes + SHARED_RESERVED_PER_BLOCK[arch])
    return by_registers, by_shared


def module_report(
    build_dir, ninja, arch, work_dir, include_dir, jobs, device_lto=False, toolkit=None
):
    """{kernel label: resources} of one build tree, and the heavy functions the kernels call."""
    work_dir.mkdir(parents=True, exist_ok=True)
    linked, ptxas_log, objects = link_module(
        build_dir, ninja, arch, work_dir, include_dir, jobs, device_lto, toolkit
    )
    nvcc = recorded_command(build_dir, ninja, objects[0], toolkit)[0]
    resources = linked_resources(str(Path(nvcc).with_name("cuobjdump")), linked)
    own = own_resources(ptxas_log, arch)
    template_bytes = dffr_template_bytes(build_dir, ninja, arch, work_dir, objects, toolkit)
    fisher_bytes = dffr_fisher_bytes(build_dir, ninja, arch, work_dir, objects, toolkit)

    kernels, callees = {}, {}
    names = list(resources)
    for name, plain in zip(names, demangle(names)):
        kernel = label(plain)
        stats = dict(resources[name], **own.get(name, {}))
        base = kernel.split(" ")[0]
        if KERNEL_NAME.match(base):
            rows = re.search(r"(\d+) rows", kernel)
            if base == FISHER_KERNEL:
                stats["dynamic_shared"] = (
                    fisher_bytes.get((FISHER_TABLE_PARAMETERS, int(rows.group(1))), 0)
                    if rows
                    else 0
                )
            else:
                per_template = template_bytes.get(int(rows.group(1)), 0) if rows else 0
                stats["dynamic_shared"] = TEMPLATES_PER_KERNEL.get(base, 1) * per_template
            stats["blocks"] = blocks_per_sm(
                stats["registers"], stats["static_shared"] + stats["dynamic_shared"], arch
            )
            kernels[kernel] = stats
        elif stats["registers"] >= CALLEE_REGISTER_FLOOR and re.search(r"(?i)stft|dffr", plain):
            callees[re.sub(r"\(.*", "", plain)] = stats["registers"]
    return kernels, callees, (template_bytes, fisher_bytes)


def cell(stats, field):
    if stats is None:
        return "-"
    if field == "spills":
        return f"{stats.get('spill_stores', '-')}/{stats.get('spill_loads', '-')}"
    if field == "blocks":
        return "{}, {}".format(*stats["blocks"])
    return str(stats.get(field, "-"))


FIELDS = (
    ("registers", "registers"),
    ("own_registers", "own registers"),
    ("stack", "stack [B]"),
    ("spills", "spills [B]"),
    ("static_shared", "static shared [B]"),
    ("dynamic_shared", "dynamic shared [B]"),
    ("blocks", "blocks per SM"),
)


def print_table(title, kernels, reference=None):
    width = 23 if reference is not None else 13
    print(f"\n{title}")
    print(f"{'kernel':44s} " + " ".join(f"{heading:>{width}s}" for _, heading in FIELDS))
    for name in sorted(kernels):
        if reference is None:
            cells = [cell(kernels[name], field) for field, _ in FIELDS]
        else:
            cells = [
                f"{cell(reference.get(name), field)} / {cell(kernels[name], field)}"
                for field, _ in FIELDS
            ]
        print(f"{name:44s} " + " ".join(f"{text:>{width}s}" for text in cells))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-dir", required=True, help="GBGPU build tree (holds build.ninja)")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-build-dir", help="GBGPU build tree of an earlier version")
    reference.add_argument("--reference-header", help="older lat_stft_kernels.hh, same build tree")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="80", choices=sorted(SHARED_PER_SM))
    parser.add_argument("--jobs", type=int, default=4, help="files compiled at once")
    parser.add_argument(
        "--device-lto", action="store_true", help="link the candidate with device LTO (-dlto)"
    )
    parser.add_argument(
        "--toolkit", default=None, help="CUDA toolkit folder whose bin/nvcc builds the candidate"
    )
    parser.add_argument(
        "--ninja", default=None, help="ninja executable (default: on PATH or next to python)"
    )
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir).resolve()
    output_dir = Path(args.output_dir).resolve() / f"sm{args.arch}"
    ninja = args.ninja or shutil.which("ninja") or str(Path(sys.executable).with_name("ninja"))

    reference_kernels = None
    if args.reference_build_dir:
        reference_kernels, _, _ = module_report(
            Path(args.reference_build_dir).resolve(),
            ninja,
            args.arch,
            output_dir / "reference",
            None,
            args.jobs,
        )
    elif args.reference_header:
        with tempfile.TemporaryDirectory() as include_dir:
            # * Only the kernel header is replaced; every other header comes from the build tree.
            shutil.copy(args.reference_header, Path(include_dir) / "lat_stft_kernels.hh")
            reference_kernels, _, _ = module_report(
                build_dir, ninja, args.arch, output_dir / "reference", include_dir, args.jobs
            )
    kernels, callees, (template_bytes, fisher_bytes) = module_report(
        build_dir,
        ninja,
        args.arch,
        output_dir / ("candidate_lto" if args.device_lto else "candidate"),
        None,
        args.jobs,
        args.device_lto,
        args.toolkit,
    )

    print(
        f"GBGPU GPU module linked for sm_{args.arch}. registers: linked, which set the blocks per "
        f"SM; own registers and spills: ptxas on the kernel's file; blocks per SM of "
        f"{THREADS_PER_BLOCK} threads: by registers, by shared memory "
        f"({SHARED_PER_SM[args.arch] // 1024} kB per SM)."
    )
    if args.device_lto:
        print("The candidate is linked with device link-time optimisation (-dlto).")
    if template_bytes:
        print(
            "DFFR template (source constants, settings, segment table): "
            + ", ".join(
                f"{size} B for {rows} rows" for rows, size in sorted(template_bytes.items())
            )
        )
    if fisher_bytes:
        print(
            f"DFFR Fisher matrix, dynamic shared memory (the table shows {FISHER_TABLE_PARAMETERS} "
            "parameters) and blocks per SM by registers and by shared memory:"
        )
        for (parameters, rows), size in sorted(fisher_bytes.items()):
            stats = kernels.get(f"{FISHER_KERNEL} (DFFR, {rows} rows)")
            blocks = (
                "{}, {}".format(
                    *blocks_per_sm(stats["registers"], stats["static_shared"] + size, args.arch)
                )
                if stats
                else "-"
            )
            print(f"  {parameters} parameters, {rows} rows: {size} B; blocks per SM {blocks}")
    if reference_kernels is None:
        print_table("Kernels", kernels)
    else:
        print_table(
            "Kernels of both versions: reference / candidate",
            {name: kernels[name] for name in kernels if name in reference_kernels},
            reference_kernels,
        )
        print_table(
            "Kernels of the candidate only",
            {name: kernels[name] for name in kernels if name not in reference_kernels},
        )
        missing = sorted(set(reference_kernels) - set(kernels))
        if missing:
            print("\nKernels of the reference only: " + ", ".join(missing))
    print(f"\nCalled functions with an entry of their own, from {CALLEE_REGISTER_FLOOR} registers")
    if not callees:
        print("  none reported: this architecture's linked code gives registers for kernels only")
    for name in sorted(callees, key=lambda key: -callees[key]):
        print(f"  {callees[name]:4d}  {name}")


if __name__ == "__main__":
    main()
