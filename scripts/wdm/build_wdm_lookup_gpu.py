"""Build a WDM sin/cos lookup table on the GPU and write a portable HDF5 file.

The output file stores plain NumPy arrays (cupy.get() is called on
serialization), so a table built on a CUDA node can be copied to a laptop
and loaded on CPU with:

    from lisatools.domains import WDMLookupTable
    table = WDMLookupTable.from_file("wdm_lookup.h5", force_backend="cpu")

Examples
--------
3-month Sangria smoke test (matches build_wdm_lookup_3mo.py defaults):

    python build_wdm_lookup_gpu.py \
        --Nf 720 --Nt 2160 --dt 5.0 \
        --min-freq 1e-4 --max-freq 2.5e-2 \
        --m-ref 21 --num-norm 1000 --num-layers-diff 5 \
        --time-layers 256 --batch-size 20 \
        --out wdm_lookup_n_ref_NF720_NT2160_3mo.h5

Sangria ~1 yr per_n build:

    python build_wdm_lookup_gpu.py \
        --Nf 730 --Nt 8244 --dt 5.0 \
        --min-freq 1e-4 --max-freq 3.5e-2 \
        --build-kind per_n --eps-freq 1e-3 \
        --out wdm_lookup_per_n_NF730_NT8244_sangria.h5
"""

import argparse
import os

import numpy as np

# fastlisaresponse registers the fastlisaresponse_* backends that
# DomainSettingsBase.supported_backends() resolves against; importing
# lisatools registers the lisatools_* backends.
import fastlisaresponse  # noqa: F401
import lisatools
from lisatools.domains import WDMLookupTable, WDMSettings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Grid
    p.add_argument("--Nf", type=int, required=True, help="Number of frequency layers.")
    p.add_argument("--Nt", type=int, required=True, help="Number of time pixels.")
    p.add_argument("--dt", type=float, required=True, help="data_dt in seconds.")
    p.add_argument("--oversample", type=int, default=16)
    p.add_argument("--min-freq", type=float, default=1e-4)
    p.add_argument("--max-freq", type=float, default=2.5e-2)

    # Sub-layer / m_diff grid (two ways: explicit counts, or eps-driven)
    p.add_argument("--m-ref", type=int, default=21)
    p.add_argument("--num-norm", type=int, default=1000,
                   help="Sub-layer frequency offsets per layer (linspace 0..layer_df).")
    p.add_argument("--num-layers-diff", type=int, default=5,
                   help="m_diffs spans 2*N+2 entries centered on 0.")
    p.add_argument("--eps-freq", type=float, default=None,
                   help="If set, build (norm_freq_single_layer, m_diffs, m_ref) "
                        "via WDMLookupTable.apply_eps_frequency(eps) instead of "
                        "--num-norm/--num-layers-diff/--m-ref.")

    # fdot grid
    p.add_argument("--eps-fdot", type=float, default=None,
                   help="If set, build symmetric fdot grid via apply_eps_fdot(eps); "
                        "otherwise fdot_vals=[0.0].")
    p.add_argument("--fdot-max-factor", type=float, default=8.0)

    # Build knobs
    p.add_argument("--build-kind", choices=("n_ref_only", "per_n"),
                   default="n_ref_only")
    p.add_argument("--time-layers", type=int, default=256,
                   help="Nt of the synthetic transform used during build "
                        "(n_ref_only only; smaller = faster, must be even).")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--nchannels", type=int, default=3)
    p.add_argument("--backend", default="gpu",
                   help="lisatools backend: cpu, cuda11x, cuda12x, cuda13x, cuda, gpu.")

    # Output
    p.add_argument("--out", required=True, help="Output HDF5 path.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace --out if it already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not lisatools.has_backend(args.backend):
        raise SystemExit(
            f"Backend {args.backend!r} unavailable. Install a "
            f"lisaanalysistools-cudaXXx wheel + matching cupy-cudaXXx."
        )

    settings = WDMSettings(
        Nf=args.Nf,
        Nt=args.Nt,
        dt=args.dt,
        oversample=args.oversample,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        force_backend=args.backend,
    )

    print(f"backend  = {args.backend}")
    print(f"layer_dt = {settings.layer_dt} s")
    print(f"layer_df = {settings.layer_df} Hz")
    print(f"Tobs     = {settings.Tobs} s")
    print(f"ind_min_f/ind_max_f = {settings.ind_min_f}/{settings.ind_max_f}")

    if args.eps_freq is not None:
        norm_freq_single_layer, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(
            args.eps_freq, settings,
            m_ref=args.m_ref, num_layers_diff=args.num_layers_diff,
        )
    else:
        norm_freq_single_layer = np.linspace(
            0.0, settings.layer_df, args.num_norm, endpoint=False,
        )
        m_diffs = (
            np.arange(2 * args.num_layers_diff + 2) - (args.num_layers_diff + 1)
        ).astype(np.int32)
        m_ref = args.m_ref

    if args.eps_fdot is None:
        fdot_vals = np.array([0.0])
    else:
        fdot_vals = WDMLookupTable.apply_eps_fdot(
            args.eps_fdot, settings, fdot_max_factor=args.fdot_max_factor,
        )

    print(f"m_ref={m_ref}, "
          f"norm_freq_single_layer: {len(norm_freq_single_layer)} entries, "
          f"m_diffs: {len(m_diffs)} entries, "
          f"fdot_vals: {len(fdot_vals)} entries")

    if os.path.exists(args.out):
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite to replace.")
        os.remove(args.out)

    print(f"Building {args.out} ({args.build_kind}, Nt={settings.Nt}, "
          f"num_f={len(norm_freq_single_layer) * len(m_diffs)}, "
          f"num_fdot={len(fdot_vals)}) ...")

    table = WDMLookupTable(
        settings,
        nchannels=args.nchannels,
        m_ref=m_ref,
        norm_freq_single_layer=norm_freq_single_layer,
        m_diffs=m_diffs,
        fdot_vals=fdot_vals,
        store_path=args.out,
        batch_size_gen=args.batch_size,
        build_kind=args.build_kind,
        time_layers=args.time_layers if args.build_kind == "n_ref_only" else None,
        verbose=True,
    )

    print("done. table_cos:", table.table_cos.shape,
          "table_sin:", table.table_sin.shape,
          "-> wrote", args.out)


if __name__ == "__main__":
    main()
