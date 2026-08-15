#!/usr/bin/env python
"""Reset the VGB band temperatures in a live run store (2026-08-15 bugfix).

The vgb branch's beta ladder collapsed to [1e-4] (general_setup.ntemps=1
seeded a 1-rung ladder whose only rung was then clobbered to 1e-4 by
stock/erebor/vgb.py -- fixed in the same commit as this script). Every
stored iteration therefore sampled VGBs with the likelihood effectively
OFF: sub_backend/vgb/band_temps is 1e-4 for all bands. Resume restores
band_temps from the h5, so pulling the code fix alone does NOT heal a
live store -- run this once before resubmitting:

    python scripts/fstat_proposal/fix_vgb_band_temps.py <store>/gf_prod_3mo_testing.h5

Writes a .bak first. Sets the stored band_temps to the proper ladder for
the STORED rung count (1 rung -> [1.0]; k rungs -> 1/1.2**i with the last
at 1e-4), for every stored iteration row so the resume row is correct
regardless of which iteration the backend restores from.

NOTE: all VGB samples stored BEFORE this fix are the SNR-gated prior,
not posteriors (and 36/55 leaves were frozen at init by the GB SNR>=5
gate, also fixed in the same commit). Post-fix iterations re-equilibrate
from the stored coords, which start at/near truth -- expect the movers
to contract onto real posteriors within a few iterations.
"""
import shutil
import sys

import h5py
import numpy as np


def main(path, nrungs_target=None):
    bak = path + ".bak_vgbtemps"
    shutil.copy2(path, bak)
    print(f"backup: {bak}")
    with h5py.File(path, "r+") as f:
        vg = f["global_fit"]["sub_backend"]["vgb"]
        nit, nbands, nrungs = vg["band_temps"].shape
        k = int(nrungs_target or nrungs)
        ladder = 1.0 / 1.2 ** np.arange(k)
        if k > 1:
            ladder[-1] = 1e-4
        before = np.array(vg["band_temps"][-1, 0, :])
        if k == nrungs:
            vg["band_temps"][...] = np.broadcast_to(
                ladder, (nit, nbands, k))
        else:
            # RUNG-COUNT CHANGE (e.g. 1 -> 8, VGB_NTEMPS=8 user ruling):
            # the loader derives the branch's ntemps from the STORED
            # band_temps shape (state.py setdefault from shape[-1]), so
            # every rung-dimensioned dataset must be recreated. Counters
            # restart at zero (they are diagnostics); swap arrays carry
            # k-1 adjacent pairs.
            def _recreate(name, shape, fill):
                if name not in vg:
                    return
                del vg[name]
                vg.create_dataset(name, data=np.broadcast_to(
                    fill, shape).copy())
                print(f"  {name} -> {shape}")
            _recreate("band_temps", (nit, nbands, k), ladder)
            for name in ("band_num_accepted", "band_num_proposed",
                         "band_num_accepted_rj", "band_num_proposed_rj"):
                _recreate(name, (nit, nbands, k), 0.0)
            for name in ("band_swaps_accepted", "band_swaps_proposed"):
                _recreate(name, (nit, nbands, max(k - 1, 0)), 0.0)
        print(f"band_temps rungs {nrungs} -> {k}: ladder {ladder} "
              f"(last stored row was {before})")
    print("done -- resubmit the run; VGB likelihood is live again.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__ + "\nOptional 2nd arg: target rung count "
                 "(e.g. 8 -- must match VGB_NTEMPS in the submit script).")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else None)
