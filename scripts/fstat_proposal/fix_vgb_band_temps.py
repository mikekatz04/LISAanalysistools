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


def main(path):
    bak = path + ".bak_vgbtemps"
    shutil.copy2(path, bak)
    print(f"backup: {bak}")
    with h5py.File(path, "r+") as f:
        bt = f["global_fit"]["sub_backend"]["vgb"]["band_temps"]
        nit, nbands, nrungs = bt.shape
        ladder = 1.0 / 1.2 ** np.arange(nrungs)
        if nrungs > 1:
            ladder[-1] = 1e-4
        before = np.array(bt[-1, :, :])
        bt[...] = np.broadcast_to(ladder, (nit, nbands, nrungs))
        print(f"band_temps {bt.shape}: rung ladder set to {ladder} "
              f"(was e.g. {before[0]} on the last stored row)")
    print("done -- resubmit the run; VGB likelihood is live again.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
