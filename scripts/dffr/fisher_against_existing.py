"""The DFFR Fisher matrices against those of the existing response, entry by entry.

    python fisher_against_existing.py EXISTING.npz DFFR.npz

Reads two captures of ``finite_difference_steps.py --all-sources``, one per response, on the same
fixtures and sources, and compares their Fisher matrices at the default steps (factor 1) on the 8
parameters of the global fit, fddot being filled. Per configuration it prints:

* the largest and the median over sources of max |F_dffr - F_existing| / sqrt(F_ii F_jj), with
  the diagonal of the existing response;
* the range of the diagonal ratios F_ii(dffr) / F_ii(existing);
* the largest difference of the normalised matrices F_ij / sqrt(F_ii F_jj);
* the range of the condition numbers of the normalised matrices, which sets how strongly the
  inverse, and so the marginal standard deviations, can amplify such differences;
* the range of the ratios of the marginal standard deviations from the inverses.

There is no threshold. The two responses model the signal differently (DFFR carries the Doppler
shift in its reception chirps, the existing Fisher uses the astro track), and each Fisher carries
its own finite-difference noise; a validation of the Fisher matrix separates the two. Recorded in
docs/response/dffr/tests.md.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from stft_golden_outputs import load_blob

#: The parameters of the global fit: every parameter but fddot.
TEST_INDS = [0, 1, 2, 4, 5, 6, 7, 8]


def normalised(info):
    diagonal = np.sqrt(np.abs(np.diagonal(info, axis1=-2, axis2=-1)))
    return info / (diagonal[..., :, None] * diagonal[..., None, :]), diagonal


def compare(existing_path, dffr_path):
    existing_blob, dffr_blob = load_blob(existing_path), load_blob(dffr_path)
    configs = sorted(
        {
            key.split("/", 1)[0]
            for key in dffr_blob
            if key.endswith("/fisher@1") and key in existing_blob
        }
    )
    for config in configs:
        selection = np.ix_(TEST_INDS, TEST_INDS)
        existing = np.stack([info[selection] for info in existing_blob[f"{config}/fisher@1"]])
        dffr = np.stack([info[selection] for info in dffr_blob[f"{config}/fisher@1"]])
        existing_normalised, existing_diagonal = normalised(existing)
        dffr_normalised, dffr_diagonal = normalised(dffr)
        scale = existing_diagonal[:, :, None] * existing_diagonal[:, None, :]
        per_source = (np.abs(dffr - existing) / scale).max(axis=(1, 2))
        diagonal_ratio = (dffr_diagonal / existing_diagonal) ** 2
        conditions = np.array(
            [
                np.linalg.cond(matrix)
                for matrix in np.concatenate([existing_normalised, dffr_normalised])
            ]
        )
        sigma_ratio = np.sqrt(
            np.abs(np.diagonal(np.linalg.inv(dffr), axis1=1, axis2=2))
            / np.abs(np.diagonal(np.linalg.inv(existing), axis1=1, axis2=2))
        )
        print(f"{config} ({existing.shape[0]} sources, 8 parameters)")
        print(
            f"  max |F_dffr - F_existing| / sqrt(F_ii F_jj): largest {per_source.max():.2e}, "
            f"median over sources {np.median(per_source):.2e}"
        )
        print(
            f"  diagonal ratio dffr / existing: {diagonal_ratio.min():.5f} to "
            f"{diagonal_ratio.max():.5f}"
        )
        print(
            "  largest difference of F_ij / sqrt(F_ii F_jj): "
            f"{np.abs(dffr_normalised - existing_normalised).max():.2e}"
        )
        print(
            f"  condition numbers of the normalised matrices: {conditions.min():.2e} to "
            f"{conditions.max():.2e}"
        )
        print(
            f"  marginal standard deviation ratio dffr / existing: {sigma_ratio.min():.4f} to "
            f"{sigma_ratio.max():.4f}, median {np.median(sigma_ratio):.4f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("existing", metavar="EXISTING_NPZ")
    parser.add_argument("dffr", metavar="DFFR_NPZ")
    args = parser.parse_args(argv)
    compare(args.existing, args.dffr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
