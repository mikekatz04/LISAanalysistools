"""Changes of the Fisher matrices between two golden captures, entry by entry.

    python fisher_entry_changes.py REF.npz NEW.npz

``stft_golden_outputs.py`` normalises a change by the largest value of each array. Here the change
of every entry F_ij is divided by its own scale sqrt(|F_ii F_jj|) in the reference, which shows the
entries whose finite differences enlarge rounding most, such as those of fddot. For every
``fisher*`` array held by both captures the table gives the largest such change, the parameter pair
where it occurs, and the largest change outside the row and column of fddot. Recorded in
docs/response/dffr/tests.md.
"""

from __future__ import annotations

import argparse

import numpy as np

#: Parameter order of STFTGBComputations.information_matrix.
PARAMETER_NAMES = ("amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta")
# * The subset of stft_golden_outputs.py (FISHER_SUBSET), kept here to avoid importing the stack.
SUBSET_NAMES = ("f0", "fdot", "lam", "beta")


def entry_changes(reference: np.ndarray, candidate: np.ndarray):
    """|F_new - F_ref| / sqrt(|F_ii F_jj|), maximised over sources: (num_params, num_params)."""
    diagonal = np.abs(np.diagonal(reference, axis1=-2, axis2=-1))
    scale = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    change = np.abs(candidate - reference)
    relative = np.divide(change, scale, out=np.zeros_like(change), where=scale > 0)
    return relative.max(axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args(argv)

    with np.load(args.reference) as reference_file, np.load(args.candidate) as candidate_file:
        keys = sorted(
            key
            for key in set(reference_file.files) & set(candidate_file.files)
            if key.split("/", 1)[-1].startswith("fisher")
        )
        print(
            f"{'array':40s} {'peak-normalised':>16s} {'largest per entry':>18s} {'at':>14s} "
            f"{'outside fddot':>14s}"
        )
        for key in keys:
            reference = np.asarray(reference_file[key])
            candidate = np.asarray(candidate_file[key])
            names = SUBSET_NAMES if reference.shape[-1] == len(SUBSET_NAMES) else PARAMETER_NAMES
            per_entry = entry_changes(reference, candidate)
            row, column = np.unravel_index(np.argmax(per_entry), per_entry.shape)
            outside = per_entry.copy()
            if "fddot" in names:
                fddot_index = names.index("fddot")
                outside[fddot_index, :] = 0.0
                outside[:, fddot_index] = 0.0
            peak = np.abs(candidate - reference).max() / np.abs(reference).max()
            print(
                f"{key:40s} {peak:16.3e} {per_entry.max():18.3e} "
                f"{names[row] + ',' + names[column]:>14s} {outside.max():14.3e}"
            )


if __name__ == "__main__":
    main()
