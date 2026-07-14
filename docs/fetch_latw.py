"""Fetch the LISA Analysis Tools Workshop (LATW) informational notebooks.

At doc-build time we pull the *informational* workshop tutorials
(``tutorials/00_*.ipynb`` .. ``tutorials/08_*.ipynb``) from the LATW repo at a
pinned dev commit and drop them into ``docs/source/latw/`` so nbsphinx can
render them (from their committed outputs -- see ``nbsphinx_execute="never"``
in ``conf.py``).  The exercises under ``tutorials/further/`` are intentionally
*not* copied: they are a linked mention in the docs, not rendered pages.

The pinned commit is read from ``docs/latw_ref.txt`` (never hardcoded here).
If the fetch fails (e.g. building offline) we print a warning and skip, so the
Sphinx build still succeeds -- it just won't contain the workshop notebooks.
Re-running overwrites the copies, so this is idempotent.
"""

import glob
import os
import shutil
import subprocess
import tempfile

LATW_REPO = "https://github.com/lisa-analysis-tools/LATW"
LATW_DEV_BRANCH = "dev"


def _read_ref(here):
    """Return the pinned LATW commit from ``docs/latw_ref.txt`` (stripped)."""
    ref_file = os.path.join(here, "latw_ref.txt")
    with open(ref_file) as fh:
        return fh.read().strip()


def fetch_latw(docs_source_dir):
    """Clone LATW at the pinned ref and copy the 00..08 notebooks into ``latw/``.

    Parameters
    ----------
    docs_source_dir : str
        Path to ``docs/source`` -- the informational notebooks are written into
        ``<docs_source_dir>/latw/``.
    """
    # ``latw_ref.txt`` and this module both live in ``docs/``.
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        ref = _read_ref(here)
    except OSError as exc:
        print(f"[fetch_latw] WARNING: could not read latw_ref.txt ({exc}); "
              "skipping LATW notebook fetch")
        return

    dest = os.path.join(docs_source_dir, "latw")
    tmp = tempfile.mkdtemp(prefix="latw_clone_")
    try:
        # Fetch just the pinned commit. GitHub permits fetching any reachable
        # SHA; fall back to the dev tip + checkout if the server rejects a bare
        # SHA fetch.
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "remote", "add", "origin", LATW_REPO],
                       check=True)
        try:
            subprocess.run(
                ["git", "-C", tmp, "fetch", "-q", "--depth", "1", "origin", ref],
                check=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "-C", tmp, "fetch", "-q", "--depth", "100", "origin",
                 LATW_DEV_BRANCH],
                check=True,
            )
        subprocess.run(["git", "-C", tmp, "checkout", "-q", ref], check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[fetch_latw] WARNING: could not fetch LATW at {ref} ({exc}); "
              "skipping -- docs will build without the workshop notebooks")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    # Copy ONLY the informational top-level tutorials (00_*..08_*); the
    # non-recursive glob naturally excludes tutorials/further/.
    notebooks = sorted(
        glob.glob(os.path.join(tmp, "tutorials", "[0-9][0-9]_*.ipynb"))
    )
    os.makedirs(dest, exist_ok=True)
    for nb in notebooks:
        shutil.copy(nb, os.path.join(dest, os.path.basename(nb)))
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"[fetch_latw] copied {len(notebooks)} LATW informational notebook(s) "
          f"at {ref} into {dest}")
    return notebooks
