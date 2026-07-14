# Examples have moved to the LISA Analysis Tools Workshop (LATW)

The tutorials and worked examples that used to live in this directory now
live in their own repository, the **LISA Analysis Tools Workshop (LATW)**:

- **LATW repository:** https://github.com/lisa-analysis-tools/LATW

## Which branch?

LATW has two branches that track the two ways you can install this stack:

- **`main`** — runs against the **pip-installable releases**. Use this if you
  installed the packages with `pip install lisaanalysistools` (and friends).
  https://github.com/lisa-analysis-tools/LATW/tree/main
- **`dev`** — runs against the **development stack** installed by
  [`LISAanalysistools/install.sh`](../install.sh). Use this if you built the
  packages from source side by side. Pinned dev commit for the current docs:
  **`9961567`**.
  https://github.com/lisa-analysis-tools/LATW/tree/dev

(If you change a dev-only API in any stack package, update the LATW `dev`
tutorials in the same session — see the tutorials/branch-policy rule in the
repo `CLAUDE.md`.)

## What is where

- **Informational tutorials** — `tutorials/00_*.ipynb` … `tutorials/08_*.ipynb`.
  These are also rendered into these Sphinx docs (committed outputs, not
  re-executed at build time). See the **Workshop tutorials** section of the
  [built documentation](https://lisa-analysis-tools.github.io/LISAanalysistools/).
- **Workshop exercises** — hands-on student notebooks (with an `answers/`
  companion set) under
  [`tutorials/further/`](https://github.com/lisa-analysis-tools/LATW/tree/dev/tutorials/further).
  These are **not** rendered into the Sphinx docs; work through them in the
  LATW repo itself.

Please do not add new notebooks to this `examples/` directory — it is a
pointer only. New tutorials go in LATW.
