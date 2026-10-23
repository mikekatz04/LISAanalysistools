# Browsing the docs locally

The development-branch docs exist only locally. All six packages publish to GitHub Pages only on a `v*`
tag or on a commit whose message contains `[ci:publish-pages]`, so the online sites show the last release,
not `dev`. Build them on the laptop instead; it takes a few minutes and needs no network (see LAT below).

## What there is

Paths below are relative to the umbrella workspace folder that holds the six repos side by side
(`~/Research/lisa_sprint_2026`, called `$WS` in the commands).

| Package | Repo folder (branch) | Built site after a build |
|---|---|---|
| lisatools | `LISAanalysistools` (`dev`) | `LISAanalysistools/docs/build/html/index.html` |
| eryn | `Eryn` (`dev`) | `Eryn/docs/build/html/index.html` |
| gbgpu | `GBGPU` (`dev`) | `GBGPU/docs/build/html/index.html` |
| bbhx | `BBHx` (`dev`) | `BBHx/docs/build/html/index.html` |
| few | `FastEMRIWaveforms` (`gpu_backend`) | `FastEMRIWaveforms/docs/build/html/index.html` |
| gpubackendtools | `GPUBackendTools` (`spline`) | `GPUBackendTools/docs/build/html/index.html` |

Each site has an API section under `user/` built from the docstrings. Besides the Sphinx sites there are
hand-written Markdown guides, which are not part of any site and are best read in an editor (VS Code:
open the file, then Cmd+Shift+V for the preview):

- every repo: `CLAUDE.md` (rules, layout) and `docs/codebase-map.md` (where each module lives);
- LAT `docs/`: `global-fit-navigation.md` (launcher, driver, recipe, moves, store, monitor),
  `global-fit-deep-dives.md` (how a run is examined), `conventions.md`, `architecture-map.md`
  (cross-repo map) and the feature notes (`emri-direct-wdm.md`, `sobbh-wdm-lookup.md`,
  `gb-para-search.md`, `warm-start-gb-proposal.md`, ...).

Six of the LAT guides (`conventions`, `architecture-map`, `codebase-map`, `global-fit-launch`,
`stock-stages-and-moves`, `multigpu-cluster-validation`) also appear in the LAT site under
"Developer guides"; edit the originals in `docs/`, not the copies the build makes.

## Build

Use the `deving` environment (Sphinx 9.1, `sphinx_rtd_theme`, `myst_parser`, `nbsphinx`,
`sphinx-tippy`; the six packages installed editable). The command is the one CI uses, the same in every
repo. One package:

```bash
conda activate deving
WS=~/Research/lisa_sprint_2026          # the umbrella folder holding the six repos
cd $WS/LISAanalysistools
OMP_NUM_THREADS=1 sphinx-build -M html docs/source docs/build -j 1
open docs/build/html/index.html
```

All six, one after another (keeps to one python process at a time):

```bash
conda activate deving
WS=~/Research/lisa_sprint_2026
for r in GPUBackendTools Eryn BBHx GBGPU FastEMRIWaveforms LISAanalysistools; do
  (cd $WS/$r && OMP_NUM_THREADS=1 sphinx-build -M html docs/source docs/build -j 1 -q)
done
```

Measured on the laptop (2026-10-22): GPUBackendTools 10 s, FastEMRIWaveforms 25 s, Eryn 41 s, BBHx 74 s,
GBGPU 76 s, LAT 158 s. `docs/build/` is gitignored in every repo. A rebuild after editing a docstring only
re-renders the changed pages; delete `docs/build` for a clean build.

The search box works on the `file://` pages. If a browser blocks it, serve the folder instead:

```bash
python -m http.server --directory docs/build/html 8000
```

and open `http://localhost:8000`.

## What a build does besides writing `docs/build`

- **LAT** clones the LATW repo at the commit pinned in `docs/latw_ref.txt` and copies its notebooks
  `00`..`08` into `docs/source/latw/` (network; offline it prints a warning and builds without them). It
  also copies the six developer guides into `docs/source/devguides/`. Both folders are untracked copies.
- **GBGPU** regenerates the tracked `docs/source/README.rst` from `README.md`. If you do not want that
  diff, undo it with `git -C $WS/GBGPU checkout -- docs/source/README.rst`.
- **BBHx** copies `examples/*.ipynb` into `docs/source/` (`mbh_tdionfly_tutorial.ipynb` there is untracked).
- autodoc imports the packages, so build in `deving`. A page whose module needs a package that is not
  installed (jax, mpi4py, phentax, or FEW/bbhx for LAT's source pages) builds with an import warning and an
  empty section.

## Warnings you will see (expected)

`unsupported theme option 'display_version'` and `html_static_path entry '_static' does not exist`
(set in each `conf.py`), sphinx-tippy failing to fetch a DOI when offline, and nbsphinx "File not found"
for links from the BBHx / GBGPU notebooks to files in sibling repos. On 2026-10-22 the totals were
LAT 4, GBGPU 8, BBHx 6, Eryn 2, GPUBackendTools 2, FastEMRIWaveforms 0, with no errors.

## Finding your way in the LAT site

The front page has four groups: **Core library** (domains, sensitivity, data and analysis containers,
response, sources, sampling, backends, JAX), **Global fit** (stock fits, recipe and moves, internals, warm
start, monitor, and a legacy table of dead or superseded modules), **Workshop tutorials** (the LATW
notebooks) and **Developer guides**. The production global fit is `erebor.all_sources`; start from the
"Stock global fits" page, then "Global-fit internals".
