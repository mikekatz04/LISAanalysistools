# LISA Analysis Tools

[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://lisa-analysis-tools.github.io/lisa-analysis-tools)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138723.svg)](https://doi.org/10.5281/zenodo.17138723)

LISA Analysis Tools (`lisaanalysistools`, imported as `lisatools`) is a package for LISA data
analysis, from a one-line SNR to the LISA global fit: detector orbits and the TDI response,
noise models, time / frequency / WDM domains, the likelihood containers, source waveforms,
and the global-fit pipeline (`lisatools.globalfit`) with its installed stock run
configurations. Python 3.12 or newer is required.

To install the latest release of `lisaanalysistools` using `pip`, run:

```sh
# For CPU-only version
pip install lisaanalysistools

# For GPU-enabled versions with CUDA 12.Y.Z (Linux x86_64)
pip install lisaanalysistools-cuda12x
```

The GPU package is a backend plugin (`lisatools_backend_cuda12x`, plus its `cupy-cuda12x`
dependency) that the main package loads lazily. The release workflow currently builds only
the CPU package and the `cuda12x` plugin; the backend code also supports `cuda11x` and
`cuda13x` (the global-fit cluster runs use `cuda13x`), which you get by building from source
(see "Installing from sources" below). To know your CUDA version, run the tool `nvidia-smi` in a terminal and check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

Now, in a python file or notebook:

```py3
import lisatools
```

You may check the currently available backends:

```py3
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda13x"]:
...     status = "available" if lisatools.has_backend(backend) else "unavailable"
...     print(f" - Backend '{backend}': {status}")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda13x': unavailable
```

`lisatools.get_backend` also accepts the aliases `"cuda"` and `"gpu"`, which return the first
loadable CUDA backend (`cuda13x`, then `cuda12x`, then `cuda11x`). `lisatools.has_backend`
does not accept the aliases: `has_backend("cuda")` and `has_backend("gpu")` raise `ValueError`,
so check the concrete names as above. With `jax` installed, a pure-JAX backend (`"jax"`) is
registered as well. Compute classes choose their backend once, at construction, with
`force_backend=` (for example `force_backend="cuda12x"`).

If you expected a backend to be available but it is not, call `lisatools.get_backend` with its
name (for example `lisatools.get_backend("cuda12x")`): the `BackendAccessException` it raises,
and the exceptions chained to it, say what is missing (typically the plugin package or CuPy).

Once LISA Analysis Tools is working and the expected backends are selected, the tutorials of
the [LISA Analysis Tools Workshop (LATW)](https://github.com/lisa-analysis-tools/LATW) show how
to start with this software (see "Tutorials" below).

## Quick start: a stock global fit

Run configurations are installed classes, not settings files. Pick a stock fit, adjust its
knobs as plain attributes, build, run:

```python
from lisatools.globalfit.stock import erebor

erebor.get_stock_options()             # [(name, description), ...]
fit = erebor.all_sources(nwalkers=4)   # the production variant (a cheap clone, no data loaded)
fit.general.num_iterations = 50        # run-wide knob (env NUM_ITERATIONS)
fit.emri.ntemps = 12                   # per-branch knob (env EMRI_NTEMPS)
fit.remove_branch("galfor")            # compose whole branches out (their moves go too)
fit.pop_move("sobbh_pe")               # or single moves
print(fit.list_moves())
fit.build()                            # the heavy stage: data load, domain, branch setups
fit.run()                              # MPI run; launch the script with mpiexec
```

Every knob default resolves as *explicit kwarg > env var > lite preset > hard default*; the
environment variable is the capitalized field name with the branch prefix (`gb.min_freq` ->
`GB_MIN_FREQ`). For a single-process loop on a laptop, `fit.sample()` yields `(model, state)`
each iteration; `erebor.blank` is the zero-branch starting point:

```python
fit = erebor.blank(nwalkers=4, ntemps=2, make_diagnostic_plots=False)
for model, state in fit.sample(iterations=3, store=False, progress=False):
    print(state.log_like)
```

The 9-month and 12-month production runs use `all_sources` through the launchers
`scripts/fstat_proposal/submit_gf_9mo_v9.sh` and `scripts/flagship/submit_gf_12mo.sh`; read
`docs/global-fit-navigation.md` before touching a run.

## Tutorials

The tutorials live in the [LISA Analysis Tools Workshop (LATW)](https://github.com/lisa-analysis-tools/LATW)
repository (this repository's `examples/` directory is a pointer only). LATW `main` runs on the
pip releases; LATW `dev` runs on the development stack installed by `install.sh` (below). The
informational notebooks are also rendered in the documentation.

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- Python 3.12 or newer
- for the development stack, LAPACKE: the GPUBackendTools, BBHx, GBGPU and FastEMRIWaveforms
  builds look for it (LISA Analysis Tools itself does not; the installer passes
  `GBT_LAPACKE_DETECT_WITH=PKGCONFIG`, i.e. detection through `pkg-config`, to every compiled
  package)

If you want to enable GPU support in LISA Analysis Tools, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`), and a CuPy build matching your CUDA
version (`cupy-cuda12x`, ...), which a source install does not pull in. The CMake option
`LISATOOLS_WITH_GPU` (`AUTO` by default: the GPU backend is built when the CUDA toolchain is
found; also `ON`, `OFF`, `ONLY`, `BARE`) is set with
`pip install --config-settings=cmake.define.LISATOOLS_WITH_GPU=ON .`.

### The development stack (`install.sh`)

The development stack is several repositories side by side: GPUBackendTools, Eryn, this
repository, BBHx, GBGPU, the LATW tutorials and FastEMRIWaveforms, plus phentax. The
installer in this repository clones (or reuses) the siblings next to this clone,
fast-forward-pulls each on its development branch and editable-installs them in dependency
order (GPUBackendTools -> Eryn -> LISA Analysis Tools -> BBHx -> GBGPU; LATW is checked out but
not installed; phentax is pip-installed from GitHub; FastEMRIWaveforms last):

```sh
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools
./install.sh              # full dev install
./install.sh --pull-only  # later: update the checkouts only (no pip, no compile)
```

Options are environment variables, for example `SKIP_FEW=1`, `SKIP_PHENTAX=1`,
`SKIP_GBGPU=1` (pull but do not recompile), `GIT_PULL=0` (offline) and
`GBT_LAPACKE_DETECT_WITH`; the header of `install.sh` lists them all.

The installer exports `PIP_CONSTRAINT=constraints/sprint.txt`, which pins `nanobind` and
`pybind11` across the stack: every compiled package must be built against the same binding
library version, or objects shared between packages stop being recognized across them.
Export the same constraint yourself before installing any stack package by hand (the
`git rev-parse` form below works from this repository's root; for a sibling repository, use the
absolute path of this clone's `constraints/sprint.txt`):

```sh
export PIP_CONSTRAINT="$(git rev-parse --show-toplevel)/constraints/sprint.txt"
pip install -e '.[dev,testing]'
```


### Installation instructions using conda

We recommend to install LISA Analysis Tools using conda in order to have the compilers all within an environment.
First clone the repo

```
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools
```

Now create an environment (these instructions work for all platforms but some
adjustements can be needed, refer to the
[detailed installation documentation](https://lisa-analysis-tools.github.io/lisa-analysis-tools) for more information):

```
conda create -n lisatools_env -y -c conda-forge --override-channels \
    python=3.12 cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate lisatools_env
```

Then we can install locally for development (with the stack constraint, see above):
```
export PIP_CONSTRAINT="$(git rev-parse --show-toplevel)/constraints/sprint.txt"
pip install -e '.[dev,testing]'
```

### Installation instructions using conda on GPUs and linux
Below is a quick set of instructions to install the LISA Analysis Tools package on GPUs and linux.

```sh
conda create -n lisatools_env -c conda-forge lisaanalysistools-cuda12x python=3.12
conda activate lisatools_env
```

Test the installation device by running python
```python
import lisatools
lisatools.get_backend("cuda12x")
```

### Running the installation

To start the from-source installation, ensure the pre-requisite are met, clone
the repository, and then simply run a `pip install` command:

```sh
# Clone the repository
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://lisa-analysis-tools.github.io/lisa-analysis-tools). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/lisa-analysis-tools/lisa-analysis-tools/issues)
or contact the developers through other means.



### Running the Tests

The tests require a few dependencies which are not installed by default. To install them, add the `[testing]` label to LISA Analysis Tools package
name when installing it. E.g:

```sh
# For CPU-only version with testing enabled
pip install lisaanalysistools[testing]

# For GPU version with CUDA 12.Y and testing enabled
pip install lisaanalysistools-cuda12x[testing]

# For from-source install with testing enabled
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools
pip install '.[testing]'
```

The tests live in the repository's `tests/` directory (they are not shipped with the package).
To run them, open a terminal in the repository root and run the `unittest` module in
`discover` mode, or one test module at a time:

```sh
$ git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
$ cd lisa-analysis-tools
$ python -m unittest discover
$ python -m unittest tests.test_detector
```

Several tests import the other stack packages (GBGPU, BBHx, FastEMRIWaveforms), and some use
CuPy or the mojito data.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop LISA Analysis Tools, install `lisatools` from source in `editable` mode
(the `install.sh` development stack above, or the single-package command). The `[dev]` extra
installs the formatting and lint tools (`black`, `isort`, `mypy`, `pre-commit`, `pylint`), the
`[testing]` extra the test dependencies and the `[doc]` extra the documentation tools
(`sphinx`, `sphinx-rtd-theme`, `myst-parser`, `nbsphinx`, ...):

```
$ git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
$ cd lisa-analysis-tools
$ pip install -e '.[dev,testing,doc]'
```

The documentation source files are in `docs/source`. To compile the documentation locally,
change to the `docs` directory and run `make html`; the pages land in `docs/build/html`. The
build copies six developer guides from `docs/*.md` (the list is in `docs/source/conf.py`) into
`docs/source/devguides/` and fetches the
LATW tutorial notebooks at the commit pinned in `docs/latw_ref.txt` (this needs network
access; offline, the build skips the notebooks with a warning).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/lisa-analysis-tools/lisa-analysis-tools/tags).

## Contributors

A (non-exhaustive) list of contributors to the LISA Analysis Tools code can be found in [CONTRIBUTORS.md](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/CONTRIBUTORS.md).

## License

This project is licensed under the Apache License - see the [LICENSE](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/LICENSE) file for details.

## Citation

Please make sure to cite LISA Analysis Tools papers and the LISA Analysis Tools software on [Zenodo](https://zenodo.org/records/17138723).
All references are detailed in the [CITATION.cff](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/CITATION.cff) file. The `lisatools_citations`
command prints them as BibTeX (`lisatools_citations` for all of them,
`lisatools_citations lisatools.detector` for one module's); classes that declare their own
references do so through `lisatools.utils.citations.Citable`.

