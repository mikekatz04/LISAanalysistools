# LISA Analysis Tools

[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://mikekatz04.github.io/LISAanalysistools)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138723.svg)](https://doi.org/10.5281/zenodo.17138723)

LISA Analysis Tools is a package for performing LISA Data Analysis tasks, including building the LISA Global Fit.  

To install the latest version of `lisaanalysistools` using `pip`, simply run:

```sh
# For CPU-only version
pip install lisaanalysistools

# For GPU-enabled versions with CUDA 11.Y.Z
pip install lisaanalysistools-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install lisaanalysistools-cuda12x
```

To know your CUDA version, run the tool `nvidia-smi` in a terminal a check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

You may also install `lisaanalysistools` directly using conda (including on Windows)
as well as its CUDA 12.x plugin (only on Linux). It is strongly advised to:

1. Ensure that your conda environment makes sole use of the `conda-forge` channel
2. Install `lisaanalysistools` directly when building your conda environment, not afterwards

```sh
# For CPU-only version, on either Linux, macOS or Windows:
conda create --name lisatools_cpu -c conda-forge --override-channels python=3.12 lisaanalysistools
conda activate lisatools_cpu

# For CUDA 12.x version, only on Linux
conda create --name lisatools_cuda -c conda-forge --override-channels python=3.12 lisaanalysistools-cuda12x
conda activate lisatools_cuda
```

Now, in a python file or notebook:

```py3
import lisatools
```

You may check the currently available backends:

```py3
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
...     print(f" - Backend '{backend}': {"available" if lisatools.has_backend(backend) else "unavailable"}")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda': unavailable
 - Backend 'gpu': unavailable
```

Note that the `cuda` backend is an alias for either `cuda11x` or `cuda12x`. If any is available, then the `cuda` backend is available.
Similarly, the `gpu` backend is (for now) an alias for `cuda`.

If you expected a backend to be available but it is not, run the following command to obtain an error
message which can guide you to fix this issue:

```py3
>>> import lisatools
>>> lisatools.get_backend("cuda12x")
ModuleNotFoundError: No module named 'lisatools_backend_cuda12x'

The above exception was the direct cause of the following exception:
...

lisatools.cutils.BackendNotInstalled: The 'cuda12x' backend is not installed.

The above exception was the direct cause of the following exception:
...

lisatools.cutils.MissingDependencies: LISAanalysistools CUDA plugin is missing.
    If you are using lisatools in an environment managed using pip, run:
        $ pip install lisaanalysistools-cuda12x

The above exception was the direct cause of the following exception:
...

lisatools.cutils.BackendAccessException: Backend 'cuda12x' is unavailable. See previous error messages.
```

Once LISA Analysis Tools is working and the expected backends are selected, check out the [examples notebooks](https://github.com/mikekatz04/LISAanalysistools/tree/master/examples/)
on how to start with this software.

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- A Python version supported by [scikit-build-core](https://github.com/scikit-build/scikit-build-core) (>=3.7 as of Jan. 2025)

If you want to enable GPU support in LISA Analysis Tools, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`).


### Installation instructions using conda

We recommend to install LISA Analysis Tools using conda in order to have the compilers all within an environment.
First clone the repo

```
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools
```

Now create an environment (these instructions work for all platforms but some
adjustements can be needed, refer to the
[detailed installation documentation](https://lisaanalysistools.readthedocs.io/en/stable/user/install.html) for more information):

```
conda create -n lisatools_env -y -c conda-forge --override-channels |
    cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate lisatools_env
```

Then we can install locally for development:
```
pip install -e '.[dev, testing]'
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
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://lisaanalysistools.readthedocs.io/en/stable/user/install.html). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/mikekatz04/LISAanalysistools/issues)
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
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools
pip install '.[testing]'
```

To run the tests, open a terminal in a directory containing the sources of LISA Analysis Tools and then run the `unittest` module in `discover` mode:

```sh
$ git clone https://github.com/mikekatz04/LISAanalysistools.git
$ cd LISAanalysistools
$ python -m lisatools.tests  # or "python -m unittest discover"
...
----------------------------------------------------------------------
Ran 20 tests in 71.514s
OK
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop LISA Analysis Tools and produce documentation, install `lisatools` from source with the `[dev]` label and in `editable` mode:

```
$ git clone https://github.com/mikekatz04/LISAanalysistools.git
$ cd LISAanalysistools
pip install -e '.[dev, testing]'
```

This will install necessary packages for building the documentation (`sphinx`, `pypandoc`, `sphinx_rtd_theme`, `nbsphinx`) and to run the tests.

The documentation source files are in `docs/source`. To compile the documentation locally, change to the `docs` directory and run `make html`.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/LISAanalysistools/tags).

## Contributors

A (non-exhaustive) list of contributors to the LISA Analysis Tools code can be found in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Citation

Please make sure to cite LISA Analysis Tools papers and the LISA Analysis Tools software on [Zenodo](https://zenodo.org/records/17138723).
We provide a set of prepared references in [PAPERS.bib](PAPERS.bib). There are other papers that require citation based on the classes used. For most classes this applies to, you can find these by checking the `citation` attribute for that class.  All references are detailed in the [CITATION.cff](CITATION.cff) file.

