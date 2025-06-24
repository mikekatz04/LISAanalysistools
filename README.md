# LISA Analysis Tools

[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://mikekatz04.github.io/LISAanalysistools)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10930980.svg)](https://doi.org/10.5281/zenodo.10930980)

LISA Analysis Tools is a package for performing LISA Data Analysis tasks, including building the LISA Global Fit.  

## 1 - Getting Started

These instructions will get you a copy of the project up and running on your local machine,
either for development and testing purposes or as an installed package.  For more information, see the documentation at https://mikekatz04.github.io/LISAanalysistools.

# Installation

You can install with `pip`:
```
pip install lisaanalysistools
```

If you want to install all of the tools associated with LISA Analysis Tools (Fast EMRI Waveforms, BBHx, GBGPU, fastlisaresponse, eryn), see the following instructions.

LISA Analysis Tools leverages conda environments to install and use necessary packages. If you do not have [Anaconda](https://www.anaconda.com/download) or [miniconda](https://docs.anaconda.com/free/miniconda/index.html) installed, you must do this first and load your `base` conda environment. Recommended components for install in your conda environment are `lapack`, `gsl`, `hdf5`, which are needed for various waveform packages. 

For an easy full install, follow these instructions.

First, clone the repo and `cd` to the `LISAanalysistools` directory.:
```
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools/
```

Install all packages necessary for the tutorials by running:
```
bash install.sh
```
Running `bash install.sh -h` will also give you some basic install options. 

If you want more flexibility, you can install each package given above separately.

To install this software for use with NVIDIA GPUs (compute capability >5.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install CuPy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDA_HOME` environment variable.

We are currently working on building wheels and making the GPU version pip installable. For now, to work with GPUs, git clone the repository and install it from source. You must run `python scripts/prebuild.py` before running the install process.  

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/LISAanalysistools/tags).

Current Version: 1.0.15

## Authors/Developers

* **Michael Katz**
* Lorenzo Speri
* Christian Chapman-Bird
* Natalia Korsakova
* Nikos Karnesis

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

```
@software{michael_katz_2024_10930980,
  author       = {Michael Katz and
                  CChapmanbird and
                  Lorenzo Speri and
                  Nikolaos Karnesis and
                  Korsakova, Natalia},
  title        = {mikekatz04/LISAanalysistools: First main release.},
  month        = apr,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.3},
  doi          = {10.5281/zenodo.10930980},
  url          = {https://doi.org/10.5281/zenodo.10930980}
}
```

