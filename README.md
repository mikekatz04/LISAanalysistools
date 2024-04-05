# LISA Analysis Tools

[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://mikekatz04.github.io/LISAanalysistools)

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

LISA Analysis Tools leverages conda environments to install and use necessary packages. If you do not have [Anaconda](https://www.anaconda.com/download) or [miniconda](https://docs.anaconda.com/free/miniconda/index.html) installed, you must do this first and load your `base` conda environment. 

First, clone the repo and `cd` to the `LATW` directory.:
```
git clone https://github.com/mikekatz04/LISAanalysistools.git
cd LISAanalysistools/
```

Install all packages necessary for the tutorials by running:
```
bash install.sh
```
Running `bash install.sh -h` will also give you some basic install options. 

If you want more flexibility, you can install each package given above separately. If you do this, you will also need # TODO: add. 

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags).

Current Version: 1.0.3

## Authors/Developers

* **Michael Katz**

### Contibutors

* Lorenzo Speri
* Christian Chapman-Bird

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

TODO.

