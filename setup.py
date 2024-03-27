# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup
from setuptools import Extension

# from Cython.Distutils import build_ext
import numpy
import shutil


# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


detector_ext = Extension(
    "lisatools.cutils.detector",
    sources=["src/Detector.cpp", "src/pycppdetector.pyx"],
    include_dirs=["./include", numpy_include],
    language="c++",
    # extra_compile_args={"gcc": [], "nvcc": []},
)

extensions = [detector_ext]

with open("README.md", "r") as fh:
    long_description = fh.read()

# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

for line in lines:
    if line.startswith("Current Version"):
        version_string = line.split("Current Version: ")[1].split("\n")[0]

with open("lisatools/_version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(version_string))
    f.write('__copyright__ = "Michael L. Katz 2024"\n')
    f.write('__name__ = "lisaanalysistools"\n')
    f.write('__author__ = "Michael L. Katz"\n')

setup(
    name="lisaanalysistools",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    ext_modules=extensions,
    # Inject our custom trigger
    packages=[
        "lisatools",
        "lisatools.sampling",
        "lisatools.sampling.moves",
        "lisatools.utils",
        "lisatools.sources",
        "lisatools.sources.emri",
        "src",
    ],
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version_string,
    url="https://github.com/mikekatz04/lisa-on-gpu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Environment :: GPU :: NVIDIA CUDA",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
)

import sys
import os.path


for tmp in os.listdir((path_to_package := numpy.__file__.split("numpy")[0])):
    if tmp[:9] == "lisatools":
        tmp2 = path_to_package + tmp + "/lisatools/cutils/"
        if not os.path.exists(tmp2 + "src"):
            os.mkdir(tmp2 + "src/")
        if not os.path.exists(tmp2 + "include"):
            os.mkdir(tmp2 + "include/")

        shutil.copy("src/Detector.cpp", tmp2 + "src/Detector.cpp")
        shutil.copy("include/Detector.hpp", tmp2 + "include/Detector.hpp")
