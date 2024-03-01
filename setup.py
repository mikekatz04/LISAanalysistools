# from future.utils import iteritems
from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

detector_ext = Extension(
    "lisatools.cutils.detector",
    ["src/Detector.cpp", "src/pycppdetector.pyx"],
    include_dirs=["./include", numpy_include],
    language="c++",
)


extensions = [detector_ext]

for e in extensions:
    e.cython_directives = {"language_level": "3"}  # all are Python-3

setup(
    name="LISAanalysistools",
    # Random metadata. there's more you can supply
    author="Michael Katz, Lorenzo Speri, Ollie Burke",
    ext_modules=cythonize(extensions),
    version="0.1",
    packages=[
        "lisatools",
        "lisatools.sampling",
        "lisatools.sampling.moves",
        "lisatools.pipeline",
        "lisatools.utils",
        "lisatools.sources",
        "lisatools.sources.emri",
    ],
)
