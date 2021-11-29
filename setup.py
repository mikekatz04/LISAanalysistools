# from future.utils import iteritems
from setuptools import setup

setup(
    name="LISAanalysistools",
    # Random metadata. there's more you can supply
    author="Michael Katz, Lorenzo Speri, Ollie Burke",
    version="0.1",
    packages=[
        "lisatools",
        "lisatools.sampling",
        "lisatools.sampling.moves",
        "lisatools.pipeline",
        "lisatools.utils",
    ],
)
