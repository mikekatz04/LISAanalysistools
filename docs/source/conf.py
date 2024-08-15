# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"

import shutil

shutil.copy(
    dir_path + "examples/lisatools_tutorial.ipynb",
    dir_path + "docs/source/tutorial/lisatools_tutorial.ipynb",
)

sys.path.insert(0, os.path.abspath("../../"))

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, "../../lisatools", "_version.py"), encoding="utf-8") as f:
    exec(f.read(), about)
# from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------

project = about["__name__"]
copyright = about["__copyright__"]
author = about["__author__"]

# The full version, including alpha/beta/rc tags
release = about["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx_autodoc_typehints",
]

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": ["../examples_ucb"],  # , "../examples_smbh"],
    # path to where to save gallery generated output
    "gallery_dirs": ["examples_ucb"],  # , "examples_smbh"],
}

autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "ArrayLike",
}

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "special-members": "__call__",
}

typehints_defaults = "comma"

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/{.major}".format(sys.version_info),
        None,
    ),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/", None),
}

latex_engine = "xelatex"
