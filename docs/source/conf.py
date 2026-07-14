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
import shutil
import sys


dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"

sys.path.insert(0, os.path.abspath("../../"))

here = os.path.abspath(os.path.dirname(__file__))

# Pull the LATW workshop's informational notebooks (tutorials/00_*..08_*) at
# the pinned dev commit recorded in docs/latw_ref.txt and copy them into
# docs/source/latw/ so nbsphinx renders them (from their committed outputs;
# see nbsphinx_execute = "never" below). The exercises under
# tutorials/further/ are a linked mention in index.rst only -- not fetched or
# rendered here. fetch_latw lives in docs/, so put docs/ on sys.path first.
docs_dir = os.path.abspath(os.path.join(here, ".."))  # .../docs
sys.path.insert(0, docs_dir)
from fetch_latw import fetch_latw  # noqa: E402

fetch_latw(here)

# Copy the cross-repo developer guides (docs/*.md) into docs/source/devguides/
# so myst_parser can render them inside the docs tree. Sphinx toctree entries
# must live within the source directory, so we cannot point a toctree at
# ../../docs/*.md directly -- we copy them in at build time instead.
_devguides = [
    "conventions",
    "architecture-map",
    "codebase-map",
    "global-fit-launch",
    "stock-stages-and-moves",
    "multigpu-cluster-validation",
]
_devguides_dest = os.path.join(here, "devguides")
os.makedirs(_devguides_dest, exist_ok=True)
for _name in _devguides:
    _src = os.path.join(docs_dir, _name + ".md")
    if os.path.exists(_src):
        shutil.copy(_src, os.path.join(_devguides_dest, _name + ".md"))
    else:
        print(f"[conf] WARNING: developer guide not found: {_src}")


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
html_theme = "sphinx_rtd_theme"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx_tippy",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

source_suffix = [".rst", ".md"]

# The LATW workshop notebooks ship with committed outputs; never re-execute
# them at build time (they need the full LISA dev stack + data to run).
nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# readme.md is pulled into index.rst via an ``.. include::`` directive, so it
# must not also be treated as a standalone source document now that ".md" is a
# recognized source suffix (avoids a spurious orphan / duplicate-heading page).
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "readme.md"]

import sphinx_rtd_theme

autodoc_member_order = "bysource"


def skip(app, what, name, obj, would_skip, options):
    if name == "__call__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_nav_header_background": "coral",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
}

