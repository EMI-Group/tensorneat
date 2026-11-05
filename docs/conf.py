"""Sphinx configuration for the TensorNEAT documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version as distribution_version

# Import the package from this checkout when it is not installed (``pip install
# -e .`` / the ``docs`` extra).  Paths are absolute so the build does not depend
# on the directory Sphinx was started from; a missing ``src/`` is not an error,
# the installed distribution is used instead.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if os.path.isdir(SRC_DIR):
    sys.path.insert(0, SRC_DIR)

# -- Project information -----------------------------------------------------

project = "TensorNEAT"
copyright = "2026, Lishuang Wang"
author = "Lishuang Wang"

try:
    release = distribution_version("tensorneat")
except PackageNotFoundError:
    release = "0.2.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
