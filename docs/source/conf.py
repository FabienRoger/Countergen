import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath("../../countergen"))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Countergen"
copyright = "2022, SaferAI"
author = "SaferAI"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

master_doc = "index"

# -- sphinx
autodoc_typehints = "description"
