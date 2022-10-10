# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "CounterGen"
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
    "sphinx.ext.intersphinx",
    # "autoapi.extension",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- autoapi
# autoapi_type = "python"
# autoapi_dirs = ["../countergen/", "../countergenedit/"]
autodoc_typehints = "description"

# Fix default arg expansion https://stackoverflow.com/questions/29257961/sphinx-documentation-how-to-disable-default-argument-expansion
import re


def remove_default_value_when_too_long(app, what, name, obj, options, signature: str, return_annotation):
    if signature:
        terms = signature[1:-1].split("'")
        signature = "({})".format("'".join([t if len(t) < 150 else "..." for t in terms]))
    return (signature, return_annotation)


def setup(app):
    print(app.connect("autodoc-process-signature", remove_default_value_when_too_long))
