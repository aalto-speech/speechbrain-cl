"""Sphinx configuration."""
project = "Speechbrain Cl"
author = "Georgios K."
copyright = "2022, Georgios K."
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "sphinxarg.ext",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
