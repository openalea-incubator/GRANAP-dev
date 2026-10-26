"""Sphinx configuration of GRANAP's documentation: `python -m sphinx -b html doc doc/_build/html`. The Markdown files
of this folder are development notes, outside the documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "GRANAP"
author = "the GRANAP authors"
extensions = [
    "sphinx.ext.autodoc",          # docstrings
    "sphinx.ext.mathjax",          # equations
    "sphinx.ext.viewcode",         # links to the source
    "openalea.metafspm.sphinx",    # metafspm components rendered from their declarations
]
exclude_patterns = ["_build", "callgraph", "*.md"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
