# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# docs/source/conf.py

import os
import sys
# This line tells Sphinx to look in the parent directory for your code
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MacroscopicQED'
copyright = '2025, Guangming Liu'
author = 'Guangming Liu'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # Google/NumPy docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",      # render .. math:: and inline math
]
autosummary_generate = True
autodoc_mock_imports = [
    # Heavy scientific / simulation deps not needed for doc builds
    "joblib",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "h5py",
    "qutip",
    "meep",
    "mpb",
    "pyvista",
    "tqdm",
    "mpi4py",
    # Hydra / OmegaConf
    "hydra",
    "omegaconf",
    # Logging
    "loguru",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -- Logo & branding ---------------------------------------------------------
html_logo = '_static/logo.png'
html_favicon = '_static/logo.png'

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82B1FF",
        "color-brand-content": "#82B1FF",
    },
}

# Force light mode as default (Furo respects data-theme on <body>)
html_context = {
    "default_mode": "light",
}
