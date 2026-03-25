"""Sphinx configuration for unite documentation."""

import unite

# -- Project information -----------------------------------------------------

project = 'unite'
author = 'Raphael Erik Hviding'
copyright = '2025, Raphael Erik Hviding'
version = unite.__version__
release = unite.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/README.rst', 'auto_tutorials/index.rst']

# Suppress cross-reference ambiguity warnings that arise because public classes
# are registered at both the package level (e.g. unite.line.LineConfiguration)
# and the submodule level (e.g. unite.line.config.LineConfiguration).
suppress_warnings = [
    'ref.python',        # cross-ref ambiguity: classes re-exported at package & submodule level
    'py.duplicate-object',  # numpydoc Attributes section + autodoc dataclass field duplication
]

# Minimum Sphinx version
needs_sphinx = '8.0'

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'

# -- Options for numpydoc ----------------------------------------------------

numpydoc_show_class_members = False  # let autodoc handle member listing

# -- Options for MyST --------------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'substitution',
]
myst_heading_anchors = 3

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'numpyro': ('https://num.pyro.ai/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#A40122",
        "color-brand-content": "#E20134",
        "color-link--visited": "#E20134",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FFC33B",
        "color-brand-content": "#FF6E3A",
        "color-link--visited": "#FF6E3A",
    },
    # 'style_nav_header_background': '#2ecc71',
}
html_title = f'unite v{version}'

sphinx_gallery_conf = {
    'examples_dirs': ['tutorials'],
    'gallery_dirs': ['auto_tutorials'],
    'filename_pattern': r'/tutorial_',
    'plot_gallery': True,
    'abort_on_example_error': True,
    'reset_modules': ('matplotlib',),
    'min_reported_time': 0,
}
