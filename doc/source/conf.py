import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Implicit Reparametrization Trick'
author = 'Matvei Kreinin, Maria Nikitina, Petr Babkin, Irina Zaboryanskaya'
release = '0.1'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.ifconfig', 'sphinx.ext.viewcode',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.autosummary', 'sphinx.ext.mathjax',
              'sphinx_rtd_theme']

autodoc_mock_imports = ["numpy", "scipy", "sklearn", "torch"]

templates_path = ['_templates']
exclude_patterns = []

html_extra_path = []

html_context = {
    "display_github": True,
    "github_user": "Intelligent-Systems-Phystech",
    "github_repo": "ProjectTemplate",
    "github_version": "master",
    "conf_py_path": "/doc/source/",
}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
