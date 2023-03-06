# python-specex [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7808292.svg)](https://doi.org/10.5281/zenodo.7808292) [![Build Status](https://github.com/mauritiusdadd/python-specex/actions/workflows/build-and-check.yml/badge.svg)](https://github.com/mauritiusdadd/python-specex/actions/workflows/build-and-check.yml) [![Documentation Status](https://readthedocs.org/projects/python-specex/badge/?version=latest)](https://python-specex.readthedocs.io/en/latest/?badge=latest)

Extract spectra from fits cubes

# SETUP

To install the latest stable version of specex, just use pip:

    $ pip install specex

To install the bleeding edge version, clone the github repository use the command:

    $ pip install .

If you want to use the rrspecex script and the correspondig module, make sure to install also redrock. If you don't already have a system wide installation of redrock, a simple script is provided that creates a python venv and downloads and installs the required packages, in this case the commands to install specex are the following:

    $ chmod +x redrock_venv_setup.sh
    $ ./redrock_venv_setup.sh
    $ . ./redrock_venv/bin/activate
    $ pip install -r rrspecex-requirements.txt
    $ pip install .

# DOCUMENTATION

To build the documentation, install the requirements and run sphinx:

    $ pip install -r docs/requirements.txt
    $ sphinx-build -b html docs/ docs/_build/html

The full documentation is also available here: [https://python-specex.readthedocs.io/en/latest/index.html]
