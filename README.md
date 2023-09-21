# python-specex [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7808292.svg)](https://doi.org/10.5281/zenodo.7808292) [![Documentation Status](https://readthedocs.org/projects/python-specex/badge/?version=latest)](https://python-specex.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://github.com/mauritiusdadd/python-specex/actions/workflows/build-and-check.yml/badge.svg)](https://github.com/mauritiusdadd/python-specex/actions/workflows/build-and-check.yml) [![Coverage Status](https://coveralls.io/repos/github/mauritiusdadd/python-specex/badge.svg?branch=main)](https://coveralls.io/github/mauritiusdadd/python-specex?branch=main)

Extract spectra from fits cubes

# SETUP

To install the latest stable version of specex, just use pip:

    $ pip install specex

Some functionalities are enabled only if some optional packages are installed, these packages can be installed by pip when installing Specex by specifing the desired configuration, for example

    $ pip install specex[animation]

will install the packages required to use the command specex-cube-anim, while with the command

    $ pip install specex[animation,regions,redrock]

also the packages to use rrspecex and to enable region file handling will be installed. To install all the optional dependencies use

    $ pip install specex[all]

To install the bleeding edge version, clone the github repository then use pip:

    $ git clone 'https://github.com/mauritiusdadd/python-specex.git'
    $ cd python-specex
    $ pip install .

If you want to use the rrspecex script and the correspondig module, make sure to install also redrock. If you don't already have a system wide installation of redrock, a simple script is provided that creates a python venv and downloads and installs the required packages, in this case the commands to install specex are the following:

    $ chmod +x redrock_venv_setup.sh
    $ ./redrock_venv_setup.sh
    $ . ./redrock_venv/bin/activate
    $ pip install .[redrock]

# DOCUMENTATION

To build the documentation, install the requirements and run sphinx:

    $ pip install -r docs/requirements.txt
    $ sphinx-build -b html docs/ docs/_build/html

The full documentation is also available here: [https://python-specex.readthedocs.io/en/latest/index.html]
