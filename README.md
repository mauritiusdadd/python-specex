# python-spex [![Build Status](https://github.com/mauritiusdadd/python-spex/actions/workflows/build-and-check.yml/badge.svg)](https://github.com/mauritiusdadd/python-spex/actions/workflows/build-and-check.yml) [![Documentation Status](https://readthedocs.org/projects/python-spex/badge/?version=latest)](https://python-spex.readthedocs.io/en/latest/?badge=latest)

Extract spectra from fits cubes

# SETUP

To install spex use the following command:

    $ pip install -r requirements.txt
    $ pip install .

If you want to use the rrspex script and the correspondig module, make sure to install also redrock. If you don't already have a system wide installation of redrock, a simple script is provided that creates a python venv and downloads and installs the required packages, in this case the commands to install spex are the following:

    $ chmod +x redrock_venv_setup.sh
    $ ./redrock_venv_setup.sh
    $ . ./redrock_venv/bin/activate
    $ pip install -r rrspex-requirements.txt
    $ pip install .


To build the documentation install the reuirements and run sphinx:

    $ pip install -r docs/requirements.txt
    $ sphinx-build -b html docs/ docs/_build/html

# DOCUMENTATION

The full documentation is available here: [https://python-spex.readthedocs.io/en/latest/index.html]
