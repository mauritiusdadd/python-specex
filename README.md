# python-specex [![Build Status](https://github.com/mauritiusdadd/python-specex/actions/workflows/build-and-check.yml/badge.svg)](https://github.com/mauritiusdadd/python-specex/actions/workflows/build-and-check.yml) [![Documentation Status](https://readthedocs.org/projects/python-specex/badge/?version=latest)](https://python-specex.readthedocs.io/en/latest/?badge=latest)

Extract spectra from fits cubes

# SETUP

To install specex use the following command:

    $ pip install -r requirements.txt
    $ pip install .

If you want to use the rrspecex script and the correspondig module, make sure to install also redrock. If you don't already have a system wide installation of redrock, a simple script is provided that creates a python venv and downloads and installs the required packages, in this case the commands to install specex are the following:

    $ chmod +x redrock_venv_setup.sh
    $ ./redrock_venv_setup.sh
    $ . ./redrock_venv/bin/activate
    $ pip install -r rrspecex-requirements.txt
    $ pip install .


To build the documentation install the reuirements and run sphinx:

    $ pip install -r docs/requirements.txt
    $ sphinx-build -b html docs/ docs/_build/html

# DOCUMENTATION

The full documentation is available here: [https://python-specex.readthedocs.io/en/latest/index.html]
