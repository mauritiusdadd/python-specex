# python-spex
[![Build Status](https://app.travis-ci.com/mauritiusdadd/python-spex.svg?token=fRNrxziGGvs3HmNyD6gZ&branch=main)](https://app.travis-ci.com/mauritiusdadd/python-spex)
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


To build the documentation run the command:

    $ sphinx-build -b html docs/source/ docs/build/html
