#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 11:45:22 2023

@author: daddona
"""
import sys

try:
    from IPython.core import ultratb
    from IPython import embed
except Exception:
    HAS_IPYTHON = False
else:
    HAS_IPYTHON = True


def get_ipython_embedder():
    """
    Embed an IPython console, if IPython is installed.

    Returns
    -------
    None.

    """
    if HAS_IPYTHON:
        return embed


def exception_handler(exception_type, value, traceback):
    """
    Start ipydb.

    Parameters
    ----------
    exception_type : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.
    traceback : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    global HAS_IPYTHON

    if HAS_IPYTHON:
        traceback_formatter = ultratb.FormattedTB(
            mode='Verbose', color_scheme='Linux', call_pdb=1
        )
        return traceback_formatter(exception_type, value, traceback)
    else:
        sys.__excepthook__(exception_type, value, traceback)
        print("\n*** IPython not installed, cannot start the debugger! ***\n")

    sys.exit(1)
