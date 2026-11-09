#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Emission and absorption line dictionary

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

    if HAS_IPYTHON:
        try:
            traceback_formatter = ultratb.FormattedTB(
                mode='Verbose', theme_name='linux', call_pdb=1
            )
        except TypeError:
            traceback_formatter = ultratb.FormattedTB(
                mode='Verbose', color_scheme='Linux', call_pdb=1
            )
        return traceback_formatter(exception_type, value, traceback)
    else:
        sys.__excepthook__(exception_type, value, traceback)
        print("\n*** IPython not installed, cannot start the debugger! ***\n")

    sys.exit(1)
