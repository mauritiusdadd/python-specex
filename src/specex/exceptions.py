#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Emission and absorption line dictionary

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
