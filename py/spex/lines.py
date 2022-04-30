"""
SPEX - SPectra EXtractor.

Emission and absorption line dictionary

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>

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

import numpy as np


# Some important lines with corresponding wavelenghts in Angstrom
RESTFRAME_LINES = [
    (10320, '[SII]', 'E'),
    (8863.0, 'TiO', 'A'),
    (8430.0, 'TiO', 'A'),
    (8195.0, 'NaI', 'A'),
    (8183.0, 'NaI', 'A'),
    (7590.0, 'TiO', 'A'),
    (7065.2, 'HeI', 'AE'),
    (6725.0, '[SII]', 'E'),
    (6562.8, 'Halpha', 'AEB'),
    (6159.0, 'TiO', 'A'),
    (5892.5, 'NaD', 'A'),
    (5603.0, 'TiO', 'A'),
    (5269.0, 'Ca,Fe', 'A'),
    (5175.4, 'MgI', 'A'),
    (5006.8, '[OIII]', 'E'),
    (4958.9, '[OIII]', 'E'),
    (4861.3, 'Hbeta', 'AEB'),
    (4340.4, 'Hgamma', 'AE'),
    (4304.4, 'Gband', 'A'),
    (4216.0, 'CN', 'A'),
    (4101.7, 'Hdelta', 'AE'),
    (4000.0, 'Balmer_Break', 'Break'),
    (4072.0, '[SII]', 'E'),
    (3968.5, 'CaII_H', 'A'),
    (3933.7, 'CaII_K', 'A'),
    (3889.1, 'Hksi,CN(H8)', 'AE'),
    (3869.0, '[NeIII]', 'E'),
    (3797.9, 'Hteta', 'AE'),
    (3770.6, 'H11', 'AE'),
    (3727.5, '[OII]', 'E'),
    (3581.0, 'FeI', 'A'),
    (3425.8, '[NeV]', 'E'),
    (3345.9, '[NeV]', 'E'),
    (2964.0, 'FeII_bump', 'E'),
    (2799.0, 'MgII', 'AEB'),
    (2626.0, 'FeII', 'E'),
    (2600.0, 'FeII', 'A'),
    (2586.7, 'FeII', 'A'),
    (2382.0, 'FeII', 'A'),
    (2374.0, 'FeII', 'A'),
    (2344.2, 'FeII', 'A'),
    (2260.0, 'FeII', 'A'),
    (2142.0, '[NII]', 'E'),
    (1909.0, '[CIII]', 'EB'),
    (1856.0, 'AlIII', 'A'),
    (1670.8, 'AlII', 'A'),
    (1640.0, 'HeII', 'AE'),
    (1608.5, 'FeII', 'A'),
    (1549.0, 'CIV', 'AEB'),
    (1526.7, 'SiII', 'A'),
    (1397.0, 'SiIV+OIV', 'AEB'),
    (1334.5, 'CII', 'AE'),
    (1303.0, 'OI', 'AE'),
    (1260.4, 'SiII', 'A'),
    (1240.0, 'NV', 'AE'),
    (1215.7, 'LyA', 'AEB'),
    (1033.0, 'OVI', 'AE'),
    (1025.6, 'LyB', 'AE'),
    (972.5, 'LyG', 'AE'),
]


def get_lines(name=None, line_type=None, wrange=None, z=0):
    """
    Return line data according to the given line name and types.

    Parameters
    ----------
    name : str or None, optional
        The name of the line (eg. CaII_H or FeI, etc...). If None, the lines
        are selected only by type. If both name and type are None, all lines
        are returned.
    line_type : str or None, optional
        Type of the line, can be 'A' (absorption), 'E' (emission) 'B' (Broad).
        If None, then all the line types are returned.
        The default is None.
    wrange : tuple/list/np.ndarray of floats or None, optional
        The wavelength ragne in which lines should be. If None, no selection
        according to the line wavelenght is made.
        The default is None.
    z : float, optional
        The redshit of the lines. The default value is 0.

    Returns
    -------
    selected_lines : list
        List of line data. Each element of the list is a 3-tuple in the form
        (wavelenght in Angstrom, Line name, Line type).

    """
    if name is None:
        selected_lines = RESTFRAME_LINES[:]
    else:
        selected_lines = [
            line
            for line in RESTFRAME_LINES
            if name.lower() == line[1].lower()
        ]
    if line_type is not None:
        selected_lines = [
            line
            for line in selected_lines
            if line_type.lower() in line[2].lower()
        ]

    selected_lines = [
        ((1 + z) * line[0], line[1], line[2])
        for line in selected_lines
    ]

    if wrange is not None:
        w_min = np.nanmin(wrange)
        w_max = np.nanmax(wrange)
        selected_lines = [
            line
            for line in selected_lines
            if w_min <= line[0] <= w_max
        ]

    return selected_lines
