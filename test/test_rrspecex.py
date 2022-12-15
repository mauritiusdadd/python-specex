#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import warnings
import unittest
from astropy.io import fits
from astropy.table import Table, join

try:
    from specex.rrspecex import rrspecex
except ImportError:
    HAS_RR = False
else:
    HAS_RR = True

from test import make_synt_specs

Z_FTOL = 0.01


class TestRRSpex(unittest.TestCase):

    @unittest.skipIf(not HAS_RR, "redrock not installed")
    def test_rrspecex_success(self):
        test_files = make_synt_specs.main()

        true_z_table = Table(
            names=['SPECID', 'TRUE_Z'],
            dtype=['U10', 'float32']
        )

        for file in test_files:
            header = fits.getheader(file, ext=0)
            true_z_table.add_row([header['ID'], header['OBJ_Z']])

        options = ['--quite', ] + test_files
        targets, zbest, scandata = rrspecex(options=options)

        zbest = join(true_z_table, zbest, keys=['SPECID'])
        print(zbest)

        for i, obj in enumerate(zbest):
            delta_z = abs(obj['TRUE_Z'] - obj['Z'])/(1 + obj['TRUE_Z'])
            if delta_z >= Z_FTOL:
                warnings.warn(
                    f"OBJ {i}: computed redshift outside f01 limit!",
                )


if __name__ == '__main__':
    if HAS_RR:
        mytest = TestRRSpex()
        mytest.test_rrspecex_success()
