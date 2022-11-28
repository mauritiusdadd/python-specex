#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import os
import unittest
import pathlib
from astropy.io import fits
from astropy.table import Table, join

try:
    from spex.rrspex import rrspex
except ImportError:
    HAS_RR = False
else:
    HAS_RR = True


TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")
Z_FTOL = 0.01


class TestRRSpex(unittest.TestCase):

    @unittest.skipIf(not HAS_RR, "redrock not installed")
    def test_rrspex_success(self):
        test_files = [
            os.path.join(TEST_DATA_PATH, x)
            for x in os.listdir(TEST_DATA_PATH)
            if x.startswith('rrspex_') and x.endswith('.fits')
        ]

        true_z_table = Table(
            names=['SPECID', 'TRUE_Z'],
            dtype=['U10', 'float32']
        )

        for file in test_files:
            header = fits.getheader(file, ext=0)
            true_z_table.add_row([header['ID'], header['OBJ_Z']])

        options = ['--quite', ] + test_files
        targets, zbest, scandata = rrspex(options=options)

        zbest = join(true_z_table, zbest, keys=['SPECID'])
        print(zbest)

        for i, obj in enumerate(zbest):
            self.assertLessEqual(
                abs(obj['TRUE_Z'] - obj['Z'])/(1 + obj['TRUE_Z']), Z_FTOL,
                msg="computed redshift outside f01 limit!"
            )


if __name__ == '__main__':
    mytest = TestRRSpex()
    mytest.test_rrspex_success()
