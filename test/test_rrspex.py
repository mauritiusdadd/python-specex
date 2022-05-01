#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import unittest
import pathlib
from astropy.io import fits


TEST_FILES = ["spec_synt_e.fits", ]
Z_FTOL = 0.01


try:
    from spex.rrspex import rrspex
except ImportError:
    HAS_RR = False
else:
    HAS_RR = True


class TestRRSpex(unittest.TestCase):

    @unittest.skipIf(not HAS_RR, "redrock not installed")
    def test_rrspex_success(self):
        test_files = [
            os.path.join(pathlib.Path(__file__).parent.resolve(), "data", x)
            for x in TEST_FILES
        ]

        obj_z = [fits.getheader(x, ext=0)['OBJ_Z'] for x in test_files]

        targets, zbest, scandata = rrspex(options=test_files)

        for i, obj in enumerate(zbest):
            self.assertTrue(
                abs(obj_z[i] - obj['Z']) <= obj['ZERR'],
                "computed - fitted redshift absolute difference is greater "
                "than computed redshift error!"
            )
            self.assertTrue(
                abs(obj_z[i] - obj['Z'])/(1 + obj_z[i]) < Z_FTOL,
                "computed redshift outside f01 limit!"
            )
