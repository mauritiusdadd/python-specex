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

from spex.spex import spex

try:
    from spex.rrspex import rrspex
except ImportError:
    HAS_RR = False
else:
    HAS_RR = True


TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")
Z_FTOL = 0.01


class TestSpex(unittest.TestCase):

    test_cube_file = os.path.join(TEST_DATA_PATH, "test_cube.fits")
    test_cat_file = os.path.join(TEST_DATA_PATH, "test_cube_cat.fits")
    test_ref_file = os.path.join(TEST_DATA_PATH, "test_cube.reg")

    def test_spex_catalog_success(self):
        spex_options = [
            '--catalog', self.test_cat_file,
            '--mode', 'circular_aperture',
            '--aperture-size', '0.8',
            '--no-nans', self.test_cube_file
        ]
        spex(options=spex_options)

    def test_spex_regionfile_success(self):
        spex_options = [
            '--regionfile', self.test_ref_file,
            '--mode', 'circular_aperture',
            '--aperture-size', '0.8',
            '--no-nans', self.test_cube_file
        ]
        spex(options=spex_options)

    """
    @unittest.skipIf(not HAS_RR, "redrock not installed")
    def test_spex_rrspex_success(self):
        spex_options = [
        ]
        spex(options=spex_options)
    """