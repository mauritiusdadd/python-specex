#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import unittest

from spex.spex import spex

from test import make_synt_cube

Z_FTOL = 0.01


class TestSpex(unittest.TestCase):

    reg_file, cat_file, cube_file = make_synt_cube.main(overwrite=False)

    def test_spex_catalog_success(self):
        spex_options = [
            '--catalog', self.cat_file,
            '--mode', 'circular_aperture',
            '--aperture-size', '0.8arcsec',
            '--no-nans', self.cube_file
        ]
        spex(options=spex_options)

    def test_spex_regionfile_success(self):
        spex_options = [
            '--regionfile', self.reg_file,
            '--mode', 'kron_ellipse',
            '--no-nans', self.cube_file
        ]
        spex(options=spex_options)


if __name__ == '__main__':
    mytest = TestSpex()
    mytest.test_spex_catalog_success()
    mytest.test_spex_regionfile_success()
