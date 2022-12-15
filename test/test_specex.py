#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import unittest

from specex.specex import specex

from test import make_synt_cube

Z_FTOL = 0.01


class TestSpex(unittest.TestCase):

    reg_file, cat_file, cube_file = make_synt_cube.main(overwrite=False)

    def test_specex_catalog_success(self):
        specex_options = [
            '--catalog', self.cat_file,
            '--mode', 'circular_aperture',
            '--aperture-size', '0.8arcsec',
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)

    def test_specex_regionfile_success(self):
        specex_options = [
            '--regionfile', self.reg_file,
            '--mode', 'kron_ellipse',
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)


if __name__ == '__main__':
    mytest = TestSpex()
    mytest.test_specex_catalog_success()
    mytest.test_specex_regionfile_success()
