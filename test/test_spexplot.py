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

from spex.spexplot import spexplot


TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")
Z_FTOL = 0.01


class TestSpexplot(unittest.TestCase):

    spec_dir_name = 'test_cube_spectra'

    def test_spexplot_success(self):
        spec_files = [
            os.path.join(self.spec_dir_name, x)
            for x in os.listdir(self.spec_dir_name)
            if x.startswith('spec_') and x.endswith('.fits')
        ]

        spexplot_options = [
            '--zcat', 'zbest_cube.fits',
            '--key-id', 'SPECID',
            '--restframe',
            *spec_files
        ]

        spexplot(options=spexplot_options)


if __name__ == '__main__':
    mytest = TestSpexplot()
    mytest.test_spexplot_success()
