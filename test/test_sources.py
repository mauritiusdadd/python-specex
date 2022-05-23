#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import pathlib
import unittest

from spex.sources import detect_from_cube


TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")


class TestSourceDetection(unittest.TestCase):

    # test_cube_file = os.path.join(TEST_DATA_PATH, "test_cube.fits")
    test_cube_file = "/home/daddona/dottorato/Y1/muse_cubes/M0416/MACSJ0416_NE_DATACUBE_FINAL_VACUUM_caminha_zap.fits"

    def test_extract_sources(self):
        detect_from_cube([self.test_cube_file])


if __name__ == '__main__':
    mytest = TestSourceDetection()
    mytest.test_extract_sources()
