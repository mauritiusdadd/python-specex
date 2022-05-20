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

from spex.sources import main


TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")


class TestSourceDetection(unittest.TestCase):

    test_cube_file = os.path.join(TEST_DATA_PATH, "test_cube.fits")

    def test_extract_sources(self):
        main([self.test_cube_file])


if __name__ == '__main__':
    mytest = TestSourceDetection()
    mytest.test_extract_sources()
