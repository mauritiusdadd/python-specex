#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import unittest

from specex.sources import detect_from_cube

from test import make_synt_cube


class TestSourceDetection(unittest.TestCase):

    reg_file, cat_file, cube_file = make_synt_cube.main(overwrite=False)

    def test_extract_sources(self):
        detect_from_cube([self.cube_file])


if __name__ == '__main__':
    mytest = TestSourceDetection()
    mytest.test_extract_sources()
