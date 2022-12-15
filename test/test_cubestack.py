#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import unittest
from specex.stack import cube_stack

from test import make_synt_cube


class TestCubeStacking(unittest.TestCase):

    reg_file, cat_file, cube_file = make_synt_cube.main(overwrite=False)

    def test_stack_cube(self):
        print(self.cube_file)
        cube_stack_options = [
            self.cube_file
        ]
        cube_stack(cube_stack_options)


if __name__ == '__main__':
    mytest = TestCubeStacking()
    mytest.test_stack_cube()
