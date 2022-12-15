#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import unittest

from specex.cube import cutout_main

from test import make_synt_cube, get_hst_test_images, TEST_DATA_PATH


NICMOS_REGFILE_DATA = """
# Region file format: DS9 version 4.1
fk5
box(182.6350214,39.4064356,5.000",3.000",262.21022)
circle(182.6356792,39.4058053,2.500")
ellipse(182.6349624,39.4062180,2.500",1.500",217.21022)
"""


class TestCutout(unittest.TestCase):

    test_hst_imgs = get_hst_test_images()

    def test_grayscale_cutout(self):
        if not self.test_hst_imgs:
            print(
                "Failed to download HST test images, skipping this test...",
                file=sys.stderr
            )
            return True
        for img in self.test_hst_imgs:
            if img.endswith('NICMOSn4hk12010_mos.fits'):
                regfile = os.path.join(
                    TEST_DATA_PATH, 'NICMOSn4hk12010_mos.reg'
                )

                with open(regfile, 'w') as f:
                    f.write(NICMOS_REGFILE_DATA)

                cutout_options = [
                    '--verbose',
                    '--regionfile', regfile,
                    img
                ]
                cutout_main(cutout_options)
                break

    def test_cube_cutout(self):
        reg_file, cat_file, cube_file = make_synt_cube.main(overwrite=False)

        regfile = os.path.join(
            TEST_DATA_PATH, 'cutout_test_cube.reg'
        )

        with open(regfile, 'w') as f:
            f.write(NICMOS_REGFILE_DATA)

        cutout_options = [
            '--verbose',
            '--regionfile', regfile,
            cube_file
        ]
        cutout_main(cutout_options)


if __name__ == '__main__':
    mytest = TestCutout()
    mytest.test_grayscale_cutout()
    mytest.test_cube_cutout()
