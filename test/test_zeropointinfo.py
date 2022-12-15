#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import sys
import unittest

from specex.zeropoints import main as zpinfo

from test import get_hst_test_images


class TestZeropointInfo(unittest.TestCase):

    test_hst_imgs = get_hst_test_images()

    def test_zeropoint_info(self):
        if not self.test_hst_imgs:
            print(
                "Failed to download HST test images, skipping this test...",
                file=sys.stderr
            )
            return True
        zpinfo(self.test_hst_imgs)


if __name__ == '__main__':
    mytest = TestZeropointInfo()
    mytest.test_zeropoint_info()
