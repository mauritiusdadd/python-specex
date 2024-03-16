#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic spectral datacube that can be used to test specex module.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import warnings
import numpy as np

from astropy.io import fits
from astropy.modeling import models
from scipy.ndimage import gaussian_filter
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.utils.exceptions import AstropyWarning
from astropy import units

from test import TEST_DATA_PATH
from specex.utils import get_sdss_spectral_templates, get_sdss_template_data


class FakeObject():
    """
    Class to handle fake object generation.

    This class contains morphological properties of an object, such as the
    the shape of the isophotal ellipse at kron radius, as well as physical
    properties as its spectral tepmlate and luminosity.
    """

    __template_dict = {}

    def __init__(self, template, x_image, y_image, z=None,
                 surface_bightness_profile=None):
        """
        Initialize the class.

        Parameters
        ----------
        template_file : str
            Path of a SDSS template file.
        x_image : float
            The x position of the center of the object in the image.
        y_image : float
            The y position of the center of the object in the image.
        z : float
            Redshift of the spectrum.
        surface_bightness_profile : astropy.modeling.Fittable2DModel, optional
            The surface brightness profile model. If None, a Sersic2D model is
            used with random ellipticity and theta. The default is None.

        Returns
        -------
        None.

        """
        self.x_image = x_image
        self.y_image = y_image
        self.z = z

        self.template_file = template

        if surface_bightness_profile is None:
            obj_type = self.template_file['type']
            obj_type = obj_type.lower()
            if obj_type == 'star':
                n_val = 99
                amp_val = 1
                e_val = 0
                r_eff_val = 2
                if self.z is None:
                    self.z = 0
            elif obj_type == 'galaxy':
                n_val = 1 + np.random.random()*5
                amp_val = 10 + np.random.random()*5
                e_val = np.random.random()*0.6
                r_eff_val = np.random.normal(loc=2)**2
                if self.z is None:
                    self.z = 0
                    while self.z < 0.2:
                        self.z = np.random.rand() * 2
            else:
                n_val = 99
                amp_val = 5.0e-6
                e_val = 0
                r_eff_val = 1.5
                if self.z is None:
                    self.z = 0
                    while self.z < 1:
                        self.z = np.random.rand() * 5

            self.surface_bightness_profile_function = models.Sersic2D(
                amplitude=amp_val,
                r_eff=r_eff_val,
                n=n_val,
                x_0=self.x_image,
                y_0=self.y_image,
                ellip=e_val,
                theta=np.random.random() * 2 * np.pi
            )
        else:
            self.surface_bightness_profile_function = surface_bightness_profile

    @property
    def template(self):
        """
        Get the spectral template of the object.

        Returns
        -------
        template : dict
            The spectral template dictionary returned by
            specex.utils.get_sdss_template_data().

        """
        return get_sdss_template_data(self.template_file['file'])

    def get_image(self, width, height, seeing=1):
        """
        Get the an image containing the object.

        Parameters
        ----------
        width : int
            The image width.
        height : int
            The image height.
        seeing: float
            Parameter of a gaussian blur filter.

        Returns
        -------
        img : 2D numpy.ndarray
            The image containing the object.

        """
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        base_image = self.surface_bightness_profile_function(x, y)

        base_image += gaussian_filter(base_image, seeing)
        return base_image

    def get_spectrum(self, wave_range, wave_step=1):
        """
        Get the spectrum of the object.

        Parameters
        ----------
        wave_range : tuple or numpy.ndarray
            The range of wavelenght, in Angstrom, the sythetic spectrum should
            cover. Only maximum and minimum values of this parameter are used.

        Returns
        -------
        wave : numpy.ndarray
            Wavelenghts, in Angstrom, used to generate spectral data
        flux : numpy.ndarray
            Fluxes, in arbitrary units, at the different wavelengths.

        """
        wave, flux = gen_fake_spectrum(
            wave_range,
            self.template,
            z=self.z,
            wave_step=wave_step
        )
        return wave, flux

    def get_cube(self, width, height, wave_range,  wave_step=1, seeing=1):
        """
        Get a datacube containing the spectrum of the object.

        Parameters
        ----------
        width : int
            The image width.
        height : int
            The image height.
        wave_range : tuple or numpy.ndarray
            The range of wavelenght, in Angstrom, the sythetic spectrum should
            cover. Only maximum and minimum values of this parameter are used.
        wave_step : float, optional
            Resolution of the spectrum in Angstrom. The default is 1.

        Returns
        -------
        cube : 3D numpy.ndarray
            The spectral datacybe of the object.

        """
        waves, flux = self.get_spectrum(wave_range, wave_step)
        image = self.get_image(width, height, seeing)
        cube = image[..., None] * flux
        noise = np.random.random(size=cube.shape) * 0.005
        cube = cube + noise
        cube = cube.transpose(2, 0, 1)
        return cube


def get_waves(wave_range, wave_step=1.0):
    """
    Generate a wavelength grid given an input wavelenght range.

    Parameters
    ----------
    wave_range : tuple or numpy.ndarray
        The range of wavelenght, in Angstrom, the sythetic spectrum should
        cover. Only maximum and minimum values of this parameter are used.

    Returns
    -------
    wave : numpy.ndarray
        Wavelenghts, in Angstrom.

    """
    # generate a wavelenght grid
    w_start = np.min(wave_range)
    w_end = np.max(wave_range)
    wave = np.arange(w_start, w_end, step=wave_step)
    return wave


def gen_fake_spectrum(wave_range, template, z, wave_step=1.0):
    """
    Generate a synthetic spectrum using a specific template.

    Parameters
    ----------
    wave_range : tuple or numpy.ndarray
        The range of wavelenght, in Angstrom, the sythetic spectrum should
        cover. Only maximum and minimum values of this parameter are used.
    template : SDSS spectral template data
        spectral data dictionary compatible with the format returned by
        specex.get_sdss_template_data()
    z : float
        Redshift of the spectrum.
    wave_step : float, optional
        Resolution of the spectrum in Angstrom. The default is 1.

    Returns
    -------
    wave : numpy.ndarray
        Wavelenghts, in Angstrom, used to generate spectral data
    flux : numpy.ndarray
        Fluxes, in arbitrary units, at the different wavelengths.
    """
    wave = get_waves(wave_range, wave_step)
    flux = np.interp(wave, template['wavelengths'] * (1 + z), template['flux'])
    return wave, flux - np.min(flux)


def gen_fake_cube(out_dir, width, height, wave_range, n_objects, wave_step=1.0,
                  seeing=1, overwrite=False):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    templates = get_sdss_spectral_templates(out_dir, use_cached=True)

    reg_file = os.path.join(out_dir, 'test_cube.reg')
    cat_file = os.path.join(out_dir, 'test_cube_cat.fits')
    cube_file = os.path.join(out_dir, 'test_cube.fits')

    out_files_list = [reg_file, cat_file, cube_file]

    if all([os.path.isfile(x) for x in out_files_list]):
        if not overwrite:
            return out_files_list

    print(f"Generating a synthetic cube to {cube_file}")

    waves = get_waves(wave_range, wave_step)

    header = fits.Header()

    header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'
    header['CRPIX1'] = 135.5
    header['CRPIX2'] = 132.0
    header['CD1_1'] = 2.86146E-06
    header['CD1_2'] = -2.07292E-05
    header['CD2_1'] = -2.09168E-05
    header['CD2_2'] = -2.83578E-06
    header['CUNIT1'] = 'deg     '
    header['CUNIT2'] = 'deg     '
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 182.6361425552
    header['CRVAL2'] = 39.40589479746

    my_wcs = WCS(header)

    w_padding = width/50
    h_padding = width/50

    obj_attributes = zip(
        w_padding + np.random.random(n_objects) * (width - 2*w_padding),
        h_padding + np.random.random(n_objects) * (height - 2*h_padding),
        np.random.choice(templates, n_objects)
    )

    objects = [
        FakeObject(
            attr[2],
            attr[0], attr[1],
        )
        for attr in obj_attributes
    ]

    base_cube = np.zeros((waves.shape[0], height, width))
    for j, obj in enumerate(objects):
        try:
            base_cube += obj.get_cube(
                width, height,
                wave_range, wave_step,
                seeing
            )
        except ValueError:
            print(f"WARNING: Skipping object {j}")
            continue
        else:
            sys.stdout.write(f"\rprogress: {j/ len(objects):.2%}\r")
            sys.stdout.flush()

    print("DONE!")

    myt = Table(
        names=[
            'NUMBER',
            'X_IMAGE',
            'Y_IMAGE',
            'A_IMAGE',
            'B_IMAGE',
            'THETA_IMAGE',
            'KRON_RADIUS',
            'ALPHA_J2000',
            'DELTA_J2000',
            'TRUE_Z',
            'TRUE_TYPE',
            'TRUE_SUBTYPE'
        ],
        dtype=[
            int, float, float, float, float, float, float, float, float, float,
            'U10', 'U10'
        ]
    )

    celestial_wcs = my_wcs.celestial
    pixelscale_x, pixelscale_y = [
        units.Quantity(sc_val, u)
        for sc_val, u in zip(
            proj_plane_pixel_scales(celestial_wcs),
            celestial_wcs.wcs.cunit
        )
    ]

    regfile_text = 'fk5\n'
    for i, obj in enumerate(objects):
        obj_x = int(obj.x_image)
        obj_y = int(obj.y_image)
        obj_kron = obj.surface_bightness_profile_function.r_eff.value
        e_val = obj.surface_bightness_profile_function.ellip.value
        obj_a_image = 3 * obj_kron * pixelscale_x
        obj_a_image = obj_a_image.to('arcsec').value
        obj_b_image = 3 * obj_kron * (1 - e_val) * pixelscale_y
        obj_b_image = obj_b_image.to('arcsec').value
        obj_theta_image = obj.surface_bightness_profile_function.theta.value
        obj_theta_image = np.rad2deg(obj_theta_image)

        sky_coords = my_wcs.pixel_to_world(obj.x_image, obj.y_image)

        obj_ra = sky_coords.ra.deg
        obj_dec = sky_coords.dec.deg

        regfile_text += f"ellipse({obj_ra:.6f}, {obj_dec:.6f}, "
        regfile_text += f"{obj_a_image:.4f}\", "
        regfile_text += f"{obj_b_image:.4f}\", {obj_theta_image:.4f}) "
        regfile_text += f"# text={{S{i:06d}}}\n"

        new_row = [
            i, obj_x, obj_y, obj_a_image, obj_b_image, obj_theta_image,
            obj_kron, obj_ra, obj_dec, obj.z, obj.template_file['type'],
            obj.template_file['sub-type']
        ]

        myt.add_row(new_row)

    with open(reg_file, "w") as f:
        f.write(regfile_text)

    myt.write(cat_file, overwrite=True)

    header['CRVAL3'] = waves[0]
    header['CRPIX3'] = 1.0
    header['CTYPE3'] = 'WAVE'
    header['CD3_3'] = wave_step
    header['CD1_3'] = 0.0
    header['CD2_3'] = 0.0
    header['CD3_1'] = 0.0
    header['CD3_2'] = 0.0
    header['CUNIT3'] = 'Angstrom'
    header['OBJECT'] = 'SYNTHETIC'
    header['PCOUNT'] = 0
    header['GCOUNT'] = 1

    cube_hdu = fits.ImageHDU(
        data=base_cube,
        name='SPECTRUM',
        header=header
    )

    cube_hdu.writeto(cube_file, overwrite=True)
    return out_files_list


def main(overwrite=False):
    """
    Run the main program of this module.

    Returns
    -------
    None.

    """
    warnings.simplefilter('ignore', category=AstropyWarning)

    wave_range = (4500, 8000)

    return gen_fake_cube(
        TEST_DATA_PATH, 256, 256, wave_range, n_objects=30, wave_step=1
    )


if __name__ == '__main__':
    main()
