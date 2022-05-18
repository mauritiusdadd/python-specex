#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic spectral datacube that can be used to test spex module.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import pathlib
import numpy as np
from astropy.io import fits
from astropy.modeling import models
from scipy.ndimage import gaussian_filter
from redrock.templates import find_templates, Template
from astropy.table import Table
from astropy.wcs import WCS


class FakeObject():
    """
    Class to handle fake object generation.

    This class contains morphological properties of an object, such as the
    the shape of the isophotal ellipse at kron radius, as well as physical
    properties as its spectral tepmlate and luminosity.
    """

    __template_dict = {}

    def __init__(self, template_file, x_image, y_image, z=None, coeffs=None,
                 surface_bightness_profile=None):
        """
        Initialize the class.

        Parameters
        ----------
        template : redrock.template.Template
            The spectral template of the object.
        x_image : float
            The x position of the center of the object in the image.
        y_image : float
            The y position of the center of the object in the image.
        coeffs : list or numpy.ndarray
            Coefficients uesd for generating the spectrum, passed to the method
            template.eval().
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

        if template_file not in self.__template_dict:
            self.__template_dict[template_file] = Template(template_file)

        self.template_file = template_file

        f = fits.open(template_file)
        try:
            has_archetypes = f[1].name == 'ARCHETYPE_COEFF'
        except IndexError:
            has_archetypes = False

        if has_archetypes:
            archetype_coeffs = f[1].data
            coeff_indexes = len(archetype_coeffs)
            self.__coeffs = archetype_coeffs[np.random.choice(coeff_indexes)]
        else:
            coeff_indexes = range(self.template.nbasis)
            self.__coeffs = np.zeros_like(coeff_indexes)
            self.__coeffs[np.random.choice(coeff_indexes)] = 1

        if surface_bightness_profile is None:
            obj_type = self.template.template_type
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
                        self.z = np.random.choice(self.template.redshifts)
            else:
                n_val = 99
                amp_val = 5.0e-6
                e_val = 0
                r_eff_val = 1.5
                if self.z is None:
                    self.z = 0
                    while self.z < 1:
                        self.z = np.random.choice(self.template.redshifts)

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
        template : redrock.template.Template
            The spectral template of the object.

        """
        return self.__template_dict[self.template_file]

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
            coeffs=self.__coeffs,
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


def gen_fake_spectrum(wave_range, template, coeffs, z, wave_step=1.0):
    """
    Generate a synthetic spectrum using a specific template.

    Parameters
    ----------
    wave_range : tuple or numpy.ndarray
        The range of wavelenght, in Angstrom, the sythetic spectrum should
        cover. Only maximum and minimum values of this parameter are used.
    template : rerdock.template.Template or custom class
        The template used to compute the spectrum. If a custom class is used,
        it should have a method named eval that accetps the following
        parameters eval(coeff, wave, z), where:

            - coeff : list or numpy.ndarray
                Coefficients uesd for generating the spectrum
            - wave : numpy.ndarray
                Wavelenghts, in Angstrom, used to generate spectral data
            - z : float
                Redshift of the spectrum.

        The eval method should return a 1D numpy.ndarray of the same lenght of
        wave and containing the flux of the synthetic spectrum in arbitrary
        units.
    coeffs : list or numpy.ndarray
        Coefficients uesd for generating the spectrum, passed to template.eval
        method.
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
    flux = template.eval(coeffs, wave, z)
    return wave, flux - np.min(flux)


def gen_fake_cube(out_dir, width, height, wave_range, n_objects, wave_step=1.0,
                  seeing=1):
    waves = get_waves(wave_range, wave_step)

    header = fits.Header()

    header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'
    header['CRPIX1'] = 1.0
    header['CRPIX2'] = 1.0
    header['CD1_1'] = 5.0e-5
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 5.0e-5
    header['CUNIT1'] = 'deg     '
    header['CUNIT2'] = 'deg     '
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 90.0
    header['CRVAL2'] = 0.0

    my_wcs = WCS(header)

    available_templates = [
        f for f in find_templates()
        if'galaxy' in f
    ]

    w_padding = width/50
    h_padding = width/50

    obj_attributes = zip(
        w_padding + np.random.random(n_objects) * (width - 2*w_padding),
        h_padding + np.random.random(n_objects) * (height - 2*h_padding),
        np.random.choice(available_templates, n_objects)
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
    regfile_text = 'fk5\n'
    for i, obj in enumerate(objects):
        obj_x = int(obj.x_image)
        obj_y = int(obj.y_image)
        obj_kron = obj.surface_bightness_profile_function.r_eff.value
        e_val = obj.surface_bightness_profile_function.ellip.value
        obj_a_image = obj_kron
        obj_b_image = obj_a_image * (1 - e_val)
        obj_theta_image = obj.surface_bightness_profile_function.theta.value
        obj_theta_image = np.rad2deg(obj_theta_image)

        sky_coords = my_wcs.pixel_to_world(obj.x_image, obj.y_image)

        obj_ra = sky_coords.ra.deg
        obj_dec = sky_coords.dec.deg

        regfile_text += f"ellipse({obj_ra:.6f}, {obj_dec:.6f}, "
        regfile_text += f"{obj_a_image:.4f}, "
        regfile_text += f"{obj_b_image:.4f}, {obj_theta_image:.4f}) "
        regfile_text += f"# text={{S{i:06d}}}\n"

        new_row = [
            i, obj_x, obj_y, obj_a_image, obj_b_image, obj_theta_image,
            obj_kron, obj_ra, obj_dec, obj.z, obj.template.template_type,
            obj.template.sub_type
        ]

        myt.add_row(new_row)

    with open(os.path.join(out_dir, 'test_cube.reg'), "w") as f:
        f.write(regfile_text)

    myt.write(os.path.join(out_dir, 'test_cube_cat.fits'), overwrite=True)

    header['CRVAL3'] = waves[0]
    header['CRPIX3'] = 1.0
    header['CTYPE3'] = 'WAVE'
    header['CD3_3'] = wave_step
    header['CD1_3'] = 0.0
    header['CD2_3'] = 0.0
    header['CD3_1'] = 0.0
    header['CD2_1'] = 0.0
    header['CUNIT3'] = 'Angstrom'
    header['OBJECT'] = 'SYNTHETIC'
    header['PCOUNT'] = 0
    header['GCOUNT'] = 1

    cube_hdu = fits.ImageHDU(
        data=base_cube,
        name='SPECTRUM',
        header=header
    )

    cube_hdu.writeto(os.path.join(out_dir, 'test_cube.fits'), overwrite=True)


def main():
    """
    Run the main program of this module.

    Returns
    -------
    None.

    """
    wave_range = (4500, 8000)
    out_dir = pathlib.Path(__file__).parent.resolve()
    gen_fake_cube(out_dir, 256, 256, wave_range, n_objects=30, wave_step=1)


if __name__ == '__main__':
    main()
