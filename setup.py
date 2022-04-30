#!/usr/bin/env python
"""Setup for python-spex."""
import setuptools
import time
import os
import codecs


def read(rel_path):
    """
    Read a file.

    Parameters
    ----------
    rel_path : str
        The path of the file to read.

    Returns
    -------
    str
        The content of the file.

    """
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    """
    Get the version of a module.

    Parameters
    ----------
    rel_path : str
        The path of the module file.

    Raises
    ------
    RuntimeError
        If there is no '__version__' line in the file raise this exception.

    Returns
    -------
    str
        The version of the module.

    """
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


BUILD_INFO = {
    'date': time.strftime("%Y-%m"),
    'asctime': time.asctime(),
    'github': "https://github.com/mauritiusdadd/python-spex"
}


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='spex',
    version=get_version("py/spex/__init__.py"),
    author='Maurizio D\'Addona',
    author_email='mauritiusdadd@gmail.com',
    url='https://github.com/mauritiusdadd/python-spex',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="py"),
    package_dir={"": "py"},
    python_requires=">=3.8",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',

        # Pick your license as you wish (should match "license" above)
        'BSD 3-Clause "New" or "Revised" License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
    ],
    keywords='spectroscopy spectra spectrum spectral cubes',
    entry_points={
        'console_scripts': [
            'spex=spex.spex:spex',
            'rrspex=spex.rrspex:rrspex',
            'zeropointinfo=spex.zeropoints:main',
            'cubestack=spex.stack:cube_stack'
        ],
    },
)
