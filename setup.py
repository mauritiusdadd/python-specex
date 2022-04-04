#!/usr/bin/env python
"""Setup for python-spex."""
import setuptools
import time

BUILD_INFO = {
    'date': time.strftime("%Y-%m"),
    'asctime': time.asctime(),
    'github': ""
}


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='spex',
    version='0.2.0',
    author='Maurizio D\'Addona',
    author_email='mauritiusdadd@gmail.com',
    url='https://github.com/mauritiusdadd/python-spex',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="py"),
    package_dir={"": "py"},
    # data_files=[
    #     ('share/licenses/spex', ['LICENSE']),
    #     ('share/man/man1', ['man/spex.1.gz']),
    #     ('share/man/it/man1', ['man/it/spex.1.gz']),
    # ],
    python_requires=">=3.7",
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
            'spex=spex:main',
            'rrspex=rrspex:main',
        ],
    },
)
