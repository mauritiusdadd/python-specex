"""
SPECEX - SPECtra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

try:
    from specex._version import version as __version__  # type: ignore
    from specex._version import version_tuple  # type: ignore
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


