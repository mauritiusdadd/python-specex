#!/bin/bash

args=$(getopt -l "help" -l "noenv" -o "h" -- "$@")

eval set -- "$args"

while [ $# -ge 1 ]; do
        case "$1" in
                --)
                    # No more options left.
                    shift
                    break
                   ;;
                --noenv)
                        NO_PYTHON_ENV=1
                        shift
                        ;;
                -h|--help)
                        echo ""
                        echo "    -h         Show this help message."
                        echo "    --noenv    Do not create a virtual python environment"
                        echo "               and install the package in the current system."
                        exit 0
                        ;;
        esac

        shift
done

#
# Utilities needed to download packages
#

if command -v curl &> /dev/null ; then
  export DLOAD_AGENT="curl -L -C - -o"
else
  if command -v dwget &> /dev/null ; then
    export DLOAD_AGENT="wget -c -O"
  else
    echo "Please install either wget or curl!"
    exit 1
  fi
fi

if ! command -v tar &> /dev/null ; then
  echo "Please install tar!"
  exit 1
fi

# This function downloads a DESI package from github given
# its name and version
function dload_desi_pkg() (
  pkgname=$1
  pkgver=$2

  if [[ -z "$pkgname" ]] || [[ -z "$pkgver" ]] ; then
    echo "Package name or version not provided"
    exit 1
  fi

  package_url="https://github.com/desihub/$pkgname/archive/refs/tags/$pkgver.tar.gz"
  remote_name="$(basename $package_url)"
  local_name="$pkgname-$remote_name"
  echo ""
  echo "Downloading $local_name"
  $DLOAD_AGENT "$local_name" "$package_url"

  echo ""
  echo "Extracting the archive"
  tar -xzf "$local_name"
)

function install_desi_pkg() (
  srcdir="$1"
  if [[ -z $srcdir ]] ; then
    echo "Package folder not specified, bailing out!"
    exit 1
  fi

  cd $srcdir
  python setup.py build
  python setup.py install
)

#
# Installation procedure
#

if [ -z $NO_PYTHON_ENV ]; then

  # Step 1: create a python virtual environment so we do not mess up the
  #         system-wide python installation. Let's also reuse the already
  #         installed packages.
  python3 -m venv --system-site-packages "redrock-venv"
  mkdir -p "redrock-venv/pkg"
  curr_dir="$(pwd)"

  # Step 2: activate the virtual environment.

  . redrock-venv/bin/activate || exit 1

  cd "redrock-venv/pkg"
fi

# Step 3: install required python packages.
pip install numpy scipy astropy numba sqlalchemy healpy requests fitsio \
                  photutils h5py

$DLOAD_AGENT "python-empca" https://github.com/sbailey/empca/archive/c34abb3f7a2bed4baaf25f15f00791d5f60f8be8.zip

# Step 4: download and extract source packages (redrock and all the needed
#         DESI dependencies).
#dload_desi_pkg "speclite" "v0.16"
#dload_desi_pkg "desiutil" "3.3.0"
#dload_desi_pkg "desitarget" "0.22.0"
#dload_desi_pkg "desimodel" "0.9.6"
#dload_desi_pkg "specter" "0.8.6"
#dload_desi_pkg "desispec" "0.51.13"
#dload_desi_pkg "desisim" "0.28.0"
dload_desi_pkg "redrock-templates" "0.8"
dload_desi_pkg "redrock" "0.15.4"

mkdir -p "redrock-0.15.4/py/redrock/templates"
cp -Rf "redrock-templates-0.8"/* "redrock-0.15.4/py/redrock/templates/"

# Step 5: install the packages in the correct order
#install_desi_pkg "speclite-0.15"
#install_desi_pkg "desiutil-3.3.0"
#install_desi_pkg "desitarget-0.22.0"
#install_desi_pkg "desimodel-0.9.6"
#install_desi_pkg "specter-0.8.6"
#install_desi_pkg "desispec-0.27.1"
#install_desi_pkg "desisim" "0.28.0"
install_desi_pkg "redrock-0.15.4"

cd "$curr_dir"
