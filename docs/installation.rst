.. |github_mark| image:: pics/github-mark.png
   :height: 1em
   :target: github_repo

Installing a Packaged Release
=============================

The simplest way to install Specex is using ``pip``

.. code-block:: bash

    pip install specex

Some functionalites of the program are considered optional and are enabled only if the appropriate python packages are installed. You can enable the optional dependencies at the Installion time using the syntax

.. code-block:: bash

    pip install specex[group1,group2,group...]

where *group1*, *group2*, etc. are the name of the functionalites you want to enable, for example the following command will install all the optional dependencies

.. code-block:: bash

    pip install specex[all]

while this command will install only the dependencies needed to run the command ``specex-cube-anim`` and to use regionfiles as input for ``specex``

.. code-block:: bash

    pip install specex[animations,regions]

This is a list of the name of all optional dependency groups

  * ``animations`` : dependencies for generatig animations with ``specex-cube-anim``
  * ``redrock`` : dependencies for estimating redshifts with ``rrspecex`` trough *Redrock*  [:ref:`1<references_installation>`]
  * ``regions`` : dependencies to enable the handling of regionfiles in ``specex``
  * ``all``: install all the previous dependencies4

.. attention::

    The dependency group ``redrock`` install only the dependencies needed to run ``rrspecex`` and not *Redrock* itself. It is your responsability to correctly install *Redrock*


Installing from GitHub
======================

If you like to use the bleeding-edge version, you can install Specex directly from the |github_mark| `GitHub repository <https://github.com/mauritiusdadd/python-specex>`_

.. code-block:: bash

    git clone 'https://github.com/mauritiusdadd/python-specex.git'
    cd python-specex
    pip install .

Also in this case, you can specify which group of dependencies you want to install by running, for example

.. code-block:: bash

    pip install .[all]

The git repository also includes a simple script that will help you to install *Redrock* in a virtual environment

.. code-block:: bash

    chmod +x redrock_venv_setup.sh
    ./redrock_venv_setup.sh
    . ./redrock_venv/bin/activate
    pip install .[redrock]

even if it is better if **YOU** install it following the installation instructions on its `GitHub page <https://github.com/desihub/redrock>`_.

.. _references_installation:

References
----------

#. `Redrock <https://github.com/desihub/redrock>`_
