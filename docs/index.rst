:html_theme.sidebar_secondary.remove: true

XEFI
====

A package for calculations of X-ray Electric Field Intensities (XEFI) using the Parratt recursive algorithm, and built to the feature-rich standards of `xraysoftmat <https://github.com/xraysoftmat>`_.

This package calculates discrete models of multi-layer structures, including the ability to slice simplistic models into arbitrary layers.
Supports the use of the `KKCalc` package to calculate the index of refraction within layers.

.. MAKE SURE TO UPDATE THE README.rst to match. These files are now diverged.

|PyPI Version| |readthedocs| |Coveralls| |Pre-commit|

|PyTest| |Linting| |Documentation|

|tool-semver| |tool-black| |tool-ruff| |tool-numpydoc|

.. |PyPI Version| image:: https://img.shields.io/pypi/v/XEFI?label=XEFI&logo=pypi
   :target: https://pypi.org/project/XEFI/
   :alt: pypi
.. |PyTest| image:: https://github.com/xraysoftmat/XEFI/actions/workflows/tests.yml/badge.svg
    :alt: PyTest
    :target: https://github.com/xraysoftmat/XEFI/actions/workflows/tests.yml
.. |Linting| image:: https://github.com/xraysoftmat/XEFI/actions/workflows/linting.yml/badge.svg
    :alt: Linting
    :target: https://github.com/xraysoftmat/XEFI/actions/workflows/linting.yml
.. |Documentation| image:: https://github.com/xraysoftmat/XEFI/actions/workflows/docs.yml/badge.svg
    :alt: Documentation
    :target: https://github.com/xraysoftmat/XEFI/actions/workflows/docs.yml
.. |Coveralls| image:: https://coveralls.io/repos/github/xraysoftmat/XEFI/badge.svg
    :alt: Coverage Status
    :target: https://coveralls.io/github/xraysoftmat/XEFI
.. |Pre-commit| image:: https://results.pre-commit.ci/badge/github/xraysoftmat/XEFI/main.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/xraysoftmat/XEFI/main
.. |readthedocs| image:: https://img.shields.io/readthedocs/XEFI?version=latest&style=flat&label=ReadtheDocs
    :alt: Documentation
    :target: https://XEFI.readthedocs.io/

.. |tool-semver| image:: https://img.shields.io/badge/versioning-Python%20SemVer-blue.svg
    :alt: Python SemVer
    :target: https://python-semantic-release.readthedocs.io/en/stable/
.. |tool-black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code style: black
    :target: https://github.com/psf/black
.. |tool-ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :alt: Ruff
    :target: https://github.com/astral-sh/ruff
.. |tool-numpydoc| image:: https://img.shields.io/badge/doc_style-numpydoc-blue.svg
    :alt: Code doc: numpydoc
    :target: https://github.com/numpy/numpydoc


.. image:: ./source/_static/graphics/basic-dev-ps_p3ht_si-XEFI_map.png
   :alt: Screenshot of an XEFI generated map.
   :align: center

The Model
#########
To make this model representative of the code, we count $N+1$ layers from $i=0$ to $i=N$ inclusive, as `python` indexes.

.. image:: ./source/_static/graphics/geometry.png
   :alt: Screenshot of the XEFI model geometry.
   :align: center

Here, layers :math:`i=0` and :math:`i=N` are semi-infinite layers, typically modelling air/vacuum and a substrate respectively. Boundary conditions allow us to set the incident amplitude :math:`T_0 = 1`, and the reflected amplitude :math:`R_{N}=0`. We define the following quantities:

+---------------------+-------------------------------------------------------------------------------------------------------+
| **Variable**        | **Description**                                                                                       |
+=====================+=======================================================================================================+
| :math:`N`           | The number of interfaces between the top and bottom layers, corresponding to :math:`N+1` layers       |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`i`           | The layer number, indexed from 0 (i.e. 0 to :math:`N`)                                                |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`z_i`         | The depth of the $i^{th}$ interface (:math:`z_i < 0`).                                                |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`d_i`         | The thickness of the $i^{th}$ layer (:math:`d_0 = d_N = ∞`)                                           |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`θ^t_i`       | The transmitted angle of incidence in layer :math:`i`.                                                |
|                     | Same as the angle of reflection :math:`θ^r_i` in layer:math:`i`.                                      |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`k_i`         | The z-component of the wavevector in the :math:`i^{th}` layer.                                        |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`T_i`         | The complex amplitude of the downward propogating electric field at interface :math:`i`.              |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`R_i`         | The complex amplitude of the upward propogating electric field at interface :math:`i`.                |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`X_i`         | The ratio of the downward and upward propogating electric field intensities at interface :math:`i`.   |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`E^{Total}_i` | The total electric field in layer :math:`i`.                                                          |
+---------------------+-------------------------------------------------------------------------------------------------------+
| :math:`E_{beam}`    | The X-ray beam energy in eV.                                                                          |
+---------------------+-------------------------------------------------------------------------------------------------------+

After recursively computing the ratio :math:`X_i`, then solving the amplitudes :math:`T_i`, :math:`R_i` at each interface, then the total electric field at depth :math:`z` in the film can then be calculated as the sum of downward and upward propogating waves:

$$E^{Total}_i(E_{beam}, θ^t_0, z) = T_i(E_{beam}, θ^t    _0) exp(-i k_i (z-z_i))   + R_i  (E_{beam}, θ^t_0) exp(i k_i (z-z_i))$$



.. list-table:: XEFI Features
  :widths: 20 80
  :header-rows: 1

Links
#####

Please raise any `issues <https://github.com/xraysoftmat/XEFI/issues>`_ here.

- Development (Github): https://github.com/xraysoftmat/XEFI/
- Releases
  - Github: https://github.com/xraysoftmat/XEFI/releases
  - PyPI: https://pypi.python.org/pypi/XEFI/
- Documentation (ReadtheDocs): https://XEFI.readthedocs.io/

.. toctree::
    :hidden:

    source/install
    source/tutorials/index
    source/contributing
    CHANGELOG
    TODOLIST
    source/api

.. hello?
