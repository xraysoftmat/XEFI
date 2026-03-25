=========
Tutorials
=========

XEFI is designed to be easy to use and make a calculation in Python3.
The package is broken into two types of important classes/functions.

``XEFI.models``: This module contains the main classes for generating X-ray electric field intensity calculations.
  * Submodule for generator methods, such as
     * ``XEFI.models.XEF_Basic``
     * ``XEFI.models.XEF_Sliced``
  * and their corresponding result objects such as
     * ``XEFI.models.BasicResult``
     * ``XEFI.models.BasicRoughResult``
     * ``XEFI.models.SlicedResult``

``XEFI.results``: Submodule for the main calculation method, with templates for ``XEFI.results.BaseResult`` and ``XEFI.results.BaseRoughResult`` classes, which result classes in ``XEFI.models`` rely on. These classes implement much of the graphing and calculation logic required, unless overriden in the model result class.

Select a model to get started.


.. toctree::
    :maxdepth: 2

    theory

.. toctree::
    :caption: Models
    :maxdepth: 2

    xef/basic
    xef/sliced

.. toctree::
    :caption: Advanced
    :maxdepth: 2

    advanced/kkcalc2
    advanced/energy
    advanced/algorithms
