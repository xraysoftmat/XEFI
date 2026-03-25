.. _changelog:

=========
CHANGELOG
=========

..
    version list

.. _changelog-v0.2.0:

v0.2.0 (2026-03-25)
===================

Bug Fixes
---------

* **BasicResult**: Fix gridding bugs (`7684828`_)

* **results.py**: Fix inconsistent angle plotting (`d22bba6`_)

Documentation
-------------

* **docs**: Major update to documentation (`15398cf`_)

* **README.rst**: Add Zenodo badge (`ed6a901`_)

Features
--------

* **results.py**: Add 1D calculation (only thickness dependence) (`1d37948`_)

.. _15398cf: https://github.com/xraysoftmat/XEFI/commit/15398cf6d1ab8afade588f3a7b6a181aa4407282
.. _1d37948: https://github.com/xraysoftmat/XEFI/commit/1d379485f643b47a26309c50b6fdd091f66b409d
.. _7684828: https://github.com/xraysoftmat/XEFI/commit/7684828a2acb94bb1cb6b584729564d954c0d1ee
.. _d22bba6: https://github.com/xraysoftmat/XEFI/commit/d22bba6626100b8a785e668709468b77a0b7e720
.. _ed6a901: https://github.com/xraysoftmat/XEFI/commit/ed6a90179cd5a270a29ca57a67f4976670b13699


.. _changelog-v0.1.0:

v0.1.0 (2026-03-19)
===================

Bug Fixes
---------

* **BaseResult**: Modify ``_sum_field_intensity`` to correctly apply bounds to correct index
  (`171d15d`_)

* **results.py**: Modify graphic functions for comphrehensive plot (`ab9fa70`_)

* **sliced.py**: Change init value to None for ``slice_thickness`` attr (`cde1354`_)

Documentation
-------------

* **04-Summed-Intensity.ipynb**: Added example layered intensity profile (`84c287f`_)

* **docs**: Add further documentation structure (`a40c493`_)

* **docs**: Create initial documentation structure (`f2019e7`_)

* **docs**: Initial documentation infrastructure (`c921a35`_)

* **XEFI_basic.ipynb**: Remove outdated example (`6c83bcb`_)

.. _171d15d: https://github.com/xraysoftmat/XEFI/commit/171d15dda909c3666bc82c0e3d8a8a22987109f1
.. _6c83bcb: https://github.com/xraysoftmat/XEFI/commit/6c83bcb34841e9439de0cff6e019c8af87fa18af
.. _84c287f: https://github.com/xraysoftmat/XEFI/commit/84c287fa0ab081a6786456ea93885dc6c305cfdc
.. _a40c493: https://github.com/xraysoftmat/XEFI/commit/a40c493435a844a1ce6447eeeb848c3b229289a8
.. _ab9fa70: https://github.com/xraysoftmat/XEFI/commit/ab9fa707a4af408f783f37a6b6af6b941203aadd
.. _c921a35: https://github.com/xraysoftmat/XEFI/commit/c921a35011f174f77773dee76aa174489dd2175e
.. _cde1354: https://github.com/xraysoftmat/XEFI/commit/cde1354530a3dcda7f099408cdcda301bc41aa2e
.. _f2019e7: https://github.com/xraysoftmat/XEFI/commit/f2019e7100f08a4be7c2aade8650023c84ed219a


.. _changelog-v0.1.0-beta.1:

v0.1.0-beta.1 (2025-10-01)
==========================

* Initial Release
