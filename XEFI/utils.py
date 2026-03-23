"""
Utility functions for XEFI.

Includes external dependencies and helper functions for the XEFI package.
"""

# External dependencies
import scipy.constants as sc

# Check for optional dependencies
HAS_KKCALC: bool
try:
    # Check for kkcalc2
    import kkcalc2

    HAS_KKCALC = True
    del kkcalc2
except ImportError:
    HAS_KKCALC = False


en2wav: float = sc.h * sc.c / sc.e * 1e10
r"""
Conversion factor from energy in eV to wavelength in angstroms.

.. math::
    \lambda = (h \times c) / (E \times e)
    wav = en2wav / E

"""
en2wvec: float = 2 * sc.pi / en2wav
r"""
Conversion factor from energy in eV to wavevector in inverse angstroms.

.. math::
    \lambda = (h \times c) / (E \times e) * 1e10
    \lambda = en2wav / E
    \k = 2 \pi / (\lambda)
    \k = en2wvec * E
"""
