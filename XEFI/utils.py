"""
Utility functions for XEFI.

Includes external dependencies and helper functions for the XEFI package.
"""

# External dependencies
import scipy.constants as sc
import numpy.typing as npt
import typing

# Check for optional dependencies
HAS_KKCALC: bool
try:
    # Check for kkcalc2
    import kkcalc2

    HAS_KKCALC = True
    del kkcalc2
except ImportError:
    HAS_KKCALC = False


_en2wav_factor: float = sc.h * sc.c / sc.e * 1e10
r"""
Conversion factor from energy in eV to wavelength in angstroms.

.. math::
    \lambda = (h \times c) / (E \times e)
    wav = en2wav / E

"""

_en2wvec_factor: float = 2 * sc.pi / _en2wav_factor
r"""
Conversion factor from energy in eV to wavevector in inverse angstroms.

.. math::
    \lambda = (h \times c) / (E \times e) * 1e10
    \lambda = en2wav / E
    \k = 2 \pi / (\lambda)
    \k = en2wvec * E
"""

T = typing.TypeVar("T", bound=npt.ArrayLike)


def en2wav(energy: T) -> T:
    r"""
    Convert energy in eV to wavelength in angstroms.

    Parameters
    ----------
    energy : npt.ArrayLike
        Energy in eV to be converted to wavelength in angstroms.

    Returns
    -------
    npt.ArrayLike
        Wavelength in angstroms corresponding to the input energy in eV.
    """
    return _en2wav_factor / energy


def en2wvec(energy: T) -> T:
    r"""
    Convert energy in eV to wavevector in inverse angstroms.

    Parameters
    ----------
    energy : npt.ArrayLike
        Energy in eV to be converted to wavevector in inverse angstroms.

    Returns
    -------
    npt.ArrayLike
        Wavevector in inverse angstroms corresponding to the input energy in eV.
    """
    return _en2wvec_factor * energy


def wav2en(wavelength: T) -> T:
    r"""
    Convert wavelength in angstroms to energy in eV.

    Parameters
    ----------
    wavelength : npt.ArrayLike
        Wavelength in angstroms to be converted to energy in eV.

    Returns
    -------
    npt.ArrayLike
        Energy in eV corresponding to the input wavelength in angstroms.
    """
    return _en2wav_factor / wavelength


def wvec2en(wvec: T) -> T:
    r"""
    Convert wavevector in inverse angstroms to energy in eV.

    Parameters
    ----------
    wvec : npt.ArrayLike
        Wavevector in inverse angstroms to be converted to energy in eV.

    Returns
    -------
    npt.ArrayLike
        Energy in eV corresponding to the input wavevector in inverse angstroms.
    """
    return wvec / _en2wvec_factor
