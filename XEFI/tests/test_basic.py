"""
Tests for the `models.basic` XEFI module.
"""

import pytest
from XEFI.models import BasicRoughResult
from XEFI.results import BaseResult, BaseRoughResult
from XEFI import XEF_method, XEF_Basic
import numpy as np


def base_attributes_defined(result: BaseResult, are_defined: bool = True):
    """
    Check that all of the base attributes are defined on the object.

    Parameters
    ----------
    result : BaseResult
        The result object to check.
    are_defined : bool
        Whether to check if attributes are defined or not.
    """
    # L,M,N:
    assert result.L is not None if are_defined else result.L is None
    assert result.M is not None if are_defined else result.M is None
    assert result.N is not None if are_defined else result.N is None
    # Energies, Angles, Z:
    assert result.energies is not None if are_defined else result.energies is None
    assert result.theta is not None if are_defined else result.theta is None
    assert result.z is not None if are_defined else result.z is None
    # Calculated Properties
    assert (
        result.angles_of_incidence is not None
        if are_defined
        else result.angles_of_incidence is None
    )
    assert (
        result.refractive_indices is not None
        if are_defined
        else result.refractive_indices is None
    )
    assert result.fresnel_r is not None if are_defined else result.fresnel_r is None
    assert result.fresnel_t is not None if are_defined else result.fresnel_t is None
    assert result.T is not None if are_defined else result.T is None
    assert result.R is not None if are_defined else result.R is None
    # Do not test X as sometimes X is not defined by the method.


def base_rough_attributes_defined(result: BaseRoughResult, are_defined: bool = True):
    """
    Check that all of the roughness attributes are defined on the object.

    Parameters
    ----------
    result : BaseResult
        The result object to check.
    are_defined : bool
        Whether to check if attributes are defined or not.
    """
    base_attributes_defined(result, are_defined)
    assert result.z_roughness is not None if are_defined else result.z_roughness is None


def basic_rough_attributes_defined(result: BasicRoughResult, are_defined: bool = True):
    """
    Check that all of the basic rough attributes are defined on the object.

    Parameters
    ----------
    result : BaseResult
        The result object to check.
    are_defined : bool
        Whether to check if attributes are defined or not.
    """
    base_rough_attributes_defined(result, are_defined)
    assert result.rough_S is not None if are_defined else result.rough_S is None
    assert result.rough_T is not None if are_defined else result.rough_T is None
    assert (
        result.fresnel_r_rough is not None
        if are_defined
        else result.fresnel_r_rough is None
    )
    assert (
        result.fresnel_t_rough is not None
        if are_defined
        else result.fresnel_t_rough is None
    )


class TestBasic:
    @pytest.mark.parametrize("method", list(XEF_method))
    def test_methods_on_3layers(self, method: XEF_method):
        beam_energy = 16.9e3  # in eV
        angles_rad = np.linspace(0.5e-3, 3.0e-3, 3000)  # in radians
        z = np.array(
            [
                0,
                -1000,
                -1500,
            ]
        )  # Define the z-coordinates for the multilayer interface
        labels = ["Air", "Poly", "Au", "Si"]
        refractive_air = 1 - (0.0) + 1j * (0.0)
        refractive_poly = 1 - (1.87e-6) + 1j * (1.18e-8)
        refractive_au = 1 - (2.18e-5) + 1j * (2.63e-6)
        refractive_si = 1 - (3.38e-6) + 1j * (1.093e-8)
        refractive_indices = [
            refractive_air,
            refractive_poly,
            refractive_au,
            refractive_si,
        ]
        result = XEF_Basic(
            energies=beam_energy,
            angles=angles_rad,
            z=z,
            refractive_indices=refractive_indices,
            layer_names=labels,
            method=XEF_method.DEV,
            angles_in_deg=False,
        )
        base_attributes_defined(result)
        result.reset()
        base_attributes_defined(result, are_defined=False)

    @pytest.mark.parametrize("method", list(XEF_method))
    def test_methods_on_rough_3layers(self, method: XEF_method):
        beam_energy = 16.9e3  # in eV
        angles_rad = np.linspace(0.5e-3, 3.0e-3, 3000)  # in radians
        z = np.array(
            [
                0,
                -1000,
                -1500,
            ]
        )  # Define the z-coordinates for the multilayer interface
        z_rough = np.array(
            [
                50,
                20,
                3,
            ]
        )

        labels = ["Air", "Poly", "Au", "Si"]
        refractive_air = 1 - (0.0) + 1j * (0.0)
        refractive_poly = 1 - (1.87e-6) + 1j * (1.18e-8)
        refractive_au = 1 - (2.18e-5) + 1j * (2.63e-6)
        refractive_si = 1 - (3.38e-6) + 1j * (1.093e-8)
        refractive_indices = [
            refractive_air,
            refractive_poly,
            refractive_au,
            refractive_si,
        ]
        result = XEF_Basic(
            energies=beam_energy,
            angles=angles_rad,
            z=z,
            z_roughness=z_rough,
            refractive_indices=refractive_indices,
            layer_names=labels,
            method=XEF_method.DEV,
            angles_in_deg=False,
        )
        basic_rough_attributes_defined(result)
        result.reset()
        basic_rough_attributes_defined(result, are_defined=False)
