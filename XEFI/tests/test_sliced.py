"""
Tests for the `model.sliced` XEFI module.
"""

import pytest
from XEFI.models import SlicedResult
from XEFI import XEF_method, XEF_Sliced
from XEFI.tests.test_basic import base_rough_attributes_defined
import numpy as np


def sliced_attributes_defined(result: SlicedResult, are_defined: bool = True):
    """
    Check that all of the sliced attributes are defined on the object.

    Parameters
    ----------
    result : BaseResult
        The result object to check.
    are_defined : bool
        Whether to check if attributes are defined or not.
    """
    base_rough_attributes_defined(result, are_defined)
    assert (
        result.slice_thickness is not None
        if are_defined
        else result.slice_thickness is None
    )
    assert result.pre_N is not None if are_defined else result.pre_N is None
    assert result.pre_z is not None if are_defined else result.pre_z is None
    assert (
        result.pre_refractive_indices is not None
        if are_defined
        else result.pre_refractive_indices is None
    )


class TestSliced:
    """
    Tests for the `models.sliced` XEFI module.
    """

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
        result = XEF_Sliced(
            energies=beam_energy,
            angles=angles_rad,
            z=z,
            z_roughness=z_rough,
            refractive_indices=refractive_indices,
            layer_names=labels,
            method=XEF_method.DEV,
            angles_in_deg=False,
        )
        sliced_attributes_defined(result)
