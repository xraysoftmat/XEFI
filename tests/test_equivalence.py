"""
Tests the consistency between methods in the XEFI Module.
"""

import pytest
from XEFI import XEF_Basic, XEF_Sliced
from .test_basic import base_rough_attributes_defined
from .test_sliced import sliced_attributes_defined
from .materials import MATERIALS
import numpy as np
import kkcalc2 as kk


class TestEquivalence:
    """
    Test the consistency between XEF models/methods in the XEFI Module.
    """

    @pytest.mark.parametrize(
        "energy, thickness, angle",
        [
            (15e3, 800, 0.1),  # eV, Angstroms, degrees
            (10e3, 500, 0.2),
            (2e3, 1000, 0.3),
        ],
    )
    def test_model_equivalence(self, energy, thickness, angle):
        """
        Test that the `XEF_Basic` and `XEF_Sliced` methods give consistent results for a simple case.
        """
        z = [0, -thickness]  # Define the z-coordinates for the multilayer interface
        z_roughness = [1, 1]  # Define the roughness for each interface

        refractive_indices: list[kk.models.asp_complex] = [
            kk.models.asp_db_complex(
                MATERIALS["Air"]["formula"], density=MATERIALS["Air"]["density"]
            ),  # Air
            kk.models.asp_db_complex(
                MATERIALS["P3HT"]["formula"], density=MATERIALS["P3HT"]["density"]
            ),  # P3HT
            kk.models.asp_db_complex(
                MATERIALS["Si"]["formula"], density=MATERIALS["Si"]["density"]
            ),  # Si
        ]

        basic_result = XEF_Basic(
            energies=energy,
            angles=angle,
            z=z,
            refractive_indices=refractive_indices,
            layer_names=["Air", "P3HT", "Si"],
            z_roughness=z_roughness,
        )
        sliced_result = XEF_Sliced(
            energies=energy,
            angles=angle,
            z=z,
            refractive_indices=refractive_indices,
            layer_names=["Air", "P3HT", "Si"],
            slice_thickness=10,  # Angstroms
            z_roughness=z_roughness,
        )

        base_rough_attributes_defined(basic_result)
        sliced_attributes_defined(sliced_result)

        # Test that the results are approximately equal (this is a very basic test and can be improved)
        z_vals = np.linspace(z[0], z[1], 100)
        basic_intensities = basic_result(z_vals)
        sliced_intensities = sliced_result(z_vals)
        difference = np.abs(basic_intensities - sliced_intensities)
        sum_diff = np.sum(difference)
        sum_basic = np.sum(np.abs(basic_intensities))
        perc_diff = difference / np.abs(basic_intensities) * 100
        ave_perc_diff = np.mean(perc_diff)
        assert ave_perc_diff < 10, (
            f"Average percentage difference is too high: {ave_perc_diff:.2f}%. Total difference: {sum_diff:.2e}, with total basic intensity: {sum_basic:.2e}."
        )
