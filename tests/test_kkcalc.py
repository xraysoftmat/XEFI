"""
Tests the integration of KKCalc with the XEFI Module.
"""

import pytest
import numpy as np
import kkcalc2 as kk
from XEFI import XEF_Basic, XEF_Sliced
from .materials import MATERIALS
from .test_basic import base_rough_attributes_defined
from .test_sliced import sliced_attributes_defined


class TestKKCalcIntegration:
    """
    Test the integration of KKCalc with the XEFI Module.
    """

    @pytest.mark.parametrize(
        "func,test",
        [
            (XEF_Basic, base_rough_attributes_defined),
            (XEF_Sliced, sliced_attributes_defined),
        ],
    )
    def test_kkcalc_integration(self, func, test):
        """
        Test that KKCalc can be used to define the refractive indices in the XEFI Module.
        """
        angles = np.linspace(0.05, 0.5, 1000)
        thickness = 800  # Angstroms
        energy = 15e3  # eV
        z = [0, -thickness]
        r = [
            5.0,
            5.0,
        ]
        refractive_indices = [
            kk.models.asp_db_complex(
                MATERIALS["Air"]["formula"], density=MATERIALS["Air"]["density"]
            ),  # Air
            kk.models.asp_db_complex(
                MATERIALS["P3HT"]["formula"], density=MATERIALS["P3HT"]["density"]
            ),  # P3HT
            kk.models.asp_db_complex(
                MATERIALS["Si"]["formula"], density=MATERIALS["Si"]["density"]
            ),  # Si, g/cm^3
        ]

        result = func(
            energies=energy,
            angles=angles,
            z=z,
            refractive_indices=refractive_indices,
            layer_names=["Air", "P3HT", "Si"],
            z_roughness=r,
        )

        test(result)
