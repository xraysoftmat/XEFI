==========
Algorithms
==========

.. warning:: This tutorial is currently under construction and may contain errors or incomplete information. Please check back later for updates.

The following algorithms are available in XEFI:

- ``DEV`` : An implementation of the original Parratt recursive algorithm, which is the default algorithm used in XEFI. This algorithm is currently the only one that is performing correctly.
- ``TOLAND`` : A modified version of the Parratt algorithm that uses a different approach to calculate the electric field intensity. This algorithm is currently under development and may not be performing correctly.
- ``ABELES`` : A matrix method that uses the Abeles matrix formalism to calculate the electric field intensity. This algorithm is currently under development and may not be performing correctly.

Each method can be selected by passing the ``method`` argument to the calculation method. For example,

.. code-block:: python

    result = XEFI.XEF_Basic(
        energies=beam_energies,
        angles=angle,
        z=z,
        refractive_indices=refractive_indices,
        method=XEFI.XEF_method.TOLAND,
    )

DEV
###

This algorithm is based on the formulation wrriten in the paper by `Dev et al., Resonance enhancement of x rays in layered materials: Application to surface enrichment in polymer blends <https://link.aps.org/doi/10.1103/PhysRevB.61.8462>`_.

The field equations are

.. math ::

    E^r_j &= a_j^2 X_j E_j^t,\\
    E^t_{j+1} &= \dfrac{a_j E^t_j t_j T_j} {1 + a_{j+1}^2 X_{j+1} r_j S_j}\\

where

.. math ::

    S &= \exp\left(-2\sigma_j^2k_{j,z}k_{j+1,z}\right),\\
    T &= \exp\left(\dfrac{\sigma_j^2(k_{j,z}-k_{j+1,z})^2}{2}\right)\\
    X_j &= \dfrac{r_j S_j + a_{j+1}^2 X_{j+1} }{1 + a_{j+1}^2 X_{j+1} r_j S_j}\\
    a_j &= \exp\left(-i k_{j,z} d_j\right)


Here :math:`S` \& :math:`T` are modifications to the Fresnel coefficients to account for roughness, while :math:`X_j` is the ratio of downward to upward propogating electric field intensities at interface :math:`j`. The algorithm is solved recursively, starting from the bottom layer and working upwards, using the boundary conditions :math:`R_N = 0` and :math:`T_0 = 1`.


TOLAND
######
.. warning:: This algorithm is currently under development and may not be performing correctly. Please check back later for updates.

ABELES
######
.. warning:: This algorithm is currently under development and may not be performing correctly. Please check back later for updates.
