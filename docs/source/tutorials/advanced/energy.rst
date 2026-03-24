=====================
Energy-Dependent XEFI
=====================

``XEFI`` also possesses the ability to calculate electric field intensity maps for a range of energies, which is particularly useful for understanding how the intensity changes near absorption edges. This is done by supplying a list of energies to the ``energies`` argument in the model generator method, and ensuring that the refractive indices supplied are either complex numbers or callables that can return complex refractive indices for each energy supplied.

In this example, we use ``energies`` rather than ``angles`` as the variable parameter. However, it is worth noting that ``XEFI`` has the capability to calculate the electric field intensity for any combination of variable parameters, such as angles and energies simultaneously, by supplying lists of values for both arguments.

Calculating a map
#################

Using the same geometry and setup as in :ref:`Basic Model Tutorial<xef_basic>`, we setup the result to cover the Sulfur K-edge (~2474 eV):

.. code-block:: python

    beam_energies = np.linspace(2300, 2900, 1000)  # in eV
    angle = 0.55 # Single angle of incidence in degrees, past the critical for all layers.
    r = [20, 40, 10] # Roughness values for each interface in Å

    result = XEFI.XEF_Basic(
        energies=beam_energies,
        angles=angle,
        z=z,
        z_roughness=r,
        refractive_indices=refractive_indices,
    )

    fig, ax = result.generate_graphic_XEFI_map(z_vals=np.linspace(0, -800, 1000))

.. plot::
    :include-source: False
    :align: center
    :alt: XEFI map showing the electric field intensity as a function of depth and energy, with a clear resonance at the Sulfur K-edge.
    :class: dark-light

    import XEFI
    import numpy as np
    import kkcalc2 as kk

    energies = np.linspace(2300, 2900, 1000)  # in eV
    angles = 0.55 # Single angle of incidence in degrees, past the critical for all layers.
    z = [0, -800, -1340]  # Z-coordinates for the multilayer interface
    r = [20, 40, 10] # Roughness values for each interface in Å
    layer_names = ["Air", "PS", "P3HT", "Si"]

    # Calculated at 8050.92 eV using kkcalc2, but could be any complex
    # refractive index or callable that returns a complex refractive index.
    refractive_indices = [
        kk.models.asp_db_complex("(N78O20Ar1)0.01", density=0.001225, name="Air"),  # Air/Vacuum
        kk.models.asp_db_complex("C8H8", density=1.05, name="PS"),  # Polystyrene (C8H8)
        kk.models.asp_db_complex("C10H14S", density=1.33, name="P3HT"),  # Poly(3-hexylthiophene) (P3HT, C10H14S)
        kk.models.asp_db_complex("Si", density=2.329, name="Si"),  # Silicon (Si)
     ]

    result: XEFI.BasicResult = XEFI.XEF_Basic(
        energies=energies,
        angles=angles,
        z=z,
        refractive_indices=refractive_indices,
        layer_names=layer_names,
        z_roughness=r,
    )

    fig, ax = result.generate_graphic_XEFI_map(
        grid_roughness=True,
        angles_in_deg=True,
    )
    ax.set_title(
        rf"X-ray Electric Field Intensity at $\theta$={angles}°",
        pad=20,
    )
    plt.show()
