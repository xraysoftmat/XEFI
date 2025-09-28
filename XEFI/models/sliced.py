"""
Module for the XEFI calculation of a sliced set of layers, decomposed into a specified thickness.
"""

import warnings
from typing import Callable
from XEFI.results import BaseRoughResult, XEF_method
from XEFI.models.basic import (
    XEF_Parratt_Dev,
    XEF_Parratt_Tolan,
    XEF_Abeles_Ohta,
    en2wvec,
)
import numpy as np
import numpy.typing as npt
from scipy import special as sp

try:
    from kkcalc.models import asp_complex

    has_KKCalc = True
except ImportError:
    has_KKCalc = False


class SlicedResult(BaseRoughResult):
    """
    Result class for the sliced XEFI model, inheriting from BaseResult.

    This class extends the BaseResult to include additional properties specific to the sliced model.

    Notably, `z_interfaces` is the list of interfaces prior to slicing, and `z` is the list of interfaces
    after slicing. `layer_names` now corresponds to the layers separated by `z_interfaces`, not `z`.

    Attributes
    ----------
    slice_thickness : float
        The thickness of each slice in Angstroms (Å).
    z : npt.NDArray[np.floating] | None
        The z-coordinate of the (N) Å layer interfaces in Angstroms (Å).
    z_interfaces : npt.NDArray[np.floating] | None
        The z-coordinate of the original pre-sliced interfaces in Angstroms (Å).
    z_roughness : npt.NDArray[np.floating] | None
        The roughness values for each original pre-sliced interfaces in angstroms (Å).
    L : int | None
        The number of beam energies considered.
    M : int | None
        The number of angles of incidence considered.
    N : int | None
        The number of interfaces, corresponding to N+1 layers.
    beam_energy : float | None
        The energy of the X-ray beam in eV.
    wavelength : float | None
        The wavelength of the X-ray in Angstroms (Å).
    theta : npt.NDArray[np.floating] | None
        The angles of incidence (M) in the first layer (i=0) in degrees.
    angles_of_incidence : npt.NDArray[np.floating] | None
        The angles of incidence in each layer in radians (N+1, M).
    refractive_indices_preslice : npt.NDArray[np.complexfloating] | None
        The complex refractive indices of each layer before slicing.
    refractive_indices : npt.NDArray[np.complexfloating] | None
        The complex refractive indices of each layer (N+1).
    wavevectors : npt.NDArray[np.floating] | None
        The z-component wavevector in each layer (N+1, M).
        Defined as a magnitude with a postitive complex phase, rather than a vector direction.
    k0 : float | None
        The incident vacuum wavevector.
    fresnel_r : npt.NDArray[np.complexfloating] | None
        The Fresnel reflection coefficients for each interface and angle (N, M).
    fresnel_t : npt.NDArray[np.complexfloating] | None
        The Fresnel transmission coefficients for each interface and angle (N, M).
    method : XEF_method | None
        The XEF calculation method used.
    layer_names : list[str] | None
        The names of the layers (N+1), if provided.
    """

    def __init__(self) -> None:
        super().__init__()
        # New Definitions
        self.slice_thickness: float = 0.0
        """The thickness of each slice in Angstroms (Å)."""
        self.z_interfaces: npt.NDArray[np.floating] | None = None
        """The z-coordinate of the original pre-sliced interfaces in Angstroms (Å)."""
        self.refractive_indices_preslice: npt.NDArray[np.complexfloating] | None = None
        """The complex refractive indices of each layer before slicing."""
        # Re-commented Definitions
        self.z: npt.NDArray[np.floating] | None = None
        """The z-coordinate of the (N) Å layer interfaces in Angstroms (Å)."""
        self.z_roughness: npt.NDArray[np.floating] | None = None
        """The roughness values for each original pre-sliced interfaces in angstroms (Å)."""


def XEF_Sliced(
    energies: list[float] | npt.NDArray[np.floating] | float,
    angles: list[float] | npt.NDArray[np.floating] | float,
    z: list[float | int] | npt.NDArray[np.floating | np.integer],
    refractive_indices: (
        list[complex]
        | npt.NDArray[np.complexfloating]
        | list[Callable]
        | list["asp_complex"]
    ),
    z_roughness: list[float] | npt.NDArray[np.floating],
    slice_thickness: float = 1.0,
    sigma: float = 4.0,
    *,
    method: XEF_method | str = XEF_method.DEV,
    layer_names: list[str] | None = None,
    angles_in_deg: bool = True,
) -> SlicedResult:
    """
    Calculate the complex X-ray Electric Field for a set of layers.

    Uses the sliced method to decompose layers into thinner slices for more accurate calculations.

    Parameters
    ----------
    energies : list[float] | npt.NDArray[np.floating] | float
        The beam energy(s) in eV. Can be a single value or an array of values.
    angles : list[float] | npt.NDArray[np.floating] | float
        The angles in degrees at which to calculate the XEFI. Can be a single value or an array of values.
    z : list[float] | npt.NDArray[np.floating]
        The interface locations between different materials in Angstroms. Must be a list or array of floats.
    refractive_indices : list[complex] | npt.NDArray[np.complexfloating] | list[Callable] | list["asp_complex"]
        The refractive indices for each energy and layer.
        Can be a list of complex numbers (L, N+1), a numpy array of complex numbers (L, N+1),
        a list of `KKCalc` `asp_complex` objects (N+1), or a list of Callable functions that return complex numbers (N+1).
    z_roughness : list[float] | npt.NDArray[np.floating] | None, optional
        The roughness of the interfaces in Angstroms. If provided, it should be a list or array of floats with the same length as `z`.
        If None, no roughness is applied, by default None.
    slice_thickness : float, optional
        The thickness of each slice in Angstroms (Å), by default 1.0 Å.
    sigma : float, optional
        The extra z-range to extend beyond the provided interfaces for roughness calculations, in units of the interface roughness.
        Default is 4.0, which corresponds to 99.38 of the roughness profile assuming a Gaussian distribution.
    method : XEF_method | str
        The method to use for the calculation, by default `XEF_method.dev`.
    layer_names : list[str] | None, optional
        The names of the layers corresponding to the refractive indices, by default None.
    angles_in_deg : bool
        Whether the angles are in degrees (True) or radians (False), by default True.

    Returns
    -------
    Sliced Result
        An instance of `SlicedResult` containing the calculated XEFI data.
    """
    # Check and convert method to enum
    if isinstance(method, str):
        if method in XEF_method:
            method = XEF_method(method)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Initialise a result
    result = SlicedResult()
    # Store the slice thickness and original interfaces and roughnesses
    assert slice_thickness > 0, "Slice thickness must be a positive value."
    result.slice_thickness = slice_thickness
    # Interface locations
    assert z is not None, "Interface locations (z) must be provided."
    assert len(z) >= 2, "At least two interface locations (z) must be provided."
    z_interfaces = np.array(z, dtype=np.float64, copy=True)
    assert np.all(np.diff(z_interfaces) < 0), (
        "Interface locations (z) must be in decending order (into the substrate)."
    )
    result.z_interfaces = z_interfaces
    # Roughness
    z_roughness = np.array(z_roughness, dtype=np.float64, copy=True)
    assert z_roughness.ndim == 1
    assert len(z_roughness) == len(z_interfaces), (
        "Roughness values (z_roughness) must match the number of interfaces provided (z)."
    )
    result.z_roughness = z_roughness

    # Beam energy
    energies = np.atleast_1d(energies)
    """The beam energy(s) in eV (with length L)"""
    result.energies = energies
    # Energy number
    L: int = energies.shape[0]
    """Number of energies."""
    result.L = L
    # Angles
    angles = np.atleast_1d(angles)
    theta = np.deg2rad(angles) if angles_in_deg else angles
    """The angles in radians (with length M)"""
    theta = np.atleast_1d(theta)
    result.theta = theta
    # Angle number
    M: int = angles.shape[0]
    """Number of angles."""
    result.M = M

    # Calculate the sliced interfaces and refractive indices
    # Interface number
    N: int = int(
        np.ceil(
            (
                (z_interfaces[0] + sigma * z_roughness[0])
                - (z_interfaces[-1] - sigma * z_roughness[-1])
            )
            / slice_thickness
        )
    )
    """Number of `sliced` interfaces (N)."""
    result.N = N
    z_sliced = np.arange(
        z_interfaces[0] + sigma * z_roughness[0],
        z_interfaces[-1] - sigma * z_roughness[-1],
        -slice_thickness,  # Decreasing z into the substrate
        dtype=np.float64,
    )
    if len(z_sliced) == N - 1:
        z_sliced = np.append(z_sliced, z_interfaces[-1] - sigma * z_roughness[-1])
    assert len(z_sliced) == N, (
        f"Calculated number of sliced interfaces {len(z_sliced)} does not match the expected number {N} \
        with `slice_thickness` {slice_thickness}, `sigma` {sigma} and `z` {z_interfaces}."
    )
    del z  # Do not use z anymore, as it is ambiguous in this code. Now use z_sliced or z_interfaces.

    # Refractive indices require verification based on supplied data type
    ref_idxs_preslice: npt.NDArray[np.complexfloating]
    """A numpy array of complex refractive indices across all energies (L) and pre-sliced layers."""
    print(refractive_indices)
    if (
        isinstance(refractive_indices, np.ndarray)
        and L > 1
        and refractive_indices.ndim == 2
        and refractive_indices.shape[1] == N + 1
        and refractive_indices.shape[0] == L
    ):
        # Valid refractive indices for multiple energies
        if issubclass(refractive_indices.dtype.type, np.complexfloating):
            ref_idxs_preslice = refractive_indices.copy()
        else:
            raise ValueError(
                f"Refractive indices must be a numpy array of complex numbers \
                              for multiple energies. Dtype was instead {refractive_indices.dtype}"
            )

    elif (
        isinstance(refractive_indices, np.ndarray)
        and L == 1
        and refractive_indices.ndim == 1
        and refractive_indices.shape[0] == N + 1
    ):
        # Valid refractive indices for a single energy
        if issubclass(refractive_indices.dtype.type, np.complex128):
            ref_idxs_preslice = refractive_indices.copy()[np.newaxis, :]
        else:
            raise ValueError(
                f"Refractive indices must be a numpy array of complex numbers \
                                for a single energy. Dtype was instead {refractive_indices.dtype}"
            )

    elif (
        isinstance(refractive_indices, list)
        and all(isinstance(n, (int, float, complex)) for n in refractive_indices)
        and L == 1
        and len(refractive_indices) == N + 1
    ):
        # Valid refractive indices for a single energy
        ref_idxs_preslice = np.array(
            refractive_indices, dtype=np.complex128, copy=True
        )[np.newaxis, :]
        if np.sum(ref_idxs_preslice.imag != 0) == 0:
            warnings.warn(
                "Refractive indices provided are all real. \
                This may not be correct for X-ray calculations.",
                UserWarning,
            )
    elif (
        isinstance(refractive_indices, (list, np.ndarray))
        and has_KKCalc
        # Allow float for vacuum or air - i.e. no material absorption.:
        and all(
            isinstance(n, (float, complex, asp_complex)) for n in refractive_indices
        )
        and len(refractive_indices) == N + 1
    ):
        # Valid refractive indices for a single energy using KKCalc
        ref_idxs_preslice = np.zeros((L, N + 1), dtype=np.complex128)
        for i, mat_n in enumerate(
            refractive_indices
        ):  # Iterate over the layers, apply the energy.
            if isinstance(mat_n, asp_complex):
                ref_idxs_preslice[:, i] = mat_n.eval_refractive_index(
                    energies
                )  # Apply the energy to the KKCalc object
            elif isinstance(mat_n, complex):
                ref_idxs_preslice[:, i] = mat_n
            else:  # float
                ref_idxs_preslice[:, i] = complex(
                    real=mat_n, imag=0
                )  # Convert to complex]
    elif (
        isinstance(refractive_indices, list)
        and len(refractive_indices) == N + 1
        and all(
            callable(n) for n, i in enumerate(refractive_indices) if i != 0
        )  # Allow for first layer to be air or vacuum with n=1
    ):
        # Valid refractive indices for multiple energies using Callable
        ref_idxs_preslice = np.zeros((L, N + 1), dtype=np.complex128)
        single_energy_calc: bool = (
            False  # Prevent callable from being called with multiple energies.
        )
        for i, mat_n in enumerate(
            refractive_indices
        ):  # Iterate over the layers, apply the energy.
            if i == 0 and (isinstance(mat_n, (int, float))):
                ref_idxs_preslice[:, i] = mat_n + 0j  # Convert to complex
                continue

            assert callable(mat_n)
            if L == 1:
                ref_idxs_preslice[0, i] = mat_n(
                    energies
                )  # Apply the energy to the Callable function
            elif single_energy_calc:
                for j in range(L):
                    ref_idxs_preslice[j, i] = mat_n(
                        energies[j]
                    )  # Apply the energy to the Callable function
            else:
                if callable(mat_n):
                    try:
                        ref_idxs_preslice[:, i] = mat_n(
                            energies
                        )  # Apply the energy to the Callable function
                    except TypeError | ValueError:
                        single_energy_calc = True
                        for j in range(L):
                            ref_idxs_preslice[j, i] = mat_n(
                                energies[j]
                            )  # Apply the energy to the Callable function

    else:
        raise ValueError(
            "Refractive index values must be a list of complex numbers, \
                            a numpy array of complex numbers, a list of `KKCalc` `asp_complex` objects \
                            or a list of Callable functions for multiple energies."
        )
    result.refractive_indices_preslice = ref_idxs_preslice

    # Convert the refractive indices to a sliced set based on the provided slice thickness
    ref_idxs: npt.NDArray[np.complexfloating] = np.zeros(
        (L, N + 1), dtype=np.complex128
    )
    for i in range(1, N):
        zi = z_sliced[i]
        # Find the index of the interface that is closest to zi
        j = np.digitize(zi, z_interfaces)
        if j > 0 and j < len(z_interfaces):
            # Check which interface is closest.
            j = (
                j - 1
                if abs(zi - z_interfaces[j - 1]) < abs(z_interfaces[j] - zi)
                else j
            )
        elif j > len(z_interfaces) - 1:
            j = len(z_interfaces) - 1

        # Get the roughness between j and j+1
        sigma_i = z_roughness[j]
        # (n1 + n2)/2 - (n1 - n2)/2 * erf(z_interface / (np.sqrt(2) * sigma_i))
        refractive_indices[i] = (
            ref_idxs_preslice[:, j] + ref_idxs_preslice[:, j + 1]
        ) / 2 - (ref_idxs_preslice[:, j] - ref_idxs_preslice[:, j + 1]) / 2 * sp.erf(
            -(zi - z_interfaces[j]) / (np.sqrt(2) * sigma_i)
        )

    # First and last layers are the same as the pre-sliced
    ref_idxs[:, 0] = ref_idxs_preslice[:, 0]
    ref_idxs[:, -1] = ref_idxs_preslice[:, -1]

    # Labels
    if layer_names is not None:
        assert len(layer_names) == N + 1, (
            "Layer names must match the number of layers (N+1)."
        )
        result.layer_names = layer_names.copy()
    else:
        result.layer_names = None
        layer_names = [f"Layer {i}" for i in range(N + 1)]

    ## Generate result data
    # Wavevector magnitude in vacuum
    k0: npt.NDArray[np.floating] = en2wvec * energies  # convert energy to wavevector.
    """The wavevector magnitude (per angstrom) in vacuum for each energy (L)."""
    result.k0 = k0

    # Wavevector-z in each layer
    wavevectors: npt.NDArray[np.complexfloating] = np.zeros(
        (L, M, N + 1), dtype=np.complex128
    )
    """Z-component wavevectors at each energy (L) and angle (M) for each layer (N + 1)."""
    wavevectors[:, :, 0] = k0[:, np.newaxis] * np.sin(theta[np.newaxis, :])

    # Angle of incidence in each layer
    angles_of_incidence: npt.NDArray[np.complexfloating] = np.zeros(
        (L, M, N + 1), dtype=np.complex128
    )
    """The angles of incidence at each energy (L) and angle (M) for each layer (N + 1) in radians."""
    angles_of_incidence[:, :, 0] = theta[
        np.newaxis, :
    ]  # Vacuum layer has no refraction

    # Calculate angles of incidence for each layer: Snell's Law
    angles_of_incidence[:, :, 1:] = np.arccos(
        np.cos(theta[np.newaxis, :, np.newaxis])
        * ref_idxs[:, np.newaxis, 0, np.newaxis]
        / ref_idxs[:, np.newaxis, 1:]
    )
    # Calculate wavevectors for each layer
    wavevectors[:, :, 1:] = k0[:, np.newaxis, np.newaxis] * np.sqrt(
        (ref_idxs[:, np.newaxis, 1:] ** 2)
        - (np.cos(theta[np.newaxis, :, np.newaxis]) ** 2)
    )
    result.wavevectors = wavevectors
    result.angles_of_incidence = angles_of_incidence

    # Calculate the Fresnel coefficients for each layer
    fresnel_r = np.zeros((L, M, N), dtype=np.complex128)
    """The Fresnel reflection coefficients for each interface (L, M, N)"""
    fresnel_t = np.zeros((L, M, N), dtype=np.complex128)
    """The Fresnel transmission coefficients for each interface (L, M, N)"""
    fresnel_t[:, :, :] = (
        2 * wavevectors[:, :, :-1] / (wavevectors[:, :, :-1] + wavevectors[:, :, 1:])
    )
    fresnel_r[:, :, :] = fresnel_t[:, :, :] - 1.0
    # fresnel_r[:, :, :] = (
    #     (wavevectors[:, :, :-1] - wavevectors[:, :, 1:]) /
    #     (wavevectors[:, :, :-1] + wavevectors[:, :, 1:])
    # )
    result.fresnel_r = fresnel_r
    result.fresnel_t = fresnel_t

    # Calculate the critical angles
    critical_angles: npt.NDArray[np.floating] = np.sqrt(2 * (1 - ref_idxs[:, 1:].real))
    """The critical angles of each energy and material interface (excluding vacuum/air) (L, N) in radians."""
    result.critical_angles = critical_angles

    # Define variable links to pass to XEF methods
    fr_t = fresnel_t
    fr_r = fresnel_r

    T, R, X = None, None, None
    result.method = method
    match method:
        case XEF_method.OHTA:
            R, T = XEF_Abeles_Ohta(
                L=L,
                M=M,
                N=N,
                wavevectors=wavevectors,
                fresnel_r=fr_r,
                fresnel_t=fr_t,
                z=z_sliced,
            )
        case XEF_method.TOLAN:
            R, T, X = XEF_Parratt_Tolan(
                L=L,
                M=M,
                N=N,
                wavevectors=wavevectors,
                fresnel_r=fr_r,
                fresnel_t=fr_t,
                z=z_sliced,
            )
        case XEF_method.DEV:
            R, T, X = XEF_Parratt_Dev(
                L=L,
                M=M,
                N=N,
                wavevectors=wavevectors,
                fresnel_r=fr_r,
                fresnel_t=fr_t,
                z=z_sliced,
            )
        case _:
            raise NotImplementedError(f"Method {method} not yet implemented.")

    # Save results.
    result.R = R
    result.T = T
    result.X = X

    # Squeeze the results.
    result.squeeze()

    # Return the result
    return result
