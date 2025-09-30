"""
Module for the XEFI calculation of a sliced set of layers, decomposed into a specified thickness.
"""

import warnings
from typing import Callable, override, Literal

from matplotlib.axes import Axes, Axes as mplAxes
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
    z_roughness : npt.NDArray[np.floating] | None
        The roughness values for each original pre-sliced interfaces in angstroms (Å).
    L : int | None
        The number of beam energies considered.
    M : int | None
        The number of angles of incidence considered.
    N : int | None
        The number of interfaces, corresponding to N+1 layers.
    energies : float | None
        The energies of the X-ray beam in eV (L).
    theta : npt.NDArray[np.floating] | None
        The angles of incidence (M) in the first layer (i=0) in degrees.
    angles_of_incidence : npt.NDArray[np.floating] | None
        The complex angles of incidence for each energy, angle and (within each) layer in radians (L, M, N+1).
    wavevectors : npt.NDArray[np.complexfloating] | None
        The unsigned complex z-component wavevector in each layer (L, M, N+1)
    refractive_indices : npt.NDArray[np.complexfloating] | None
        The complex refractive indices at each energy of each layer (L, N+1).
    k0 : npt.NDArray[np.floating] | None
        The incident vacuum wavevector for each energy (L).
    fresnel_r : npt.NDArray[np.complexfloating] | None
        The Fresnel reflection coefficients for each energy, angle and interface (L, M, N).
    fresnel_t : npt.NDArray[np.complexfloating] | None
        The Fresnel transmission coefficients for each energy, angle and interface (L, M, N).
    X : npt.NDArray[np.complexfloating] | None
        The complex ratio of downward and upward propagating fields for each energy, angle and interface (L, M, N).
    method : XEF_method | None
        The XEF calculation method used.
    z_roughness : npt.NDArray[np.floating] | None
        The z-coordinates corresponding to the roughness profile in angstroms (Å).
    pre_N : int | None
        The pre-sliced number of layers.
    pre_z : npt.NDArray[np.floating] | None
        The z-coordinate of the original pre-sliced interfaces in Angstroms (Å).
    pre_refractive_indicies : npt.NDArray[np.complexfloating] | None
        The complex refractive indices of each layer before slicing (pre_N).
    critical_angles : npt.NDArray[np.floating] | None
        The critical angles for total internal reflection for each beam energy and interface (L, pre_N) in radians.
    layer_names : list[str] | None
        The names of the pre-sliced layers (pre_N + 1), if provided.
    """

    @override
    def __init__(self) -> None:
        super().__init__()
        # New Definitions
        self.slice_thickness: float = 0.0
        """The thickness of each slice in Angstroms (Å)."""
        self.pre_N: int | None
        """The pre-sliced number of layers."""
        self.pre_z: npt.NDArray[np.floating] | None = None
        """The z-coordinate of the original pre-sliced interfaces (pre_N) in Angstroms (Å)."""
        self.pre_refractive_indices: npt.NDArray[np.complexfloating] | None = None
        """The complex refractive indices at each energy of each layer before slicing (L, pre_N)."""
        # Re-commented Definitions
        self.z: npt.NDArray[np.floating] | None = None
        """The z-coordinate of the (N) `slice_thickness` Å layer interfaces in Angstroms (Å)."""
        self.z_roughness: npt.NDArray[np.floating] | None = None
        """The roughness values for each original pre-sliced interfaces (pre_N) in angstroms (Å)."""
        self.critical_angles: npt.NDArray[np.floating] | None
        """The critical angles for total internal reflection for each beam energy and interface (L, pre_N) in radians."""

    @override
    def reset(self) -> None:
        """
        Reset all attributes to their initial state.
        """
        # Reset parent properties
        super().reset()
        # Reset new properties
        self.slice_thickness = 0.0
        self.pre_N = None
        self.pre_z = None
        self.pre_refractive_indices = None

    @override
    def graph_refractive_indexes(
        self,
        ax_re: Axes | None = None,
        ax_im: Axes | None = None,
        l_index: int | None = None,
    ) -> tuple[Axes, Axes] | None:
        #
        result = super().graph_refractive_indexes(
            ax_re=ax_re,
            ax_im=ax_im,
            l_index=l_index,
        )
        if ax_re is None and result is not None and result[0] is not None:
            ax_re = result[0]
        if ax_im is None and result is not None and result[1] is not None:
            ax_im = result[1]

        # Also plot the pre-sliced refractive indices if available.
        if ax_re is not None or ax_im is not None:
            pre_N = self.pre_N
            L = self.L
            pre_ref_idxs = self.pre_refractive_indices
            assert pre_N is not None
            assert pre_ref_idxs is not None
            if l_index is not None:
                assert L is not None
                if l_index < 0 or l_index >= L:
                    raise ValueError(f"l_index {l_index} is out of bounds for L={L}.")
                pre_ref_idxs = pre_ref_idxs[l_index, :]
            # Create singular horizontal line
            for i in range(pre_N + 1):
                if ax_im is not None:
                    ax_im.axhline(
                        y=pre_ref_idxs[i].imag, color="C1", linestyle="--", alpha=0.5
                    )
                if ax_re is not None:
                    ax_re.axhline(
                        y=pre_ref_idxs[i].real, color="C1", linestyle="--", alpha=0.5
                    )

        return

    # @override
    @BaseRoughResult.critical_angles_deg.getter
    def critical_angles_deg(self) -> npt.NDArray[np.floating] | None:
        """
        The critical angles for each energy and interface (L, pre_N) in degrees.

        Returns
        -------
        npt.NDArray[floating] | float | None
            The critical angles, or None if theta is not set.
        """
        return (
            np.rad2deg(self.critical_angles)
            if self.critical_angles is not None
            else None
        )

    @override
    def _require_singular_x_data(
        self,
        l_index: int | None = None,
        m_index: int | None = None,
        angles_in_degrees: bool = True,
    ) -> tuple[Literal["theta", "energy"], np.ndarray, np.ndarray | None]:
        """
        Ensure that the data for the x-axis (either `theta` or `energy`) is singular (1D).

        This is a helper function for 2D plotting functions, which require either `theta` or `energy`
        to be singular (1D). If both dimensions are greater than 1, the user must specify either `m_index`
        or `l_index` to select a singular index.

        Modified from `BaseResult` to serve a `SlicedResult`.

        Parameters
        ----------
        l_index : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        m_index : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.
        angles_in_degrees : bool, optional
            Whether to return the angles of incidence and the critical angles
            in degrees (True) or radians (False).

        Returns
        -------
        x_datatype : Literal["theta", "energy"] | None
            The type of data selected for the x-axis, either "theta" or "energy".
            None if neither is singular.
        x_data : np.ndarray | None
            The data for the x-axis, either `theta` or `energy`.
            None if neither is singular.
        critical_angles : np.ndarray | None
            The pre-sliced critical angles for total internal reflection for each beam energy and
            interface (L, pre_N) in radians. Subset if l_index is not None. None if not defined.
        """
        L, M = self.L, self.M
        theta = self.theta_deg if angles_in_degrees else self.theta
        energies = self.energies
        critical_angles = (
            self.critical_angles_deg if angles_in_degrees else self.critical_angles
        )

        x_selection: Literal["theta", "energy"] | None = None
        if L is not None and L > 1 and M is not None and M > 1:
            assert isinstance(theta, np.ndarray)
            assert isinstance(energies, np.ndarray)
            if m_index is not None and l_index is not None:
                raise ValueError(
                    "Values for both `l` and `m` are incompatible, as data is required to be 2D."
                )
            elif m_index is not None:
                theta = theta[m_index]
                x_data = energies
                x_selection = "energy"
                M = 1  # Reduce M to 1 for plotting
            elif l_index is not None:
                energies = energies[l_index]
                x_data = theta
                x_selection = "theta"
                L = 1  # Reduce L to 1 for plotting
                if critical_angles is not None:
                    critical_angles = critical_angles[l_index]
            else:
                raise ValueError(
                    "This method is designed for 2D plotting. Ensure that either L or M are singular, \
                    or choose a singular index using `l` or `m` function parameters."
                )
        elif L is not None and L > 1:
            assert isinstance(energies, np.ndarray)
            x_selection = "energy"
            x_data = energies
        elif M is not None and M > 1:
            assert isinstance(theta, np.ndarray)
            x_selection = "theta"
            x_data = theta
        else:
            raise ValueError(
                "This method is designed for 2D plotting. Ensure that either L or M are singular, \
                or choose a singular index using `l` or `m` function parameters."
            )

        return x_selection, x_data, critical_angles

    @override
    def _add_gridding_to_XEFI(
        self,
        ax: Axes,
        l_index: int | None = None,
        m_index: int | None = None,
        grid_z: bool = True,
        grid_labels: bool = True,
        grid_crit: bool = True,
        labels: list[str] | None = None,
    ) -> None:
        """
        To add gridding lines.

        Modified for a sliced model to only add grid lines of non-sliced interfaces.
        You can apply `BaseResult._add_gridding_to_XEFI` to the result if you want
        to grid the sliced layers.

        Grid lines can be added for specified:
        - `grid_z`: `z` values of interfaces
        - `grid_labels`: `labels` of different layers.
        - `grid_crit`: the `critical_angles` of the pre-sliced interfaces.

        The second dimension is either `theta` (angle of incidence) or `beam_energy`.
        This dimension is automatically chosen if one of the dimensions is singular (i.e., has length 1).
        Otherwise, the user must specify either `m_index` or `l_index` to choose a singular index.

        Parameters
        ----------
        ax : Axes
            The matplotlib axes to add the gridding.
        l_index : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        m_index : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.
        grid_z : bool, optional
            Whether to plot the layer grid. Defaults to True.
        grid_labels : bool, optional
            Whether to plot the z layer labels. Defaults to True.
        grid_crit : float, optional
            Whether to plot the critical angles grid. Defaults to True.
        labels : list[str] | None, optional
            The labels for the z layers. If None, defaults to automatic labels.
        """
        pre_N, pre_z = self.pre_N, self.pre_z
        assert pre_N is not None
        assert pre_z is not None

        # Get the sliced x-data.
        x_selection, x_data, critical_angles = self._require_singular_x_data(
            m_index=m_index,
            l_index=l_index,
        )

        ### Add gridding and labels to the layers
        if labels is None:
            result_labels = self.layer_names
            if result_labels is not None:
                labels = result_labels
            else:
                labels = [f"Layer {i}" for i in range(pre_N + 1)]
        # Layers:
        if grid_z:
            for zi in pre_z:
                ax.axhline(zi, color="white", linestyle="--", alpha=0.2, linewidth=0.5)
            if grid_labels and labels is not None:
                for i, zi in enumerate(pre_z):
                    name = labels[i + 1]
                    ax.text(
                        x=1.0,
                        y=zi,
                        s=name,
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="top",
                        color="white",
                    )
                ax.text(
                    x=1.0,
                    y=pre_z[0],
                    s=labels[0],
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    color="white",
                )

        # Critical Angles
        if (
            grid_crit and x_selection == "theta" and critical_angles is not None
        ):  # Is a angle plot.
            for ang in critical_angles:
                if ang > x_data.min() and ang < x_data.max():
                    ax.axvline(
                        x=ang, color="white", linestyle="--", alpha=0.2, linewidth=0.5
                    )
            if grid_labels and labels is not None:
                for i, ang in enumerate(critical_angles):
                    if ang > x_data.min() and ang < x_data.max():
                        name = labels[i + 1]
                        ax.text(
                            x=ang,
                            y=0.00,
                            s=name,
                            transform=ax.get_xaxis_transform(),
                            color="white",
                            ha="center",
                        )

    @override
    def _add_rough_gridding_to_XEFI(
        self,
        ax: mplAxes,
        l_index: int | None = None,
        m_index: int | None = None,
    ):
        """
        Represent the roughness on the grid by transparent bars.

        Parameters
        ----------
        ax : mplAxes
            The axis on which the XEFI map has been created.
        l_index : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        m_index : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.
        """
        z = self.pre_z
        zr = self.z_roughness

        x_selection, x_data, _ = self._require_singular_x_data(
            l_index=l_index,
            m_index=m_index,
        )

        if x_selection is None:
            raise ValueError(
                "Requires a singular dependent variable (theta or energy)."
            )
        if z is None:
            raise ValueError("`z` is not defined.")
        if zr is None:
            raise ValueError("`z_roughness` is not defined on the result.")
        for i, zi in enumerate(z):
            ax.fill_between(
                x=x_data,
                y1=zi - zr[i] / 2,
                y2=zi + zr[i] / 2,
                alpha=0.1,
                hatch="/",
                color="white",
            )


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
    sigmas: float = 4.0,
    enforce_boundary: bool = True,
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
    sigmas : float, optional
        The extra z-range to extend beyond the provided interfaces for roughness calculations, in units of the interface roughness.
        Default is 4.0, which corresponds to 99.38 of the roughness profile assuming a Gaussian distribution.
    enforce_boundary : bool, optional
        Ensure that the sliced model has the first and last slice matching the top and bottom refractive index,
        rather than a calculation of the rough combination across an interface. By default True.
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
    result.pre_z = z_interfaces
    # Number of pre-sliced interfaces
    pre_N: int = z_interfaces.shape[0]
    result.pre_N = pre_N
    # Roughness
    z_roughness = np.array(z_roughness, dtype=np.float64, copy=True)
    assert z_roughness.ndim == 1
    assert len(z_roughness) == pre_N, (
        "Roughness values (z_roughness) must match the number of interfaces provided (pre_N)."
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
    # Note, we need to include extra z layers depending on interface roughness.
    # Interface number
    N: int = int(
        np.ceil(
            (
                (
                    z_interfaces[0] + sigmas * z_roughness[0]
                )  # sigma roughness above the first interface
                - (
                    z_interfaces[-1] - sigmas * z_roughness[-1]
                )  # sigma roughness below the last interface
            )
            / slice_thickness
        )
    )
    """Number of `sliced` interfaces (N)."""
    result.N = N
    z_sliced = np.arange(
        z_interfaces[0] + sigmas * z_roughness[0],
        z_interfaces[-1] - sigmas * z_roughness[-1],
        -slice_thickness,  # Decreasing z into the substrate
        dtype=np.float64,
    )
    if len(z_sliced) == N - 1:
        z_sliced = np.append(z_sliced, z_interfaces[-1] - sigmas * z_roughness[-1])
    assert len(z_sliced) == N, (
        f"Calculated number of sliced interfaces {len(z_sliced)} does not match the expected number {N} \
        with `slice_thickness` {slice_thickness}, `sigma` {sigmas} and `z` {z_interfaces}."
    )
    result.z = z_sliced
    del z  # Do not use z anymore, as it is ambiguous in this code. Now use z_sliced or z_interfaces.

    # Refractive indices require verification based on supplied data type
    pre_ref_idxs: npt.NDArray[np.complexfloating]
    """A numpy array of complex refractive indices across all energies (L) and pre-sliced layers (pre_z)."""
    if (
        isinstance(refractive_indices, np.ndarray)
        and L > 1
        and refractive_indices.ndim == 2
        and refractive_indices.shape[1] == pre_N + 1
        and refractive_indices.shape[0] == L
    ):
        # Valid refractive indices for multiple energies
        if issubclass(refractive_indices.dtype.type, np.complexfloating):
            pre_ref_idxs = refractive_indices.copy()
        else:
            raise ValueError(
                f"Refractive indices must be a numpy array of complex numbers \
                              for multiple energies. Dtype was instead {refractive_indices.dtype}"
            )

    elif (
        isinstance(refractive_indices, np.ndarray)
        and L == 1
        and refractive_indices.ndim == 1
        and refractive_indices.shape[0] == pre_N + 1
    ):
        # Valid refractive indices for a single energy
        if issubclass(refractive_indices.dtype.type, np.complex128):
            pre_ref_idxs = refractive_indices.copy()[np.newaxis, :]
        else:
            raise ValueError(
                f"Refractive indices must be a numpy array of complex numbers \
                                for a single energy. Dtype was instead {refractive_indices.dtype}"
            )

    elif (
        isinstance(refractive_indices, list)
        and all(isinstance(n, (int, float, complex)) for n in refractive_indices)
        and L == 1
        and len(refractive_indices) == pre_N
    ):
        # Valid refractive indices for a single energy
        pre_ref_idxs = np.array(refractive_indices, dtype=np.complex128, copy=True)[
            np.newaxis, :
        ]
        if np.sum(pre_ref_idxs.imag != 0) == 0:
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
        and len(refractive_indices) == pre_N + 1
    ):
        # Valid refractive indices for a single energy using KKCalc
        pre_ref_idxs = np.zeros((L, pre_N + 1), dtype=np.complex128)
        for i, mat_n in enumerate(
            refractive_indices
        ):  # Iterate over the layers, apply the energy.
            if isinstance(mat_n, asp_complex):
                pre_ref_idxs[:, i] = mat_n.eval_refractive_index(
                    energies
                )  # Apply the energy to the KKCalc object
            elif isinstance(mat_n, complex):
                pre_ref_idxs[:, i] = mat_n
            else:  # float
                pre_ref_idxs[:, i] = complex(real=mat_n, imag=0)  # Convert to complex]
    elif (
        isinstance(refractive_indices, list)
        and len(refractive_indices) == pre_N + 1
        and all(
            callable(n) if (n != 1) else True for n in refractive_indices
        )  # Allow for any layer to be air or vacuum with n=1
    ):
        # Valid refractive indices for multiple energies using Callable
        pre_ref_idxs = np.zeros((L, N + 1), dtype=np.complex128)
        single_energy_calc: bool = (
            False  # Prevent callable from being called with multiple energies.
        )
        for i, mat_n in enumerate(
            refractive_indices
        ):  # Iterate over the layers, apply the energy.
            if isinstance(mat_n, (int, float, complex)) and mat_n == 0:
                pre_ref_idxs[:, i] = mat_n + 0j  # Convert to complex
                continue

            assert callable(mat_n)
            if L == 1:
                pre_ref_idxs[0, i] = mat_n(
                    energies
                )  # Apply the energy to the Callable function
            elif single_energy_calc:
                for j in range(L):
                    pre_ref_idxs[j, i] = mat_n(
                        energies[j]
                    )  # Apply the energy to the Callable function
            else:
                if callable(mat_n):
                    try:
                        pre_ref_idxs[:, i] = mat_n(
                            energies
                        )  # Apply the energy to the Callable function
                    except TypeError | ValueError:
                        single_energy_calc = True
                        for j in range(L):
                            pre_ref_idxs[j, i] = mat_n(
                                energies[j]
                            )  # Apply the energy to the Callable function

    else:
        raise ValueError(
            "Refractive index values must be a list of complex numbers, \
                            a numpy array of complex numbers, a list of `KKCalc` `asp_complex` objects \
                            or a list of Callable functions for multiple energies."
        )
    result.pre_refractive_indices = pre_ref_idxs

    # Convert the refractive indices to a sliced set based on the provided slice thickness
    ref_idxs: npt.NDArray[np.complexfloating] = np.zeros(
        (L, N + 1), dtype=np.complex128
    )
    # Apply boundary (or single interface) for first and last slices:
    ref_idxs[:, 0] = (
        (
            (pre_ref_idxs[:, 0] + pre_ref_idxs[:, 1]) / 2
            - (pre_ref_idxs[:, 0] - pre_ref_idxs[:, 1])
            / 2
            * sp.erf(-(z_sliced[0] - z_interfaces[j]) / (np.sqrt(2) * z_roughness[0]))
        )
        if not enforce_boundary
        else pre_ref_idxs[:, 0]
    )
    ref_idxs[:, -1] = (
        (
            (pre_ref_idxs[:, -2] + pre_ref_idxs[:, -1]) / 2
            - (pre_ref_idxs[:, -2] - pre_ref_idxs[:, -1])
            / 2
            * sp.erf(
                -(z_sliced[-1] - z_interfaces[-1]) / (np.sqrt(2) * z_roughness[-1])
            )
        )
        if not enforce_boundary
        else pre_ref_idxs[:, -1]
    )
    result.refractive_indices = ref_idxs

    # Calculate between two interfaces for the remaining slices:
    for i in range(1, N):
        zi = z_sliced[i]
        # Find the index of the two interfaces that are closest to zi
        j = np.digitize(zi, z_interfaces)  # 0 before first interface, 1 after, etc.
        # j should always be j > 0, and j < N

        if j == 0:
            ref_idxs[:, i] = (pre_ref_idxs[:, 0] + pre_ref_idxs[:, 1]) / 2 - (
                pre_ref_idxs[:, 0] - pre_ref_idxs[:, 1]
            ) / 2 * sp.erf(-(zi - z_interfaces[j]) / (np.sqrt(2) * z_roughness[0]))
        elif j == pre_N:
            ref_idxs[:, i] = (pre_ref_idxs[:, -2] + pre_ref_idxs[:, -1]) / 2 - (
                pre_ref_idxs[:, -2] - pre_ref_idxs[:, -1]
            ) / 2 * sp.erf(
                -(z_sliced[-1] - z_interfaces[-1]) / (np.sqrt(2) * z_roughness[-1])
            )
        else:
            # 0 < j < pre_N
            ## NEW CONTINUOUS METHOD CONSIDERING TWO INTERFACES
            sigma_j = z_roughness[j - 1]
            sigma_jp1 = z_roughness[j]
            # Calculate the changing refractive indexes going through the two interfaces
            rj = (pre_ref_idxs[:, j - 1] + pre_ref_idxs[:, j]) / 2 - (
                pre_ref_idxs[:, j - 1] - pre_ref_idxs[:, j]
            ) / 2 * sp.erf(-(zi - z_interfaces[j - 1]) / (np.sqrt(2) * sigma_j))
            rjp1 = (pre_ref_idxs[:, j] + pre_ref_idxs[:, j + 1]) / 2 - (
                pre_ref_idxs[:, j] - pre_ref_idxs[:, j + 1]
            ) / 2 * sp.erf(-(zi - z_interfaces[j]) / (np.sqrt(2) * sigma_jp1))
            # Mix them
            scale = (zi - z_interfaces[j - 1]) / (z_interfaces[j] - z_interfaces[j - 1])
            ref_idxs[:, i] = (
                (1 - scale) * rj  # Proximity to z_interfaces[j]
                + scale * rjp1  # Proximity to z_interfaces[j+1]
            )

        ## OLD DISCONTINUOUS METHOD CONSIDERING ONLY ONE INTERFACE
        # if j > 0 and j < len(z_interfaces):
        #     # Check which interface is closest.
        #     j = (
        #         j - 1
        #         if abs(zi - z_interfaces[j - 1]) < abs(z_interfaces[j] - zi)
        #         else j
        #     )
        # elif j > len(z_interfaces) - 1:
        #     j = len(z_interfaces) - 1

        # # Get the roughness between j and j+1
        # sigma_i = z_roughness[j]
        # # (n1 + n2)/2 - (n1 - n2)/2 * erf(z_interface / (np.sqrt(2) * sigma_i))
        # refractive_indices[i] = (
        #     ref_idxs_preslice[:, j] + ref_idxs_preslice[:, j + 1]
        # ) / 2 - (ref_idxs_preslice[:, j] - ref_idxs_preslice[:, j + 1]) / 2 * sp.erf(
        #     -(zi - z_interfaces[j]) / (np.sqrt(2) * sigma_i)
        # )

    # Labels
    if layer_names is not None:
        assert len(layer_names) == pre_N + 1, (
            "Layer names must match the number of pre-sliced layers (pre_N+1)."
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
    critical_angles: npt.NDArray[np.floating] = np.sqrt(
        2 * (1 - pre_ref_idxs[:, 1:].real)
    )
    """The critical angles of each energy and material interface (excluding vacuum/air) (L, pre_N) in radians."""
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
