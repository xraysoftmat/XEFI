"""
A module for basic result handling in XEFI.
"""

from typing import Literal, TypeVar, override, Iterable
from abc import ABCMeta
from enum import Enum
import warnings

import numpy as np
import numpy.typing as npt

# TODO: Make matplotlib and plotly optional dependencies.
# try:
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as mplFig, SubFigure as mplSubFig
from matplotlib.axes import Axes as mplAxes
from matplotlib.colors import Colormap, LogNorm, Normalize
#     has_mpl = True
# except ImportError:
#     has_mpl = False

# try:
#     import plotly.express as px  # noqa: F401
#     from plotly.graph_objects import Figure as pxFig  # noqa: F401
#     has_plotly = True
# except ImportError:
# has_plotly = False

# import scipy.constants as sc

T = TypeVar("T", bound=np.float64)
"""Type variable for floating intensity results."""


class XEF_method(Enum):
    """
    An enumerate for the XEF calculation method.
    """

    TOLAN = "tolan"  # https://doi.org/10.1007/bfb0112834
    DEV = "dev"  # https://doi.org/10.1103/PhysRevB.61.8462
    OHTA = "ohta"  # https://doi.org/10.1364/AO.29.001952


class BaseResult(metaclass=ABCMeta):
    """
    An abstract base class for handling results in XEFI.

    Can be called to calculate the total electric field for a given (set of) z-coordinate(s),
    if a result has been calculated.

    Multi-dimensional results can be calculated along `theta` and `beam_energy` dimensions.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to initialize attributes of the class.

    Attributes
    ----------
    z : npt.NDArray[np.floating] | None
        The z-coordinate of the (N) interfaces in Angstroms (Å).
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

    def __init__(self, **kwargs) -> None:
        self.z: npt.NDArray[np.floating] | None
        """The z-coordinate of the (N) interfaces in Angstroms (Å)."""
        self.L: int | None
        """The number of beam energies considered."""
        self.M: int | None
        """The number of angles of incidence."""
        self.N: int | None
        """The number of interfaces, corresponding to N+1 layers."""
        self.energies: npt.NDArray[np.floating] | float | None
        """The energy(s) of the X-ray beam in eV."""
        self.theta: npt.NDArray[np.floating] | float | None
        """The angle(s) of incidence (M) in the first layer (i=0) in radians."""
        self.angles_of_incidence: npt.NDArray[np.complexfloating] | None
        """The complex angles of incidence in each layer in radians (L, M, N+1)."""
        self.wavevectors: npt.NDArray[np.complexfloating] | None
        """The unsigned complex z-component wavevector in each layer (L, M, N+1)."""
        self.refractive_indices: npt.NDArray[np.complexfloating] | None
        """The complex refractive indices of each layer (N+1)."""
        self.critical_angles: npt.NDArray[np.floating] | None
        """The critical angles for total internal reflection for each beam energy and interface (L, N) in radians."""
        self.k0: npt.NDArray[np.floating] | float | None
        """The incident vacuum wavevector for each energy (L)."""
        self.fresnel_r: npt.NDArray[np.complexfloating] | None
        """The Fresnel reflection coefficients for each energy, angle and interface (L, M, N)."""
        self.fresnel_t: npt.NDArray[np.complexfloating] | None
        """The Fresnel transmission coefficients for each energy, angle and interface (L, M, N)."""
        self.T: npt.NDArray[np.complexfloating] | None
        """The complex transmission amplitude for each energy, angle and interface (L, M, N)."""
        self.R: npt.NDArray[np.complexfloating] | None
        """The complex reflection amplitude, for each energy, angle and interface (L, M, N)."""
        self.X: npt.NDArray[np.complexfloating] | None
        """The complex ratio of downward and upward propagating fields for each energy, angle and interface (L, M, N)."""
        self.method: XEF_method | None
        """The XEF calculation method used."""
        self.layer_names: list[str] | None
        """The names of the layers (N+1), if provided."""

        # Initialize all attributes to None
        self.reset()

        # Initialize attributes with provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"BaseResult has no attribute '{key}'")

    @property
    def wavelength(self) -> npt.NDArray[np.floating] | float | None:
        """
        The wavelength of the X-ray in Angstroms (Å), from the beam energy (eV).

        Calculated from the beam energy (eV) using the formula:
        .. math::
            \\lambda &= \\frac{hc}{q E} * 10^{10}
                     &= \\frac{12398.42}{E}

        Returns
        -------
        float | None
            The wavelength of the X-ray in Angstroms (Å), or None if beam_energy is not set.
        """
        # return sc.c * sc.h / (sc.e * self.beam_energy) if self.beam_energy is not None else None
        return 12398.42 / self.energies if self.energies is not None else None

    @property
    def theta_deg(self) -> npt.NDArray[np.floating] | float | None:
        """
        The angle(s) of incidence (M) in the first layer (i=0) in degrees.

        Returns
        -------
        float | None
            The angle(s) of incidence in degrees, or None if theta is not set.
        """
        return np.rad2deg(self.theta) if self.theta is not None else None

    def reset(self) -> None:
        """
        Clear/initialise the result object attributes to None.
        """
        self.z = None
        self.L = None
        self.M = None
        self.N = None
        self.theta = None
        self.angles_of_incidence = None
        self.refractive_indices = None
        self.wavevectors = None
        self.k0 = None
        self.fresnel_r = None
        self.fresnel_t = None
        self.layer_names = None
        self.R = None
        self.T = None
        self.X = None

    def squeeze(self) -> None:
        """
        Reduce the dimensions of the result if L or M is 1.

        All results are calculated over energy, angle and layer indexes.
        Sometimes energies (L) or angles (M) may be singular, and can be squeezed.
        """
        if self.X is not None:
            self.X = np.squeeze(self.X)[()]
        if self.R is not None:
            self.R = np.squeeze(self.R)[()]
        if self.T is not None:
            self.T = np.squeeze(self.T)[()]
        if self.wavevectors is not None:
            self.wavevectors = np.squeeze(self.wavevectors)[()]
        if self.angles_of_incidence is not None:
            self.angles_of_incidence = np.squeeze(self.angles_of_incidence)[()]
        if self.fresnel_r is not None:
            self.fresnel_r = np.squeeze(self.fresnel_r)[()]
        if self.fresnel_t is not None:
            self.fresnel_t = np.squeeze(self.fresnel_t)[()]
        if self.refractive_indices is not None:
            self.refractive_indices = np.squeeze(self.refractive_indices)[()]
        if self.critical_angles is not None:
            self.critical_angles = np.squeeze(self.critical_angles)[()]

    def electric_field(self, z_vals: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        """
        Calculate the total electric field at given z-coordinates.

        The electric field result has dimensions (L, M, N), where L is the number
        of z-coordinates and M (`self.M`) is the number of angles of incidence in `self.theta`.

        Parameters
        ----------
        z_vals : npt.ArrayLike
            The z-coordinates at which to calculate the electric field in angstroms (Å).

        Returns
        -------
        npt.NDArray[np.complexfloating]
            The total electric field at the specified z-coordinates and angle theta. Dimensions are (L, M).
        """
        # Get the indexes
        L, M, N = self.L, self.M, self.N

        # Check if required attributes are set
        if self.z is None:
            raise ValueError(
                "The z-coordinates of the interfaces (self.z) must be set before calculating the electric field."
            )
        # if self.M is None:
        #     raise ValueError(
        #         "The number of angles of incidence (self.M) must be set before calculating the electric field."
        #     )
        if N is None:
            raise ValueError(
                "The number of interfaces (self.N) must be set before calculating the electric field."
            )
        if self.T is None or self.R is None:
            raise ValueError(
                "The transmission (self.T) and reflection (self.R) coefficients must be set before calculating the electric field."
            )
        if self.wavevectors is None:
            raise ValueError(
                "The wavevectors (self.wavevectors) must be set before calculating the electric field."
            )

        if M is None:
            M = 1
        if L is None:
            L = 1

        # Ensure z_vals is a numpy array
        z_vals = np.asarray(z_vals, dtype=np.float64)

        # Find the indices of z_vals in self.z
        z0 = self.z
        layer_idxs = np.digitize(z_vals, z0)
        # Initialize the electric field array

        E_total = np.zeros((L, M, len(z_vals)), dtype=np.complex128)

        # Top of layer definitions
        z0 = np.r_[z0[0], z0]  # Include the top of the first layer for air/vacuum

        if L != 1 and M != 1:
            raise NotImplementedError("Nope")
        elif L == 1 and M == 1:
            raise NotImplementedError("Nope")
        else:
            assert L == 1 or M == 1

            T = self.T
            R = self.R

            # if self.method == XEF_method.dev:
            #     # Invert complex sign
            #     T.imag *= -1
            #     R.imag *= -1

            # # For each layer
            for i in range(N + 1):
                # For each layer
                subset = layer_idxs == i  # Get the indices for this layer
                z_subset = z_vals[subset]  # Get the z values for this layer

                # Calculate the distance into the layer from the top of the layer.
                d = (
                    z_subset - z0[i]
                )  # for semi-infinite i=0, we flip, for i=N, we use the last z value.

                wvs = self.wavevectors[:, i]
                assert wvs.shape == (self.M,) or wvs.shape == (self.L,), (
                    f"Is: {wvs.shape}"
                )

                # wvs.imag[wvs.imag < 0] *= -1 # Invert complex sign of negatively calculated wavevectors.
                phase = (
                    -1j * wvs[:, np.newaxis] * d[np.newaxis, :]
                )  # indx: angles|energies, z values
                transmission = T[
                    :, i, np.newaxis
                ] * np.exp(  # indx: angles|energies, z values
                    phase
                )
                reflection = (
                    (R[:, i, np.newaxis] * np.exp(-phase))
                    if i < N
                    else np.zeros((len(wvs), len(d)))
                )
                if L == 1:
                    E_total[0][:, subset] = transmission + reflection
                elif M == 1:
                    E_total[:, 0, subset] = transmission + reflection

            return np.squeeze(E_total)  # Remove excess dimensions.

    def electric_field_intensity(
        self, z_vals: npt.ArrayLike
    ) -> npt.NDArray[np.floating]:
        """
        Calculate the total electric field intensity at given z-coordinates.

        .. math::
            I(z) = |E(z)|^2 = E(z) \cdot E^*(z)

        The electric field intensity result has dimensions (L, M), where L is the number
        of z-coordinates and M (`self.M`) is the number of angles of incidence in `self.theta`.

        Parameters
        ----------
        z_vals : npt.ArrayLike
            The z-coordinates at which to calculate the electric field in angstroms (Å).

        Returns
        -------
        npt.NDArray[np.complexfloating]
            The total electric field at the specified z-coordinates and angle theta. Dimensions are (L, M).
        """
        field = self.electric_field(z_vals)
        intensity = (field * np.conj(field)).real
        return intensity

    __call__ = (
        electric_field_intensity  # is this sufficient for documentation? probably not.
    )

    # def __call__(self, z_vals: npt.ArrayLike) -> npt.NDArray[np.floating]:
    #     """
    #     Calculate the total electric field intensity at given z-coordinates.

    #     .. math::
    #         I(z) = |E(z)|^2 = E(z) \cdot E^*(z)

    #     The electric field intensity result has dimensions (L, M), where L is the number
    #     of z-coordinates and M (`self.M`) is the number of angles of incidence in `self.theta`.

    #     Parameters
    #     ----------
    #     z_vals : npt.ArrayLike
    #         The z-coordinates at which to calculate the electric field in angstroms (Å).

    #     Returns
    #     -------
    #     npt.NDArray[np.complexfloating]
    #         The total electric field at the specified z-coordinates and angle theta. Dimensions are (L, M).
    #     """
    #     return self.electric_field_intensity(z_vals)

    def _require_singular_x_data(
        self,
        l_index: int | None = None,
        m_index: int | None = None,
    ) -> tuple[Literal["theta", "energy"] | None, np.ndarray | None, np.ndarray | None]:
        """
        Ensure that the data for the x-axis (either `theta` or `energy`) is singular (1D).

        This is a helper function for 2D plotting functions, which require either `theta` or `energy`
        to be singular (1D). If both dimensions are greater than 1, the user must specify either `m_index`
        or `l_index` to select a singular index.

        Parameters
        ----------
        l_index : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        m_index : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.

        Returns
        -------
        x_datatype : Literal["theta", "energy"] | None
            The type of data selected for the x-axis, either "theta" or "energy".
            None if neither is singular.
        x_data : np.ndarray | None
            The data for the x-axis, either `theta` or `energy`.
            None if neither is singular.
        critical_angles : np.ndarray | None
            The critical angles for total internal reflection for each beam energy and
            interface (L, N) in radians. Subset if l_index is not None. None if not defined.
        """
        L, M = self.L, self.M
        theta, energies = self.theta, self.energies
        critical_angles = self.critical_angles

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

    def generate_pretty_figure_XEFI(
        self,
        z_vals: npt.NDArray[np.floating] | list[float | int] | None = None,
        fig: mplFig | mplSubFig | None = None,
        ax: mplAxes | None = None,
        cbar_loc: Literal["fig", "ax"] = "fig",
        cmap: Colormap = plt.get_cmap("viridis"),
        norm: Literal["linear", "log"] | Normalize = "linear",
        l_index: int | None = None,
        m_index: int | None = None,
        labels: list[str] | None = None,
        angles_in_deg: bool = True,
        grid_z: bool = True,
        grid_labels: bool = True,
        grid_crit: bool = True,
    ) -> tuple[mplFig | mplSubFig, mplAxes]:
        """
        Generate a pretty 2D plot of the X-ray electric field intensity as a function of depth.

        The second dimension is either `theta` (angle of incidence) or `beam_energy`.
        This dimension is automatically chosen if one of the dimensions is singular (i.e., has length 1).
        Otherwise, the user must specify either `m` or `l` to choose a singular index.

        Parameters
        ----------
        z_vals : npt.NDArray[np.floating] | list[float | int] | None, optional
            The z-coordinates at which to calculate the electric field intensity, in angstroms (Å).
            If None, uses the z-coordinates from the result object, with 10% padding.
        fig : Figure | SubFigure | None, optional
            The matplotlib figure to use for the plot. If None, a new figure is created.
        ax : Axes | None, optional
            The matplotlib axes to use for the plot. If None, a new axes is created.
        cbar_loc : Literal["fig", "ax"], optional
            The location of the colorbar. If "fig", the colorbar is padded to the right of the figure;
            if "ax", it is padded to the right of the axes. Defaults to "fig".
        cmap : Colormap, optional
            The colormap to use for the plot. Defaults to matplotlib's "viridis".
        norm : Literal["linear", "log"] | matplotlib.colors.Normalize, optional
            The normalization to use for the colormap. If "linear", uses a linear normalization;
            if "log", uses a logarithmic normalization. Defaults to "linear".
            Can also provide a custom normalization object.
        l_index : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        m_index : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.
        labels : list[str] | None, optional
            The labels for the z layers. If None, defaults to automatic labels.
        angles_in_deg : bool, optional
            Whether the angles are in degrees (True) or radians (False). Defaults to True.
        grid_z : bool, optional
            Whether to plot the layer grid. Defaults to True.
        grid_labels : bool, optional
            Whether to plot the z layer labels. Defaults to True.
        grid_crit : float, optional
            Whether to plot the critical angles grid. Defaults to True.

        Returns
        -------
        tuple[Figure | SubFigure, Axes]
            A tuple containing the matplotlib figure and axes used for the plot.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
        elif fig is None and ax is not None:
            fig = ax.figure
        else:
            # fig is not None and ax is None
            assert fig is not None
            ax = fig.add_subplot(1, 1, 1)

        # Ensure L or M is singular for 2D plotting.
        N = self.N
        x_selection: Literal["theta", "energy"]
        x_data: npt.NDArray[np.floating]

        # Get the sliced x-data.
        x_selection, x_data, critical_angles = self._require_singular_x_data(
            m_index=m_index,
            l_index=l_index,
        )

        plot_z: npt.NDArray[np.float64]
        z = self.z
        assert z is not None and N is not None and len(z) == N, (
            "Interface z-coordinates (self.z) must be set before generating the figure."
        )
        if z_vals is None:
            # Create a linspace of the z-coordinates with 10% padding
            width = abs(z[-1] - z[0])
            plot_z = np.linspace(
                z[0] - 0.1 * width,
                z[-1] + 0.1 * width,
                1000,
                dtype=np.float64,
            )
        else:
            plot_z = np.asarray(z_vals, dtype=np.float64)

        # Calculate the electric field intensity at the specified z-coordinates
        intensity: npt.NDArray[np.floating] = self(plot_z)

        # Generate a normalisation
        norm_fn: Normalize | LogNorm
        if norm == "linear":
            norm_fn = Normalize(vmin=intensity.min(), vmax=intensity.max())
        elif norm == "log":
            norm_fn = LogNorm(vmin=intensity[intensity > 0].min(), vmax=intensity.max())
        elif isinstance(norm, Normalize) or issubclass(norm, Normalize):
            norm_fn = norm
        else:
            raise ValueError("`norm` must be either 'linear' or 'log'.")

        # Display the colormesh
        ax.pcolormesh(
            x_data,
            plot_z,
            intensity.T,
            cmap=cmap,
            norm=norm_fn,
            shading="nearest",
            rasterized=True,
        )

        # Finishing up
        ax.set_xlabel(
            f"Angle of Incidence ({'degrees' if angles_in_deg else 'radians'})"
            if x_selection == "theta"
            else "Energy (eV)"
        )
        ax.set_ylabel("$z$ (Å)")
        ax.set_title("X-ray Electric Field Intensity")

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_fn)
        if cbar_loc == "fig":
            fig.colorbar(sm, ax=fig.axes, label="Electric Field Intensity (A.U.)")
        elif cbar_loc == "ax":
            fig.colorbar(sm, cax=ax, label="Electric Field Intensity (A.U.)")
        else:
            raise ValueError("`cbar_loc` must be either 'fig' or 'ax'.")

        ### Add gridding and labels to the layers
        if labels is None:
            result_labels = self.layer_names
            if result_labels is not None:
                labels = result_labels
            else:
                labels = [f"Layer {i}" for i in range(N + 1)]
        # Layers:
        if grid_z:
            for zi in z:
                ax.axhline(zi, color="white", linestyle="--", alpha=0.2, linewidth=0.5)
            if grid_labels and labels is not None:
                for i, zi in enumerate(z):
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
                    y=z[0],
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
        return fig, ax

    def generate_pretty_figure_XEFI_intensity(
        self,
        z_vals: npt.NDArray[np.floating] | list[float | int] | None = None,
        fig: mplFig | mplSubFig | None = None,
        ax: mplAxes | None = None,
    ) -> tuple[mplFig | mplSubFig, mplAxes]:
        """
        Generate a pretty plot of the summed XEFI intensity within each layer.

        Parameters
        ----------
        z_vals : npt.NDArray[np.floating] | list[float | int] | None, optional
            The z-coordinates at which to calculate the electric field intensity.
            If None, uses the z-coordinates from the result object, with 10% padding.
        fig : Figure | SubFigure | None, optional
            The matplotlib figure to use for the plot. If None, a new figure is created.
        ax : Axes | None, optional
            The matplotlib axes to use for the plot. If None, a new axes is created.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
        elif fig is None and ax is not None:
            fig = ax.figure
        else:
            # fig is not None and ax is None
            assert fig is not None
            ax = fig.add_subplot(1, 1, 1)

    def generate_pretty_figure(
        self,
        z_vals: npt.NDArray[np.floating] | list[float | int] | None = None,
        fig: mplFig | mplSubFig | None = None,
        ax: mplAxes | None = None,
    ) -> tuple[mplFig | mplSubFig, mplAxes]:
        """
        Generate a pretty plot of the XEFI intensity, and the summed XEFI intensity within each layer.

        Combines the functionality of `generate_pretty_figure_XEFI` and `generate_pretty_figure_XEFI_intensity`.

        Parameters
        ----------
        z_vals : npt.NDArray[np.floating] | list[float | int] | None, optional
            The z-coordinates at which to calculate the electric field intensity.
            If None, uses the z-coordinates from the result object, with 10% padding.
        fig : Figure | SubFigure | None, optional
            The matplotlib figure to use for the plot. If None, a new figure is created.
        ax : Axes | None, optional
            The matplotlib axes to use for the plot. If None, a new axes is created.

        Returns
        -------
        fig : Figure | SubFigure
            The matplotlib figure containing the plot.
        ax : Axes
            The matplotlib axes containing the plot.
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
        elif fig is None and ax is not None:
            fig = ax.figure
        else:
            # fig is not None and ax is None
            assert fig is not None
            ax = fig.add_subplot(1, 1, 1)

        return fig, ax

    def graph_wavevectors(
        self,
        ax_re: mplAxes | None = None,
        ax_im: mplAxes | None = None,
    ) -> tuple[mplAxes, mplAxes] | None:
        """
        Plot the wavevectors for each layer as a function of angle/energy.

        Parameters
        ----------
        ax_re : mplAxes | None, optional
            The matplotlib axes to use for the real part of the wavevector plot.
            If `ax_im` is also None, new axes are created.
        ax_im : mplAxes | None, optional
            The matplotlib axes to use for the imaginary part of the wavevector plot.
            If `ax_re` is also None, new axes are created.

        Returns
        -------
        ax_re : matplotlib.Axes
            The axes handle for the real component wavevector plot.
        ax_im : matplotlib.Axes
            The axes handle for the imaginary component wavevector plot.
        """
        L, M, N = self.L, self.M, self.N
        assert L is not None and M is not None and N is not None
        # Create a graph
        if ax_re is None and ax_im is None:
            fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True)
            ax_re.set_ylabel("Wavevector Re")
            ax_im.set_ylabel("Wavevector Im")
            ax_im.set_xlabel("Angle (degrees)")

        for i in range(L):
            for j in range(N + 1):
                if ax_re is not None:
                    ax_re.plot(
                        self.theta_deg,
                        self.wavevectors[i, :, j].real,
                        label=f"Layer {j}",
                    )
                if ax_im is not None:
                    ax_im.plot(
                        self.theta_deg,
                        self.wavevectors[i, :, j].imag,
                        label=f"Layer {j}",
                    )
        return ax_re, ax_im

    def graph_fresnel(
        self,
        ax_re: mplAxes | None = None,
        ax_im: mplAxes | None = None,
        interfaces: int
        | list[int]
        | range
        | npt.NDArray[np.integer]
        | tuple[int, ...]
        | None = None,
        angles_in_deg: bool = True,
        plot_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[mplAxes, mplAxes] | None:
        """
        Plot the Fresnel coefficients for each interface as a function of angle/energy.

        Parameters
        ----------
        ax_re : mplAxes | None, optional
            The matplotlib axes to use for the real part of the Fresnel coefficients plot.
            If `ax_im` is also None, new axes are created.
        ax_im : mplAxes | None, optional
            The matplotlib axes to use for the imaginary part of the Fresnel coefficients plot.
            If `ax_re` is also None, new axes are created.
        interfaces : int | list[int] | range | npt.NDArray[np.integer] | None, optional
            The index or indices of the interfaces to plot. If None, all interfaces are plotted.
        angles_in_deg : bool, optional
            Whether to plot the angles in degrees (True) or radians (False). By default True.
        plot_kwargs : dict | None, optional
            Additional keyword arguments to pass to the plot functions for the main axes.
            By default None.
        **kwargs
            Equivalent to `plot_kwargs` for compatibility. Used to update `plot_kwargs` if both
            are provided.

        Returns
        -------
        ax_re : matplotlib.Axes
            The axes handle for the real component Fresnel coefficients plot.
        ax_im : matplotlib.Axes
            The axes handle for the imaginary component Fresnel coefficients plot.
        """  # numpydoc ignore=PR06
        L, M, N = self.L, self.M, self.N
        assert L is not None and M is not None and N is not None
        if ax_re is None and ax_im is None:
            fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True)
            ax_im.set_xlabel(f"Angle ({'degrees' if angles_in_deg else 'radians'})")
            ax_re.set_ylabel("Fresnel Coefficients (Re)")
            ax_im.set_ylabel("Fresnel Coefficients (Im)")

        indices: list[int] | range | npt.NDArray[np.integer] | tuple[int, ...]
        if interfaces is None:
            indices = range(N - 1)
        elif isinstance(interfaces, int):
            indices = [interfaces]
        elif isinstance(interfaces, (range, list, np.ndarray, tuple)):
            indices = interfaces
        else:
            raise ValueError("`interfaces` must be an int, list of int, or range.")

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.update(kwargs)
        theta = self.theta if not angles_in_deg else self.theta_deg
        for l_index in range(L):
            for i in indices:
                ax_re.plot(
                    theta,
                    self.fresnel_r[l_index, :, i].real,
                    label=f"Fr R_{{{i},{i + 1}}} (Re)",
                    **plot_kwargs,
                )
                ax_im.plot(
                    theta,
                    self.fresnel_r[l_index, :, i].imag,
                    label=f"Fr R_{{{i},{i + 1}}} (Im)",
                    **plot_kwargs,
                )
                ax_re.plot(
                    theta,
                    self.fresnel_t[l_index, :, i].real,
                    label=f"Fr T_{{{i},{i + 1}}} (Re)",
                    **plot_kwargs,
                )
                ax_im.plot(
                    theta,
                    self.fresnel_t[l_index, :, i].imag,
                    label=f"Fr T_{{{i},{i + 1}}} (Im)",
                    **plot_kwargs,
                )
        ax_re.legend()
        ax_im.legend()

        return ax_re, ax_im

    def graph_fresnel_magnitude(
        self,
        ax: mplAxes | None = None,
        interfaces: int
        | list[int]
        | range
        | npt.NDArray[np.integer]
        | tuple[int, ...]
        | None = None,
        transmission: bool = True,
        reflection: bool = True,
        angles_in_deg: bool = True,
        plot_kwargs: dict | None = None,
        **kwargs,
    ) -> mplAxes | None:
        """
        Plot the Fresnel coefficients for each interface as a function of angle/energy.

        Parameters
        ----------
        ax : mplAxes | None, optional
            The matplotlib axes to use for the Fresnel coefficients plot.
            If None, a new figure/axis is created.
        interfaces : int | list[int] | range | npt.NDArray[np.integer] | None, optional
            The index or indices of the interfaces to plot. If None, all interfaces are plotted.
        transmission : bool, optional
            Whether to plot the transmission coefficients. By default True.
        reflection : bool, optional
            Whether to plot the reflection coefficients. By default True.
        angles_in_deg : bool, optional
            Whether to plot the angles in degrees (True) or radians (False). By default True.
        plot_kwargs : dict | None, optional
            Additional keyword arguments to pass to the plot functions for the main axes.
            By default None.
        **kwargs
            Equivalent to `plot_kwargs` for compatibility. Used to update `plot_kwargs` if both
            are provided.

        Returns
        -------
        matplotlib.Axes
            The axes handle for the Fresnel coefficients plot.
        """  # numpydoc ignore=PR06
        L, M, N = self.L, self.M, self.N
        assert L is not None and M is not None and N is not None
        if ax is None:
            fig, ax = plt.subplots(1, 1, sharex=True)
            ax.set_xlabel(f"Angle ({'degrees' if angles_in_deg else 'radians'})")
            if transmission and reflection:
                ax.set_ylabel("Fresnel Coefficients $|r_{i}|,|t_{i}|$")
            elif transmission:
                ax.set_ylabel("Fresnel Coefficients $|t_{i}|$")
            elif reflection:
                ax.set_ylabel("Fresnel Coefficients $|r_{i}|$")

        indices: list[int] | range | npt.NDArray[np.integer] | tuple[int, ...]
        if interfaces is None:
            indices = range(N - 1)
        elif isinstance(interfaces, int):
            indices = [interfaces]
        elif isinstance(interfaces, (range, list, np.ndarray, tuple)):
            indices = interfaces
        else:
            raise ValueError("`interfaces` must be an int, list of int, or range.")

        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.update(kwargs)
        theta = self.theta if not angles_in_deg else self.theta_deg
        if L == 1:
            for i in indices:
                if reflection:
                    ax.plot(
                        theta,
                        np.abs(self.fresnel_r[:, i]),
                        label=f"$|r_{{{i},{i + 1}}}|$",
                        **plot_kwargs,
                    )
                if transmission:
                    ax.plot(
                        theta,
                        np.abs(self.fresnel_t[:, i]),
                        label=f"$|t_{{{i},{i + 1}}}|$",
                        **plot_kwargs,
                    )
        elif L > 1:
            for l_index in range(L):
                for i in indices:
                    if reflection:
                        ax.plot(
                            theta,
                            np.abs(self.fresnel_r[l_index, :, i]),
                            label=f"$|r_{{{i},{i + 1}}}|$",
                            **plot_kwargs,
                        )
                    if transmission:
                        ax.plot(
                            theta,
                            np.abs(self.fresnel_t[l_index, :, i]),
                            label=f"$|t_{{{i},{i + 1}}}|$",
                            **plot_kwargs,
                        )
        else:
            raise ValueError("L must be >= 1")
        return ax

    def graph_field_coefficients(
        self,
        ax_R: mplAxes | None = None,
        ax_T: mplAxes | None = None,
        ax_X: mplAxes | None = None,
        layers: int | list[int] | range | None = None,
        inset: bool = True,
        inset_loc: Literal[
            "upper left",
            "upper right",
            "center left",
            "center right",
            "upper center",
            "lower center",
            "center",
            "lower left",
            "lower right",
        ] = "upper right",
        scale: Literal["linear", "log"] = "log",
        add_labels: bool = True,
        angles_in_deg: bool = True,
        plot_kwargs: dict | None = None,
        inset_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[mplAxes | None, mplAxes | None, mplAxes | None]:
        """
        Plot the electric field amplitudes solved within each layer via the interfaces.

        Insets can be added to show the the same data on a logarithmic scale.

        Parameters
        ----------
        ax_R : mplAxes | None
            The axes to plot the reflectance on.
        ax_T : mplAxes | None
            The axes to plot the transmittance on.
        ax_X : mplAxes | None
            The axes to plot the ratio of reflected to transmitted fields.
        layers : int | list[int] | None
            The index or indices of the layers to plot.
        inset : bool, optional
            Whether to add insets to the plots. By default True.
        inset_loc : Literal["upper left", "upper right", "center left", "center right",
                           "upper center", "lower center", "center",
                           "lower left", "lower right"], optional
            The location of the inset axes. By default "upper right".
        scale : Literal["linear", "log"], optional
            The scale of the main axes. By default "log".
        add_labels : bool, optional
            Whether to add labels to provided axes / titles. By default True.
        angles_in_deg : bool, optional
            Whether to plot the angles in degrees (True) or radians (False). By default True.
        plot_kwargs : dict | None, optional
            Additional keyword arguments to pass to the plot
            functions for the main axes. By default None.
        inset_kwargs : dict | None, optional
            Additional keyword arguments to pass to the inset
            axes. By default copies (and updates) plot_kwargs.
        **kwargs
            Equivalent to `plot_kwargs` for compatibility. Used to update
            `plot_kwargs` if both are provided.

        Returns
        -------
        tuple[mplAxes | None, mplAxes | None, mplAxes | None]
            The axes used for the reflectance, transmittance, and ratio of reflected to transmitted fields.
        """
        X, R, T = self.X, self.R, self.T
        L, M, N = self.L, self.M, self.N
        critical_angles = self.critical_angles
        if critical_angles is not None and angles_in_deg:
            critical_angles = np.rad2deg(critical_angles)
        angles = self.theta_deg if angles_in_deg else self.theta
        fresnel_r = self.fresnel_r

        assert N is not None and N >= 2, "N (interfaces) must be >= 2"
        assert angles is not None
        assert fresnel_r is not None

        if layers is None:
            layers = range(N + 1)
        elif isinstance(layers, int):
            layers = [layers]

        layer_names = self.layer_names
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in layers]

        if ax_R is None and ax_T is None and ax_X is None:
            if X is not None:
                ax: Iterable[mplAxes]
                fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
                (ax_R, ax_T, ax_X) = ax
                ax_X.set_ylabel("Reflected / Transmitted Field Ratio")
            else:
                fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
                (ax_R, ax_T) = ax
            ax[-1].set_xlabel(f"Angle ({'degrees' if angles_in_deg else 'radians'})")
            ax_R.set_ylabel("Reflectance")
            ax_T.set_ylabel("Transmittance")
            fig.suptitle(f"XEFI Calculation using {self.method.value} method")
        else:
            ax = []
            if ax_R is not None:
                ax.append(ax_R)
            if ax_T is not None:
                ax.append(ax_T)
            if ax_X is not None:
                ax.append(ax_X)
            if add_labels:
                if ax_R is not None:
                    ax_R.set_ylabel("Reflectance")
                if ax_T is not None:
                    ax_T.set_ylabel("Transmittance")
                if ax_X is not None:
                    ax_X.set_ylabel("Reflected / Transmitted Field Ratio")
                ax[-1].set_xlabel(
                    f"Angle ({'degrees' if angles_in_deg else 'radians'})"
                )

        loc: tuple[float, float, float, float]
        match inset_loc:
            case "upper right":
                loc = (0.5, 0.5, 0.45, 0.4)
            case "upper left":
                loc = (0.05, 0.5, 0.45, 0.4)
            case "lower left":
                loc = (0.05, 0.1, 0.45, 0.4)
            case "lower right":
                loc = (0.5, 0.1, 0.45, 0.4)
            case "center left":
                loc = (0.05, 0.3, 0.45, 0.4)
            case "center right":
                loc = (0.5, 0.3, 0.45, 0.4)
            case "upper center":
                loc = (0.275, 0.5, 0.45, 0.4)
            case "lower center":
                loc = (0.275, 0.1, 0.45, 0.4)
            case "center":
                loc = (0.275, 0.3, 0.45, 0.4)
            case _:
                warnings.warn("Unknown inset location, defaulting to upper right.")
                loc = (0.5, 0.5, 0.45, 0.4)

        if inset:
            # Check if existing inset:
            axin1, axin2 = None, None
            if ax_R:
                for child in ax_R.get_children():
                    if isinstance(child, mplAxes):
                        axin1 = child
                        break
                if axin1 is None:
                    axin1 = (
                        ax_R.inset_axes(loc) if ax_R else None
                    )  # [x, y, width, height]
            if ax_T:
                for child in ax_T.get_children():
                    if isinstance(child, mplAxes):
                        axin2 = child
                        break
                if axin2 is None:
                    axin2 = (
                        ax_T.inset_axes(loc) if ax_T else None
                    )  # [x, y, width, height]
            if X is not None and ax_X:
                axin3 = None
                for child in ax_X.get_children():
                    if isinstance(child, mplAxes):
                        axin3 = child
                        break
                if axin3 is None:
                    axin3 = (
                        ax_X.inset_axes(loc) if ax_X else None
                    )  # [x, y, width, height]

        # Set default kwargs
        default_kwargs = {
            "alpha": 0.7,
        }
        if plot_kwargs is None:
            plot_kwargs = default_kwargs.copy()
        else:
            default_kwargs.update(plot_kwargs)
            plot_kwargs = default_kwargs
        plot_kwargs.update(kwargs)
        if inset_kwargs is None:
            inset_kwargs = plot_kwargs.copy()
        else:
            base = plot_kwargs.copy()
            base.update(inset_kwargs)
            inset_kwargs = base

        # Plot the data
        if L == 1:
            assert M and M > 1
            for i in layers:
                if ax_R:
                    ax_R.plot(
                        angles,
                        np.abs(R[:, i]) ** 2,
                        label=f"$|R_{i}|^2$ {layer_names[i]}",
                        **plot_kwargs,
                    )
                if ax_T:
                    ax_T.plot(
                        angles,
                        np.abs(T[:, i]) ** 2,
                        label=f"$|T_{i}|^2$ {layer_names[i]}",
                        **plot_kwargs,
                    )
                if X is not None and ax_X:
                    ax_X.plot(
                        angles,
                        np.abs(X[:, i]) ** 2,
                        label=f"$|X_{i}|^2$ {layer_names[i]}",
                        **plot_kwargs,
                    )
                if inset:
                    if ax_R:
                        axin1.plot(
                            angles,
                            np.abs(R[:, i]) ** 2,
                            label=f"$|R_{i}|^2$ {layer_names[i]}",
                            **inset_kwargs,
                        )
                    if ax_T:
                        axin2.plot(
                            angles,
                            np.abs(T[:, i]) ** 2,
                            label=f"$|T_{i}|^2$ {layer_names[i]}",
                            **inset_kwargs,
                        )
                    if X is not None and ax_X:
                        axin3.plot(
                            angles,
                            np.abs(X[:, i]) ** 2,
                            label=f"$|X_{i}|^2$ {layer_names[i]}",
                            **inset_kwargs,
                        )
        if inset:
            axins = []
            if ax_R:
                axins += [axin1]
            if ax_T:
                axins += [axin2]
            if X is not None and ax_X:
                axins += [axin3]

        if critical_angles is not None and critical_angles.ndim == 1:
            # Use critical angles within desired range
            critical_angles = critical_angles[
                (critical_angles > angles.min()) & (critical_angles < angles.max())
            ]
            if inset:
                for axin in axins:
                    # Set the inset to examine the critical angles in detail
                    axin.set_xlim(
                        np.min(critical_angles[:]) - 0.02,
                        np.max(critical_angles[:]) + 0.02,
                    )
            for l_index in range(len(ax)):
                if L == 1:
                    for i in layers:
                        j = i - 1  # critical angle index is one before
                        if j >= 0:
                            # Add vertical line
                            ax[l_index].axvline(
                                x=critical_angles[j],
                                color="k",
                                linestyle="--",
                                linewidth=0.5,
                                alpha=0.2,
                            )
                            ax[l_index].text(
                                x=critical_angles[j],
                                y=1.05,
                                s=f"{layer_names[i]}",
                                rotation=0,
                                color="k",
                                alpha=0.5,
                                ha="center",
                                va="top",
                                transform=ax[l_index].get_xaxis_transform(),
                            )
                            ax[l_index].set_ylim(1e-4, 5.0)
                            if inset:
                                for axin in axins:
                                    axin.axvline(
                                        x=critical_angles[j],
                                        color="k",
                                        linestyle="--",
                                        linewidth=0.5,
                                        alpha=0.2,
                                    )

                if inset and np.max(axins[l_index].get_ylim()) > 10:
                    axins[l_index].set_ylim(np.min(axins[l_index].get_ylim()), 1e1)
                ax[l_index].set_yscale(scale)

        return ax_R, ax_T, ax_X


class BaseRoughResult(BaseResult, metaclass=ABCMeta):
    """
    An abstract base class for handling roughness results in XEFI.

    Inherits from `BaseResult` and extends it with attributes and methods specific to roughness calculations.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to initialize attributes of the class.

    Attributes
    ----------
    sigma : npt.NDArray[np.floating] | None
        The roughness values for each interface (N) in angstroms (Å).
    roughness_profile : npt.NDArray[np.floating] | None
        The roughness profile as a function of z-coordinate.
    roughness_z : npt.NDArray[np.floating] | None
        The z-coordinates corresponding to the roughness profile in angstroms (Å).
    """

    def __init__(self, **kwargs) -> None:
        # Parent constructor.
        super().__init__(**kwargs)
        # Declare the properties
        self.z_roughness: npt.NDArray[np.floating] | None = None
        """The roughness of the (N) interfaces in Angstroms (Å)."""

    @override
    def reset(self) -> None:
        """
        Clear/initialise the result object attributes to None.
        """
        # Reset the parent properties
        super().reset()
        # Reset new properties
        self.roughness_z = None
        return

    @override
    def generate_pretty_figure_XEFI(
        self,
        z_vals: npt.NDArray[np.floating] | list[float | int] | None = None,
        fig: mplFig | mplSubFig | None = None,
        ax: mplAxes | None = None,
        cbar_loc: Literal["fig", "ax"] = "fig",
        cmap: Colormap = plt.get_cmap("viridis"),
        norm: Literal["linear", "log"] | Normalize = "linear",
        l_index: int | None = None,
        m_index: int | None = None,
        labels: list[str] | None = None,
        angles_in_deg: bool = True,
        grid_z: bool = True,
        grid_labels: bool = True,
        grid_crit: bool = True,
        grid_roughness: bool = True,
    ) -> tuple[mplFig | mplSubFig, mplAxes]:
        """
        Generate a pretty 2D plot of the X-ray electric field intensity as a function of depth.

        The second dimension is either `theta` (angle of incidence) or `beam_energy`.
        This dimension is automatically chosen if one of the dimensions is singular (i.e., has length 1).
        Otherwise, the user must specify either `m` or `l` to choose a singular index.

        Parameters
        ----------
        z_vals : npt.NDArray[np.floating] | list[float | int] | None, optional
            The z-coordinates at which to calculate the electric field intensity, in angstroms (Å).
            If None, uses the z-coordinates from the result object, with 10% padding.
        fig : Figure | SubFigure | None, optional
            The matplotlib figure to use for the plot. If None, a new figure is created.
        ax : Axes | None, optional
            The matplotlib axes to use for the plot. If None, a new axes is created.
        cbar_loc : Literal["fig", "ax"], optional
            The location of the colorbar. If "fig", the colorbar is padded to the right of the figure;
            if "ax", it is padded to the right of the axes. Defaults to "fig".
        cmap : Colormap, optional
            The colormap to use for the plot. Defaults to matplotlib's "viridis".
        norm : Literal["linear", "log"] | matplotlib.colors.Normalize, optional
            The normalization to use for the colormap. If "linear", uses a linear normalization;
            if "log", uses a logarithmic normalization. Defaults to "linear".
            Can also provide a custom normalization object.
        l_index : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        m_index : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.
        labels : list[str] | None, optional
            The labels for the z layers. If None, defaults to automatic labels.
        angles_in_deg : bool, optional
            Whether the angles are in degrees (True) or radians (False). Defaults to True.
        grid_z : bool, optional
            Whether to plot the layer grid. Defaults to True.
        grid_labels : bool, optional
            Whether to plot the z layer labels. Defaults to True.
        grid_crit : float, optional
            Whether to plot the critical angles grid. Defaults to True.
        grid_roughness : bool, optional
            Whether to plot the roughness profile. Defaults to True.

        Returns
        -------
        tuple[Figure | SubFigure, Axes]
            A tuple containing the matplotlib figure and axes used for the plot.
        """
        fig, ax = super().generate_pretty_figure_XEFI(
            z_vals=z_vals,
            fig=fig,
            ax=ax,
            cbar_loc=cbar_loc,
            cmap=cmap,
            norm=norm,
            m_index=m_index,
            l_index=l_index,
            labels=labels,
            grid_z=grid_z,
            grid_labels=grid_labels,
            grid_crit=grid_crit,
            angles_in_deg=angles_in_deg,
        )

        # Add lines to display the roughness
        if grid_roughness:
            z = self.z
            zr = self.z_roughness

            x_selection, x_data, critical_angles = self._require_singular_x_data(
                l_index=l_index,
                m_index=m_index,
            )

            if x_selection is None:
                raise ValueError("Requires a singular")
            for i, zi in enumerate(z):
                ax.fill_between(
                    x=x_data,
                    y1=zi - zr[i] / 2,
                    y2=zi + zr[i] / 2,
                    alpha=0.1,
                    hatch="/",
                    color="white",
                )
        return fig, ax
