"""
A module for basic result handling in XEFI.
"""

import numpy as np, numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from typing import Literal, TypeVar
from abc import ABCMeta
# import scipy.constants as sc 

T = TypeVar("T", bound=np.float64)
"""Type variable for floating intensity results."""


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
    k0 : float | None
        The incident vacuum wavevector.
    fresnel_r : npt.NDArray[np.complexfloating] | None
        The Fresnel reflection coefficients for each interface and angle (N, M).
    fresnel_t : npt.NDArray[np.complexfloating] | None
        The Fresnel transmission coefficients for each interface and angle (N, M).
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
        """The complex z-component wavevector in each layer (L, M, N+1)."""
        self.refractive_indices: npt.NDArray[np.complexfloating] | None
        """The complex refractive indices of each layer (N+1)."""
        self.critical_angles: npt.NDArray[np.floating] | None
        """The critical angles for total internal reflection for each beam energy and interface (L, N) in radians."""
        self.k0: npt.NDArray[np.floating] | float | None
        """The incident vacuum wavevector for each energy (L)."""
        self.fresnel_r: npt.NDArray[np.complexfloating] | None
        """The Fresnel reflection coefficients for each energy, angle and interface (L, M, M)."""
        self.fresnel_t: npt.NDArray[np.complexfloating] | None
        """The Fresnel transmission coefficients for each energy, angle and interface (L, M, M)."""
        self.T: npt.NDArray[np.complexfloating] | None
        """The complex transmission amplitude for each energy, angle and interface (L, M, M)."""
        self.R: npt.NDArray[np.complexfloating] | None
        """The complex reflection amplitude, for each energy, angle and interface (L, M, M)."""
        self.X: npt.NDArray[np.complexfloating] | None
        """The complex ratio of downward and upward propagating fields for each energy, angle and interface (L, M, M)."""
        self.fig: Figure | SubFigure | None
        """The matplotlib figure for plotted results."""
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
        return

    def __call__(self, z_vals: npt.ArrayLike) -> npt.NDArray[np.floating]:
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
        return self.electric_field_intensity(z_vals)

    def electric_field(self, z_vals: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        """
        Calculate the total electric field at given z-coordinates.

        The electric field result has dimensions (L, M), where L is the number
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
        layer_idxs = np.digitize(z_vals, self.z)
        # Initialize the electric field array
        
        E_total = np.zeros((L, M, len(z_vals)), dtype=np.complex128)

        # Top of layer definitions
        z0 = np.r_[self.z[0], self.z]  # Include 0 for the top of the first layer (air)

        if L != 1 and M != 1:
            for i in range(N + 1):
                # For each layer
                subset = layer_idxs == i  # Get the indices for this layer
                z_subset = z_vals[subset]  # Get the z values for this layer

                # Calculate the distance into the layer from the top of the layer.
                d = (
                    z_subset - z0[i]
                )  # for semi-infinite i=0, we flip, for i=N, we use the last z value.

                transmission = (
                    # Amplitude
                    self.T[:, :, i, np.newaxis] # newaxis for the specific z values
                    # Downward propogating phase
                    * np.exp(-1j * self.wavevectors[:, :, i, np.newaxis] * d[np.newaxis, np.newaxis, :])
                )
                reflection = (
                    (
                        # Amplitude
                        self.R[:, :, i, np.newaxis]
                        # Upward propogating phase
                        * np.exp(
                            1j * self.wavevectors[:, :, i, np.newaxis] * d[np.newaxis, np.newaxis, :]
                        )
                    )
                    if i < N
                    else 0.0
                )
                E_total[:, :, subset] = transmission + reflection
        elif L == 1 and M == 1:
            for i in range(N + 1):
                # For each layer
                subset = layer_idxs == i  # Get the indices for this layer
                z_subset = z_vals[subset]  # Get the z values for this layer

                # Calculate the distance into the layer from the top of the layer.
                d = (
                    z_subset - z0[i]
                )  # for semi-infinite i=0, we flip, for i=N, we use the last z value.

                transmission = (
                    # Amplitude
                    self.T[i] # newaxis for the specific z values
                    # Downward propogating phase
                    * np.exp(-1j * self.wavevectors[i] * d)
                )
                reflection = (
                    (
                        # Amplitude
                        self.R[i]
                        # Upward propogating phase
                        * np.exp(
                            1j * self.wavevectors[i] * (d)
                        )
                    )
                    if i < N
                    else 0.0
                )
                E_total[:, :, subset] = transmission + reflection
        else:
            assert L == 1 or M == 1
            for i in range(N + 1):
                # For each layer
                subset = layer_idxs == i  # Get the indices for this layer
                z_subset = z_vals[subset]  # Get the z values for this layer

                # Calculate the distance into the layer from the top of the layer.
                d = (
                    z_subset - z0[i]
                )  # for semi-infinite i=0, we flip, for i=N, we use the last z value.

                transmission = (
                    # Amplitude
                    self.T[:, i, np.newaxis] # newaxis for the specific z values
                    # Downward propogating phase
                    * np.exp(-1j * self.wavevectors[:, i, np.newaxis] * d[np.newaxis, :])
                )
                reflection = (
                    (
                        # Amplitude
                        self.R[:, i, np.newaxis]
                        # Upward propogating phase
                        * np.exp(
                            1j * self.wavevectors[:, i, np.newaxis] * (d[np.newaxis, :])
                        )
                    )
                    if i < N
                    else 0.0
                )
                E_total[:, :, subset] = transmission + reflection

        return np.squeeze(E_total) # Remove excess dimensions.

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

    def generate_pretty_figure_XEFI(
        self,
        z_vals: npt.NDArray[np.floating] | list[float | int] | None = None,
        fig: Figure | SubFigure | None = None,
        ax: Axes | None = None,
        cbar_loc: Literal["fig", "ax"] = "fig",
        cmap: Colormap = plt.cm.get_cmap("viridis"),
        norm: Literal["linear", "log"] = "linear",
        m: int | None = None,
        l: int | None = None,
        grid_z: bool = True,
        grid_labels : bool = True,
        grid_crit: bool = True,
        labels: list[str] | None = None,
    ) -> tuple[Figure | SubFigure, Axes]:
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
        norm : Literal["linear", "log"], optional
            The normalization to use for the colormap. If "linear", uses a linear normalization;
            if "log", uses a logarithmic normalization. Defaults to "linear".
        m : int | None, optional
            A singular index to consider for the angles of incidence. Defaults to None.
        l : int | None, optional
            A singular index to consider for the beam energies. Defaults to None.
        grid_z : bool
            Whether to plot the layer grid. Defaults to True.
        grid_crit : float
            Whether to plot the critical angles grid. Defaults to True.
        grid_z_labels : bool
            Whether to plot the z layer labels. Defaults to True.
        labels : list[str] | None
            The labels for the z layers. If None, defaults to automatic labels.

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
        L, M, N = self.L, self.M, self.N
        x_selection: Literal["theta", "energy"]
        x_data: npt.NDArray[np.floating]
        theta = self.theta
        energy = self.energies
        critical_angles = self.critical_angles
        if critical_angles is not None:        
            critical_angles = np.rad2deg(critical_angles)
        if L is not None and L > 1 and M is not None and M > 1:
            assert isinstance(theta, np.ndarray)
            assert isinstance(energy, np.ndarray)
            if m is not None and l is not None:
                raise ValueError(
                    "Values for both `l` and `m` are incompatible, as data is required to be 2D."
                )
            elif m is not None:
                theta = theta[m]
                x_data = energy
                x_selection = "energy"
                M = 1  # Reduce M to 1 for plotting
            elif l is not None:
                energy = energy[l]
                x_data = theta
                x_selection = "theta"
                L = 1  # Reduce L to 1 for plotting
                if critical_angles:
                    critical_angles = critical_angles[l]
            else:
                raise ValueError(
                    "This method is designed for 2D plotting. Ensure that either L or M are singular, \
                    or choose a singular index using `l` or `m` function parameters."
                )
        elif L is not None and L > 1:
            assert isinstance(energy, np.ndarray)
            x_selection = "energy"
            x_data = energy
        elif M is not None and M > 1:
            assert isinstance(theta, np.ndarray)
            x_selection = "theta"
            x_data = theta
        else:
            raise ValueError(
                "This method is designed for 2D plotting. Ensure that either L or M are singular, \
                or choose a singular index using `l` or `m` function parameters."
            )
            
        if x_selection == "theta":
            # Convert to degrees
            x_data = np.rad2deg(x_data)

        plot_z: npt.NDArray[np.float64]
        z = self.z
        assert (
            z is not None and N is not None and len(z) == N
        ), "Interface z-coordinates (self.z) must be set before generating the figure."
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
            norm_fn = LogNorm(
                vmin=intensity[intensity > 0].min(), vmax=intensity.max()
            )
        else:
            raise ValueError("`norm` must be either 'linear' or 'log'.")

        # Display the colormesh
        pixmap = ax.pcolormesh(
            x_data,
            plot_z,
            intensity.T,
            cmap=cmap,
            norm=norm_fn,
            shading="nearest",
            rasterized=True,
        )

        # Finishing up
        ax.set_xlabel("Angle of Incidence (degrees)" if x_selection == "theta" else "Energy (eV)")
        ax.set_ylabel("$z$ (Å)")
        ax.set_title("X-ray Electric Field Intensity")

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_fn)
        if cbar_loc == "fig":
            cbar = fig.colorbar(
                sm, ax=fig.axes, label="Electric Field Intensity (A.U.)"
            )
        elif cbar_loc == "ax":
            cbar = fig.colorbar(sm, cax=ax, label="Electric Field Intensity (A.U.)")
        else:
            raise ValueError("`cbar_loc` must be either 'fig' or 'ax'.")
        
        ### Add gridding and labels to the layers
        if labels is None:
            result_labels = self.layer_names
            if result_labels is not None:
                labels = result_labels
            else:
                labels = [f"Layer {i}" for i in range(N+1)]
        # Layers:
        if grid_z:
            for zi in z:
                ax.axhline(zi, color='white', linestyle='--', alpha=0.2, linewidth=0.5)
            if grid_labels and labels is not None:
                for i, zi in enumerate(z):
                    name = labels[i+1]
                    ax.text(x=1.0, y=zi, s=name, transform=ax.get_yaxis_transform(), ha='right', va='top', color='white')
                ax.text(x=1.0, y=z[0], s=labels[0], transform=ax.get_yaxis_transform(), ha='right', va='bottom', color='white')
        # Critical Angles
        if grid_crit and x_selection == "theta" and critical_angles is not None: # Is a angle plot.
            for ang in critical_angles:
                ax.axvline(x=ang, color='white', linestyle='--', alpha=0.2, linewidth=0.5)    
            if grid_labels and labels is not None:
                for i, ang in enumerate(critical_angles):    
                    name = labels[i+1]
                    ax.text(x=ang, y=0.00, s=name, 
                            transform=ax.get_xaxis_transform(), 
                            color="white", ha='center')
                    
        return fig, ax

    def generate_pretty_figure_XEFI_intensity(
        self,
        z_vals: npt.NDArray[np.floating] | list[float | int] | None = None,
        fig: Figure | SubFigure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure | SubFigure, Axes]:
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
        fig: Figure | SubFigure | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure | SubFigure, Axes]:
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
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
        elif fig is None and ax is not None:
            fig = ax.figure
        else:
            # fig is not None and ax is None
            assert fig is not None
            ax = fig.add_subplot(1, 1, 1)
