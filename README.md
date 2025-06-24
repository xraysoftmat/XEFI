# XEFI
A package for calculations for X-ray Electric Field Intensities using the Parratt recursive algorithm.

This package calculates discrete models of multi-layer structures, including the ability to slice simplistic models into arbitrary layers.
Supports the use of the `KKCalc` package to calculate the index of refraction within layers.

### The Model

![Model](docs/geometry.png)
Using the geometry shown above, we define the following quantities:
| Variable     | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| $N$          | The number of interfaces between the semi-infinite vacuum and substrate, corresponding to $N+1$ layers |
| $i$          | The layer number, indexed from 0 (i.e. 0 to $N$)                                       |
| $z_i$        | The depth of the $i^{th}$ interface.                                                   | 
| $d_i$        | The thickness of the $i^{th}$ layer ($z_0 = z_N = \infty$)                             |
| $\theta^t_i$ | The transmitted angle of incidence in layer $i$. Same as the angle of reflection $\theta^r_i$ in layer $i$.
| $k_i$        | The z-component of the wavevector in the $i^{th}$ layer.
| $T_i$        | The complex amplitude of the transmitted electric field.

The total electric field can then be calculated as the sum of downward and upward propogating waves:

$$ E^{Total}_i = T_i(0) \exp\left(-i k_i \left(z-z_i\right)\right) + R_i(0) \exp\left(i k_i \left(z-z_i\right)\right) $$