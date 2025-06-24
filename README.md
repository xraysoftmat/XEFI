# XEFI
A package for calculations of X-ray Electric Field Intensities (XEFI) using the Parratt recursive algorithm.

This package calculates discrete models of multi-layer structures, including the ability to slice simplistic models into arbitrary layers. 
Supports the use of the `KKCalc` package to calculate the index of refraction within layers.

### The Model

![Model](docs/geometry.png)
Here, layers $i=0$ and $i=N+1$ are semi-infinite layers, typically modelling air/vacuum and a substrate respectively. Boundary conditions allow us to set the incident amplitude $T_0 = 1$, and the reflected amplitude $R_{N+1}=0$. We define the following quantities:
| Variable     | Description
| -            | -
| $N$          | The number of interfaces between the top and bottom layers, corresponding to $N+1$ layers
| $i$          | The layer number, indexed from 0 (i.e. 0 to $N$)
| $z_i$        | The depth of the $i^{th}$ interface ($z_i < 0$).
| $d_i$        | The thickness of the $i^{th}$ layer ($d_0 = d_N = \infty$)
| $\theta^t_i$ | The transmitted angle of incidence in layer $i$. Same as the angle of reflection $\theta^r_i$ in layer $i$.
| $k_i$        | The z-component of the wavevector in the $i^{th}$ layer.
| $T_i$        | The complex amplitude of the downward propogating electric field in layer i.
| $R_i$        | The complex amplitude of the upward propogating electric field in layer i.
| $X_i$        | The ratio of the downward and upward propogating electric field intensities.

The total electric field can then be calculated as the sum of downward and upward propogating waves:

$$ E^{Total}_i = T_i(0) \exp\left(-i k_i \left(z-z_i\right)\right) + R_i(0) \exp\left(i k_i \left(z-z_i\right)\right) $$