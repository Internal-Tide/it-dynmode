# Ocean Vertical Mode Solver for Internal Waves

This project provides a robust and efficient Python tool, `dynmodes.py`, for calculating the vertical structure of oceanic internal waves. It solves the Sturm-Liouville eigenvalue problem for a given buoyancy frequency profile (Brunt-Väisälä frequency, $N^2$) to decompose the complex wave field into a series of orthogonal vertical modes.

This code is adapted from the original MATLAB version by John Klinck (1999) and has been significantly refactored for performance using NumPy's vectorized operations. It also incorporates modern normalization schemes as described in Kelly (2016).

![Internal Wave Modes](https://www.ocean-sci.net/12/1105/2016/os-12-1105-2016-f01-web.png)
*<p align="center">Figure: Conceptual illustration of the first three baroclinic modes of internal waves in a stratified fluid. Source: Kelly, S. M. (2016).</p>*

---

## 1. Physical and Mathematical Background

Internal waves are gravity waves that oscillate within a density-stratified fluid, rather than on its surface. In the ocean, the continuous stratification allows for the existence of a theoretically infinite set of internal wave modes. Any complex vertical structure of velocity or displacement can be represented as a linear superposition of these modes.

The vertical structure of these waves is governed by the Taylor-Goldstein equation, which, under certain assumptions (e.g., no background flow, linear, Boussinesq approximation), simplifies to a Sturm-Liouville problem. The core of this problem is to solve for the vertical structure of vertical velocity, $\hat{w}(z)$, which is described by the following differential equation:

$$
\frac{d^2 \hat{w}}{dz^2} + \frac{N^2(z) - \omega^2}{\omega^2 - f^2} k_h^2 \hat{w} = 0
$$

where:
- $z$ is the vertical coordinate (positive upwards).
- $N^2(z)$ is the Brunt-Väisälä frequency squared, representing the stratification.
- $\omega$ is the wave frequency.
- $f$ is the Coriolis frequency.
- $k_h$ is the horizontal wavenumber.

This project solves a simplified form of this equation to find the eigenvalues ($c_n$, the modal speeds) and eigenfunctions ($\phi_n(z)$, the vertical structures) for pressure and velocity. The governing equation solved numerically is:

$$
\frac{d^2 W}{dz^2} + \frac{N^2(z)}{c_n^2} W = 0
$$

where $W(z)$ is the vertical structure function for vertical velocity, and $c_n$ is the eigenvalue corresponding to the modal speed of the $n$-th mode.

### Boundary Conditions

The solver supports two types of surface boundary conditions:

1.  **Rigid-Lid (`boundary='rigid'`)**: Assumes the sea surface is a fixed, flat lid. This implies that the vertical velocity at the surface is zero ($W(0) = 0$). This is a good approximation for baroclinic (internal) modes, as their surface expression is very small.
2.  **Free-Surface (`boundary='free'`)**: Allows the sea surface to move freely. This condition is necessary to resolve the barotropic mode (mode 0), which represents a depth-uniform flow.

At the seabed ($z=-H$), the vertical velocity is always assumed to be zero ($W(-H) = 0$).

---

## 2. Features

- **High Performance**: Utilizes NumPy's vectorized operations to replace slow `for` loops, ensuring fast computation even for high-resolution vertical profiles.
- **Flexible Grid**: Accepts non-uniformly spaced depth levels.
- **Multiple Boundary Conditions**: Supports both `rigid-lid` and `free-surface` conditions.
- **Robust Normalization**: Implements the energy-based normalization scheme from Kelly (2016), ensuring physically consistent amplitude scaling.
- **Error Handling**: Gracefully handles cases where the number of requested modes exceeds the number of physically valid modes that can be resolved from the stratification profile.
- **Helper Function**: Includes a `preprocess_for_dynmodes` function to automatically clean input data containing `NaN` values.

---

## 3. Installation and Dependencies

The script requires standard scientific Python libraries. You can install them using `pip`:

```bash
pip install numpy scipy
```

Then, simply download or clone the `dynmodes.py` file into your project directory.

---

## 4. Quick Start Tutorial

Here is a simple example of how to use `dynmodes` to compute the vertical modes for a canonical exponential stratification profile.

```python
import numpy as np
import matplotlib.pyplot as plt
from dynmodes import dynmodes, preprocess_for_dynmodes

# 1. Generate a sample stratification profile (N^2)
# Exponentially decaying N^2 profile
depth_h = 1000  # Total depth
n_levels = 200
depth_vec = -np.linspace(0, depth_h, n_levels) # Depth vector (negative values)

N0 = 5.2e-3  # Surface buoyancy frequency [rad/s]
b = 1300     # e-folding depth scale [m]
Nsq_vec = N0**2 * np.exp(2 * depth_vec / b)

# (Optional) Add some noise or NaNs to test preprocessing
# Nsq_vec[-10:] = np.nan
# Nsq_clean, depth_clean, is_valid = preprocess_for_dynmodes(Nsq_vec, depth_vec)

# 2. Call the dynmodes solver
n_modes_to_calc = 4
wmodes, pmodes, ce, z_grid, zh_grid = dynmodes(
    Nsq_vec,
    depth_vec,
    nmodes=n_modes_to_calc,
    boundary='rigid' # Use 'rigid' for pure baroclinic modes
)

# 3. Print the modal speeds
print(f"Calculated modal speeds (c_n) for the first {n_modes_to_calc} modes:")
for i, speed in enumerate(ce):
    print(f"  Mode {i+1}: {speed:.2f} m/s")

# 4. Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

# Plot W modes (vertical velocity)
for i in range(n_modes_to_calc):
    ax1.plot(wmodes[:, i], z_grid, label=f'Mode {i+1}')
ax1.set_title('Vertical Velocity Modes (W)')
ax1.set_xlabel('Amplitude (normalized)')
ax1.set_ylabel('Depth (m)')
ax1.legend()
ax1.grid(True, linestyle=':')

# Plot P modes (horizontal velocity / pressure)
for i in range(n_modes_to_calc):
    ax2.plot(pmodes[:, i], zh_grid, label=f'Mode {i+1}')
ax2.set_title('Horizontal Velocity Modes (P)')
ax2.set_xlabel('Amplitude (normalized)')
ax2.legend()
ax2.grid(True, linestyle=':')

plt.suptitle('Calculated Vertical Modes')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
```

---

## 5. API Reference

### `dynmodes(Nsq, depth, nmodes, boundary='rigid', grav=9.81)`

**Description:**
Calculates the ocean's vertical dynamic modes based on a given stratification profile.

**Parameters:**
- `Nsq` (array-like): 1D array of Brunt-Väisälä buoyancy frequency squared ($s^{-2}$).
- `depth` (array-like): 1D array of corresponding water depths (m). Must be negative and monotonically increasing.
- `nmodes` (int): The number of vertical modes to calculate and return.
- `boundary` (str, optional): The surface boundary condition. Can be `'rigid'` (default) for pure baroclinic modes or `'free'` to include the barotropic mode.
- `grav` (float, optional): Gravitational acceleration ($m/s^2$). Default is `9.81`.

**Returns:**
- `wmodes` (ndarray): A 2D array (`len(z)`, `nmodes`) containing the vertical velocity mode structures.
- `pmodes` (ndarray): A 2D array (`len(z)-1`, `nmodes`) containing the horizontal velocity/pressure mode structures, defined at mid-depth levels.
- `ce` (ndarray): A 1D array (`nmodes`,) of the modal speeds (eigenvalues) in m/s.
- `z` (ndarray): The vertical grid (m) on which `wmodes` are defined.
- `zh` (ndarray): The vertical grid (m) on which `pmodes` are defined.

---

## 6. References

- **Klinck, J. M. (1999).** The original MATLAB implementation which inspired this project.
- **Kelly, S. M. (2016).** The vertical mode decomposition of internal waves. *Ocean Science*, 12(5), 1105-1122. [doi:10.5194/os-12-1105-2016](https://doi.org/10.5194/os-12-1105-2016). This paper provides an excellent overview and details the normalization scheme used.
