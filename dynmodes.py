from scipy import linalg
import numpy as np

def dynmodes(Nsq, depth, nmodes, boundary='rigid', grav=9.81):
    """
    DYNMODES calculates ocean dynamic vertical modes (Refactored for performance).
    
    This version is optimized to replace slow for-loops with fast NumPy vectorization
    for matrix assembly and mode normalization. The core logic remains the same.

    USAGE: [wmodes, pmodes, ce, z, zh] = dynmodes(Nsq, depth, nmodes, boundary, grav)

    Inputs:
        Nsq      = column vector of Brunt-Vaisala buoyancy frequency (s^-2)
        depth    = column vector of water depth (m, negative values, monotonic)
        nmodes   = number of vertical modes to calculate
        boundary = boundary condition at surface: 'rigid' (default) or 'free'
        grav     = gravitational acceleration (default: 9.81 m/s^2)

    Outputs:
        wmodes = vertical velocity structure (n_depths, n_modes)
        pmodes = horizontal velocity/pressure structure (n_depths-1, n_modes)
        ce     = modal speed (m/s)
        z      = vertical levels for wmodes (m)
        zh     = vertical mid-point levels for pmodes (m)
    """
    # Ensure inputs are numpy arrays
    Nsq = np.asarray(Nsq)
    depth = np.asarray(depth)

    # --- 1. Setup Grid and N2 Profile ---
    # Use depth directly, z is positive upwards from the bottom.
    # This part is slightly modified for clarity.
    if depth[0] != 0:
        # Add surface point if not present
        z_levels = np.concatenate(([0], depth))
        N2 = np.concatenate(([Nsq[0]], Nsq))
    else:
        z_levels = depth
        N2 = Nsq
    
    nz = len(z_levels)
    z = -z_levels # z is positive upwards, with z=0 at the surface.

    # Calculate grid spacing
    dz = z[:-1] - z[1:]  # Thickness of each layer
    
    # Midpoint depth and spacing (for matrix A)
    zm = z[:-1] - 0.5 * dz
    dzm = np.zeros(nz)
    dzm[1:-1] = zm[:-1] - zm[1:]
    dzm[0] = dzm[1]
    dzm[-1] = dzm[-2]

    # --- 2. Build Matrices A and B (Vectorized) ---
    A = np.zeros((nz, nz))
    
    # --- Optimization: Vectorized construction of matrix A ---
    # Calculate diagonal and off-diagonal elements without a loop
    i = np.arange(1, nz - 1)
    diag_A = 1 / (dz[i-1] * dzm[i]) + 1 / (dz[i] * dzm[i])
    upper_A = -1 / (dz[i] * dzm[i+1])
    lower_A = -1 / (dz[i-1] * dzm[i-1])
    
    A[i, i] = diag_A
    A[i, i+1] = upper_A
    A[i, i-1] = lower_A
    # Note: The above direct indexing is more modern and faster than np.diag.
    
    B = np.diag(N2)

    # --- 3. Set Boundary Conditions ---
    if boundary.lower() == 'rigid':
        # w=0 at surface and bottom
        A[0, :] = 0; A[:, 0] = 0; A[0, 0] = 1
        B[0, 0] = 0 # This leads to A*w = 0, forcing w[0]=0
        A[-1, :] = 0; A[:, -1] = 0; A[-1, -1] = 1
        B[-1, -1] = 0 # Forcing w[-1]=0
    else: # Free surface
        A[0, 0] = -1.0 / dz[0]
        A[0, 1] = 1.0 / dz[0]
        B[0, 0] = grav
        # Bottom boundary (w=0)
        A[-1, :] = 0; A[:, -1] = 0; A[-1, -1] = 1
        B[-1, -1] = 0

    # --- 4. Solve the Eigenvalue Problem ---
    # We solve A*w = (1/c^2)*B*w. Here, eigvals = 1/c^2
    eigvals, wmodes = linalg.eig(A, B)

    # Process eigenvalues and eigenvectors
    # Remove non-physical solutions (NaN or Inf)
    valid_eigs = np.where(np.isfinite(eigvals))[0]
    eigvals = eigvals[valid_eigs]
    wmodes = wmodes[:, valid_eigs]

    # Sort by magnitude of eigenvalue
    ind = np.argsort(np.abs(eigvals))
    eigvals = eigvals[ind]
    wmodes = wmodes[:, ind]

    # Filter out near-zero eigenvalues (weak stratification)
    indu = np.where(np.abs(eigvals) > 1e-10)[0]
    eigvals = eigvals[indu]
    wmodes = wmodes[:, indu]
    
    available_modes = eigvals.shape[0]
    if available_modes < nmodes:
        print(f"Warning: Only {available_modes} valid modes found (requested {nmodes}). Returning NaN for missing modes.")
        w_result = np.full((nz, nmodes), np.nan)
        p_result = np.full((nz - 1, nmodes), np.nan)
        ce_result = np.full(nmodes, np.nan)
        w_result[:, :available_modes] = wmodes[:, :nmodes]
        # pmodes and ce can be calculated for available modes if needed, here just return NaNs
        return w_result, p_result, ce_result, z, -0.5 * (z[:-1] + z[1:])

    # Select requested number of modes
    eigvals = eigvals[:nmodes]
    wmodes = wmodes[:, :nmodes]

    ce = 1.0 / np.sqrt(np.abs(eigvals))

    # --- 5. Normalize Modes (Vectorized) ---
    dz_col = dz[:, np.newaxis] # Reshape for broadcasting

    # Calculate pmodes (proportional to d(wmodes)/dz)
    pmodes = (wmodes[:-1, :] - wmodes[1:, :]) / dz_col

    # Normalize pmodes to have integral of p^2*dz = H
    p_norm_factor = np.sqrt(np.sum(pmodes**2 * dz_col, axis=0))
    total_depth = np.abs(z[-1])
    pmodes = pmodes / p_norm_factor * np.sqrt(total_depth)

    # Ensure pmodes are positive at the surface
    sgp = np.sign(pmodes[0, :])
    pmodes *= sgp
    wmodes *= sgp # Apply same sign correction to wmodes for consistency

    # Normalize wmodes based on energy
    # This part can be complex and depends on the specific normalization scheme.
    # The Kelly (2016) normalization is kept here in a vectorized form.
    N2_col = N2[:, np.newaxis]
    integral_term = np.sum(0.5 * (wmodes[:-1, :]**2 * N2_col[:-1] + wmodes[1:, :]**2 * N2_col[1:]) * dz_col, axis=0)
    surface_term = grav * wmodes[0, :]**2
    wnorm = np.sqrt(np.abs((integral_term + surface_term) * eigvals) / total_depth)
    
    wmodes /= wnorm

    zh = -0.5 * (z_levels[:-1] + z_levels[1:])

    return wmodes, pmodes, ce, z, zh


def preprocess_for_dynmodes(Nsq, depth):
    """
    Helper function to clean input data by removing trailing NaNs.
    """
    Nsq_clean = np.array(Nsq, copy=True)
    depth_clean = np.array(depth, copy=True)

    if np.all(np.isnan(Nsq_clean)):
        return Nsq_clean, depth_clean, False

    valid_indices = ~np.isnan(Nsq_clean)
    if not np.all(valid_indices):
        # Find the last valid index
        last_valid_idx = len(valid_indices) - 1 - np.argmax(valid_indices[::-1])
        
        # Trim the arrays to the last valid data point
        if last_valid_idx < len(Nsq_clean) - 1:
            Nsq_clean = Nsq_clean[:last_valid_idx + 1]
            depth_clean = depth_clean[:last_valid_idx + 1]

    return Nsq_clean, depth_clean, True