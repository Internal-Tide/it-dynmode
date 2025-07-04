from scipy import linalg
import numpy as np

def dynmodes(Nsq, depth, nmodes, boundary='rigid', grav=9.81):
    """
    DYNMODES calculates ocean dynamic vertical modes
    taking a column vector of Brunt-Vaisala values (Nsq) at
    different depths and calculating some number of dynamic modes (nmodes).
    TZW (modified 2025.04.10) core code from matlab version in github, but have some revisions.
    Note: The input depths need not be uniformly spaced,
          and the deepest depth is assumed to be the bottom.
    USAGE: [wmodes, pmodes, ce] = dynmodes(Nsq, depth, nmodes, boundary, grav)

    Inputs:
        Nsq      = column vector of Brunt-Vaisala buoyancy frequency (s^-2)
        depth    = column vector of water depth (m, negative values)
        nmodes   = number of vertical modes to calculate
        boundary = boundary condition at surface: 'rigid' (default) or 'free' #(my test prove that rigid is pure baroclinic modes, free surface is barotropic(mode0)+bc modes)
        grav     = gravitational acceleration (default: 9.81 m/s^2) #(is not important)

    Outputs:
        wmodes = vertical velocity structure
        pmodes = horizontal velocity structure
        ce     = modal speed (m/s)
        z      = vertical levels of wmodes
        zp     = vertical levels of pmodes (midway between z levels)
    """

    p = -depth

    # Check for surface value
    n = len(p)
    if p[0] > 0:
        # Add surface pressure with top Nsq value
        z = np.zeros(n+1)
        z[1:] = -p
        N2 = np.zeros(n + 1)
        N2[0] = Nsq[0]  # Constant N2 at surface
      # Alternative: Linear N2 at surface: N2[0] = 2*Nsq[0] - Nsq[1]
        N2[1:] = Nsq
        nz = n + 1
    else:
        z = -p
        N2 = Nsq
        nz = n

    # Calculate depths and spacing
    dz = z[:-1] - z[1:]
    # Midpoint depth
    zm = z[:-1] - 0.5 * dz
    # Midpoint spacing
    dzm = np.zeros(nz)
    dzm[1:nz-1] = zm[:nz-2] - zm[1:nz-1]
    dzm[0] = dzm[1]
    dzm[nz-1] = dzm[nz-2]

    # Calculate midpoint depths for pmodes
    zh = 0.5 * (z[:-1] + z[1:])

    # Direct eigenvalue solution for vertical modes
    A = np.zeros((nz, nz))
    B = np.zeros((nz, nz))

    # Create matrices
    for i in range(1, nz-1):
        A[i,i] = 1/(dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
        A[i,i-1] = -1/(dz[i-1]*dzm[i])
        A[i,i+1] = -1/(dz[i]*dzm[i])
    for i in range(nz):
        B[i,i] = N2[i]

    # Set boundary conditions
    is_rigid = boundary.lower() == 'rigid'
    if is_rigid:
        A[0,0] = 0
        A[nz-1,nz-1] = 0
        B[0,0] = 1
        B[nz-1,nz-1] = 1
    else:
        # Free surface
        A[0,0] = -1.0/dz[0]
        A[0,1] = 1.0/dz[0]
        B[0,0] = grav + 0.5*dz[0]*N2[0]
        A[nz-1,nz-1] = 0
        B[nz-1,nz-1] = 1

    # Solve eigenvalue problem
    eigvals, wmodes = linalg.eig(A, B)

    # Process eigenvalues and eigenvectors
    ind = np.argsort(np.abs(eigvals))
    eigvals = eigvals[ind]
    wmodes = wmodes[:, ind]

    # Filter out near-zero eigenvalues
    indu = np.where(np.abs(eigvals) > 1e-8)[0] # if the N2 too weak,  so the eigvals too small, the modes do not exits(tzw modified 2025.04.20)
    eigvals = eigvals[indu]
    wmodes = wmodes[:, indu]
    available_modes = eigvals.shape[0]


    if available_modes < nmodes:
        print(f"警告: 只有 {available_modes} 个有效模态可用 (请求 {nmodes})，返回全NaN结果")
        result_wmodes = np.full((nz, nmodes), np.nan)
        result_pmodes = np.full((nz-1, nmodes), np.nan)
        result_ce = np.full(nmodes, np.nan)
        return result_wmodes, result_pmodes, result_ce, z, zh
    # Select requested number of modes
    eigvals = eigvals[:nmodes]
    wmodes = wmodes[:, :nmodes]

    # Eigen speed
    ce = 1.0 / np.sqrt(np.abs(eigvals))

    # Create pressure structure
    nm = nmodes
    pmodes = np.zeros((nz-1, nm))
    depth_abs = abs(z[-1])

    for i in range(nm):
        # Calculate first derivative of vertical modes
        pr = wmodes[:-1, i] - wmodes[1:, i]
        # Most accurate at half nodes
        pr = pr / dz

        # Normalize as in Kelly's 2016 paper
        sgp = np.sign(pr[0])
        pr = pr / (np.sqrt(np.sum(pr**2 * dz))) * np.sqrt(-z[-1]) * sgp
        pmodes[:, i] = pr

        # Normalize wmodes according to Kelly's 2016 paper
        wnorm = (np.sum(0.5 * (wmodes[:-1, i]**2 * N2[:-1] + wmodes[1:, i]**2 * N2[1:]) * dz) +
                grav * wmodes[0, i]**2) * eigvals[i]
        wnorm = np.sqrt(abs(wnorm) / depth_abs)
        wmodes[:, i] = wmodes[:, i] / wnorm * sgp

    return wmodes[:, :nmodes], pmodes[:, :nmodes], ce[:nmodes], z, zh

def preprocess_for_dynmodes(Nsq, depth):

    Nsq_clean = np.array(Nsq, copy=True)
    depth_clean = np.array(depth, copy=True)

    if np.all(np.isnan(Nsq_clean)):
        return Nsq_clean, depth_clean, False

    valid_indices = ~np.isnan(Nsq_clean)
    if not np.all(valid_indices):
        last_valid_idx = len(valid_indices) - 1 - np.argmax(valid_indices[::-1])

        if last_valid_idx < len(Nsq_clean) - 1:
            Nsq_clean = Nsq_clean[:last_valid_idx + 1]
            depth_clean = depth_clean[:last_valid_idx + 1]


    return Nsq_clean, depth_clean, True
