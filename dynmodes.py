from scipy import linalg
import numpy as np
from scipy.integrate import cumulative_trapezoid

def dynmodes(Nsq, depth, nmodes, boundary='rigid', grav=9.81, method='direct'):
    """
    DYNMODES calculates ocean dynamic vertical modes
    taking a column vector of Brunt-Vaisala values (Nsq) at
    different depths and calculating some number of dynamic modes (nmodes).

    Note: The input depths need not be uniformly spaced, 
          and the deepest depth is assumed to be the bottom.

    USAGE: [wmodes, pmodes, ce] = dynmodes(Nsq, depth, nmodes, boundary, grav, method)

    Inputs:  
        Nsq      = column vector of Brunt-Vaisala buoyancy frequency (s^-2)
        depth    = column vector of water depth (m, negative values)
        nmodes   = number of vertical modes to calculate 
        boundary = boundary condition at surface: 'rigid' (default) or 'free'
        grav     = gravitational acceleration (default: 9.81 m/s^2)
        method   = solution method: 'direct' (default) or 'wkb' (WKB approximation)
               
    Outputs: 
        wmodes = vertical velocity structure
        pmodes = horizontal velocity structure
        ce     = modal speed (m/s)
        z      = vertical levels of wmodes
        zp     = vertical levels of pmodes (midway between z levels)
    """
    p = -depth
    rho0 = 1028

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
    
    # Choose solution method
    if method.lower() == 'wkb':
        return _wkb_solution(N2, z, zh, dz, nz, nmodes, boundary, grav)
    else:  # Default to direct method
        return _direct_solution(N2, z, zh, dz, dzm, nz, nmodes, boundary, grav)

def _direct_solution(N2, z, zh, dz, dzm, nz, nmodes, boundary, grav):
    """
    Direct eigenvalue solution for vertical modes
    """
    # Get dynamic modes
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
        # Rigid lid
        A[0,0] = 0
        A[nz-1,nz-1] = 0
        B[0,0] = 1
        B[nz-1,nz-1] = 1
    else:
        # Free surface (dynamical boundary conditions)
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
    indu = np.where(np.abs(eigvals) > 1e-8)[0]
    eigvals = eigvals[indu]
    wmodes = wmodes[:, indu]
    
    # Select requested number of modes
    eigvals = eigvals[:nmodes]
    wmodes = wmodes[:, :nmodes]
    
    # Phase speed
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

def _wkb_solution(N2, z, zh, dz, nz, nmodes, boundary, grav):
    """
    WKB approximation for vertical modes
    """
    # Safety check for N2
    N2_safe = np.maximum(N2, 1e-10)  # Avoid negative or zero N2
    N = np.sqrt(N2_safe)
    
    # Total water depth
    H = abs(z[-1])
    
    # Initialize output arrays
    wmodes = np.zeros((nz, nmodes))
    pmodes = np.zeros((nz-1, nmodes))
    ce = np.zeros(nmodes)
    
    # Calculate WKB phase speeds
    for m in range(1, nmodes+1):
        # Modal number (first mode is m=1)
        ce[m-1] = 1.0 / (m * np.pi) * np.trapz(N, z)
    
    # Calculate modes
    for m in range(1, nmodes+1):
        # Mode number starts at 1
        mode_num = m
        
        # WKB phase
        theta = np.zeros(nz)
        theta[1:] = cumulative_trapezoid(N, z)
        
        # Vertical mode structure (vertical velocity)
        if boundary.lower() == 'rigid':

            wmodes[:, m-1] = -np.sin(mode_num * np.pi * theta / theta[-1])
        else:
            # Free surface
            wmodes[:, m-1] = -np.cos(mode_num * np.pi * theta / theta[-1])
        
        # Horizontal velocity structure
        for i in range(nz-1):
            pmodes[i, m-1] = (wmodes[i, m-1] - wmodes[i+1, m-1]) / dz[i]
        

        weights = np.ones(nz)
        weights[0] = 0.5
        weights[-1] = 0.5
        if nz > 2:
            weights[1:-1] = 1.0
            
        # 对wmodes进行归一化
        w_int = np.sum(weights * wmodes[:, m-1]**2 * np.append(dz, dz[-1]))
        wnorm = np.sqrt(w_int / H)
        wmodes[:, m-1] = wmodes[:, m-1] / wnorm
        
        # 对pmodes进行归一化 
        p_weights = np.ones(nz-1)
        p_weights[0] = 0.5
        p_weights[-1] = 0.5
        if nz-1 > 2:
            p_weights[1:-1] = 1.0
            
        p_int = np.sum(p_weights * pmodes[:, m-1]**2 * dz)
        pnorm = np.sqrt(p_int / H)
        pmodes[:, m-1] = pmodes[:, m-1] / pnorm
        

    
    return wmodes, pmodes, ce, z, zh