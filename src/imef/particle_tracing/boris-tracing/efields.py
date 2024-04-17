import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def crt_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)  # radial distance
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    return r, theta, phi


"""
Calculate last closed equipotential (LCE)

From Matsui et al.:
In order to calculate such contours, we first search for maximum potential 
in the radial direction at a fixed MLT value, and then get the minimum of 
these maximum potentials in the whole MLT range.
"""


def get_LCE(phidat, udat, pgrid, dr):
    # list to hold maximum potential at every MLT
    umax = []

    # find all coordinate locations at a fixed phi/MLT value, phi[N]
    for i in range(0, dr):
        phi_loc = np.where(phidat == pgrid[i])

        uvals = []

        for j in range(0, dr):
            # find locations where phi[N] exists
            c1 = phi_loc[0][j]
            c2 = phi_loc[1][j]

            # find all potential values at phi[N]
            uvals.append(udat[c1][c2])

        # find maximum potential
        umax.append(max(uvals))

    # compute minimum of max potentials in the whole MLT range
    LCE = min(umax)
    return LCE


""" 
corotation fields
"""


def corotation_efield(coords, sph=False):
    """
    Compute corotation electric field in the equatorial plane from given coordinates.
    
    Args:
        coords (ndarray): array of coordinates to compute efield in (x, y, z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        sph (bool, optional): determines if input coordinates are in Cartesian (False) or spherical (True). Defaults to False.
        
    Returns:
        ndarray: corotation E-field in 3 dimensions in units [mV/m]
    """
    
    if sph == False:
        # Cartesian coordinates provided
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])
    else:
        # Spherical coordinates provided
        rgeo = coords[0]
        theta = coords[1] if len(coords) > 1 else 0.0  # Default to 0 if not provided
        phi = coords[2] if len(coords) > 2 else 0.0  # Default to 0 if not provided

    # Equatorial magnetic field strength at the surface of Earth [T]
    BE = 3.1e-5

    # Angular velocity of the Earth’s rotation [rad/s]
    omega = 7.2921e-5

    # Equatorial radius of Earth [m]
    RE = 6371000

    # Radial component with zero check
    ER0 = (omega * BE * RE**3) / rgeo**2 if rgeo != 0 else np.nan

    # Convert to [mV/m]
    ER0 = ER0 * 1000

    # Convert back to Cartesian coordinates
    x = ER0 * np.sin(theta) * np.cos(phi)
    y = ER0 * np.sin(theta) * np.sin(phi)
    z = ER0 * np.cos(theta)

    # Return array [mV/m]
    ER = np.array([x, y, z])
    return ER


def corotation_potential(coords, sph=False):
    """
    Compute corotation potential field in equatorial plane from given coordinates.

    Args:
        coords (ndarray): array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        sph (bool, optional): deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
     ndarray: magnitude of corotation potential in units [kV]
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]

    # equatorial magnetic field strength at surface of Earth [T]
    BE = 3.1e-5

    # angular velocity of the Earth’s rotation [rad/s]
    omega = 7.2921e-5

    # equatorial radius of earth [m]
    RE = 6371000

    # calculate potential [V]
    # 5/24/23 - changed sign from (-) to (+)
    UR = (omega * RE**3 * BE) / rgeo if rgeo != 0 else np.nan

    # return in units [kV]
    return UR * 1e-3


"""
Volland-Stern
"""


def convection_field_A0(kp):
    """
    Compute uniform convection electric field strength in equatorial plane in [kV/m^2] with given kp index.
    Based on Maynard and Chen 1975 (doi:10.1029/JA080i007p01009).

    Args:
        kp (float): kp index

    Returns:
        float: A0; convection electric field strength in [mV/m^2]
    """
    # equatorial radius of earth [m]
    RE = 6371000

    # uniform convection electric field strength in equatorial plane [kV/m^2]
    A0 = 0.045 / ((1 - (0.159 * kp) + (0.0093 * kp**2)) ** 3 * (RE**2.0))

    # convert to [mV/m^2]
    A0 = A0 * 1e6

    return A0


def vs_potential(coords, gs, kp, sph=False):
    """
    Compute Volland-Stern potential field.
    Model based on Volland 1973 (doi:10.1029/JA078i001p00171) and Stern 1975 (doi:10.1029/JA080i004p00595).

    Args:
        coords (ndarray):          array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        gs (floay):             shielding constant
        kp (float):             kp index
        sph (bool, optional):   deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        float: potential in units [kV]
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]
        theta = coords[1]
        phi = coords[2]

    # equatorial radius of earth [m]
    RE = 6371000

    # uniform convection electric field strength in equatorial plane [mV/m^2]
    A0 = convection_field_A0(kp)

    # VS potential [mV]
    # 5/24/23 - might need to fix, add -92.4 component
    U = -A0 * (rgeo**gs) * np.sin(phi)

    # return potential in [kv]
    return -U * 1e-6


def vs_efield(coords, gs, kp, sph=False):
    """
    Compute Volland-Stern conevction electric field.
    Model based on Volland 1973 (doi:10.1029/JA078i001p00171) and Stern 1975 (doi:10.1029/JA080i004p00595).

    Args:
        coords (ndarray):          array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        gs (floay):             shielding constant
        kp (float):             kp index
        sph (bool, optional):   deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        ndarray: volland-stern field in 3 dimensions in units [mV/m].
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]
        theta = coords[1]
        phi = coords[2]

    # uniform convection electric field strength in equatorial plane [mV/m^2]
    A0 = convection_field_A0(kp)

    # radial componenet [mV/m] - rho
    EC0 = (-92.4 / rgeo**2) - (A0 * gs * (rgeo ** (gs - 1)) * (np.sin(phi)))
    EC0

    ## ! one more term

    # polar component [mV/m] - theta
    EC1 = 0.0

    # azimuthal component [mV/m] - phi
    EC2 = A0 * (rgeo ** (gs - 1)) * (1 / (np.sin(theta))) * (np.cos(phi))

    # convert to cartesian 
    X = np.cos(EC1) * np.cos(EC2) * EC0
    Y = np.sin(EC1) * np.cos(EC2) * EC0
    Z = np.sin(EC2) * EC0

    # return array [mV/m]
    #EC = np.array([EC0, EC1, EC2])

    EC = np.array([X, Y, Z])
    
    return EC

