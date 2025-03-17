"""
Various tools needed for calculations in the simulation.
"""
from numpy import array, sin, cos, sinc, arctan2, arcsin, arccos, trace, sqrt, eye, sign, pi
from numpy.linalg import norm, det
from scipy.interpolate import splprep, splev

"""
wrap chi_1, so that it is within +-pi of chi_2
"""
import numpy as np

def wrap(chi_1, chi_2):
    while chi_1 - chi_2 > np.pi:
        chi_1 = chi_1 - 2.0 * np.pi
    while chi_1 - chi_2 <= -np.pi:
        chi_1 = chi_1 + 2.0 * np.pi
    return chi_1

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

def bound_angle(angle, lower, upper):
    while angle > upper:
        angle -= 2*np.pi
    while angle <=lower:
        angle += 2*np.pi  
        
    return angle

def inHalfSpace(pos, halfspace_r, halfspace_n):
        if (pos - halfspace_r).T @ halfspace_n >= 0:
            return True
        else:
            return False


def quaternion_to_euler(quaternion):
    """
    converts a quaternion attitude to an euler angle attitude
    :param quaternion: the quaternion to be converted to euler angles
    :return: the euler angle equivalent (phi, theta, psi) in a array
    """
    q0 = quaternion.item(0)
    qx = quaternion.item(1)
    qy = quaternion.item(2)
    qz = quaternion.item(3)
    phi = arctan2(2.0 * (qy * qz + q0 * qx), q0**2.0 - qx*2.0 - qy**2.0 + qz**2.0 )
    theta = arcsin(2.0 * (q0 * qy - qx * qz))
    psi = arctan2(2.0 * (qx * qy + q0 * qz), q0**2.0 + qx**2.0 - qy**2.0 - qz**2.0)
    return phi, theta, psi


def euler_to_quaternion(phi=0., theta=0., psi=0.):
    """
    Converts an euler angle attitude to a quaternian attitude
    :return: Quaternian attitude in array(e0, e1, e2, e3)
    """

    e0 = cos(psi/2.0) * cos(theta/2.0) * cos(phi/2.0) + sin(psi/2.0) * sin(theta/2.0) * sin(phi/2.0)
    e1 = cos(psi/2.0) * cos(theta/2.0) * sin(phi/2.0) - sin(psi/2.0) * sin(theta/2.0) * cos(phi/2.0)
    e2 = cos(psi/2.0) * sin(theta/2.0) * cos(phi/2.0) + sin(psi/2.0) * cos(theta/2.0) * sin(phi/2.0)
    e3 = sin(psi/2.0) * cos(theta/2.0) * cos(phi/2.0) - cos(psi/2.0) * sin(theta/2.0) * sin(phi/2.0)

    return array([[e0],[e1],[e2],[e3]])


def euler_to_rotation(phi=0., theta=0., psi=0.):
    """
    Converts euler angles to rotation matrix (R_b^i)
    """
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)

    R_roll = array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])
    R_pitch = array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
    R_yaw = array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])
    R = R_yaw @ R_pitch @ R_roll
    return R


def quaternion_to_rotation(quaternion):
    """
    converts a quaternion attitude to a rotation matrix
    """
    e0, e1, e2, e3 = quaternion

    R = array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                  [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                  [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
    R = R/det(R)
    return R


def rotation_to_quaternion(R):
    """
    converts a rotation matrix to a unit quaternion
    """
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    tmp0=r11+r22+r33
    if tmp0>0:
        q0 = 0.5*sqrt(1+tmp0)
    else:
        q0 = 0.5*sqrt(((r12-r21)**2+(r13-r31)**2+(r23-r32)**2)/(3-tmp0))

    tmpx=r11-r22-r33
    if tmpx>0:
        qx = 0.5*sqrt(1+tmpx)
    else:
        qx = 0.5*sqrt(((r12+r21)**2+(r13+r31)**2+(r23-r32)**2)/(3-tmpx))
    qx = sign(r32-r23) * qx

    tmpy=-r11+r22-r33
    if tmpy>0:
        qy = 0.5*sqrt(1+tmpy)
    else:
        qy = 0.5*sqrt(((r12+r21)**2+(r13-r31)**2+(r23+r32)**2)/(3-tmpy))
    qy = sign(r13-r31) * qy

    tmpz=-r11+-22+r33
    if tmpz>0:
        qz = 0.5*sqrt(1+tmpz)
    else:
        qz = 0.5*sqrt(((r12-r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmpz))
    qz = sign(r21-r12) * qz

    return array([[q0], [qx], [qy], [qz]])

def rotation_to_euler(R):
    """
    converts a rotation matrix to euler angles
    """
    # quat = rotation_to_quaternion(R)
    # phi, theta, psi = quaternion_to_euler(quat)
    if abs(R[2][0])!=1:
        th1 = -arcsin(R[2][0])
        #th2 = pi - th1
        phi1 = arctan2(R[2][1]/cos(th1), R[2][2]/cos(th1))
        #phi2 = arctan2(R[2][1]/cos(th2), R[2][2]/cos(th2))
        psi1 = arctan2(R[1][0]/cos(th1), R[0][0]/cos(th1))
        #psi2 = arctan2(R[1][0]/cos(th2), R[0][0]/cos(th2))
        # both solutions (phi1, theta1, psi1) and (phi2, theta2, psi2) are correct
        theta = th1
        phi = phi1
        psi = psi1
    else:
        psi = 0
        if R[2][0]==-1:
            theta = pi/2
            phi = psi + arctan2(R[0][1], R[0][2])
        else:
            theta = -pi/2
            phi = -psi + arctan2(-R[0][1], -R[0][2])
    return phi, theta, psi

def generate_3d_helix(num_points=100, radius=1, pitch=1, height=10):
    """
    Generates a 3D helix of points.
    
    Args:
        num_points (int): Number of points in the helix.
        radius (float): Radius of the helix.
        pitch (float): Vertical distance between successive loops.
        height (float): Total height of the helix.
    
    Returns:
        np.ndarray: Array of 3D points representing the helix.
    """
    # Create a parameter t that goes from 0 to the total height / pitch
    t = np.linspace(0, height / pitch * 2 * np.pi, num_points)
    
    # Compute the x, y, and z coordinates of the helix
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2 * np.pi)
    
    # Stack into a single array of shape (num_points, 3)
    helix_points = np.column_stack((x, y, z))
    
    return helix_points

def create_circular_path(radius: float, num_points: int) -> np.ndarray:
    """
    Generates a circular path in 3D space.

    Args:
        radius (float): The radius of the circular path.
        num_points (int): The number of points in the circular path.

    Returns:
        np.ndarray: A 3xN array representing the circular path.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angles
    x = radius * np.cos(theta)  # X coordinates
    y = radius * np.sin(theta)  # Y coordinates
    z = np.zeros(num_points)     # Z coordinates (flat circle)
    
    return np.vstack((x, y, z))

def fit_b_spline(data_points, num_samples=2000, smoothing=0):
    """
    Fits a B-spline curve to the given data points and returns a dense sampling of points along the curve.

    Args:
        data_points (np.ndarray): Array of shape (num_points, 3) representing X, Y, and Z coordinates.
        num_samples (int): The number of points to sample along the B-spline curve. Default is 2000.
        smoothing (float): Smoothing factor for the spline fit. Default is 0 (no smoothing).

    Returns:
        np.ndarray: Array of shape (num_samples, 3) representing sampled points on the fitted B-spline curve.
    """
    # Ensure the input data has the correct shape
    if data_points.shape[1] != 3:
        raise ValueError("data_points array must be of shape (num_points, 3)")
    
    # Prepare the spline parameters using SciPy's splprep function
    tck, u = splprep(data_points.T, s=smoothing)
    
    # Generate the parameter values for the desired number of samples
    u_dense = np.linspace(0, 1, num_samples)
    
    # Evaluate the B-spline at the dense sample points
    spline_points = splev(u_dense, tck)
    
    # Convert the spline points back to (num_samples, 3) format
    spline_points = np.array(spline_points).T
    
    return spline_points

# def interpolate_points(data_points, num_samples=2000):
#     """
#     Linearly interpolates between data points to achieve the desired density.

#     Args:
#         data_points (np.ndarray): Array of shape (num_points, 3) representing X, Y, and Z coordinates.
#         num_samples (int): The total number of interpolated points. Default is 2000.

#     Returns:
#         np.ndarray: Array of shape (num_samples, 3) with linearly interpolated points.
#     """
#     # Calculate the distances between consecutive points
#     distances = np.sqrt(np.sum(np.diff(data_points, axis=0)**2, axis=1))
#     cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
#     # Generate evenly spaced target points along the cumulative distance
#     target_distances = np.linspace(0, cumulative_distances[-1], num_samples)
    
#     # Interpolate for each dimension (X, Y, Z) independently
#     interpolated_points = np.empty((num_samples, 3))
#     for i in range(3):  # For X, Y, and Z
#         interpolated_points[:, i] = np.interp(target_distances, cumulative_distances, data_points[:, i])
    
#     return interpolated_points


def interpolate_points(data_points, num_samples=2000):
    """
    Linearly interpolates between data points to achieve the desired density.

    Args:
        data_points (np.ndarray): Array of shape (num_points, 3) representing X, Y, and Z coordinates.
        num_samples (int): The number of interpolated points to generate between each consecutive point. Default is 2000.

    Returns:
        np.ndarray: Array of shape (num_points + (num_samples * (num_points - 1)), 3) with linearly interpolated points including original points.
    """
    # If there are fewer than 2 points, return them as is
    if len(data_points) < 2:
        return data_points

    # Initialize a list to hold the result points
    result_points = []

    # Iterate over each segment between consecutive points
    for i in range(len(data_points) - 1):
        start_point = data_points[i]
        end_point = data_points[i + 1]

        # Generate interpolated points between start_point and end_point
        segment_interpolated = np.linspace(start_point, end_point, num_samples + 2)[1:-1]  # Exclude endpoints
        result_points.append(start_point)  # Include the starting point
        result_points.extend(segment_interpolated)  # Add the interpolated points

    # Append the last point
    result_points.append(data_points[-1])

    # Convert the result to a NumPy array
    result_points = np.array(result_points)

    return result_points