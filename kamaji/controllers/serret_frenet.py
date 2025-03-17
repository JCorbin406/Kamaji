import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kamaji.controllers.pid_control import PIDControl

class PathFollower():
    def __init__(self, path, gains, t_go):
        self.path = path
        self.T, self.N, self.B, self.k, self.tau = self.calculate_frenet_frame()
        self.arc_length = self.calculate_arc_length()

        # Compute path speed based on time to go and arc length
        self.s_dot = self.arc_length / t_go

        self.k1, self.k2, self.k3 = gains

        # Assemble the gain matrix, K
        self.K = np.array([[gains[0], 0, 0], [0, gains[1], 0], [0, 0, gains[2]]])
        self.t_go = t_go

    def calculate_frenet_frame(self):
        """
        Calculate the Frenet-Serret frame (tangent (T), normal (N), and binormal (B) unit vectors),
        curvature (k), and torsion (τ) for a given 3D path, with outputs having the same length as the
        original path array. The Frenet-Serret frame is used in differential geometry to describe the 
        orientation and curvature of a smooth curve in space.

        Args:
            path (np.ndarray): Array of shape (num_points, 3) representing a series of discrete 
                points along a 3D path. Each row is a 3D coordinate [x, y, z] representing a point 
                on the path.

        Returns:
            tuple: (T, N, B, k, τ) where:
                - T (np.ndarray): Tangent unit vectors of shape (num_points, 3), representing the
                direction of the curve at each point.
                - N (np.ndarray): Normal unit vectors of shape (num_points, 3), orthogonal to T and
                indicating the direction of curvature.
                - B (np.ndarray): Binormal unit vectors of shape (num_points, 3), orthogonal to both 
                T and N, forming a right-handed orthonormal basis for each point.
                - k (np.ndarray): Curvature values of shape (num_points, 1), representing the rate of 
                change of the tangent vector with respect to the curve parameter.
                - τ (np.ndarray): Torsion values of shape (num_points, 1), representing the rate of 
                change of the binormal vector with respect to the curve parameter.

        Notes:
            The Frenet-Serret formulas define a local coordinate system along a space curve.
            They are given by:
                dT/ds = k * N
                dN/ds = -k * T + τ * B
                dB/ds = -τ * N
            where:
                - s is the arc length,
                - k is the curvature (magnitude of dT/ds),
                - τ is the torsion (magnitude of dB/ds).
            
            - Curvature (k) is calculated as the magnitude of the cross product of the first
            and second derivatives of the path, divided by the norm of the first derivative
            cubed.
            - Torsion (τ) is calculated using the third derivative and the normal vector, by
            taking the dot product between the cross product of the first and second derivatives
            and the third derivative.

        References:
            - https://mymathapps.com/mymacalc-sample/MYMACalc3/Part%20I%20-%20Geometry%20&%20Vectors/CurveProps/Torsion.html
            - https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
        """
        # Calculate first, second, and third derivatives using np.gradient
        drdu = np.gradient(self.path, axis=0)  # First derivative
        drdu_norm = np.linalg.norm(drdu, axis=1, keepdims=True)
        d2rdu2 = np.gradient(drdu, axis=0)  # Second derivative
        d3rdu3 = np.gradient(d2rdu2, axis=0)  # Third derivative

        # Cross product of first and second derivatives for numerator of curvature
        numerator = np.cross(drdu, d2rdu2)
        numerator_norm = np.linalg.norm(numerator, axis=1, keepdims=True)

        # Tangent vector T
        T = drdu / drdu_norm  # Tangential unit vector

        # Binormal vector B (derived from the numerator)
        B = numerator / numerator_norm
        # B = numerator / (numerator_norm + np.finfo(float).eps)

        # Normal vector N (cross product of B and T)
        N = np.cross(B, T)

        # Curvature (k)
        k = numerator_norm / drdu_norm**3

        # Torsion (τ)
        tau = np.linalg.norm(np.gradient(T, axis=0), axis=1)

        return T, N, B, k, tau
    
    def calculate_arc_length(self):
        """
        Calculate the total arc length of a 3D path given an array of discrete points.
        The arc length is the sum of the Euclidean distances between each pair of 
        consecutive points along the path.

        Args:
            path (np.ndarray): Array of shape (num_points, 3) representing the 3D path, 
                where each row is a point [x, y, z] along the path.

        Returns:
            float: The total arc length of the path.
        """
        # Calculate the differences between consecutive points
        diffs = np.diff(self.path, axis=0)
        
        # Compute the Euclidean distances for each segment
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        # Sum up all segment lengths to get the total arc length
        arc_length = np.sum(segment_lengths)
        
        return arc_length

    def find_closest_point_on_path(self, point):
        """
        Finds the closest point on a given 3D path to an arbitrary point.

        Args:
            path (np.ndarray): Array of shape (num_points, 3) representing the discrete points of a 3D path.
            point (np.ndarray): Array of shape (3,) representing the arbitrary point in 3D space.

        Returns:
            tuple: (closest_index, distances) where
                - closest_index is the index of the closest point on the path.
                - distances is a tuple (dx, dy, dz) containing the distances in each direction
                between the provided point and the closest point on the path.
        """
        # Calculate the distance from the point to each point on the path in X, Y, and Z
        deltas = self.path - point  # Broadcasting subtraction
        distances_squared = np.sum(deltas ** 2, axis=1)  # Squared Euclidean distances
        
        # Find the index of the closest point
        closest_index = np.argmin(distances_squared)
    
        r = self.path[closest_index]

        return closest_index, r
    
    def compute_kinematic_control(self, point):
        self.closest_idx, r = self.find_closest_point_on_path(point)

        T = self.T[self.closest_idx]
        N = self.N[self.closest_idx]
        B = self.B[self.closest_idx]

        self.S = np.column_stack((T, N, B))

        r = np.expand_dims(r, axis=1)
        p = np.expand_dims(point, axis=1)

        self.d = self.S.T @ (p - r)

        self.compute_affine_system()

        self.X = -self.K @ self.d

        vel = np.linalg.inv(self.G) @ (self.X - self.F)

        return vel
    
    def compute_desired_vel(self):
        d1, d2, d3 = self.d

        k = self.k[self.closest_idx]
        tau = self.tau[self.closest_idx]

        T = self.T[self.closest_idx]
        N = self.N[self.closest_idx]
        B = self.B[self.closest_idx]

        s1 = d2 * self.k2 + self.s_dot * (d1 * k - d3 * tau)
        s2 = d1 * self.k1 - self.s_dot * (d2 * k - 1)
        s3 = d3 * self.k3 + self.s_dot * d2 * tau

        p1 = -T[0] * s2 - N[0] * s1 - B[0] * s3
        p2 = -T[1] * s2 - N[1] * s1 - B[1] * s3
        p3 = -T[2] * s2 - N[2] * s1 - B[2] * s3

        vel = np.row_stack((p1, p2, p3))

        return vel

    def compute_affine_system(self):
        self.G = self.S.T

        d1, d2, d3 = self.d

        k = self.k[self.closest_idx]
        tau = self.tau[self.closest_idx]

        fx = 1 - k * d2
        fy = k * d1 - tau * d3
        fz = tau * d2

        self.F = -self.s_dot * np.row_stack((fx, fy, fz))

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

if __name__ == "__main__":
    helix = generate_3d_helix(2000, 10, 5, 25)
    test = PathFollower(helix, (100, 100, 100), 1.0)

    test_point = np.array([11, 0, -1])

    print(test.compute_kinematic_control(test_point))

    print(f"Curvature (k): {test.k[5]}")
    print(f"Torsion (tau): {test.tau[0]}")

    # Select a subset of points to plot the coordinate system (e.g., every 20th point)
    subset_indices = np.arange(0, len(test.path), 100)
    subset_points = test.path[subset_indices]
    subset_T = test.T[subset_indices]
    subset_N = test.N[subset_indices]
    subset_B = test.B[subset_indices]

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the helix
    ax.plot(test.path[:, 0], test.path[:, 1], test.path[:, 2], color='k')

    # Plot the T, N, B coordinate system at each subset point
    for i in range(len(subset_points)):
        # Starting point for the vectors
        x, y, z = subset_points[i]
        
        # Plot the T (tangent), N (normal), and B (binormal) vectors
        ax.quiver(x, y, z, subset_T[i, 0], subset_T[i, 1], subset_T[i, 2], color='r', length=2, normalize=True, label='Tangent' if i == 0 else "")
        ax.quiver(x, y, z, subset_N[i, 0], subset_N[i, 1], subset_N[i, 2], color='g', length=2, normalize=True, label='Normal' if i == 0 else "")
        ax.quiver(x, y, z, subset_B[i, 0], subset_B[i, 1], subset_B[i, 2], color='b', length=2, normalize=True, label='Binormal' if i == 0 else "")


    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Helix')

    # Set equal scaling
    max_range = np.array([test.path[:, 0].max() - test.path[:, 0].min(),
                        test.path[:, 1].max() - test.path[:, 1].min(),
                        test.path[:, 2].max() - test.path[:, 2].min()]).max() / 2.0

    mid_x = (test.path[:, 0].max() + test.path[:, 0].min()) * 0.5
    mid_y = (test.path[:, 1].max() + test.path[:, 1].min()) * 0.5
    mid_z = (test.path[:, 2].max() + test.path[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()

    # Show the plot
    plt.show()