import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kamaji.controllers.pid_control import PIDControl

class PathFollower():
    def __init__(self, path, dt, t_go):
        self._path = path
        # self.compute_accelerations_full_path()
        self.e_prior = 0.0
        self.dt = dt
        self.pid = PIDControl(2.0, 0.01, 0.5, dt, 0.05, 10000, -10000, "error", 10)
        self.path_length = self.compute_path_length()
        # self.accel = (2 * self.path_length) / t_go**2
        self.num_points = self._path.shape[1]
        self.delta_t = t_go / self.num_points

        self.calculate_velocity()
        self.calculate_acceleration()
        self.calculate_tangent_and_normal()
        self.accel_tan  = np.einsum('ij,ij->j', self.acceleration, self._tangent)*self._tangent
        self.accel_norm = self.acceleration - self.accel_tan

    def calculate_velocity(self):
        velocity = np.diff(self._path, axis=1) / self.delta_t
        # Pad by repeating the last velocity entry
        last_velocity = velocity[:, -1].reshape(3, 1)  # Extract the last velocity and reshape to (3, 1)
        velocity_padded = np.hstack((velocity, last_velocity))  # Pad the last velocity
        self.velocity = velocity_padded

    def calculate_acceleration(self):
        acceleration = np.diff(self.velocity, axis=1) / self.delta_t
        # Step 3: Pad by repeating the last acceleration entry
        last_acceleration = acceleration[:, -1].reshape(3, 1)  # Extract the last acceleration and reshape to (3, 1)
        acceleration_padded = np.hstack((acceleration, last_acceleration))  # Pad the last acceleration
        self.acceleration = acceleration_padded

    def calculate_control(self, target_point: np.ndarray):
        self.calculate_error(target_point)
        u_mag = self.pid.update(self._path[:, self.closest_idx], target_point)
        u_dir = self.unit_vector_to_nearest_point(target_point)
        u = u_mag * u_dir

        # curr_tan = self._tangent[:, self.closest_idx]
        # curr_norm = self._normal[:, self.closest_idx]
        # a_norm = 5 * curr_norm
        # a_tan = 5 * curr_tan

        # a_norm = self.accel_norm[:, self.closest_idx]
        # a_tan = self.accel_tan[:, self.closest_idx]

        # print(a_norm)
        # print(a_tan)

        # u_total = u + a_tan + a_norm

        u_total = u + self.acceleration[:, self.closest_idx]

        return u_total

    def compute_path_length(self):
        # Number of points
        n_points = self._path.shape[1]
        
        # Initialize total length
        total_length = 0.0
        
        # Iterate through consecutive points to compute the distances
        for i in range(n_points - 1):
            p_i = self._path[:, i]      # Current point
            p_ip1 = self._path[:, i + 1]  # Next point
            
            # Compute Euclidean distance between current point and next point
            distance = np.linalg.norm(p_ip1 - p_i)
            
            # Add to total length
            total_length += distance
        
        return total_length

    def closest_index_and_distance(self, target_point: np.ndarray) -> tuple:
        """
        Finds the index of the closest point in the array and the corresponding distance.

        Args:
            points (np.ndarray): A 3xN array where each column is a point [X, Y, Z].
            target_point (np.ndarray): A 1D array of shape (3,) representing the target point [X, Y, Z].

        Returns:
            tuple: (index, distance) where `index` is the index of the closest point and `distance` is the Euclidean distance.
        """
        # Calculate the Euclidean distances from the target point to all points
        distances = np.linalg.norm(self._path - target_point[:, np.newaxis], axis=0)

        # Find the index of the minimum distance
        closest_index = np.argmin(distances)

        # Get the minimum distance value
        closest_distance = distances[closest_index]

        return closest_index, closest_distance
    
    def calculate_tangent_and_normal(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the unit tangent and normal vectors for each segment of the 3D path
        and pads the result so that both vectors have N columns.

        Args:
            positions (np.ndarray): A 3xN array where each column is a point [X, Y, Z].
        """
        # Compute the difference between consecutive points to get tangent vectors
        tangents = np.diff(self._path, axis=1) / self.delta_t
        
        # Normalize the tangent vectors to get unit tangents
        tangent_magnitudes = np.linalg.norm(tangents, axis=0)
        
        # Avoid division by zero by adding a small epsilon where magnitudes are zero
        tangent_magnitudes[tangent_magnitudes == 0] = np.finfo(float).eps
        unit_tangents = tangents / tangent_magnitudes
        
        # Compute the difference between consecutive unit tangents to get normal vectors
        normals = np.diff(unit_tangents, axis=1) / self.delta_t
        
        # Normalize the normal vectors to get unit normals
        normal_magnitudes = np.linalg.norm(normals, axis=0)
        
        # Avoid division by zero for normal magnitudes
        normal_magnitudes[normal_magnitudes == 0] = np.finfo(float).eps
        unit_normals = normals / normal_magnitudes
        
        # Pad the last column of the tangent array with the last available value
        self._tangent = np.hstack([unit_tangents, unit_tangents[:, -1].reshape(3, 1)])
        
        # Pad the last two columns of the normal array with the last available value
        self._normal = np.hstack([unit_normals, np.tile(unit_normals[:, -1].reshape(3, 1), 2)])  

    def calculate_error(self, target_point: np.ndarray):
        self.closest_idx, self.e = self.closest_index_and_distance(target_point)
        self.e_dot = (self.e - self.e_prior) / self.dt
        self.e_prior = self.e

    def unit_vector_to_nearest_point(self, target_point: np.ndarray) -> np.ndarray:
        """
        Returns the unit vector pointing from the target point to the nearest point on the path.

        Args:
            points (np.ndarray): A 3xN array where each column is a point [X, Y, Z].
            target_point (np.ndarray): A 1D array of shape (3,) representing the target point [X, Y, Z].

        Returns:
            np.ndarray: A 1D array representing the unit vector pointing from the target point to the nearest point.
        """
        # Find the index and distance to the nearest point
        closest_index, _ = self.closest_index_and_distance(target_point)

        # Get the nearest point on the path
        nearest_point = self._path[:, closest_index]

        # Compute the vector from the target point to the nearest point
        vector_to_nearest = nearest_point - target_point

        # Normalize the vector to get the unit vector
        unit_vector = vector_to_nearest / np.linalg.norm(vector_to_nearest)

        return unit_vector

    # def compute_accelerations_full_path(self):
    #     n_points = self._path.shape[1]  # Number of points (columns)
        
    #     # Initialize arrays to store results
    #     tangential_accelerations = np.zeros((3, n_points))  # 3D vector for tangential accelerations
    #     normal_accelerations = np.zeros((3, n_points))      # 3D vector for normal accelerations
        
    #     # Iterate over the path
    #     for i in range(1, n_points - 1):
    #         p_im1 = self._path[:, i - 1]  # Position at i-1 (column i-1)
    #         p_i = self._path[:, i]        # Position at i (column i)
    #         p_ip1 = self._path[:, i + 1]  # Position at i+1 (column i+1)
            
    #         # Call the previous function to compute tangential and normal accelerations
    #         a_tangent, a_normal = self.compute_accelerations_from_positions(p_im1, p_i, p_ip1)
            
    #         # Store results
    #         tangential_accelerations[:, i] = a_tangent
    #         normal_accelerations[:, i] = a_normal
        
    #     # Handle boundary points (forward/backward difference)
    #     if n_points > 1:
    #         # For the first point (i=0), use forward difference
    #         p_i = self._path[:, 0]
    #         p_ip1 = self._path[:, 1]
    #         p_ip2 = self._path[:, 2] if n_points > 2 else p_ip1  # Handle small arrays
            
    #         # Estimate velocities using forward difference
    #         delta_s = np.linalg.norm(p_ip1 - p_i)
    #         v_i = (p_ip1 - p_i) / delta_s
    #         v_ip1 = (p_ip2 - p_ip1) / delta_s
            
    #         speed_i = np.linalg.norm(v_i)
    #         speed_ip1 = np.linalg.norm(v_ip1)
            
    #         # Tangential acceleration for first point
    #         a_tangent = (speed_ip1 - speed_i) / delta_s * v_i  # Direction of v_i
    #         tangential_accelerations[:, 0] = a_tangent
    #         normal_accelerations[:, 0] = np.zeros(3)  # Normal acceleration is zero at boundaries
            
    #         # For the last point (i=n-1), use backward difference
    #         p_in1 = self._path[:, -2]
    #         p_in = self._path[:, -1]
            
    #         delta_s = np.linalg.norm(p_in - p_in1)
    #         v_in = (p_in - p_in1) / delta_s
    #         speed_in = np.linalg.norm(v_in)
            
    #         tangential_accelerations[:, -1] = np.zeros(3)  # No next point to compare
    #         normal_accelerations[:, -1] = np.zeros(3)  # Last point, no curvature to estimate

    #     self.norm_accel = normal_accelerations
    #     self.tan_accel = tangential_accelerations

    # def compute_accelerations_from_positions(self, p_im1, p_i, p_ip1):
    #     # Compute tangential velocity vector
    #     v_tangent = p_ip1 - p_im1  # Change in position
    #     v_tangent /= np.linalg.norm(v_tangent)  # Normalize to get the unit tangent vector

    #     # Compute tangential acceleration (magnitude)
    #     delta_s = np.linalg.norm(p_ip1 - p_i)  # Distance between points
    #     a_tangent_magnitude = (np.linalg.norm(p_ip1 - p_i) - np.linalg.norm(p_i - p_im1)) / delta_s
        
    #     # Calculate tangential acceleration vector
    #     a_tangent = a_tangent_magnitude * v_tangent

    #     # Calculate normal acceleration (assuming a simple path curvature)
    #     # Normal vector is perpendicular to the tangent vector
    #     p_prev = p_i - p_im1
    #     p_next = p_ip1 - p_i
    #     v_prev = p_prev / np.linalg.norm(p_prev) if np.linalg.norm(p_prev) > 0 else np.zeros_like(p_prev)
    #     v_next = p_next / np.linalg.norm(p_next) if np.linalg.norm(p_next) > 0 else np.zeros_like(p_next)

    #     # Approximate normal vector using the cross product
    #     normal_vector = np.cross(v_tangent, (v_next + v_prev))  # Cross product gives a perpendicular direction
    #     normal_vector /= np.linalg.norm(normal_vector) if np.linalg.norm(normal_vector) > 0 else np.zeros_like(normal_vector)
        
    #     # Compute normal acceleration magnitude
    #     a_normal_magnitude = np.linalg.norm(p_next - p_prev) / delta_s  # Approximation of curvature

    #     # Calculate normal acceleration vector
    #     a_normal = a_normal_magnitude * normal_vector
        
    #     return a_tangent, a_normal


def plot_path_with_tangent_and_normal(positions: np.ndarray, tangents: np.ndarray, normals: np.ndarray, arrow_length: float = 2.0, head_size: float = 0.01):
    """
    Plots the 3D path along with tangent and normal vectors at each point.

    Args:
        positions (np.ndarray): A 3xN array where each column is a point [X, Y, Z].
        tangents (np.ndarray): A 3xN array of unit tangent vectors.
        normals (np.ndarray): A 3xN array of unit normal vectors.
        arrow_length (float): Length of the arrows.
        head_size (float): Size of the arrowhead.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D path
    ax.plot(positions[0], positions[1], positions[2], label='Path', color='blue')

    # Add tangent and normal vectors at each point
    for i in range(positions.shape[1]):
        # Tangent vectors (arrows in red)
        ax.quiver(positions[0, i], positions[1, i], positions[2, i], 
                  tangents[0, i] * arrow_length, tangents[1, i] * arrow_length, tangents[2, i] * arrow_length,
                  color='red', length=arrow_length, normalize=True, 
                  arrow_length_ratio=head_size, label='Tangent' if i == 0 else "")
        
        # Normal vectors (arrows in green)
        ax.quiver(positions[0, i], positions[1, i], positions[2, i], 
                  normals[0, i] * arrow_length, normals[1, i] * arrow_length, normals[2, i] * arrow_length,
                  color='green', length=arrow_length, normalize=True, 
                  arrow_length_ratio=head_size, label='Normal' if i == 0 else "")
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot title and legend
    ax.set_title('3D Circular Path with Tangent and Normal Vectors')
    ax.legend()
    
    # Show plot
    plt.show()

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

if __name__ == "__main__":
    # points = np.array([[1.0, 2.0, 3.0, 4.0],
    #                   [1.0, 2.0, 1.0, 0.0],
    #                   [1.0, 1.0, 2.0, 3.0]])

    # Generate a circular path with specified radius and number of points
    radius = 10.0
    num_points = 200
    points = create_circular_path(radius, num_points)
    
    dt = 0.01
    control = PathFollower(points, dt, 2, 0.5, 5)

    tangents_padded, normals_padded = control._tangent, control._normal

    test_point = np.array([2.5, 2.5, 0])

    print(control.unit_vector_to_nearest_point(test_point))
    print(control.calculate_control(test_point))

    # Plot the path with tangent and normal vectors
    plot_path_with_tangent_and_normal(points, tangents_padded, normals_padded)
