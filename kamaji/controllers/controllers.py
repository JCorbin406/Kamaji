from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Controls(ABC):
    """
    Base class for various controllers.

    This class serves as a foundation for different controllers.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to compute the control input.
        
        Args:
            **kwargs: Arguments required by the specific controller implementation.

        Returns:
            np.ndarray: Control input vector.
        """
        pass

class PID(Controls):
    def __init__(self, state_names, goals, Kp, Ki, Kd, dt=0.01,
                 lower_limit=None, upper_limit=None, derivative_smoothing=0.1):
        super().__init__()
        self.state_names = state_names
        self.goals = np.array(goals)
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral = np.zeros(len(Kp))
        self.prev_error = np.zeros(len(Kp))
        self.derivative = np.zeros(len(Kp))
        self.alpha = derivative_smoothing  # smoothing factor for derivative
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def update(self, t, state_dict: dict) -> np.ndarray:
        current_state = np.array([state_dict[name] for name in self.state_names])
        error = self.goals - current_state
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10, 10)  # prevent windup

        # Smooth derivative to reduce noise
        raw_derivative = (error - self.prev_error) / self.dt
        self.derivative = self.alpha * raw_derivative + (1 - self.alpha) * self.derivative
        self.prev_error = error

        control = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative

        if self.lower_limit is not None or self.upper_limit is not None:
            lower = -np.inf if self.lower_limit is None else np.full_like(control, self.lower_limit)
            upper = np.inf if self.upper_limit is None else np.full_like(control, self.upper_limit)
            control = np.clip(control, lower, upper)

        return control

    def set_goal(self, new_goals: list) -> None:
        self.goals = np.array(new_goals)


class Constant(Controls):
    def __init__(self, control: np.ndarray):
        super().__init__()
        self.control = np.array(control)

    def update(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.control
    
# class PathFollowerDyn(Controls):
#     """
#     Placeholder
#     """
#     def __init__(self, initial_setpoint: dict, path: np.ndarray, gains: np.ndarray, t_go: float, dt: float):
#         """
#         Constructor to initialize needed terms for the controller class

#         The setpoint dict for this subclass is defined as:
#             _setpoint:
#                 'pos': np vec [x, y, z]
#         """
#         super().__init__(initial_setpoint)
#         self.path = path
        
#         self.g = PhysConst.g
#         self.dt = dt
#         self.T, self.N, self.B, self.k, self.tau = self.calculate_frenet_frame()
#         self.arc_length = self.calculate_arc_length()

#         self.obstacles = self._setpoint["obstacles"]
#         self.num_obstacles = len(self.obstacles)

#         # Compute path speed based on time to go and arc length
#         self.s_dot = self.arc_length / t_go

#         self.k1, self.k2, self.k3 = gains

#         # Assemble the gain matrix, K
#         self.K = np.array([[gains[0], 0, 0], [0, gains[1], 0], [0, 0, gains[2]]])
#         self.t_go = t_go
#         self.time = 0.0

#         kp = 500.0
#         ki = 0.0
#         kd = 50.0
#         Ts = dt
#         sigma = 0.05
#         upper_limit = 1e6
#         lower_limit = -1e6
#         anti_windup = "error_dot"
#         anti_windup_limit = 100
        
#         self.control_x = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
#                                                 anti_windup_limit)
#         self.control_y = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
#                                                 anti_windup_limit)
#         self.control_z = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
#                                                 anti_windup_limit)

#         # self.control_x = pid_control.PControl(kp, 1e9, -1e9)
#         # self.control_y = pid_control.PControl(kp, 1e9, -1e9)
#         # self.control_z = pid_control.PControl(kp, 1e9, -1e9)
        
#         """CBF Stuff V1"""
#         # # Define variables
#         # x1, x2, x3, x4, x5, x6, xc, yc, zc, r, k1, k2 = symbols('x1 x2 x3 x4 x5 x6 xc yc zc r k1 k2')
#         # ux, uy, uz = symbols('ux uy uz')
#         # # Define h(x) function
#         # h = sqrt((x1 - xc)**2 + (x2 - yc)**2 + (x3 - zc)**2) - r
#         # # Define f(x) and g(x) as matrices
#         # f = Matrix([x4, x5, x6, 0, 0, 0])
#         # g = Matrix([0, 0, 0, 1, 1, 1])
#         # # Compute Lf(h) (Lie derivative of h along f)
#         # Lf_h = sum(diff(h, x_i) * f[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))
#         # # Compute Lgh (lie derivative of h along g)
#         # Lg_h = sum(diff(h, x_i) * g[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))
#         # # Compute Lf2(h) (second Lie derivative along f)
#         # Lf2_h = sum(diff(Lf_h, x_i) * f[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))
#         # # Compute LgLf(h) (Lie derivative of Lf(h) along g)
#         # Lg_Lf_h = sum(diff(Lf_h, x_i) * g[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))


#     def update(self, t: float, state: np.ndarray):
#         """
#         The primary method call for the path following controller

#         Required Inputs:
#             t: float, timestamp for this control call
#             state: 1d numpy vector of the quad states
#                 state vector: x, y, z, psi
#         """
#         desired_vel = self.compute_kinematic_control(state[0:3])
#         curr_vel = np.expand_dims(state[3:], axis=1)

#         control_input_x = self.control_x.update(desired_vel[0], curr_vel[0]).item()
#         control_input_y = self.control_y.update(desired_vel[1], curr_vel[1]).item()
#         control_input_z = self.control_z.update(desired_vel[2], curr_vel[2]).item()

#         control_input = (control_input_x, control_input_y, control_input_z)

#         # self.obstacles = self._setpoint["obstacles"]
#         # self.num_obstacles = len(self.obstacles)

#         # _h = np.empty((self.num_obstacles,))
#         # _g = np.empty((self.num_obstacles, 3))

#         # for idx, obstacle in enumerate(self.obstacles):
#         #     xc = obstacle[0]
#         #     yc = obstacle[1]
#         #     zc = obstacle[2]
#         #     r = obstacle[3]

#         #     k1 = 2.1
#         #     k2 = 5.0

#         #     x1, x2, x3, x4, x5, x6 = state[0], state[1], state[2], state[3], state[4], state[5]
#         #     # xc, yc, zc, r = 25, 0.0, 10.0, 5.0
#         #     # xc, yc, zc, r = -25, 0., -25, 5.0
#         #     h = self.h_func(x1, xc, x2, yc, x3, zc, r)
#         #     f = self.f_func(x4, x5, x6)
#         #     g = self.g_func()
#         #     Lf_h = self.Lf_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)
#         #     Lg_h = self.Lg_h_func()
#         #     Lf2_h = self.Lf2_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)
#         #     Lg_Lf_h = self.Lg_Lf_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)

#         #     h_obs = np.array(Lf2_h + Lf_h*(k1+k2)+k1*k2*h)
#         #     g_obs = -Lg_Lf_h
#         #     _h[idx] = h_obs
#         #     _g[idx, :] = g_obs

#         # _h = np.array(Lf2_h + Lf_h*(k1+k2)+k1*k2*h)
#         # _g = -Lg_Lf_h
#         # _g = _g * np.ones((1, 3))

#         # p = 2*np.eye(3)
#         # q = -2*np.array(control_input).reshape(3,)

#         # u_filtered = solve_qp(p, q, _g, _h, solver="cvxopt")

#         self.time += self.dt

#         return control_input
#         # return u_filtered

#     def calculate_frenet_frame(self):
#         """
#         Calculate the Frenet-Serret frame (tangent (T), normal (N), and binormal (B) unit vectors),
#         curvature (k), and torsion (τ) for a given 3D path, with outputs having the same length as the
#         original path array. The Frenet-Serret frame is used in differential geometry to describe the
#         orientation and curvature of a smooth curve in space.

#         Args:
#             path (np.ndarray): Array of shape (num_points, 3) representing a series of discrete
#                 points along a 3D path. Each row is a 3D coordinate [x, y, z] representing a point
#                 on the path.

#         Returns:
#             tuple: (T, N, B, k, τ) where:
#                 - T (np.ndarray): Tangent unit vectors of shape (num_points, 3), representing the
#                 direction of the curve at each point.
#                 - N (np.ndarray): Normal unit vectors of shape (num_points, 3), orthogonal to T and
#                 indicating the direction of curvature.
#                 - B (np.ndarray): Binormal unit vectors of shape (num_points, 3), orthogonal to both
#                 T and N, forming a right-handed orthonormal basis for each point.
#                 - k (np.ndarray): Curvature values of shape (num_points, 1), representing the rate of
#                 change of the tangent vector with respect to the curve parameter.
#                 - τ (np.ndarray): Torsion values of shape (num_points, 1), representing the rate of
#                 change of the binormal vector with respect to the curve parameter.

#         Notes:
#             The Frenet-Serret formulas define a local coordinate system along a space curve.
#             They are given by:
#                 dT/ds = k * N
#                 dN/ds = -k * T + τ * B
#                 dB/ds = -τ * N
#             where:
#                 - s is the arc length,
#                 - k is the curvature (magnitude of dT/ds),
#                 - τ is the torsion (magnitude of dB/ds).

#             - Curvature (k) is calculated as the magnitude of the cross product of the first
#             and second derivatives of the path, divided by the norm of the first derivative
#             cubed.
#             - Torsion (τ) is calculated using the third derivative and the normal vector, by
#             taking the dot product between the cross product of the first and second derivatives
#             and the third derivative.

#         References:
#             - https://mymathapps.com/mymacalc-sample/MYMACalc3/Part%20I%20-%20Geometry%20&%20Vectors/CurveProps/Torsion.html
#             - https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
#         """
#         # Calculate first, second, and third derivatives using np.gradient
#         drdu = np.gradient(self.path, axis=0)  # First derivative
#         drdu_norm = np.linalg.norm(drdu, axis=1, keepdims=True)
#         d2rdu2 = np.gradient(drdu, axis=0)  # Second derivative
#         d3rdu3 = np.gradient(d2rdu2, axis=0)  # Third derivative

#         # Cross product of first and second derivatives for numerator of curvature
#         numerator = np.cross(drdu, d2rdu2)
#         numerator_norm = np.linalg.norm(numerator, axis=1, keepdims=True)

#         # Tangent vector T
#         T = drdu / drdu_norm  # Tangential unit vector

#         # Binormal vector B (derived from the numerator)
#         B = numerator / numerator_norm
#         # B = numerator / (numerator_norm + np.finfo(float).eps)

#         # Normal vector N (cross product of B and T)
#         N = np.cross(B, T)

#         # Curvature (k)
#         k = numerator_norm / drdu_norm ** 3

#         # Torsion (τ)
#         tau = np.linalg.norm(np.gradient(T, axis=0), axis=1)

#         return T, N, B, k, tau

#     def calculate_arc_length(self):
#         """
#         Calculate the total arc length of a 3D path given an array of discrete points.
#         The arc length is the sum of the Euclidean distances between each pair of
#         consecutive points along the path.

#         Args:
#             path (np.ndarray): Array of shape (num_points, 3) representing the 3D path,
#                 where each row is a point [x, y, z] along the path.

#         Returns:
#             float: The total arc length of the path.
#         """
#         # Calculate the differences between consecutive points
#         diffs = np.diff(self.path, axis=0)

#         # Compute the Euclidean distances for each segment
#         self.segment_lengths = np.linalg.norm(diffs, axis=1)

#         # Sum up all segment lengths to get the total arc length
#         arc_length = np.sum(self.segment_lengths)

#         return arc_length

#     def find_closest_point_on_path(self, point):
#         """
#         Finds the closest point on a given 3D path to an arbitrary point.

#         Args:
#             path (np.ndarray): Array of shape (num_points, 3) representing the discrete points of a 3D path.
#             point (np.ndarray): Array of shape (3,) representing the arbitrary point in 3D space.

#         Returns:
#             tuple: (closest_index, distances) where
#                 - closest_index is the index of the closest point on the path.
#                 - distances is a tuple (dx, dy, dz) containing the distances in each direction
#                 between the provided point and the closest point on the path.
#         """
#         # Calculate the distance from the point to each point on the path in X, Y, and Z
#         deltas = self.path - point  # Broadcasting subtraction
#         distances_squared = np.sum(deltas ** 2, axis=1)  # Squared Euclidean distances

#         # Find the index of the closest point
#         closest_index = np.argmin(distances_squared)

#         r = self.path[closest_index]

#         return closest_index, r

#     def compute_kinematic_control(self, point):
#         self.closest_idx, r = self.find_closest_point_on_path(point)

#         self.time_remaining = self.t_go - self.time
#         dist_remaining = np.sum(self.segment_lengths[self.closest_idx:])
#         self.s_dot = dist_remaining / self.time_remaining

#         T = self.T[self.closest_idx]
#         N = self.N[self.closest_idx]
#         B = self.B[self.closest_idx]

#         self.S = np.column_stack((T, N, B))

#         r = np.expand_dims(r, axis=1)
#         p = np.expand_dims(point, axis=1)

#         self.d = self.S.T @ (p - r)

#         self.compute_affine_system()

#         self.X = -self.K @ self.d

#         vel = np.linalg.inv(self.G) @ (self.X - self.F)

#         return vel

#     def compute_affine_system(self):
#         self.G = self.S.T

#         d1, d2, d3 = self.d

#         k = self.k[self.closest_idx]
#         tau = self.tau[self.closest_idx]

#         fx = 1 - k * d2
#         fy = k * d1 - tau * d3
#         fz = tau * d2

#         self.F = -self.s_dot * np.row_stack((fx, fy, fz))