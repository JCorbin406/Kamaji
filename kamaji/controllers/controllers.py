# controllers.py
import copy
from abc import ABC, abstractmethod
from math import cos, sin
import math

import numpy as np
import scipy.constants as consts
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as Rot

from kamaji.parameters.geometric_control_params import (GeometricParameters,
                                                     geom_ctl_param_x500)
from kamaji.parameters.quadcopter_params import QuadParameters, x500_exp_params
from kamaji.tools.tools import wrap
from kamaji.controllers import pid_control
import kamaji.tools.physcial_constants as PhysConst
from sympy import symbols, sqrt, Matrix, simplify, diff, init_printing, lambdify
from qpsolvers import solve_qp
import sympy as sp

class Controller(ABC):
    """
    Base class for various controllers.

    This class serves as a foundation for different control models, providing a
    structure for defining the setpoints, control outputs, and methods for updating
    the state based on control dynamics.

    Attributes:
        _setpoint (dict): Current setpoint the controller should drive to, which should be defined in
        subclasses.

    Methods:
        update: Computes the new control inputs needed to drive the system to the setpoint.
        set_setpoint: Updates the current setpoint of the system.
    """
    def __init__(self, initial_setpoint: dict) -> None:
        """
        Initializes the dynamics model with a given initial setpoint.

        Args:
            initial_setpoint (dict): The initial setpoint of the system.
        """
        self._setpoint = initial_setpoint

    @abstractmethod
    def update(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Abstract method to update the control inputs of the system from the controller.
        Must be implemented by subclasses.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        pass

    def set_setpoint(self, setpoint: dict) -> None:
        """
        Updates the setpoint the controller is driving to.

        Args:
            setpoint (dict): New setpoint for the controller.
        """
        self._setpoint = setpoint

    @property
    def setpoint(self) -> dict:
        """
        Returns the current setpoint of the system.

        Returns:
            dict: The current setpoint of the system.
        """
        return self._setpoint

class GeometricController(Controller):
    """
    This module contains the class implementation of the geometric controllers
    developed by T Lee and his lab
    """
    def __init__(self, initial_setpoint: dict, ctl_params: GeometricParameters = geom_ctl_param_x500,
                 quad_params: QuadParameters = x500_exp_params) -> None:
        """
        Constructor to initialize needed terms for the controller class

        The setpoint dict for this subclass is defined as:
            _setpoint:
                'pos': np vec [x, y, z]
                'vel': tuple (x, y, z) or None in elements,
                'accel': np vec [x, y, z] or None
                'quat': np vec [qx, qy, qz, qw] or None
                'psi': float or None
                'heading_vec': np 3x3 vec Q-frame x-axis: b1d, b1d_dot, b1d_ddot
                    or None
                'omega': np vec [x, y, z] or None
                'Tvec': np vec [T, taux, tauy, tauz]
                'rotor': np vec [w1, w2, ..., wn] or None
        """
        super().__init__(initial_setpoint)
        # Assign initialization arguments to class
        self.ctl_params = ctl_params  # Controller parameters
        self.quad_params = quad_params  # Quadcopter parameters
        # self.env_params = env_params # Environmental parameters

        # Setpoint variables
        self.pos_cmd = np.zeros(3)  # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3)  # Linear velocity setpoint in {A}
        self.q_cmd = np.array([0, 0, 0, 1])

        # Variables for storing previous state information
        self.w_vec_last = self.quad_params.w_min * np.array([1, -1, 1, -1])
        self.T_vec_last = np.zeros(4)  # Thrust and torque commands

        # Integral term variables
        self.ei = np.zeros(3)  # Position error integral
        self.eI1 = 0  # X-axis attitude error integral
        self.eI2 = 0  # Y-axis attitude error integral
        self.eIy = 0  # Yaw attitude error integral

        # Timing variables
        self.geom_prev_t = 0  # Time for last control loop run

        # Control allocation parameters
        theta = np.radians(45)
        CT = quad_params.CT
        d = quad_params.d
        CQ = quad_params.CQ
        self.Gamma = np.array([
            [CT, CT, CT, CT],
            [-sin(theta) * d * CT, -sin(theta) * d * CT, sin(theta) * d * CT,
             sin(theta) * d * CT],
            [cos(theta) * d * CT, -cos(theta) * d * CT, -cos(theta) * d * CT,
             cos(theta) * d * CT],
            [-CQ, CQ, -CQ, CQ]])
        self.Gamma_inv = LA.inv(self.Gamma)

    def update(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        The primary method call for the geometric controller

        Required Inputs:
            t: float, timestamp for this control call
            state: 1d numpy vector of the quad states
                state vector: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt,
                                omega_x, omega_y, omega_z
        """

        # Geometric controller call loop [200Hz]
        geom_dt = t - self.geom_prev_t
        if geom_dt >= 1 / self.ctl_params.geom_freq:

            # Pull setpoints from dictionary
            self.pos_cmd = self._setpoint['pos']
            self.vel_cmd = self._setpoint['vel']
            b1d = self._setpoint['heading_vec'][0]
            b1d_dot = self._setpoint['heading_vec'][1]
            b1d_ddot = self._setpoint['heading_vec'][2]

            self.geom_prev_t = t  # Save time of this call

            # Compute the Q to W rotation
            q = state[3:7]
            R_Q_W = Rot.from_quat(q)

            # Setup desired unit vectors
            # e1 = np.array([1, 0, 0])
            # e2 = np.array([0, 1, 0])
            e3 = np.array([0, 0, 1])
            # b1 = R_Q_W.apply(e1)
            # b2 = R_Q_W.apply(e2)
            b3 = R_Q_W.apply(e3)

            # Compute the thrust controller inputs
            ex = state[:3].astype(float) - self.pos_cmd
            ev = state[7:10].astype(float) - self.vel_cmd
            om = state[10:13]

            # Call thrust controller
            A, A_dot, A_ddot = self.thrust_control_yaw_decoupled(ex, ev, R_Q_W,
                                                                 om, geom_dt)

            # Compute the attitude controller inputs
            b3c, b3c_dot, b3c_ddot = self.deriv_unit_vec(-A, -A_dot, -A_ddot)

            A2 = -self.hat(b1d) @ b3c
            A2_dot = -self.hat(b1d_dot) @ b3c - self.hat(b1d) @ b3c_dot
            A2_ddot = -self.hat(b1d_ddot) @ b3c - 2 * self.hat(b1d_dot) @ b3c_dot - \
                      self.hat(b1d) @ b3c_ddot

            b2c, b2c_dot, b2c_ddot = self.deriv_unit_vec(A2, A2_dot, A2_ddot)

            b2c_vec = np.array([b2c, b2c_dot, b2c_ddot])
            b3c_vec = np.array([b3c, b3c_dot, b3c_ddot])

            b1c, b1c_dot, b1c_ddot = self.b1c_gen(b2c_vec, b3c_vec)

            Rc_Q_W = Rot.from_matrix([b1c, b2c, b3c])
            self.q_cmd = Rc_Q_W.inv().as_quat()
            Rc_Q_W_dot = Rot.from_matrix([b1c_dot, b2c_dot, b3c_dot])
            Rc_Q_W_ddot = Rot.from_matrix([b1c_ddot, b2c_ddot, b3c_ddot])

            R_int = Rc_Q_W.inv() * Rc_Q_W_dot
            wc = self.vee(R_int.as_matrix())

            R_int2 = Rc_Q_W.inv() * Rc_Q_W_ddot
            wc_dot = self.vee(R_int2.as_matrix() - self.hat(wc) @ self.hat(wc))
            wc3 = np.dot(b3, Rc_Q_W.apply(wc))
            wc3_dot = np.dot(b3, Rc_Q_W.apply(wc_dot)) + \
                      np.dot(R_Q_W.apply(self.hat(om)) @ e3, Rc_Q_W.apply(wc))

            # Call M1/M2 attitude controller
            M1, M2 = self.m1m2_control_yaw_decoupled(b3c, b3c_dot, b3c_ddot,
                                                     R_Q_W, om, geom_dt)

            # Call M3 yaw controller
            M3 = self.m3_control_yaw_decoupled(wc3, wc3_dot, b1c, R_Q_W, om,
                                               geom_dt)

            # Compute the thrust force f
            f = -np.dot(A, b3)
            # th_cmd_W = -A[2]
            # z_W = np.array([0, 0, 1])
            # zQ_Q = np.array([0, 0, 1])
            # zQ_W = R_Q_W .apply(zQ_Q)
            # alpha_curr = acos(np.minimum(z_W@zQ_W, 1.0))
            # f = th_cmd_W/cos(alpha_curr)

            # Call control allocation
            T_vec = np.array([f, M1, M2, M3])
            w_vec = self.control_allocation(t, T_vec)

        else:  # If not time, pass last w_vec setpoint through

            # Assign Tvec and w_vec from previous time
            T_vec = self.T_vec_last
            w_vec = self.w_vec_last

        # Save to previous command vectors
        self.T_vec_last = T_vec
        self.w_vec_last = w_vec

        # Save setpoints to vector
        num_states = 13
        setpoint_vec = np.zeros(num_states)  # Initialize state setpoint vector
        setpoint_vec[0:3] = self.pos_cmd  # Position
        setpoint_vec[3:7] = self.q_cmd
        setpoint_vec[7:10] = self.vel_cmd  # Linear velocity
        # setpoint_vec[10:] = np.zeros(3) # Angular velocity unused

        return w_vec  # Return w_vec, Tvec and setpoints not returned

    def thrust_control_yaw_decoupled(self, ex: np.ndarray, ev: np.ndarray,
                                     R_Q_W: Rot, om: np.ndarray, dt: float, xd_ddot=np.zeros(3),
                                     xd_dddot=np.zeros(3), xd_ddddot=np.zeros(3)) -> np.ndarray:
        """
        This method takes in the position-related error vectors from the UAV,
        computes the desired theoretical force vector A, and uses that to
        return the desired z-axis b3d and rotor thrust f
        """

        # Divide out the needed controller gains and system constants
        kx = self.ctl_params.kx
        kv = self.ctl_params.kv
        ki = self.ctl_params.ki
        sigma = self.ctl_params.sigma
        c1 = self.ctl_params.c1
        m = self.quad_params.m
        g = consts.g

        ex = ex.astype(float)
        ev = ev.astype(float)

        # Divide out needed UAV states
        # x = state_uav[:3]
        # q = state_uav[3:7]
        # dx_dt = state_uav[7:10]
        # om = state_uav[10:]
        # R_Q_W = Rot.from_quat(q) # Quad to inertial rotation

        # Setup desired unit vectors
        # e1 = np.array([1, 0, 0])
        # e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        # b1 = R_Q_W.apply(e1)
        # b2 = R_Q_W.apply(e2)
        b3 = R_Q_W.apply(e3)
        b3_dot = R_Q_W.as_matrix() * self.hat(om) @ e3

        # Compute alternative UAV body-fixed integral components
        R_W_Q = R_Q_W.inv()
        ex_Q = R_W_Q.apply(ex)
        ev_Q = R_W_Q.apply(ev)
        ei_dot_Q = (ev_Q + np.multiply(c1, ex_Q))
        ei_Q = (dt * ei_dot_Q)

        self.ei += ei_Q
        np.clip(self.ei, -sigma, sigma, out=self.ei)

        ei_Q_sat = self.sat_dot(sigma, self.ei, ei_dot_Q)

        ei_W = R_Q_W.apply(self.ei)
        ei_dot_W = R_Q_W.apply(ei_Q_sat)

        # Compute the integral error term
        # self.ei += dt*(ev + np.multiply(c1, ex)).astype(float)
        # ei_dot = ev + np.multiply(c1, ex)
        # np.clip(self.ei, -sigma, sigma, out=self.ei)

        # Form the theoretical force vector A
        # A = -np.multiply(kx, ex) - np.multiply(kv, ev) - \
        #     np.multiply(ki, self.ei) - m*g*e3 + m*xd_ddot
        A = -np.multiply(kx, ex) - np.multiply(kv, ev) - \
            np.multiply(ki, ei_W) - m * g * e3 + m * xd_ddot

        # ex_z = np.array([0, 0, ex[2]])
        # ev_z = np.array([0, 0, ev[2]])
        # ei_z = np.array([0, 0, self.ei[2]])
        # A_z = -np.multiply(kx, ex_z) - np.multiply(kv, ev_z) - \
        #     np.multiply(ki, ei_z) - m*g*e3 + m*xd_ddot

        # Form A_dot
        f = -np.dot(A, b3)
        ea = g * e3 - (f / m) * b3 - xd_ddot  # + x_delta / m - x_delta unknown
        # A_dot = -kx*ev - kv*ea + m*xd_dddot - \
        #     ki*self.sat_dot(sigma, self.ei, ei_dot)
        A_dot = -kx * ev - kv * ea + m * xd_dddot - ki * ei_dot_W

        # Form A_ddot
        ei_ddot = ea + np.multiply(c1, ev)
        f_dot = -np.dot(A_dot, b3) - np.dot(A, b3_dot)
        eb = -(f_dot / m) * b3 - (f / m) * b3_dot - xd_dddot
        A_ddot = -kx * ea - kv * eb + m * xd_ddddot - \
                 ki * self.sat_dot(sigma, self.ei, ei_ddot)

        return A, A_dot, A_ddot

    def m1m2_control_yaw_decoupled(self, b3d: np.ndarray, b3d_dot: np.ndarray,
                                   b3d_ddot: np.ndarray, R_Q_W: Rot, om: np.ndarray,
                                   dt: float) -> np.ndarray:
        """
        This method takes current UAV attitude state information, along with
        the desired z-axis b3d and computes the required pitch and roll
        moments M1 and M2
        """

        # Divide out the needed controller gains and system constants
        kb = self.ctl_params.kb
        kw = self.ctl_params.kw
        kI = self.ctl_params.kI
        c2 = self.ctl_params.c2
        sigma2 = self.ctl_params.sigma2
        J1 = self.quad_params.I[0, 0]
        J3 = self.quad_params.I[2, 2]

        # Setup desired unit vectors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        b1 = R_Q_W.apply(e1)
        b2 = R_Q_W.apply(e2)
        b3 = R_Q_W.apply(e3)

        # Compute the needed angular velocity terms
        w12 = om[0] * b1 + om[1] * b2
        w12d = self.hat(b3d) @ b3d_dot
        w12d_dot = self.hat(b3d) @ b3d_ddot
        b3_dot = self.hat(w12) @ b3

        # Compute the attitude error terms
        eb = np.cross(b3d, b3)
        ew = w12 + (self.hat(b3) @ self.hat(b3)) @ w12d

        # Compute the integral attitude error terms
        self.eI1 += dt * np.dot(ew + c2 * eb, b1)
        self.eI2 += dt * np.dot(ew + c2 * eb, b2)
        self.eI1 = max(min(self.eI1, sigma2), -sigma2)
        self.eI2 = max(min(self.eI2, sigma2), -sigma2)
        # np.clip(self.eI1, -sigma2, sigma2, out=self.eI1)
        # np.clip(self.eI2, -sigma2, sigma2, out=self.eI2)

        # Compute the overall control moment tau
        tau = -np.multiply(kb, eb) - np.multiply(kw, ew) - kI * self.eI1 * b1 - \
              - kI * self.eI2 * b2 - J1 * np.dot(b3, w12d) * b3_dot - \
              J1 * (self.hat(b3) @ self.hat(b3)) @ w12d_dot

        # Convert tau into M1 and M2
        tau1 = np.dot(tau, b1)
        tau2 = np.dot(tau, b2)

        M1 = tau1 + J3 * om[1] * om[2]
        M2 = tau2 - J3 * om[0] * om[2]

        return M1, M2

    def m3_control_yaw_decoupled(self, wc3: float, wc3_dot: float,
                                 b1c: np.ndarray, R_Q_W: Rot, om: np.ndarray, dt: float) -> np.ndarray:
        """
        This method takes the current UAV attitude along with a desired heading
        vector, and computes the required yaw moment M3
        """

        # Divide out the needed controller gains and system constants
        ky = self.ctl_params.ky
        kwy = self.ctl_params.kwy
        kIy = self.ctl_params.kIy
        c3 = self.ctl_params.c3
        J3 = self.quad_params.I[2, 2]

        # Setup desired unit vectors
        # e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        # e3 = np.array([0, 0, 1])
        # b1 = R_Q_W.apply(e1)
        b2 = R_Q_W.apply(e2)
        # b3 = R_Q_W.apply(e3)

        # Compute the yaw error terms
        ey = -np.dot(b2, b1c)
        ewy = om[2] - wc3

        # Compute the yaw integral error term
        self.eIy += dt * (ewy + c3 * ey)

        # Compute the moment M3
        M3 = -ky * ey - kwy * ewy - kIy * self.eIy + J3 * wc3_dot

        return M3

    def control_allocation(self, t: float, T_vec: np.ndarray) -> np.ndarray:
        """
        Method to be called to compute control allocation for the uav. Outputs
        a vector of rotor velocities required for a desired vector of torques
        and overall thrust

        The desired vector is given: T_vec = [T_tot M1 M2 M3]

        The output vector is given: w_vec = [w1 w2 w3 w4]
        """

        w_vec_sq = self.Gamma_inv @ T_vec  # Control allocation

        w_vec_sq[w_vec_sq < 0] = 0.0  # Apply minimum bounds on squared

        w_vec = np.sqrt(w_vec_sq)  # Account for magnitude squared

        w_vec[1] *= -1  # Correct spin of propellers 2 and 4
        w_vec[3] *= -1

        # Add min and max saturation for motors
        for i in range(4):

            # Maximum saturation
            if abs(w_vec[i]) > self.quad_params.w_max:
                warn_msg = ("[Ctl-control_allocation] t = %0.3f: " \
                            "Motor %i maximum saturation") % (t, i)
                # warnings.warn(warn_msg)
                w_vec[i] = self.quad_params.w_dir[i] * self.quad_params.w_max
                w_vec[i] *= 0.999

            # Minimum saturation (small overage to avoid warnings)
            if abs(w_vec[i]) < self.quad_params.w_min:
                warn_msg = ("[Ctl-control_allocation] t = %0.3f: " \
                            "Motor %i minimum saturation") % (t, i)
                # warnings.warn(warn_msg)
                w_vec[i] = self.quad_params.w_dir[i] * self.quad_params.w_min
                w_vec[i] *= 1.001

        return w_vec

    def reset_ctl(self) -> np.ndarray:
        """
        Method to resent controller parameters for repeated use in batch
        simulation runs
        """

        # Reset variables for storing previous state information
        self.w_vec_last = self.quad_params.w_min * np.array([1, -1, 1, -1])
        self.T_vec_last = np.zeros(4)  # Thrust and torque commands

        # Rest setpoint variables
        self.pos_cmd = np.zeros(3)  # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3)  # Linear velocity setpoint in {A}
        self.q_cmd = np.array([0, 0, 0, 1])

        # Rest integral term variables
        self.ei = np.zeros(3)  # Position error integral
        self.eI1 = 0  # X-axis attitude error integral
        self.eI2 = 0  # Y-axis attitude error integral
        self.eIy = 0  # Yaw attitude error integral

        # Reset timing variables
        self.geom_prev_t = 0

    def hat(self, vec: np.ndarray) -> np.ndarray:
        """
        This method implements the hat operator from the T Lee papers. The hat
        operator simply turns a vector of length 3 into a 3x3 skew-symetric
        matrix
        """

        skew_sym_mat = np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]])

        return skew_sym_mat

    def vee(self, mat: np.ndarray) -> np.ndarray:
        """
        This method implements the vee operator from the Lee papers. It takes
        a skew-symetric matrix and turns it into a vector
        """

        vec = np.array([-mat[1, 2], mat[0, 2], -mat[0, 1]])

        return vec

    def deriv_unit_vec(self, A: np.ndarray, A_dot: np.ndarray,
                       A_ddot: np.ndarray) -> np.ndarray:
        """
        This method takes in a vector and its derivatives and gives the unit-
        vector derivatives as outputs
        """

        nA = LA.norm(A)

        b = A / nA
        b_dot = A_dot / nA - (A * np.dot(A, A_dot)) / (nA ** 3)
        b_ddot = A_ddot / nA - \
                 A_dot * ((2 * np.dot(A, A_dot)) / (nA ** 3)) - \
                 A * ((np.dot(A_dot, A_dot) + np.dot(A, A_dot)) / (nA ** 3)) + \
                 A * ((3 * np.dot(A, A_dot) ** 2) / (nA ** 5))

        return b.astype(float), b_dot.astype(float), b_ddot.astype(float)

    def b1c_gen(self, b2c_vec: np.ndarray, b3c_vec: np.ndarray) -> np.ndarray:
        """
        This method computes the commanded vectors and their derivatives for
        b1c from b2c and b3c vectors and derivatives
        """

        # Split out the input vectors
        b2c = b2c_vec[0]
        b2c_dot = b2c_vec[1]
        b2c_ddot = b2c_vec[2]

        b3c = b3c_vec[0]
        b3c_dot = b3c_vec[1]
        b3c_ddot = b3c_vec[2]

        # Compute the b1c terms
        b1c = self.hat(b2c) @ b3c
        b1c_dot = self.hat(b2c_dot) @ b3c + self.hat(b2c) @ b3c_dot
        b1c_ddot = self.hat(b2c_ddot) @ b3c + 2 * self.hat(b2c_dot) @ b3c_dot + \
                   self.hat(b2c) @ b3c_ddot

        return b1c, b1c_dot, b1c_ddot

    def sat_dot(self, sigma: np.ndarray, y: np.ndarray,
                y_dot: np.ndarray) -> np.ndarray:
        """
        Implementation of the satdot function from the Lee papers with a vector
        sigma
        """

        z = np.zeros(len(y))

        for i in range(len(y)):
            if (y[i] < sigma[i]) and (y[i] > -sigma[i]):
                z[i] = y_dot[i]

        return z

class PathFollower(Controller):
    """
    Placeholder
    """
    def __init__(self, initial_setpoint: dict, path: np.ndarray, gains: np.ndarray, t_go: float, dt: float):
        """
        Constructor to initialize needed terms for the controller class

        The setpoint dict for this subclass is defined as:
            _setpoint:
                'pos': np vec [x, y, z]
        """
        super().__init__(initial_setpoint)
        self.path = path
        self.g = 9.81
        self.dt = dt
        self.T, self.N, self.B, self.k, self.tau = self.calculate_frenet_frame()
        self.arc_length = self.calculate_arc_length()

        # Compute path speed based on time to go and arc length
        self.s_dot = self.arc_length / t_go

        self.k1, self.k2, self.k3 = gains

        # Assemble the gain matrix, K
        self.K = np.array([[gains[0], 0, 0], [0, gains[1], 0], [0, 0, gains[2]]])
        self.t_go = t_go

    def update(self, t: float, state: np.ndarray):
        """
        The primary method call for the path following controller

        Required Inputs:
            t: float, timestamp for this control call
            state: 1d numpy vector of the quad states
                state vector: x, y, z, psi
        """
        vel = self.compute_kinematic_control(state[0:3]).T[0]
        vel_b = copy.deepcopy(vel)

        # Transform inertial to body-fixed
        vel_b[0] = vel[1]
        vel_b[1] = vel[0]
        vel_b[2] = -vel[2]

        # Speed calculation
        V = np.linalg.norm(vel)

        # Commanded pitch calculation
        theta_c = np.arcsin(-vel_b[2]/V)

        # Commanded roll calculation
        psi = wrap(state[3], np.pi)
        psi_s = wrap(np.arctan2(vel_b[1], vel_b[0]), np.pi)
        e_psi = psi_s - psi
        if e_psi > np.pi:
            e_psi += -np.pi
        psi_dot = e_psi / self.dt
        phi_c = np.arctan2(psi_dot*V, self.g)

        return V, theta_c, phi_c

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
        k = numerator_norm / drdu_norm ** 3

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

    def compute_affine_system(self):
        self.G = self.S.T

        d1, d2, d3 = self.d

        k = self.k[self.closest_idx]
        tau = self.tau[self.closest_idx]

        fx = 1 - k * d2
        fy = k * d1 - tau * d3
        fz = tau * d2

        self.F = -self.s_dot * np.row_stack((fx, fy, fz))

class PathFollowerDyn(Controller):
    """
    Placeholder
    """
    def __init__(self, initial_setpoint: dict, path: np.ndarray, gains: np.ndarray, t_go: float, dt: float):
        """
        Constructor to initialize needed terms for the controller class

        The setpoint dict for this subclass is defined as:
            _setpoint:
                'pos': np vec [x, y, z]
        """
        super().__init__(initial_setpoint)
        self.path = path
        
        self.g = PhysConst.g
        self.dt = dt
        self.T, self.N, self.B, self.k, self.tau = self.calculate_frenet_frame()
        self.arc_length = self.calculate_arc_length()

        self.obstacles = self._setpoint["obstacles"]
        self.num_obstacles = len(self.obstacles)

        # Compute path speed based on time to go and arc length
        self.s_dot = self.arc_length / t_go

        self.k1, self.k2, self.k3 = gains

        # Assemble the gain matrix, K
        self.K = np.array([[gains[0], 0, 0], [0, gains[1], 0], [0, 0, gains[2]]])
        self.t_go = t_go
        self.time = 0.0

        kp = 500.0
        ki = 0.0
        kd = 50.0
        Ts = dt
        sigma = 0.05
        upper_limit = 1e6
        lower_limit = -1e6
        anti_windup = "error_dot"
        anti_windup_limit = 100
        
        self.control_x = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
                                                anti_windup_limit)
        self.control_y = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
                                                anti_windup_limit)
        self.control_z = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
                                                anti_windup_limit)

        # self.control_x = pid_control.PControl(kp, 1e9, -1e9)
        # self.control_y = pid_control.PControl(kp, 1e9, -1e9)
        # self.control_z = pid_control.PControl(kp, 1e9, -1e9)
        
        """CBF Stuff V1"""
        # # Define variables
        # x1, x2, x3, x4, x5, x6, xc, yc, zc, r, k1, k2 = symbols('x1 x2 x3 x4 x5 x6 xc yc zc r k1 k2')
        # ux, uy, uz = symbols('ux uy uz')
        # # Define h(x) function
        # h = sqrt((x1 - xc)**2 + (x2 - yc)**2 + (x3 - zc)**2) - r
        # # Define f(x) and g(x) as matrices
        # f = Matrix([x4, x5, x6, 0, 0, 0])
        # g = Matrix([0, 0, 0, 1, 1, 1])
        # # Compute Lf(h) (Lie derivative of h along f)
        # Lf_h = sum(diff(h, x_i) * f[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))
        # # Compute Lgh (lie derivative of h along g)
        # Lg_h = sum(diff(h, x_i) * g[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))
        # # Compute Lf2(h) (second Lie derivative along f)
        # Lf2_h = sum(diff(Lf_h, x_i) * f[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))
        # # Compute LgLf(h) (Lie derivative of Lf(h) along g)
        # Lg_Lf_h = sum(diff(Lf_h, x_i) * g[i] for i, x_i in enumerate([x1, x2, x3, x4, x5, x6]))


    def update(self, t: float, state: np.ndarray):
        """
        The primary method call for the path following controller

        Required Inputs:
            t: float, timestamp for this control call
            state: 1d numpy vector of the quad states
                state vector: x, y, z, psi
        """
        desired_vel = self.compute_kinematic_control(state[0:3])
        curr_vel = np.expand_dims(state[3:], axis=1)

        control_input_x = self.control_x.update(desired_vel[0], curr_vel[0]).item()
        control_input_y = self.control_y.update(desired_vel[1], curr_vel[1]).item()
        control_input_z = self.control_z.update(desired_vel[2], curr_vel[2]).item()

        control_input = (control_input_x, control_input_y, control_input_z)

        self.obstacles = self._setpoint["obstacles"]
        self.num_obstacles = len(self.obstacles)

        _h = np.empty((self.num_obstacles,))
        _g = np.empty((self.num_obstacles, 3))

        for idx, obstacle in enumerate(self.obstacles):
            xc = obstacle[0]
            yc = obstacle[1]
            zc = obstacle[2]
            r = obstacle[3]

            k1 = 2.1
            k2 = 5.0

            x1, x2, x3, x4, x5, x6 = state[0], state[1], state[2], state[3], state[4], state[5]
            # xc, yc, zc, r = 25, 0.0, 10.0, 5.0
            # xc, yc, zc, r = -25, 0., -25, 5.0
            h = self.h_func(x1, xc, x2, yc, x3, zc, r)
            f = self.f_func(x4, x5, x6)
            g = self.g_func()
            Lf_h = self.Lf_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)
            Lg_h = self.Lg_h_func()
            Lf2_h = self.Lf2_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)
            Lg_Lf_h = self.Lg_Lf_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)

            h_obs = np.array(Lf2_h + Lf_h*(k1+k2)+k1*k2*h)
            g_obs = -Lg_Lf_h
            _h[idx] = h_obs
            _g[idx, :] = g_obs

        # _h = np.array(Lf2_h + Lf_h*(k1+k2)+k1*k2*h)
        # _g = -Lg_Lf_h
        # _g = _g * np.ones((1, 3))

        p = 2*np.eye(3)
        q = -2*np.array(control_input).reshape(3,)

        u_filtered = solve_qp(p, q, _g, _h, solver="cvxopt")

        self.time += self.dt

        # return control_input
        return u_filtered

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
        k = numerator_norm / drdu_norm ** 3

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
        self.segment_lengths = np.linalg.norm(diffs, axis=1)

        # Sum up all segment lengths to get the total arc length
        arc_length = np.sum(self.segment_lengths)

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

        self.time_remaining = self.t_go - self.time
        dist_remaining = np.sum(self.segment_lengths[self.closest_idx:])
        self.s_dot = dist_remaining / self.time_remaining

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

    def compute_affine_system(self):
        self.G = self.S.T

        d1, d2, d3 = self.d

        k = self.k[self.closest_idx]
        tau = self.tau[self.closest_idx]

        fx = 1 - k * d2
        fy = k * d1 - tau * d3
        fz = tau * d2

        self.F = -self.s_dot * np.row_stack((fx, fy, fz))

class PathFollowerDyn2D(Controller):
    """
    Placeholder
    """
    def __init__(self, initial_setpoint: dict, path: np.ndarray, gains: np.ndarray, t_go: float, dt: float):
        """
        Constructor to initialize needed terms for the controller class

        The setpoint dict for this subclass is defined as:
            _setpoint:
                'pos': np vec [x, y, z]
        """
        super().__init__(initial_setpoint)
        self.path = path
        
        self.g = PhysConst.g
        self.dt = dt

        self.obstacles = self._setpoint["obstacles"]
        self.num_obstacles = len(self.obstacles)

        self.k1, self.k2, self.k3 = gains

        # Assemble the gain matrix, K
        self.K = np.array([[gains[0], 0, 0], [0, gains[1], 0], [0, 0, gains[2]]])
        self.t_go = t_go
        self.time = 0.0

        kp = 0.1
        ki = 0.0
        kd = 0.001
        Ts = dt
        sigma = 0.05
        upper_limit = 1e6
        lower_limit = -1e6
        anti_windup = "error_dot"
        anti_windup_limit = 100
        
        self.control_x = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
                                                anti_windup_limit)
        self.control_y = pid_control.PIDControl(kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup,
                                                anti_windup_limit)
        
        """CBF Stuff V2"""
        x1, x2, x3, y1, y2, y3, r, alpha, gamma = symbols('x1 x2 x3 y1 y2 y3 r alpha gamma')
        # Define h(x) functions
        h1 = sqrt((x1-x2)**2 + (y1-y2)**2) - r
        h2 = sqrt((x1-x3)**2 + (y1-y3)**2) - r
        h3 = sqrt((x2-x3)**2 + (y2-y3)**2) - r
        h_funcs = [h1, h2, h3]
        # Define f(x) and g(x) as matrices
        f = Matrix([0, 0])
        g = Matrix([
            [1, 0],  # Position derivatives (p1, p2, p3) do not depend on control inputs
            [0, 1]
        ])
        sum = 0
        for idx, cbf in enumerate(h_funcs):
            sum += sp.exp(-gamma*cbf)
        h = -(1/gamma)*sum
        state_vars1 = Matrix([x1, y1])
        state_vars2 = Matrix([x2, y2])
        state_vars3 = Matrix([x3, y3])
        
        grad_h1 = h.diff(state_vars1).T
        grad_h2 = h.diff(state_vars2).T
        grad_h3 = h.diff(state_vars3).T
        Lf1_h = grad_h1 * f
        Lg1_h = grad_h1 * g
        Lf2_h = grad_h2 * f
        Lg2_h = grad_h2 * g
        Lf3_h = grad_h3 * f
        Lg3_h = grad_h3 * g
        
        self.h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), h)
        self.f_func = lambdify((), f)
        self.g_func = lambdify((), g)
        self.Lf1_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lf1_h)
        self.Lf2_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lf2_h)
        self.Lf3_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lf3_h)
        self.Lg1_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lg1_h)
        self.Lg2_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lg2_h)
        self.Lg3_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lg3_h)

    def update(self, t: float, state: np.ndarray):
        """
        The primary method call for the path following controller

        Required Inputs:
            t: float, timestamp for this control call
            state: 1d numpy vector of the quad states
                state vector: x, y, z, psi
        """
        desired_pos = self.path[-1]
        curr_pos = state[:2]

        control_input_x = self.control_x.update(desired_pos[0], curr_pos[0]).item()
        control_input_y = self.control_y.update(desired_pos[1], curr_pos[1]).item()

        control_input = (control_input_x, control_input_y)

        # print(control_input)

        # self.obstacles = self._setpoint["obstacles"]
        # self.num_obstacles = len(self.obstacles)

        # _h = np.empty((self.num_obstacles,))
        # _g = np.empty((self.num_obstacles, 3))

        # for idx, obstacle in enumerate(self.obstacles):
        #     xc = obstacle[0]
        #     yc = obstacle[1]
        #     zc = obstacle[2]
        #     r = obstacle[3]

        #     k1 = 2.1
        #     k2 = 5.0

        #     x1, x2, x3, x4, x5, x6 = state[0], state[1], state[2], state[3], state[4], state[5]
        #     # xc, yc, zc, r = 25, 0.0, 10.0, 5.0
        #     # xc, yc, zc, r = -25, 0., -25, 5.0
        #     h = self.h_func(x1, xc, x2, yc, x3, zc, r)
        #     f = self.f_func(x4, x5, x6)
        #     g = self.g_func()
        #     Lf_h = self.Lf_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)
        #     Lg_h = self.Lg_h_func()
        #     Lf2_h = self.Lf2_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)
        #     Lg_Lf_h = self.Lg_Lf_h_func(x1, xc, x2, yc, x3, zc, x4, x5, x6)

        #     h_obs = np.array(Lf2_h + Lf_h*(k1+k2)+k1*k2*h)
        #     g_obs = -Lg_Lf_h
        #     _h[idx] = h_obs
        #     _g[idx, :] = g_obs

        # # _h = np.array(Lf2_h + Lf_h*(k1+k2)+k1*k2*h)
        # # _g = -Lg_Lf_h
        # # _g = _g * np.ones((1, 3))

        # p = 2*np.eye(3)
        # q = -2*np.array(control_input).reshape(3,)

        # u_filtered = solve_qp(p, q, _g, _h, solver="cvxopt")

        # self.time += self.dt

        # return control_input
        return control_input[:2]