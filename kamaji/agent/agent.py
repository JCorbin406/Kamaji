# agent.py
from typing import Callable, List, Optional, Tuple

import numpy as np
from numba import njit
from qpsolvers import solve_qp
from sympy import Matrix, diff, init_printing, simplify, sqrt, symbols

import kamaji.tools.ode_solvers as ode  # Importing the RK4_step function from the external file
# from kamaji.controllers.path_follower import PathFollower
from kamaji.controllers.controllers import *
from kamaji.dynamics.dynamics import *


class Agent:
    def __init__(self, initial_state: np.ndarray, dynamics_model: str, control_model: Optional[str] = None,
                 path: Optional[list[dict]] = None, dt: Optional[float] = 0.01, **kwargs) -> None:
        """
        Initializes the Agent with a specified dynamics model and initial state.

        Args:
            initial_state (np.ndarray): The initial state of the agent.
            dynamics_model (str): A string corresponding to the desired dynamics model.
            control_model (Optional[str]): If specified, a string corresponding to the desired control model.
            path (Optional[str]): If specified, a list of the complete setpoints that define the path for the agent.
                The agent will start at the first setpoint and step to the last, remaining there if the sim continues.
            dt (Optional[float]): The step time for the simulation, in seconds. By default, 0.01 seconds.
        """
        # Assign kwargs to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dt = dt
        # Assign a dynamics model to the agent
        self.dynamics_model = Dynamics
        self.assign_dynamics(initial_state, dynamics_model)
        # Assign a controller and path to the agent
        self.control_model = None
        self.path = []
        self.path_index = 0
        self.assign_controller(control_model, path)
        # Initialize state and control history
        self._state_history = np.array([initial_state])
        self._control_history = np.array([0] * self.dynamics_model.num_control_inputs)

    def assign_dynamics(self, initial_state: np.ndarray, dynamics_model: str) -> None:
        """
        Assign a dynamics model to the agent based on a given string. 

        Args:
            initial_state (np.ndarray): The initial state of the agent. 
            dynamics_model (str): A string corresponding to the desired dynamics model.

        Raises:
            NotImplementedError: _description_
        """
        if dynamics_model == "unicycle":
            self.dynamics_model = Unicycle(initial_state)
        elif dynamics_model == "cruise":
            self.dynamics_model = CruiseControl(initial_state)
        elif dynamics_model == "Quad6DOF":
            self.dynamics_model = Quad6DOF(initial_state)
        elif dynamics_model == "Plane3DOF":
            self.dynamics_model = Plane3DOF(initial_state)
        elif dynamics_model == "Tilt6DOF":
            self.dynamics_model = Tiltrotor6DOF(initial_state)
        elif dynamics_model == "SingleIntegrator1DOF":
            self.dynamics_model = SingleIntegrator1DOF(initial_state)
        elif dynamics_model == "SingleIntegrator2DOF":
            self.dynamics_model = SingleIntegrator2DOF(initial_state)
        elif dynamics_model == "DoubleIntegrator1DOF":
            self.dynamics_model = DoubleIntegrator1DOF(initial_state)
        elif dynamics_model == "SingleIntegrator3DOF":
            self.dynamics_model = SingleIntegrator3DOF(initial_state)
        elif dynamics_model == "DoubleIntegrator2DOF":
            self.dynamics_model = DoubleIntegrator2DOF(initial_state)
        elif dynamics_model == "DoubleIntegrator3DOF":
            self.dynamics_model = DoubleIntegrator3DOF(initial_state)
        else:
            raise NotImplementedError(f"{dynamics_model} not a valid dynamics model option")

    def assign_controller(self, controller: str, path: list[dict]) -> None:
        """
        Assign a controller and path to the agent.

        Args:
            controller (str): A string corresponding to the desired controller.
            path (list[dict]): A list of setpoints for the controller to folow.

        Raises:
            NotImplementedError: _description_
        """
        if path is None and controller is None:
            self.control_model = None
        elif path is None:
            raise ValueError('At least an initial setpoint must be added to path with control model enabled.')
        else:
            self.path = path
            # If there is only one setpoint in the path, it is assumed that the path will be updated dynamically,
            # so the controller should always pull from the latest value.
            if len(self.path) == 1:
                self.path_index = -1
            # Assign controller
            if controller == "PID":
                pass
                # self.control_model = pid_control.PIDControl()
            elif controller == "Geometric":
                self.control_model = GeometricController(self.path[0])
            elif controller == "PathFollower":
                path = np.array([item['pos'] for item in path])
                self.control_model = PathFollower(self.path[0], path, self.gains, self.t_go, self.dt)
            elif controller == "PathFollowerDyn":
                path_points = path[0]["path"]
                gains = path[0]["gains"]
                t_go = path[0]["t_go"]
                obstacles = path[0]["obstacles"]
                self.control_model = PathFollowerDyn(path[0], path_points, gains, t_go, self.dt)
            elif controller == "PathFollowerDyn2D":
                path_points = path[0]["path"]
                gains = path[0]["gains"]
                t_go = path[0]["t_go"]
                obstacles = path[0]["obstacles"]
                self.control_model = PathFollowerDyn2D(path[0], path_points, gains, t_go, self.dt)
            else:
                raise NotImplementedError(f"{controller} not a valid controller option")

    def update_path(self, setpoint: dict) -> None:
        """
        Updates the path for the agent to follow. Appends the setpoint to the total path.

        Args:
            setpoint (dict): The setpoint to add to the path.
        """
        self.path.append(setpoint)

    def reset(self, initial_state: np.ndarray) -> None:
        """
        Resets the agent's state and clears the history.

        Args:
            initial_state (np.ndarray): The new initial state of the agent.
        """
        self.dynamics_model.set_state(initial_state)
        self._state_history = np.array([initial_state])  # Reset state history with new initial state
        # self._control_history = np.zeros((0, len(initial_state)))  # Reset control history

    def control_step(self, t: float, dt: float, update_path=True) -> None:
        """
        Manually calculates the next control input for the agent.

        Args:
            t (float): The current time of the simulation.
            dt (float): The time step size for the simulation.
        """
        # Update path setpoint in controller and get new control input
        if self.control_model is not None:
            self.control_model.set_setpoint(self.path[self.path_index])
            control_input = self.control_model.update(t, self.dynamics_model.state)
            if update_path:
                if self.path_index == -1 or self.path_index == len(self.path)-1:
                    self.path_index = -1
                else:
                    self.path_index += 1
        else:
            try:
                control_input = self.u
            except AttributeError:
                control_input = np.array([0] * self.dynamics_model.num_control_inputs)

        return control_input


    def step(self, t: float, dt: float, control_input) -> None:
        """
        Advances the state of the agent by one time step using RK4 integration.

        Args:
            t (float): The current time of the simulation.
            dt (float): The time step size for the simulation.
        """
        # Update path setpoint in controller and get new control input
        # control_input = self.control_step(t, dt, update_path=True)

        # Update to new state
        _, new_state = ode.rk4_step(self.dynamics_model.dynamics, t, self.dynamics_model.state, control_input, dt)
        self.dynamics_model.set_state(new_state)

        # Append new state and control input to state history
        self._state_history = np.vstack([self._state_history, new_state])  # Append new state
        self._control_history = np.vstack([self._control_history, control_input])  # Append control input

    def control_nom(self, x):
        K = 1000.0
        return K * (24 - x[1])

    def control_safe(self, x):
        # https://github.com/Berk-Tosun/cbf-cartpole/blob/master/acc.py
        u_nom = self.control_nom(x)

        m = 1650
        T_h = 1.8
        v_0 = 14
        c_d = 0.3
        g = 9.81

        alpha = 0.1

        f0 = 0.1
        f1 = 5
        f2 = 0.25

        F_r = f0 + f1 * x[1] + f2 * x[1] ** 2

        p = np.array([1.])
        q = np.array([-u_nom])

        term = 1 / m * (T_h + (x[1] - v_0) / (c_d * g))
        _g = np.array([term])
        _h = np.array([term * F_r + (v_0 - x[1]) + alpha * self.h(x)])

        u_filtered = solve_qp(p, q, _g, _h, solver="cvxopt")

        if u_filtered is not None:
            u_filtered = u_filtered.item()  # Ensure u_filtered is a scalar

        # cbf_t.append(cbf_cstr(x, u_filtered))
        # h_t.append(h(x))
        # u_nom_t.append(u_nom)
        # u_filtered_t.append(u_filtered)

        return np.array([u_filtered])

    def control_safe_di(self, x, u_nom):
        p = np.array([1.])
        q = np.array([-u_nom])

        u_filtered = solve_qp(p, q, _g, _h, solver="cvxopt")

        if u_filtered is not None:
            u_filtered = u_filtered.item()  # Ensure u_filtered is a scalar

        # cbf_t.append(cbf_cstr(x, u_filtered))
        # h_t.append(h(x))
        # u_nom_t.append(u_nom)
        # u_filtered_t.append(u_filtered)

        return np.array([u_filtered])

    def h(self, x):
        T_h = 1.8
        v_0 = 14
        c_d = 0.3
        g = 9.81
        return x[2] - T_h * x[1] - 1 / 2 * (x[1] - v_0) ** 2 / (c_d * g)

    @property
    def state(self) -> np.ndarray:
        """
        Get the current state of the agent.

        Returns:
            np.ndarray: The current state of the agent.
        """
        return self.dynamics_model.state

    @property
    def state_log(self) -> np.ndarray:
        """
        Returns the logged states of the agent.

        Returns:
            np.ndarray: A history of the agent's states over time.
        """
        return self._state_history

    @property
    def control_log(self) -> np.ndarray:
        """
        Returns the logged control inputs of the agent.

        Returns:
            np.ndarray: A history of the agent's control inputs over time.
        """
        return self._control_history
