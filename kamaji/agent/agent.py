# agent.py
from typing import Callable, List, Optional, Tuple
import pandas as pd

import numpy as np
from numba import njit
from qpsolvers import solve_qp
from sympy import Matrix, diff, init_printing, simplify, sqrt, symbols

import kamaji.tools.ode_solvers as ode  # Importing the RK4_step function from the external file
# from kamaji.controllers.path_follower import PathFollower
from kamaji.controllers.controllers_old import *
from kamaji.dynamics.dynamics import *
from kamaji.controllers.controllers import *

class Agent:
    def __init__(self, agent_config, t = 0.0, dt = 0.01, **kwargs):
        # Assign kwargs to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Assign configuration parameters
        self._agent_config = agent_config

        # Set the simulation timestep for the dynamics updates
        self._dt = dt
        # Assign the agent's ID number
        self._id = agent_config['id']
        # Grab the list of state variable names
        self._state_list = list(agent_config['initial_state'].keys())
        # Set the current agent state as the initial states from the config
        self._state = agent_config['initial_state']
        # Create a dataframe to hold the state history - name the columns with the state variable names
        self._state_history = pd.DataFrame(columns=['time'] + self._state_list)
        # Assign the initial state to the state history
        self._state_history.loc[len(self._state_history)] = {'time': t} | self._state

        # Determine number of control inputs from controller config (assumes fixed-length control vector)
        num_controls = agent_config.get('num_controls', None)
        if num_controls is None and 'controller' in agent_config:
            # If not explicitly defined, try to infer from controller spec
            controller_type = agent_config['controller']['type']
            if controller_type == 'PID':
                num_controls = len(agent_config['controller']['specs'])
            else:
                num_controls = 1  # fallback
        self._control_list = [f'u{i}' for i in range(num_controls)]
        # Initialize control history DataFrame
        self._control_history = pd.DataFrame(columns=['time'] + self._control_list)

        # Assign the agent's dynamics model and controller
        self.assign_dynamics()
        self.assign_controller()

    def assign_dynamics(self) -> None:
        """
        Assign a dynamics model to the agent based on a given string. 

        Args:
            initial_state (np.ndarray): The initial state of the agent. 
            dynamics_model (str): A string corresponding to the desired dynamics model.

        Raises:
            NotImplementedError: _description_
        """

        dynamics_model = self._agent_config['dynamics_model']

        if dynamics_model == "Unicycle":
            self.dynamics_model = Unicycle(self._dt)
        elif dynamics_model == "CruiseControl":
            self.dynamics_model = CruiseControl(self._dt)
        elif dynamics_model == "SingleIntegrator1DOF":
            self.dynamics_model = SingleIntegrator1DOF(self._dt)
        elif dynamics_model == "SingleIntegrator2DOF":
            self.dynamics_model = SingleIntegrator2DOF(self._dt)
        elif dynamics_model == "DoubleIntegrator1DOF":
            self.dynamics_model = DoubleIntegrator1DOF(self._dt)
        elif dynamics_model == "SingleIntegrator3DOF":
            self.dynamics_model = SingleIntegrator3DOF(self._dt)
        elif dynamics_model == "DoubleIntegrator2DOF":
            self.dynamics_model = DoubleIntegrator2DOF(self._dt)
        elif dynamics_model == "DoubleIntegrator3DOF":
            self.dynamics_model = DoubleIntegrator3DOF(self._dt)
        else:
            raise NotImplementedError(f"{dynamics_model} not a valid dynamics model option")

    def assign_controller(self) -> None:
        """
        Assign a controller to the agent.

        Args:
            controller_config

        Raises:
            NotImplementedError: _description_
        """

        controller = self._agent_config['controller']

        controller_type = controller['type']

        if controller_type == "PathFollower":
            try:
                self.control_model = PathFollower(self.path)
            except AttributeError:
                raise ValueError("PathFollower requires a path to be set.")
        elif controller_type == "PID":
            try:
                pid_specs = controller['specs']
                state_names = [s['state'] for s in pid_specs]
                goals = [s['goal'] for s in pid_specs]
                Kp = [s['gains']['Kp'] for s in pid_specs]
                Ki = [s['gains']['Ki'] for s in pid_specs]
                Kd = [s['gains']['Kd'] for s in pid_specs]
                self.control_model = PID(state_names, goals, Kp, Ki, Kd, dt=self._dt)
            except AttributeError:
                raise ValueError("PID requires gains to be set.")
        else:
            raise NotImplementedError(f"{controller} not a valid controller option")

    def compute_control(self, t) -> np.ndarray:
        """
        Compute the new control signal using the agent's controller.

        Args:
            t (float): The current time of the simulation.

        Returns:
            np.ndarray: The computed control input.
        """
        if not hasattr(self, 'control_model'):
            raise AttributeError("Agent does not have a controller assigned.")

        current_state = self._state

        control_input = self.control_model.update(t, current_state)

        return control_input
    
    def compute_dynamics(self, t, control_input) -> np.ndarray:
        """
        Compute the new state of the agent using the dynamics model.

        Returns:
            np.ndarray: The new state of the agent.
        """
        if not hasattr(self, 'dynamics_model'):
            raise AttributeError("Agent does not have a dynamics model assigned.")

        current_state = self._state

        # Update the state using the dynamics model
        deriv = self.dynamics_model.dynamics(t, current_state, control_input)

        return deriv

    def step(self, t: float, control_input: np.ndarray) -> None:
        """
        Advances the state of the agent by one time step using RK4 integration.

        Args:
            t (float): Current simulation time.
            control_input (np.ndarray): Control input to apply at this timestep.
        """
        # Store state keys in order for conversion
        self._state_order = list(self._state.keys())
        
        # Convert current state dict to vector
        state_vec = np.array([self._state[key] for key in self._state_order])

        # Define a local wrapper function compatible with rk4_step
        def compute_dynamics(t_local: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
            state_dict = {key: y[i] for i, key in enumerate(self._state_order)}
            return self.dynamics_model.dynamics(t_local, state_dict, u)

        # RK4 integration step
        _, new_state_vec = ode.rk4_step(compute_dynamics, t, state_vec, control_input, self._dt)

        # Convert new state vector back to dict
        new_state_dict = {key: new_state_vec[i] for i, key in enumerate(self._state_order)}

        # Update internal state and history
        self._state = new_state_dict
        self._state_history.loc[len(self._state_history)] = {'time': t} | self._state

        # Log control input
        control_row = {'time': t}
        control_row.update({f'u{i}': control_input[i] for i in range(len(control_input))})
        self._control_history.loc[len(self._control_history)] = control_row


    def set_valuation(self, valuation_fn: Callable) -> None:
        self.valuation_fn = valuation_fn

    def set_marginal_valuation(self, marginal_valuation_fn: Callable) -> None:
        self.marginal_valuation_fn = marginal_valuation_fn

    def valuation(self, x):
        """Evaluate this agent's valuation at allocation x."""
        return self.valuation_fn(x)

    def marginal_valuation(self, x):
        """Evaluate this agent's marginal valuation at allocation x."""
        return self.marginal_valuation_fn(x)

    @property
    def state(self) -> np.ndarray:
        """
        Get the current state of the agent.

        Returns:
            np.ndarray: The current state of the agent.
        """
        return self._state

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
