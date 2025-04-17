from typing import Callable
import pandas as pd
import numpy as np

import kamaji.tools.ode_solvers as ode
from kamaji.dynamics.dynamics import *
from kamaji.controllers.controllers import *


class Agent:
    def __init__(self, agent_config, t=0.0, dt=0.01, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._agent_config = agent_config
        self.manual_control_input = None
        self._dt = dt
        self._id = agent_config['id']

        self._state = agent_config['initial_state']
        self._state_list = list(self._state.keys())
        self._state_history = pd.DataFrame(columns=['time'] + self._state_list)
        self._state_history.loc[len(self._state_history)] = {'time': t} | self._state

        self.assign_dynamics()
        self.assign_controller()

        self._control_list = self.dynamics_model.control_variables()
        self._control_history = pd.DataFrame(columns=['time'] + self._control_list)

        # Validate control alignment
        required_controls = len(self._control_list)
        if hasattr(self, 'control_model') and isinstance(self.control_model, dict):
            provided = len(self.control_model)
            if required_controls != provided:
                raise ValueError(
                    f"[Agent: {self._id}] Control mismatch: dynamics model expects {required_controls} controls "
                    f"but controller provides {provided}."
                )

    def assign_dynamics(self):
        model_map = {
            "Unicycle": Unicycle,
            "CruiseControl": CruiseControl,
            "SingleIntegrator1DOF": SingleIntegrator1DOF,
            "SingleIntegrator2DOF": SingleIntegrator2DOF,
            "SingleIntegrator3DOF": SingleIntegrator3DOF,
            "DoubleIntegrator1DOF": DoubleIntegrator1DOF,
            "DoubleIntegrator2DOF": DoubleIntegrator2DOF,
            "DoubleIntegrator3DOF": DoubleIntegrator3DOF,
        }
        model_name = self._agent_config['dynamics_model']
        if model_name not in model_map:
            raise NotImplementedError(f"{model_name} is not a valid dynamics model.")
        self.dynamics_model = model_map[model_name](self._dt)

    def assign_controller(self):
        controller_cfg = self._agent_config.get("controller", {})
        if not isinstance(controller_cfg, dict):
            raise ValueError("Controller must be a dictionary of control channels with 'type' fields.")

        self.control_model = {}

        for ctrl_name, ctrl_data in controller_cfg.items():
            ctrl_type = ctrl_data["type"]
            if ctrl_type == "Constant":
                val = ctrl_data["value"]
                self.control_model[ctrl_name] = lambda t, state, v=val: v
            elif ctrl_type == "PID":
                spec = ctrl_data.get("specs", [])[0]
                self.control_model[ctrl_name] = PID(
                    [spec["state"]],
                    [spec["goal"]],
                    [spec["kp"]],
                    [spec["ki"]],
                    [spec["kd"]],
                    dt=self._dt
                )
            else:
                raise ValueError(f"Unknown controller type '{ctrl_type}' for {ctrl_name}")

    def compute_control(self, t) -> np.ndarray:
        """
        Compute the full control vector by combining per-channel control outputs.

        Args:
            t (float): Current simulation time.

        Returns:
            np.ndarray: Control vector of shape (n_controls,)
        """
        current_state = self._state

        # For backward compatibility, support legacy controller format
        if isinstance(self.control_model, dict):
            control_channels = sorted(self.control_model.keys())
            control_vector = []

            for channel in control_channels:
                controller = self.control_model[channel]
                val = controller.update(t, current_state) if hasattr(controller, "update") else controller(t, current_state)
                if isinstance(val, (list, tuple, np.ndarray)):
                    control_vector.extend(np.asarray(val).flatten())
                else:
                    control_vector.append(val)

            return np.array(control_vector)
        else:
            # Fallback for legacy single controller
            return self.control_model.update(t, current_state)


    def compute_dynamics(self, t, control_input: np.ndarray) -> np.ndarray:
        return self.dynamics_model.dynamics(t, self._state, control_input)

    def step(self, t: float, control_input: np.ndarray) -> None:
        self._state_order = list(self._state.keys())
        state_vec = np.array([self._state[k] for k in self._state_order])

        def compute_dynamics(t_local, y, u):
            return self.dynamics_model.dynamics(t_local, {k: y[i] for i, k in enumerate(self._state_order)}, u)

        _, new_state_vec = ode.rk4_step(compute_dynamics, t, state_vec, control_input, self._dt)
        new_state_dict = {k: new_state_vec[i] for i, k in enumerate(self._state_order)}

        self._state = new_state_dict
        self._state_history.loc[len(self._state_history)] = {'time': t} | self._state

        control_row = {'time': t}
        control_row.update({name: control_input[i] for i, name in enumerate(self._control_list)})
        self._control_history.loc[len(self._control_history)] = control_row

    def set_valuation(self, fn: Callable): self.valuation_fn = fn
    def set_marginal_valuation(self, fn: Callable): self.marginal_valuation_fn = fn
    def valuation(self, x): return self.valuation_fn(x)
    def marginal_valuation(self, x): return self.marginal_valuation_fn(x)

    @property
    def state(self): return self._state
    @property
    def state_log(self): return self._state_history
    @property
    def control_log(self): return self._control_history
