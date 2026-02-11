from typing import Any, Callable, Optional
import pandas as pd
import numpy as np
import kamaji.tools.ode_solvers as ode
from kamaji.dynamics.dynamics import (
    Dynamics, Unicycle, CruiseControl,
    SingleIntegrator1DOF, SingleIntegrator2DOF, SingleIntegrator3DOF,
    DoubleIntegrator1DOF, DoubleIntegrator2DOF, DoubleIntegrator3DOF,
)
from kamaji.controllers.controllers import Controls, PID, Constant, SARLPolicy
import gymnasium as gym


class Agent:
    """A simulation agent that combines a dynamics model with one or more controllers.

    Each agent owns a dynamics model (defining its equations of motion), a
    set of per-channel controllers (one per control input), optional RL
    integration, and logging of state/control history. Agents are typically
    created by the ``Simulator`` from a YAML configuration dictionary, but
    can also be constructed programmatically.

    The control pipeline each time step is:

    1. Each controller channel computes its scalar (or vector) output.
    2. Channel outputs are concatenated into the full control vector ``u``.
    3. If a ``CBFSystem`` is attached to the simulator, ``u`` is filtered
       through the safety QP before being applied.
    4. The dynamics model is integrated forward one step (RK4 by default).
    5. State and control histories are logged.

    Args:
        agent_config: Configuration dictionary. Required keys:

            - ``id`` (str): Unique agent identifier.
            - ``initial_state`` (dict): Maps state variable names to initial values.
            - ``dynamics_model`` (str): Name of the dynamics class (e.g. ``"SingleIntegrator2DOF"``).
            - ``controller`` (dict): Per-channel controller specifications.

        t: Initial simulation time.
        dt: Integration time step.
        **kwargs: Additional attributes set directly on the agent instance
            (e.g. custom parameters for auction valuations).

    Attributes:
        dynamics_model: The ``Dynamics`` subclass instance.
        control_model: Dictionary mapping control channel names to controller
            objects or callables.
        trainable: ``True`` if the agent has an RL (SARL) controller.
        sarl_policy: The ``SARLPolicy`` instance, if any.
        radius: Collision radius used by the simulator's pairwise collision check.
        budget: Resource budget (used by auction mechanisms).
    """

    def __init__(self, agent_config: dict, t: float = 0.0, dt: float = 0.01, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.budget = 1
        self.radius = .25

        self._agent_config = agent_config
        self.manual_control_input = None
        self._dt = dt
        self._id = agent_config['id']
        self.trainable = False
        self.sarl_info = {}
        self.sarl_policy = None

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
        """Instantiate the dynamics model from the agent configuration.

        Reads ``self._agent_config['dynamics_model']`` (a string like
        ``"SingleIntegrator2DOF"``) and maps it to the corresponding
        ``Dynamics`` subclass.

        Raises:
            NotImplementedError: If the dynamics model name is not recognised.
        """
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
        """Parse the controller configuration and create per-channel controllers.

        Each key in the ``controller`` config block corresponds to a control
        channel (e.g. ``velocity_x``). The value specifies the controller
        ``type`` (``"PID"``, ``"Constant"``, or ``"SARLPolicy"``) and a
        single-item ``specs`` list with the controller parameters.

        If any channels use ``SARLPolicy``, all SARL channels are aggregated
        into a single ``SARLPolicy`` instance (one RL model per agent). The
        agent is then marked as ``trainable``.

        Raises:
            ValueError: If controller configuration is malformed or
                multiple different SARL model names are specified.
        """
        controller_cfg = self._agent_config.get("controller", {})
        if not isinstance(controller_cfg, dict):
            raise ValueError("Controller must be a dictionary of control channels with 'type' and 'specs'.")

        self.control_model = {}

        SARL_state_features = set()
        SARL_action_outputs = set()
        SARL_models = set()
        for ctrl_name, ctrl_data in controller_cfg.items():
            ctrl_type = ctrl_data["type"]
            specs = ctrl_data.get("specs", [])

            if not isinstance(specs, list) or len(specs) != 1:
                raise ValueError(f"[Agent: {self._id}] Controller '{ctrl_name}' must have a single-item specs list.")

            spec = specs[0]

            if ctrl_type == "Constant":
                if "value" not in spec:
                    raise ValueError(f"Constant controller for '{ctrl_name}' must specify 'value'")
                val = spec["value"]
                self.control_model[ctrl_name] = lambda t, state, v=val: v

            elif ctrl_type == "PID":
                required_keys = ["state", "goal", "kp", "ki", "kd"]
                if not all(k in spec for k in required_keys):
                    raise ValueError(f"PID controller for '{ctrl_name}' must contain {required_keys}")
                self.control_model[ctrl_name] = PID(
                    [spec["state"]],
                    [spec["goal"]],
                    [spec["kp"]],
                    [spec["ki"]],
                    [spec["kd"]],
                    dt=self._dt
                )
            elif ctrl_type == "SARLPolicy":
                required_keys = ["state_features", "model"]
                if not all(k in spec for k in required_keys):
                    raise ValueError(f"SARL controller for '{ctrl_name}' must contain {required_keys}")
                for feature in spec["state_features"]:
                    SARL_state_features.add(feature)
                SARL_models.add(spec["model"])
                SARL_action_outputs.add(ctrl_name)
            else:
                raise ValueError(f"Unknown controller type '{ctrl_type}' for {ctrl_name}")

        if len(SARL_models) > 0:
            if len(SARL_models) != 1:
                raise ValueError(f"Only one SARL model can be specified per agent. Found: {SARL_models}")
            self.sarl_policy = SARLPolicy(
                state_inputs=list(SARL_state_features),
                action_outputs=list(SARL_action_outputs),
                model_name=spec["model"]
            )
            for ctrl_name in SARL_action_outputs:
                self.control_model[ctrl_name] = self.sarl_policy
            self.trainable = True
            self.sarl_info = {
                "state_features": list(SARL_state_features),
                "action_outputs": list(SARL_action_outputs)
            }


    def assign_gym_env(self, env: gym.Env):
        """Attach a Gymnasium environment and initialise the SARL model.

        Args:
            env: A ``WrapperEnv`` instance for this agent.

        Raises:
            ValueError: If the agent has no SARL controller.
        """
        if self.sarl_policy is None:
            raise ValueError(f"[Agent: {self._id}] Cannot assign gym env without SARL controller. Either no SARL controller is defined, or controller has not yet been assigned.")
        self.sarl_policy.init_model(env)


    def compute_control(self, t: float, rl_action: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the full control vector by combining per-channel outputs.

        Iterates over control channels in sorted order, queries each
        controller, and concatenates the results. If an RL action is
        provided (during training), it overrides the SARL policy's own
        prediction for the corresponding channels.

        Args:
            t: Current simulation time.
            rl_action: Optional externally-provided RL action that overrides
                the SARL policy prediction (used during training).

        Returns:
            Control vector of shape ``(n_controls,)``.

        Raises:
            ValueError: If ``rl_action`` is provided but no SARL controller
                exists.
        """
        current_state = self._state

        # For backward compatibility, support legacy controller format
        if isinstance(self.control_model, dict):
            control_channels = sorted(self.control_model.keys())
            control_vector = []

            # Get SARL model if exists
            hasSARL = False
            SARL_model = None
            for channel in control_channels:
                if isinstance(self.control_model[channel], SARLPolicy):
                    hasSARL = True
                    SARL_model = self.control_model[channel]
                    break

            # predict SARL actions if needed
            if rl_action is not None and not hasSARL:
                raise ValueError("RL actions provided but no SARL controller found.")
            if hasSARL:
                SARL_state = np.array([current_state[feature] for feature in self.sarl_info["state_features"]])
                # Use provided rl_action rather than direct model prediction, allowing for proper handling of actions during training
                if rl_action is not None:
                    SARL_pred = rl_action
                else:
                    SARL_pred = SARL_model.update(SARL_state)
                SARL_control_channels = dict(zip(SARL_model.action_outputs, SARL_pred))

            for channel in control_channels:
                controller = self.control_model[channel]
                if isinstance(controller, SARLPolicy):
                    val = SARL_control_channels[channel]
                else:
                    val = controller.update(t, current_state) if hasattr(controller, "update") else controller(t, current_state)
                if isinstance(val, (list, tuple, np.ndarray)):
                    control_vector.extend(np.asarray(val).flatten())
                else:
                    control_vector.append(val)

            return np.array(control_vector)
        else:
            # Fallback for legacy single controller
            return self.control_model.update(t, current_state)


    def compute_dynamics(self, t: float, control_input: np.ndarray) -> np.ndarray:
        """Evaluate the dynamics at the current state (without stepping).

        Args:
            t: Current simulation time.
            control_input: Control vector.

        Returns:
            State derivative vector (dx/dt).
        """
        return self.dynamics_model.dynamics(t, self._state, control_input)

    def step(self, t: float, control_input: np.ndarray, log: bool = True) -> None:
        """Advance the agent one time step using RK4 integration.

        Integrates the dynamics model forward by ``dt`` seconds, updates the
        internal state dictionary, and (optionally) appends the new state and
        control input to the history DataFrames.

        Args:
            t: Current simulation time (before the step).
            control_input: Control vector to apply during this step.
            log: If ``True``, record the state and control to history.
        """
        self.dynamics_state_order = list(self.dynamics_model.state_variables())
        dynamics_state_vec = np.array([self._state[k] for k in self.dynamics_state_order])

        def compute_dynamics(t_local, y, u):
            return self.dynamics_model.dynamics(t_local, {k: y[i] for i, k in enumerate(self.dynamics_state_order)}, u)

        _, new_state_vec = ode.rk4_step(compute_dynamics, t, dynamics_state_vec, control_input, self._dt)
        dynamics_new_state_dict = {k: new_state_vec[i] for i, k in enumerate(self.dynamics_state_order)}
        new_state_dict = self._state.copy()
        new_state_dict.update(dynamics_new_state_dict)

        self._state = new_state_dict
        if log:
            self._state_history.loc[len(self._state_history)] = {'time': t} | self._state

        control_row = {'time': t}
        control_row.update({name: control_input[i] for i, name in enumerate(self._control_list)})
        if log:
            self._control_history.loc[len(self._control_history)] = control_row

    def set_valuation(self, fn: Callable): self.valuation_fn = fn
    def set_marginal_valuation(self, fn: Callable): self.marginal_valuation_fn = fn
    def valuation(self, x): return self.valuation_fn(x)
    def marginal_valuation(self, x): return self.marginal_valuation_fn(x)

    def reset(self, t=0.0):
        """Reset the agent to its initial state from the config."""
        self._state = self._agent_config['initial_state']


    @property
    def state(self): return self._state
    @property
    def state_log(self): return self._state_history
    @property
    def control_log(self): return self._control_history
