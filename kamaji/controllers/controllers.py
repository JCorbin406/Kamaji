from abc import ABC, abstractmethod
import stable_baselines3 as sb3
import gymnasium as gym
from typing import Any
import numpy as np

class Controls(ABC):
    """Abstract base class for Kamaji controllers.

    Every controller in Kamaji inherits from this class and implements the
    ``update()`` method, which maps the current time and state to a control
    input vector.

    To create a custom controller, subclass ``Controls`` and implement
    ``update()``. The returned array must match the dimensionality expected
    by the dynamics model's control channel(s) that this controller is
    assigned to.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self, **kwargs: Any) -> np.ndarray:
        """Compute the control input.

        Args:
            **kwargs: Arguments required by the specific controller
                implementation (typically ``t`` and ``state_dict``).

        Returns:
            Control input vector.
        """
        pass

class PID(Controls):
    """Proportional-Integral-Derivative (PID) controller.

    Computes a control signal from the error between the current state and
    a goal setpoint. Includes anti-windup clamping on the integral term and
    exponential smoothing on the derivative term to reduce noise.

    The controller reads specific state variables by name from the agent's
    state dictionary, so it can target individual degrees of freedom (e.g.
    only ``position_x``).

    Args:
        state_names: Ordered list of state variable names to track.
        goals: Target values for each tracked state variable.
        Kp: Proportional gains (one per tracked variable).
        Ki: Integral gains.
        Kd: Derivative gains.
        dt: Controller time step (should match the simulation time step).
        lower_limit: Lower saturation limit for the control output (scalar,
            applied element-wise). ``None`` means no lower limit.
        upper_limit: Upper saturation limit for the control output. ``None``
            means no upper limit.
        derivative_smoothing: Exponential-moving-average coefficient for the
            derivative term (0 = no update, 1 = no smoothing). Default 0.1.

    Attributes:
        integral: Accumulated integral error (clamped to [-10, 10]).
        derivative: Smoothed derivative of the error.
    """

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
        """Compute the PID control output.

        Args:
            t: Current simulation time (unused, kept for interface consistency).
            state_dict: Agent state dictionary mapping variable names to values.

        Returns:
            Control vector: ``Kp * e + Ki * integral(e) + Kd * d(e)/dt``.
        """
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
        """Update the target setpoint at runtime.

        Args:
            new_goals: New goal values (one per tracked state variable).
        """
        self.goals = np.array(new_goals)


class Constant(Controls):
    """Controller that always returns a fixed control vector.

    Useful for open-loop testing, constant-velocity agents, or as a
    placeholder while developing other components.

    Args:
        control: The constant control vector to return on every call.
    """

    def __init__(self, control: np.ndarray):
        super().__init__()
        self.control = np.array(control)

    def update(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the constant control vector (state and time are ignored).

        Args:
            t: Current simulation time (unused).
            state: Current agent state (unused).

        Returns:
            The fixed control vector provided at construction.
        """
        return self.control

class SARLPolicy(Controls):
    """Single-Agent Reinforcement Learning (SARL) policy controller.

    Wraps a Stable-Baselines3 RL model so it can be used as a Kamaji
    controller. The lifecycle is:

    1. **Construction** — Specify which state variables the RL policy
       observes (``state_inputs``), which control channels it produces
       (``action_outputs``), and the SB3 algorithm name (``model_name``).
    2. **Model initialisation** — Call ``init_model(env)`` with a Gymnasium
       environment to instantiate the SB3 model. This is done automatically
       by ``Agent.assign_gym_env()``.
    3. **Inference** — ``update()`` feeds the current observation through
       the trained policy and returns the action.

    Supported SB3 algorithms:
        PPO, SAC, DDPG, A2C, TD3

    Args:
        state_inputs: Ordered list of state variable names used as the
            observation vector.
        action_outputs: Ordered list of control channel names produced
            by the policy.
        model_name: Name of the Stable-Baselines3 algorithm
            (e.g. ``"PPO"``, ``"SAC"``).
    """

    def __init__(self, state_inputs: list[str], action_outputs: list[str], model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = None

        self.state_inputs = state_inputs
        self.action_outputs = action_outputs

    def init_model(self, env: gym.Env, policy="MlpPolicy",  **kwargs) -> None:
        """Instantiate the SB3 model with the given Gymnasium environment.

        Args:
            env: A Gymnasium-compatible environment (typically a ``WrapperEnv``).
            policy: SB3 policy class name. Default ``"MlpPolicy"``.
            **kwargs: Additional keyword arguments forwarded to the SB3 model
                constructor (e.g. ``learning_rate``, ``n_steps``).

        Raises:
            ValueError: If ``model_name`` is not one of the supported algorithms.
            AssertionError: If the observation space dimension does not match
                the length of ``state_inputs``.
        """
        if self.model_name == "PPO":
            self.model = sb3.PPO(policy=policy, env=env, verbose=1, **kwargs)
        elif self.model_name == "SAC":
            self.model = sb3.SAC(policy=policy, env=env, verbose=1,**kwargs)
        elif self.model_name == "DDPG":
            self.model = sb3.DDPG(policy=policy, env=env, verbose=1,**kwargs)
        elif self.model_name == "A2C":
            self.model = sb3.A2C(policy=policy, env=env, verbose=1,**kwargs)
        elif self.model_name == "TD3":
            self.model = sb3.TD3(policy=policy, env=env, verbose=1, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")
        assert len(self.state_inputs) == self.model.observation_space.shape[0], \
            "State names length must match model's observation space dimension."

    def update(self, state: np.ndarray) -> np.ndarray:
        """Run the trained policy to produce a control action.

        Args:
            state: Observation vector (ordered according to ``state_inputs``).

        Returns:
            Action vector (ordered according to ``action_outputs``).

        Raises:
            RuntimeError: If ``init_model()`` has not been called yet.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() before using the controller.")
        action, _states = self.model.predict(state, deterministic=True)
        return action
