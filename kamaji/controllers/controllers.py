from abc import ABC, abstractmethod
import stable_baselines3 as sb3
import gymnasium as gym
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
    
class SARLPolicy(Controls):
    def __init__(self, state_inputs: list[str], action_outputs: list[str], model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = None

        self.state_inputs = state_inputs
        self.action_outputs = action_outputs

    def init_model(self, env: gym.Env, policy="MlpPolicy",  **kwargs) -> None:
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
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() before using the controller.")
        action, _states = self.model.predict(state, deterministic=True)
        return action