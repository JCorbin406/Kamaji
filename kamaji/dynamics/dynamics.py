from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as Rot

class Dynamics(ABC):
    def __init__(self, dt=0.01) -> None:
        self._dt = dt
        self._state = {}
        self._state_order = []

    @abstractmethod
    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def set_state(self, state_dict: dict) -> None:
        self._state = state_dict.copy()
        self._state_order = list(state_dict.keys())

    def get_state_vector(self) -> np.ndarray:
        return np.array([self._state[key] for key in self._state_order])

    def state_info(self) -> dict:
        return self._state.copy()

class CruiseControl(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs) -> np.ndarray:
        m = kwargs.get('m', 1650)
        v_0 = kwargs.get('v_0', 14)

        p = state_dict["p"]
        v = state_dict["v"]
        z = state_dict["z"]

        dp = v
        dv = -1 / m * self.F_r(v) + 1 / m * u.item()
        dz = v_0 - v

        return np.array([dp, dv, dz])

    def F_r(self, v) -> float:
        f0, f1, f2 = 0.1, 5, 0.25
        return f0 + f1 * v + f2 * v ** 2

class Unicycle(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        v, omega = u
        x = state_dict["x"]
        y = state_dict["y"]
        theta = state_dict["theta"]

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        return np.array([dx, dy, dtheta])

class SingleIntegrator1DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        velocity = u[0]
        return np.array([velocity])

class SingleIntegrator2DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        vx, vy = u
        return np.array([vx, vy])

class SingleIntegrator3DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        vx, vy, vz = u
        return np.array([vx, vy, vz])

class DoubleIntegrator1DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        x = state_dict["x"]
        vx = state_dict["vx"]
        ax = u[0]
        return np.array([vx, ax])

class DoubleIntegrator2DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        x = state_dict["x"]
        y = state_dict["y"]
        vx = state_dict["vx"]
        vy = state_dict["vy"]
        ax, ay = u
        return np.array([vx, vy, ax, ay])

class DoubleIntegrator3DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        x = state_dict["x"]
        y = state_dict["y"]
        z = state_dict["z"]
        vx = state_dict["vx"]
        vy = state_dict["vy"]
        vz = state_dict["vz"]
        ax, ay, az = u
        return np.array([vx, vy, vz, ax, ay, az])
