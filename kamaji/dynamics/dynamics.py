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

    @staticmethod
    @abstractmethod
    def state_variables() -> list[str]:
        pass

    @staticmethod
    @abstractmethod
    def control_variables() -> list[str]:
        pass

    def set_state(self, state_dict: dict) -> None:
        self._state = state_dict.copy()
        self._state_order = list(state_dict.keys())

    def get_state_vector(self) -> np.ndarray:
        return np.array([self._state[key] for key in self._state_order])

    def state_info(self) -> dict:
        return self._state.copy()
    
    def control_dimension(self):
        return len(self.control_variables())

class CruiseControl(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        m = kwargs.get('m', 1650)
        v_0 = kwargs.get('v_0', 14)

        p = state_dict["position"]
        v = state_dict["velocity"]
        z = state_dict["z"]

        dp = v
        dv = -1 / m * self.F_r(v) + 1 / m * u.item()
        dz = v_0 - v

        return np.array([dp, dv, dz])

    def F_r(self, v) -> float:
        f0, f1, f2 = 0.1, 5, 0.25
        return f0 + f1 * v + f2 * v ** 2

    @staticmethod
    def state_variables():
        return ["position", "velocity", "z"]
    
    @staticmethod
    def control_variables():
        return ["throttle_force"]


class Unicycle(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        v, omega = u
        x = state_dict["position_x"]
        y = state_dict["position_y"]
        theta = state_dict["theta"]

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        return np.array([dx, dy, dtheta])

    @staticmethod
    def state_variables():
        return ["position_x", "position_y", "theta"]
    
    @staticmethod
    def control_variables():
        return ["velocity", "angular_velocity"]


class SingleIntegrator1DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        velocity = u[0]
        return np.array([velocity])

    @staticmethod
    def state_variables():
        return ["position_x"]
    
    @staticmethod
    def control_variables():
        return ["velocity_x"]


class SingleIntegrator2DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        vx, vy = u
        return np.array([vx, vy])

    @staticmethod
    def state_variables():
        return ["position_x", "position_y"]

    @staticmethod
    def control_variables():
        return ["velocity_x", "velocity_y"]


class SingleIntegrator3DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        vx, vy, vz = u
        return np.array([vx, vy, vz])

    @staticmethod
    def state_variables():
        return ["position_x", "position_y", "position_z"]

    @staticmethod
    def control_variables():
        return ["velocity_x", "velocity_y", "velocity_z"]


class DoubleIntegrator1DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        x = state_dict["position_x"]
        vx = state_dict["velocity_x"]
        ax = u[0]
        return np.array([vx, ax])

    @staticmethod
    def state_variables():
        return ["position_x", "velocity_x"]

    @staticmethod
    def control_variables():
        return ["acceleration_x"]


class DoubleIntegrator2DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        x = state_dict["position_x"]
        y = state_dict["position_y"]
        vx = state_dict["velocity_x"]
        vy = state_dict["velocity_y"]
        ax, ay = u
        return np.array([vx, vy, ax, ay])

    @staticmethod
    def state_variables():
        return ["position_x", "position_y", "velocity_x", "velocity_y"]

    @staticmethod
    def control_variables():
        return ["acceleration_x", "acceleration_y"]


class DoubleIntegrator3DOF(Dynamics):
    def __init__(self, dt=0.01) -> None:
        super().__init__(dt)

    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        x = state_dict["position_x"]
        y = state_dict["position_y"]
        z = state_dict["position_z"]
        vx = state_dict["velocity_x"]
        vy = state_dict["velocity_y"]
        vz = state_dict["velocity_z"]
        ax, ay, az = u
        return np.array([vx, vy, vz, ax, ay, az])

    @staticmethod
    def state_variables():
        return [
            "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z"
        ]

    @staticmethod
    def control_variables():
        return ["acceleration_x", "acceleration_y", "acceleration_z"]
