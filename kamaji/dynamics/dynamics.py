from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as Rot


class Dynamics(ABC):
    """Abstract base class for agent dynamics models.

    All dynamics models in Kamaji inherit from this class. Subclasses must
    implement ``dynamics()``, ``state_variables()``, and ``control_variables()``
    to define the equations of motion, the state vector layout, and the control
    input layout, respectively.

    To create a custom dynamics model, subclass ``Dynamics`` and implement
    the three abstract methods. The ``dynamics()`` method should return the
    time derivative of the full state vector as a NumPy array whose ordering
    matches ``state_variables()``.

    Args:
        dt: Integration time step in seconds.

    Attributes:
        _dt: Integration time step.
        _state: Current state as a ``{name: value}`` dictionary.
        _state_order: Ordered list of state variable names.
    """

    def __init__(self, dt=0.01) -> None:
        self._dt = dt
        self._state = {}
        self._state_order = []

    @abstractmethod
    def dynamics(self, t: float, state_dict: dict, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the state derivative given the current state and control.

        Args:
            t: Current simulation time.
            state_dict: Dictionary mapping state variable names to current values.
            u: Control input vector, ordered according to ``control_variables()``.
            **kwargs: Additional model-specific parameters (e.g. mass, reference speed).

        Returns:
            State derivative vector (dx/dt), ordered according to ``state_variables()``.
        """
        pass

    @staticmethod
    @abstractmethod
    def state_variables() -> list[str]:
        """Return the ordered list of state variable names.

        The ordering defines the layout of the state vector used by
        ``dynamics()`` and ``get_state_vector()``.
        """
        pass

    @staticmethod
    @abstractmethod
    def control_variables() -> list[str]:
        """Return the ordered list of control input names.

        The ordering defines the layout of the control vector ``u`` passed
        to ``dynamics()``.
        """
        pass

    def set_state(self, state_dict: dict) -> None:
        """Set the internal state from a dictionary.

        Args:
            state_dict: Dictionary mapping state variable names to values.
        """
        self._state = state_dict.copy()
        self._state_order = list(state_dict.keys())

    def get_state_vector(self) -> np.ndarray:
        """Return the current state as a flat NumPy array.

        The ordering matches ``_state_order`` (set by the most recent
        ``set_state()`` call).
        """
        return np.array([self._state[key] for key in self._state_order])

    def state_info(self) -> dict:
        """Return a copy of the current state dictionary."""
        return self._state.copy()

    def control_dimension(self):
        """Return the number of control inputs."""
        return len(self.control_variables())

class CruiseControl(Dynamics):
    """Longitudinal cruise-control dynamics (1-D vehicle model).

    Models a vehicle traveling along a single axis with aerodynamic drag.
    The state vector is ``[position, velocity, z]`` where *z* is the integral
    tracking error ``z = integral(v_0 - v) dt``.

    Equations of motion:

        dp/dt = v
        dv/dt = -F_r(v)/m + u/m
        dz/dt = v_0 - v

    where ``F_r(v) = f0 + f1*v + f2*v^2`` is the resistive force
    (default coefficients: f0=0.1, f1=5, f2=0.25).

    Keyword Args (passed via ``**kwargs`` to ``dynamics()``):
        m: Vehicle mass in kg (default 1650).
        v_0: Reference (desired) velocity in m/s (default 14).

    State variables:
        position, velocity, z

    Control variables:
        throttle_force
    """

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
        """Compute the resistive (drag) force: F_r(v) = f0 + f1*v + f2*v^2."""
        f0, f1, f2 = 0.1, 5, 0.25
        return f0 + f1 * v + f2 * v ** 2

    @staticmethod
    def state_variables():
        return ["position", "velocity", "z"]

    @staticmethod
    def control_variables():
        return ["throttle_force"]


class Unicycle(Dynamics):
    """Planar unicycle (non-holonomic) dynamics.

    Models a rigid body in 2-D whose heading is controlled by angular
    velocity. The control inputs are forward speed *v* and angular rate
    *omega*.

    Equations of motion:

        dx/dt     = v * cos(theta)
        dy/dt     = v * sin(theta)
        dtheta/dt = omega

    State variables:
        position_x, position_y, theta

    Control variables:
        velocity, angular_velocity
    """

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
    """Single-integrator dynamics in 1-D: dy/dt = u.

    The simplest possible model — position is directly driven by a
    velocity command.

    State variables:
        position_x

    Control variables:
        velocity_x
    """

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
    """Single-integrator dynamics in 2-D: dp/dt = u.

    A point mass in the plane whose velocity is the control input.

    State variables:
        position_x, position_y

    Control variables:
        velocity_x, velocity_y
    """

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
    """Single-integrator dynamics in 3-D: dp/dt = u.

    A point mass in three-dimensional space whose velocity is the
    control input.

    State variables:
        position_x, position_y, position_z

    Control variables:
        velocity_x, velocity_y, velocity_z
    """

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
    """Double-integrator dynamics in 1-D.

    Models a point mass where acceleration is the control input.

    Equations of motion:

        dp_x/dt = v_x
        dv_x/dt = u  (acceleration)

    State variables:
        position_x, velocity_x

    Control variables:
        acceleration_x
    """

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
    """Double-integrator dynamics in 2-D.

    Models a point mass in the plane where acceleration is the control
    input. The state vector stores positions first, then velocities.

    Equations of motion:

        dp_x/dt = v_x,   dp_y/dt = v_y
        dv_x/dt = a_x,   dv_y/dt = a_y

    State variables:
        position_x, position_y, velocity_x, velocity_y

    Control variables:
        acceleration_x, acceleration_y
    """

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
    """Double-integrator dynamics in 3-D.

    Models a point mass in three-dimensional space where acceleration
    is the control input.

    Equations of motion:

        dp_i/dt = v_i   for i in {x, y, z}
        dv_i/dt = a_i   for i in {x, y, z}

    State variables:
        position_x, position_y, position_z, velocity_x, velocity_y, velocity_z

    Control variables:
        acceleration_x, acceleration_y, acceleration_z
    """

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
