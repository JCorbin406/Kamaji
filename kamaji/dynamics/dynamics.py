# dynamics.py
from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as Rot

class Dynamics(ABC):
    """
    Base class for various dynamics models.

    This class serves as a foundation for different dynamic models, providing a
    structure for defining the state, control inputs, and methods for updating
    the state based on control dynamics.

    Attributes:
        _state (np.ndarray): Current state of the system, which should be defined in subclasses.

    Methods:
        dynamics: Computes the time derivative of the state given the current control inputs.
        set_state: Updates the current state of the system.
        state_info: Returns a dictionary of state variable names and their current values.
        set_num_control_inputs: Sets the number of control inputs to the system for use in the Agent class.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        """
        Initializes the dynamics model with a given initial state.

        Args:
            initial_state (Union[List[float], np.ndarray]): The initial state of the system.
        """
        self._state = np.array(initial_state, dtype=float)
        self._num_control_inputs = 0

    @abstractmethod
    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to compute the time derivative of the state.
        Must be implemented by subclasses.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system.
            u (np.ndarray): Control input vector.
            **kwargs: Optional keyword arguments for additional parameters.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        pass

    def set_state(self, x: Union[List[float], np.ndarray]) -> None:
        """
        Updates the current state of the system.

        Args:
            x (Union[List[float], np.ndarray]): New state of the system.
        """
        self._state = np.array(x, dtype=float)

    @property
    def state(self) -> np.ndarray:
        """
        Returns the current state of the system.

        Returns:
            np.ndarray: The current state of the system.
        """
        return self._state

    @abstractmethod
    def state_info(self) -> dict:
        """
        Abstract method to provide state variable names and their current values.
        Must be implemented by subclasses.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
        """
        pass

    def set_num_control_inputs(self, n: int) -> None:
        """
        Updates the number of control inputs for the dynamics model.

        Args:
            n (float): Number of control inputs.
        """
        self._num_control_inputs = n

    @property
    def num_control_inputs(self) -> int:
        """
        Returns the number of control inputs in the system.

        Returns:
            int: The number of control inputs for the system.
        """
        return self._num_control_inputs


class CruiseControl(Dynamics):
    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        super().__init__(initial_state)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs) -> np.ndarray:
        """
    Computes the time derivative of the state for the cruise control system.

    This function models the dynamics of a vehicle under cruise control. The system
    evolves based on the current position, velocity, and relative distance to a 
    lead vehicle. Control input is applied as a wheel force, and optional parameters 
    like vehicle mass and reference speed can be specified via keyword arguments.

    Args:
        t (float): Current time (s).
        state (np.ndarray): Current state of the system [p, v, z], where:

            - p: Position of the vehicle (m).
            - v: Velocity of the vehicle (m/s).
            - z: Distance to the lead vehicle (m).
        u (np.ndarray): Control input vector [u], where:

            - u: Wheel force applied to the vehicle (N).
        **kwargs: Optional keyword arguments for additional parameters:

            - m (float): Mass of the vehicle (kg).
            - v_0 (float): Desired reference speed (m/s).

    Returns:
        np.ndarray: Time derivative of the state [dp/dt, dv/dt, dz/dt].

    Raises:
        ValueError: If 'm' (mass) or 'v_0' (reference speed) are not provided in kwargs.
    """
        # Extract m and v_0 from kwargs, or raise an error if not found
        # m = kwargs.get('m', None)
        # v_0 = kwargs.get('v_0', None)

        # if m is None or v_0 is None:
        #     raise ValueError("Both 'm' (mass) and 'v_0' (reference speed) must be provided.")

        m = 1650
        v_0 = 14

        dxdt = np.zeros_like(state)

        p = state[0]
        v = state[1]
        z = state[2]

        dxdt[0] = v
        dxdt[1] = -1 / m * self.F_r(v) + 1 / m * u.item()
        dxdt[2] = v_0 - v

        return dxdt

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:

                    - "p": Current position of the ego vehicle (m).
                    - "v": Current velocity of the ego vehicle (m/s).
                    - "z": Distance between ego and lead vehicle (m).
        """
        return {
            "p": self._state[0],
            "v": self._state[1],
            "z": self._state[2]
        }

    def F_r(self, v, **kwargs) -> float:
        """
        Computes the resistive force acting on the vehicle as a function of velocity.

        Args:
            v (float): Velocity of the vehicle (m/s).
            **kwargs: Optional keyword arguments for resistive force parameters:
                - f0 (float): Constant resistive force term (N).
                - f1 (float): Linear velocity-dependent resistive force term (N/(m/s)).
                - f2 (float): Quadratic velocity-dependent resistive force term (N/(m/s^2)).

        Returns:
            float: The computed resistive force based on the given velocity.
        
        Raises:
            ValueError: If f0, f1, or f2 are not provided.
        """
        # Extract f0, f1, and f2 from kwargs, or raise an error if not found
        # f0 = kwargs.get('f0', None)
        # f1 = kwargs.get('f1', None)
        # f2 = kwargs.get('f2', None)

        # if f0 is None or f1 is None or f2 is None:
        #     raise ValueError("f0, f1, and f2 must be provided.")

        f0 = 0.1
        f1 = 5
        f2 = 0.25

        return f0 + f1 * v + f2 * v ** 2


class Unicycle(Dynamics):
    """
    Unicycle model for representing motion in 2D space.

    This class models the dynamics of a unicycle, where the state is represented
    by the position (x, y) and orientation (theta) of the vehicle. The control
    inputs are the linear velocity and angular velocity.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        super().__init__(initial_state)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Computes the dynamics of the unicycle model.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system [x, y, theta].
            u (np.ndarray): Control input vector [v, omega], where:

                - v: Linear velocity (m/s)
                - omega: Angular velocity (rad/s)
            **kwargs: Optional keyword arguments for additional parameters.

        Returns:
            np.ndarray: Time derivative of the state.
        """
        v, omega = u
        x, y, theta = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        return np.array([dx, dy, dtheta])

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:

                    - "x": Current x position of the unicycle.
                    - "y": Current y position of the unicycle.
                    - "theta": Current orientation of the unicycle (rad).
        """
        return {
            "x": self._state[0],
            "y": self._state[1],
            "theta": self._state[2]
        }

class SingleIntegrator1DOF(Dynamics):
    """
    Single Integrator model for a single degree of freedom system.

    This class models the dynamics of a single integrator system, where the state
    is represented by the position of the system and the control input is the
    velocity.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        super().__init__(initial_state)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Computes the dynamics of the single integrator system.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system [position].
            u (np.ndarray): Control input vector [velocity].
            **kwargs (Any): Optional keyword arguments for additional parameters.

        Returns:
            np.ndarray: Time derivative of the state.
        """
        velocity = u[0]
        position = state[0]
        dposition = velocity
        return np.array([dposition])

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:

                    - "position": Current position of the system.
        """
        return {
            "position": self._state[0]
        }

class SingleIntegrator2DOF(Dynamics):
    """
    Single Integrator model for a 2DOF system.

    This class models the dynamics of a single integrator system in 2 dimensions (x, y),
    where the state is represented by the position in each dimension, and the control 
    input is the velocity in each dimension.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        """
        Initializes the 2DOF Single Integrator model with the given initial state.

        Args:
            initial_state (Union[List[float], np.ndarray]): The initial state of the system [x, y].
        """
        super().__init__(initial_state)
        self.set_num_control_inputs(2)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Computes the dynamics of the single integrator system in 2DOF.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system [x, y], 
                                where (x, y) are positions.
            u (np.ndarray): Control input vector [vx, vy], representing the velocity in each dimension.
            **kwargs: Optional keyword arguments for additional parameters.

        Returns:
            np.ndarray: Time derivative of the state, [dx, dy].
        """
        # Control inputs (accelerations)
        vx, vy = u

        # Time derivatives
        dx = vx
        dy = vy

        return np.array([dx, dy])

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:
                    - "x": Position in the x-direction.
                    - "y": Position in the y-direction.
        """
        return {
            "x": self._state[0],
            "y": self._state[1]
        }

class DoubleIntegrator1DOF(Dynamics):
    """
    Double Integrator model for a single degree of freedom system.

    This class models the dynamics of a double integrator system, where the state
    is represented by the position and velocity of the system, and the control input
    is the acceleration.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        super().__init__(initial_state)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Computes the dynamics of the double integrator system.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system [position, velocity].
            u (np.ndarray): Control input vector [acceleration].
            **kwargs (Any): Optional keyword arguments for additional parameters.

        Returns:
            np.ndarray: Time derivative of the state.
        """
        acceleration = u[0]
        position, velocity = state
        dposition = velocity
        dvelocity = acceleration
        return np.array([dposition, dvelocity])

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:

                    - "position": Current position of the system.
                    - "velocity": Current velocity of the system.
        """
        return {
            "position": self._state[0],
            "velocity": self._state[1]
        }

class DoubleIntegrator2DOF(Dynamics):
    """
    Double Integrator model for a 2DOF system.

    This class models the dynamics of a double integrator system in 2 dimensions (x, y),
    where the state is represented by the position and velocity in each dimension, and the control 
    input is the acceleration in each dimension.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        """
        Initializes the 2DOF Double Integrator model with the given initial state.

        Args:
            initial_state (Union[List[float], np.ndarray]): The initial state of the system [x, y, vx, vy].
        """
        super().__init__(initial_state)
        self.set_num_control_inputs(2)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Computes the dynamics of the double integrator system in 2DOF.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system [x, y, vx, vy], 
                                where (x, y) are positions and (vx, vy) are velocities.
            u (np.ndarray): Control input vector [ax, ay], representing the acceleration in each dimension.
            **kwargs: Optional keyword arguments for additional parameters.

        Returns:
            np.ndarray: Time derivative of the state, [dx, dy, dvx, dvy].
        """
        # Control inputs (accelerations)
        ax, ay = u

        # State variables
        x, y, vx, vy = state

        # Time derivatives
        dx = vx
        dy = vy
        dvx = ax
        dvy = ay

        return np.array([dx, dy, dvx, dvy])

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:
                    - "x": Position in the x-direction.
                    - "y": Position in the y-direction.
                    - "vx": Velocity in the x-direction.
                    - "vy": Velocity in the y-direction.
        """
        return {
            "x": self._state[0],
            "y": self._state[1],
            "vx": self._state[2],
            "vy": self._state[3]
        }

class SingleIntegrator3DOF(Dynamics):
    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        super().__init__(initial_state)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        # Placeholder for the actual dynamics implementation
        pass

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:
                    # Add state variable names and values here
        """
        return {
            # Add state variable names and values here
        }


class DoubleIntegrator3DOF(Dynamics):
    """
    Double Integrator model for a 3DOF system.

    This class models the dynamics of a double integrator system in 3 dimensions (x, y, z),
    where the state is represented by the position and velocity in each dimension, and the control 
    input is the acceleration in each dimension.
    """

    def __init__(self, initial_state: Union[List[float], np.ndarray]) -> None:
        """
        Initializes the 3DOF Double Integrator model with the given initial state.

        Args:
            initial_state (Union[List[float], np.ndarray]): The initial state of the system [x, y, z, vx, vy, vz].
        """
        super().__init__(initial_state)
        self.set_num_control_inputs(3)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Computes the dynamics of the double integrator system in 3DOF.

        Args:
            t (float): Current time.
            state (np.ndarray): Current state of the system [x, y, z, vx, vy, vz], 
                                where (x, y, z) are positions and (vx, vy, vz) are velocities.
            u (np.ndarray): Control input vector [ax, ay, az], representing the acceleration in each dimension.
            **kwargs: Optional keyword arguments for additional parameters.

        Returns:
            np.ndarray: Time derivative of the state, [dx, dy, dz, dvx, dvy, dvz].
        """
        # Control inputs (accelerations)
        ax, ay, az = u

        # State variables
        x, y, z, vx, vy, vz = state

        # Time derivatives
        dx = vx
        dy = vy
        dz = vz
        dvx = ax
        dvy = ay
        dvz = az

        return np.array([dx, dy, dz, dvx, dvy, dvz])

    def state_info(self) -> dict:
        """
        Provides a dictionary of state variable names and their current values.

        Returns:
            dict: A dictionary mapping state variable names to their current values.
                Keys:
                    - "x": Position in the x-direction.
                    - "y": Position in the y-direction.
                    - "z": Position in the z-direction.
                    - "vx": Velocity in the x-direction.
                    - "vy": Velocity in the y-direction.
                    - "vz": Velocity in the z-direction.
        """
        return {
            "x": self._state[0],
            "y": self._state[1],
            "z": self._state[2],
            "vx": self._state[3],
            "vy": self._state[4],
            "vz": self._state[5]
        }
