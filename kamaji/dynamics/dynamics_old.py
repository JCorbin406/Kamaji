import numpy as np
from abc import ABC, abstractmethod

class Dynamics(ABC):
    """
    Base class for agent dynamics. Subclasses must implement the `update` method to compute
    the derivatives based on the current state and control inputs.
    """
    def __init__(self):
        pass

    @abstractmethod
    def update(self, state, control_input, time_step):
        """
        Compute the derivatives of the state based on the current state and control input.

        Args:
            state (dict): The current state of the agent (position, velocity, etc.).
            control_input (dict): The control inputs (e.g., accelerations, forces).
            time_step (float): The time step for integration.

        Returns:
            dict: The derivatives of the state.
        """
        pass


class SingleIntegrator2D(Dynamics):
    def __init__(self):
        super().__init__()

    def update(self, state, control_input, time_step):
        """
        Update the state of a unicycle agent based on control input.
        
        Args:
            state (dict): The current state (position, velocity, orientation).
            control_input (dict): Control input (linear and angular velocity).
            time_step (float): Time step for the integration.
        
        Returns:
            dict: The updated state (position, velocity, orientation).
        """
        position = np.array(state['position'], dtype=float)
        velocity = np.array(control_input['velocity'], dtype=float)

        derivative = velocity

        # Return the derivatives (state change)
        return {'position': position.tolist(), 'velocity': velocity.tolist(), 'orientation': orientation.tolist()}
