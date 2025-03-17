import numpy as np

class Agent:
    def __init__(self, config):
        """
        Initializes an agent based on the provided configuration.

        Args:
            config (dict): Configuration dictionary for the agent.
        """
        self.id = config['id']
        self.type = config['type']
        
        # Initialize states dynamically based on config
        self.state = self.initialize_state(config['initial_state'])
        self.dynamics_model = config['dynamics_model']

    def initialize_state(self, state_config):
        """
        Initializes the state of the agent from the configuration.
        
        Args:
            state_config (dict): A dictionary representing the agent's state variables.

        Returns:
            np.array: The initial state vector.
        """
        state = {}
        for key, value in state_config.items():
            state[key] = np.array(value, dtype=float)  # Ensure it's a numpy array
        return state

    def step(self, time_step, control_input=None):
        """
        Advances the agent by one time step based on its dynamics model and control input.

        Args:
            time_step (float): The time step for the simulation.
            control_input (dict, optional): Optional control input to influence state dynamics.
        """
        # Apply dynamics model (or a simple placeholder model for now)
        self.apply_dynamics(time_step, control_input)

    def apply_dynamics(self, time_step, control_input):
        """
        Updates the agent's state based on its dynamics model and control input.

        Args:
            time_step (float): The time step for the simulation.
            control_input (dict, optional): Control input (could be acceleration or other variables).
        """
        # Here we can assume a simple model where control_input affects the state.
        if control_input is not None:
            for state_key, change in control_input.items():
                if state_key in self.state:
                    self.state[state_key] += change * time_step

    def get_state(self):
        """
        Returns the current state of the agent.

        Returns:
            dict: The current state of the agent.
        """
        return self.state