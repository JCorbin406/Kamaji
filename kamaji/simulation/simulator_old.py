import yaml
import numpy as np

class Simulator:
    def __init__(self, config_file=None):
        """
        Initializes the simulator. If a config file is provided, it loads the simulation parameters from the file.
        Otherwise, it sets the default simulation parameters.

        Args:
            config_file (str, optional): Path to a YAML configuration file. Defaults to None.
        """
        if config_file:
            self.config = self.load_config(config_file)
        else:
            # Set default parameters or manually set them after initializing
            self.config = self.get_default_config()

        self.duration = self.config['simulation']['duration']
        self.time_step = self.config['simulation']['time_step']
        self.logging_enabled = self.config['logging']['enabled']
        self.obstacles = []
        self.agents = []
        self.environment = self.initialize_environment()
        self.agents = self.initialize_agents(self.config['agents'])
        
    def load_config(self, file_path):
        """
        Loads the configuration from a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: Loaded configuration data.
        """
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get_default_config(self):
        """
        Returns a default configuration if no file is provided.

        Returns:
            dict: Default configuration data.
        """
        return {
            'simulation': {
                'duration': 100.0,
                'time_step': 0.1,
            },
            'logging': {
                'enabled': True,
                'log_format': 'csv',
                'log_interval': 10,
            },
            'agents': [
                {
                    'id': 'agent_1',
                    'type': 'unicycle',
                    'initial_state': {'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0]},
                    'dynamics_model': 'unicycle_dynamics',
                    'controller': {'type': 'PID', 'parameters': {'k_p': 1.0, 'k_i': 0.1, 'k_d': 0.05}},
                },
                {
                    'id': 'agent_2',
                    'type': 'drone',
                    'initial_state': {'position': [10.0, 10.0, 5.0], 'velocity': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0]},
                    'dynamics_model': 'drone_dynamics',
                    'controller': {'type': 'LQR', 'parameters': {'Q': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'R': [[1, 0], [0, 1]]}},
                }
            ],
            'environment': {
                'obstacles': [
                    {'type': 'sphere', 'position': [5.0, 5.0, 2.5], 'radius': 1.0},
                    {'type': 'box', 'position': [12.0, 15.0, 0.0], 'dimensions': [2.0, 2.0, 3.0]},
                ],
                'terrain': {'type': 'flat', 'parameters': {'slope': 0.05}},
            },
        }

    def initialize_agents(self, agent_configs):
        """
        Initializes the agents based on the configuration.

        Args:
            agent_configs (list): List of agent configurations.

        Returns:
            list: A list of agent objects.
        """
        agents = []
        for agent_config in agent_configs:
            # Assuming Agent class exists that takes in an agent configuration
            agent = Agent(agent_config)
            agents.append(agent)
        return agents

    def initialize_environment(self):
        """
        Initializes the environment by setting gravity and creating obstacles from the config.
        """
        # Set gravity for the environment
        self.gravity = np.array(self.config['environment']['gravity'])
        
        # Create obstacles
        self.initialize_obstacles()

        print(f"Gravity set to: {self.gravity}")
        print(f"Obstacles initialized: {len(self.obstacles)}")

    def initialize_obstacles(self):
        """
        Initializes obstacles based on their type and properties.
        """
        for obs_config in self.config['environment']['obstacles']:
            obstacle_type = obs_config['type']
            position = np.array(obs_config['position'])
            if obstacle_type == "circle":
                # Circle: 2D obstacle with a radius
                radius = obs_config.get('radius', 1.0)
                obstacle = {'id': obs_config['id'], 'type': obstacle_type, 'position': position, 'radius': radius}
            
            elif obstacle_type == "sphere":
                # Sphere: 3D obstacle with a radius
                radius = obs_config.get('radius', 1.0)
                obstacle = {'id': obs_config['id'], 'type': obstacle_type, 'position': position, 'radius': radius}
            
            elif obstacle_type == "box":
                # Box: 3D obstacle with dimensions (length, width, height)
                dimensions = np.array(obs_config.get('dimensions', [1.0, 1.0, 1.0]))
                obstacle = {'id': obs_config['id'], 'type': obstacle_type, 'position': position, 'dimensions': dimensions}
            
            else:
                # Handle unexpected obstacle types
                raise ValueError(f"Unknown obstacle type: {obstacle_type}")
            
            # Add the obstacle to the list
            self.obstacles.append(obstacle)

    def run(self):
        """
        Runs the simulation.

        """
        for t in range(int(self.duration / self.time_step)):
            for agent in self.agents:
                agent.step(self.time_step)
            if self.logging_enabled:
                self.log_data()

    def log_data(self):
        """
        Logs the simulation data.
        """
        pass
