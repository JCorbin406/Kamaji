# simulator.py
from time import time
from typing import Optional
from tqdm import tqdm
import numpy as np
from kamaji.plotting.simulation_plotter import SimulationPlotter
from kamaji.logging.simulation_logger import SimulationLogger
from kamaji.agent.agent import Agent

class Simulator:
    """
    A class to manage and run multi-agent simulations with configurable dynamics,
    control, and collision avoidance mechanisms. Includes support for agent tracking,
    control input updates, manual overrides, logging, and visualization.

    Attributes:
        sim_time (float): Current simulation time.
        active_agents (list): List of currently active agents.
        inactive_agents (list): List of agents that have completed simulation.
        agent_ids (set): Set of agent IDs in the simulation.
        verbose (bool): Flag for printing debug/output information.
        plot (SimulationPlotter): Visualization utility.
        logger (SimulationLogger): Logging utility for simulation data.
    """

    def __init__(self, config=None) -> None:
        """
        Initialize the simulation with optional configuration parameters.

        Args:
            config (dict, optional): Dictionary containing simulation, agent, and logging parameters.
        """
        self._config = config
        self._current_real_time = time()
        self._elapsed_real_time = 0.0
        self.sim_time = 0.0
        self.active_agents = []
        self.inactive_agents = []
        self.agent_ids = set()
        self.verbose = True
        self.plot = SimulationPlotter(self)
        self.logging_params = config.get("logging", {}) if config else {}
        self.logger = SimulationLogger(self)

        if config is not None:
            self.load_from_config(config)
        else:
            if self.verbose:
                print("No config provided, using default values.")

    def set_sim_params(self, sim_params=None):
        """
        Set core simulation parameters such as timestep, duration, and integration method.

        Args:
            sim_params (dict, optional): Dictionary with keys 'time_step', 'duration', and 'integrator'.

        Raises:
            ValueError: If required parameters are missing or invalid.
            TypeError: If parameter types are incorrect.
        """
        if sim_params is None:
            self.dt = 0.01
            self.duration = 10.0
            self.num_timesteps = int(self.duration / self.dt)
            self.integrator = 'RK4'
            self.verbose = True
            return

        self.verbose = sim_params.get('verbose', True)
        required = ['time_step', 'duration', 'integrator']
        missing = [key for key in required if key not in sim_params]
        if missing:
            raise ValueError(f"Missing simulation parameters: {missing}")

        time_step = sim_params['time_step']
        duration = sim_params['duration']
        integrator = sim_params['integrator']

        if not isinstance(time_step, (float, int)) or time_step <= 0:
            raise ValueError("'time_step' must be a positive number.")
        if not isinstance(duration, (float, int)) or duration <= 0:
            raise ValueError("'duration' must be a positive, nonzero number.")
        if not isinstance(integrator, str):
            raise TypeError("'integrator' must be a string.")

        valid_integrators = {'RK4', 'Euler', 'RK2', 'RK45'}
        if integrator not in valid_integrators:
            raise ValueError(f"Unsupported integrator '{integrator}'. Valid options: {valid_integrators}")

        self.dt = float(time_step)
        self.duration = float(duration)
        self.num_timesteps = max(1, int(self.duration / self.dt))
        self.integrator = integrator

    def load_from_config(self, config):
        """
        Load simulation parameters, agent configuration, environment, and logging settings.

        Args:
            config (dict): YAML-style configuration dictionary.
        """
        self.set_sim_params(config.get('simulation', {}))
        self.add_agents(config.get('agents', {}))
        self.env_params = config.get('environment', {})
        self.logging_params = config.get('logging', {})

    def add_agents(self, agents) -> None:
        """
        Add one or more agents to the simulation.

        Args:
            agents (dict | tuple | dict): Dictionary of agents, (config, id) tuple, or a single config.

        Raises:
            TypeError: If format of input is not recognized.
        """
        if isinstance(agents, dict):
            if all(isinstance(v, dict) for v in agents.values()):
                for agent_id, agent_config in agents.items():
                    self._add_single_agent(agent_config, agent_id)
            else:
                self._add_single_agent(agents)
        elif isinstance(agents, tuple):
            if not isinstance(agents[0], dict) or not isinstance(agents[1], str):
                raise TypeError("Expected (agent_config: dict, agent_id: str)")
            self._add_single_agent(agents[0], agents[1])
        else:
            raise TypeError(
                "Expected one of: dict of agents, (config, id) tuple, or single agent config dict."
            )

    def _add_single_agent(self, agent_config: dict, agent_id: Optional[str] = None):
        """
        Add a single agent with the provided configuration and optional ID.

        Args:
            agent_config (dict): Configuration dictionary for the agent.
            agent_id (str, optional): Explicit ID for the agent. Auto-generated if None.

        Raises:
            ValueError: If required fields are missing or types are invalid.
            RuntimeError: If the agent fails to initialize.
        """
        required_fields = ['type', 'initial_state', 'dynamics_model', 'controller']
        missing = [k for k in required_fields if k not in agent_config]
        if missing:
            raise ValueError(f"Missing required fields in agent config: {missing}")

        if not isinstance(agent_config['initial_state'], dict):
            raise TypeError("initial_state must be a dictionary.")
        if not isinstance(agent_config['controller'], dict):
            raise TypeError("controller must be a dictionary.")

        controller_block = agent_config['controller']
        if 'type' in controller_block and 'specs' in controller_block:
            if not isinstance(controller_block['specs'], list):
                raise TypeError("Controller 'specs' must be a list.")
        else:
            for ctrl_name, ctrl_conf in controller_block.items():
                if not isinstance(ctrl_conf, dict):
                    raise TypeError(f"Controller for '{ctrl_name}' must be a dict.")
                if 'type' not in ctrl_conf:
                    raise ValueError(f"Controller '{ctrl_name}' must include a 'type' field.")
                if 'specs' not in ctrl_conf:
                    raise ValueError(f"Controller '{ctrl_name}' must include a 'specs' field.")
                if not isinstance(ctrl_conf['specs'], list):
                    raise TypeError(f"Controller '{ctrl_name}' 'specs' must be a list.")

        if agent_id is None:
            base = "agent"
            i = 1
            while f"{base}_{i}" in self.agent_ids:
                i += 1
            agent_id = f"{base}_{i}"

        if agent_id in self.agent_ids:
            raise ValueError(f"Agent ID '{agent_id}' already exists.")

        agent_config['id'] = agent_id
        try:
            self.active_agents.append(Agent(agent_config, self.sim_time, self.dt))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Agent '{agent_id}': {e}")
        self.agent_ids.add(agent_id)

        if self.verbose:
            print(f"[Simulator] Agent '{agent_id}' added.")

    def simulate(self, on_step=None) -> None:
        """
        Run the simulation loop for all time steps.

        Args:
            on_step (Callable[[Simulator, int], None], optional): Callback executed before each step.
        """
        start_time = time()
        iterator = tqdm(range(self.num_timesteps), desc="Simulating", unit="step") if self.verbose else range(self.num_timesteps)

        for step_idx in iterator:
            if on_step:
                on_step(self, step_idx)
            self.step()

        self.sim_time = time() - start_time
        self.inactive_agents.extend(self.active_agents)
        self.active_agents.clear()

        if self.logging_params.get("enabled", True):
            if self.logging_params.get("format", "hdf5") == "hdf5":
                self.logger.log_to_hdf5()

        if self.verbose:
            print(f"Sim time: {self.sim_time:.4f}")

    def step(self) -> None:
        """
        Advance the simulation by one time step, updating agent states using control input.
        """
        for idx, agent in enumerate(self.active_agents):
            control = agent.manual_control_input if agent.manual_control_input is not None else agent.compute_control(self.sim_time)
            agent.step(self.sim_time, control)

    def clear_manual_control(self, agent_id: str) -> None:
        """
        Clear any manually assigned control input for a specific agent.

        Args:
            agent_id (str): ID of the agent to clear manual control for.
        """
        for agent in self.active_agents:
            if agent._id == agent_id:
                agent.manual_control_input = None
                return

    def set_manual_control(self, agent_id: str, control: np.ndarray) -> None:
        """
        Manually assign a control input to a specific agent.

        Args:
            agent_id (str): ID of the agent.
            control (np.ndarray): Manual control input.

        Raises:
            ValueError: If the agent ID is not found.
        """
        for agent in self.active_agents:
            if agent._id == agent_id:
                agent.manual_control_input = control
                if self.verbose:
                    print(f"[Simulator] Manual control set for agent '{agent_id}'.")
                return
        raise ValueError(f"Agent with ID '{agent_id}' not found.")

    def remove_agent(self, agent: Agent) -> bool:
        """
        Remove an agent from the active list and move to the inactive list.

        Args:
            agent (Agent): The agent to be removed.

        Returns:
            bool: True if removal was successful.

        Raises:
            ValueError: If the agent is not in the active list.
        """
        if agent not in self.active_agents:
            raise ValueError(f"Agent {agent.id} is not an active Agent.")
        self.inactive_agents.append(self.active_agents.pop(self.active_agents.index(agent)))
        return True
