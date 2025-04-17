# simulator.py
from time import time
from typing import Optional
from tqdm import tqdm
import numpy as np
from kamaji.plotting.simulation_plotter import SimulationPlotter
from kamaji.logging.simulation_logger import SimulationLogger
from kamaji.agent.agent import Agent


# Add agent IDs, whatever number they were initialized in
class Simulator:
    def __init__(self, config = None) -> None:
        """
        Initializes a Simulation with a specified total time and step size.

        Args:
            total_time (float): The total time the simulation should run for, in seconds
            dt (float): The step size of the simulation, in seconds.
            agents (Optional[list[Agent]]): Initial agents to add to the sim, if desired.
        """
        self._config = config
        self._current_real_time = time()
        self._elapsed_real_time = 0.0
        self.sim_time = 0.0
        self.active_agents = []
        self.inactive_agents = []
        self.agent_ids = set()
        self.verbose = True # Default in case config is None
        self.plot = SimulationPlotter(self)
        
        self.logging_params = config.get("logging", {})
        self.logger = SimulationLogger(self)

        if config is not None:
            self.load_from_config(config)
        else:
            if self.verbose:
                print("No config provided, using default values.")

    def set_sim_params(self, sim_params=None):
        """
        Set simulation parameters including time step, duration, and integrator.

        Args:
            sim_params (dict, optional): Dictionary containing keys 'time_step', 'duration', and 'integrator'.

        Raises:
            ValueError: If required keys are missing or invalid.
            TypeError: If any value has the wrong type.
        """
        # Use defaults if none provided
        if sim_params is None:
            self.dt = 0.01
            self.duration = 10.0
            self.num_timesteps = int(self.duration / self.dt)
            self.integrator = 'RK4'
            self.verbose = True # Default if config is not used
            return
        
        # Verbose flag: default to True
        self.verbose = sim_params.get('verbose', True)

        # Required keys
        required = ['time_step', 'duration', 'integrator']
        missing = [key for key in required if key not in sim_params]
        if missing:
            raise ValueError(f"Missing simulation parameters: {missing}")

        # Type and value checks
        time_step = sim_params['time_step']
        duration = sim_params['duration']
        integrator = sim_params['integrator']

        if not isinstance(time_step, (float, int)) or time_step <= 0:
            raise ValueError("'time_step' must be a positive number.")
        if not isinstance(duration, (float, int)) or duration <= 0:
            raise ValueError("'duration' must be a positive, nonzero number.")
        if not isinstance(integrator, str):
            raise TypeError("'integrator' must be a string.")

        # Valid integrators (optional)
        valid_integrators = {'RK4', 'Euler', 'RK2', 'RK45'}
        if integrator not in valid_integrators:
            raise ValueError(f"Unsupported integrator '{integrator}'. Valid options: {valid_integrators}")

        # Assign values
        self.dt = float(time_step)
        self.duration = float(duration)
        self.num_timesteps = max(1, int(self.duration / self.dt))
        self.integrator = integrator
    
    def load_from_config(self, config):
        self.set_sim_params(config.get('simulation', {}))
        self.add_agents(config.get('agents', {}))
        self.env_params = config.get('environment', {})
        self.logging_params = config.get('logging', {})

    def add_agents(self, agents) -> None:
        """
        Adds one or more agents to the simulation. Supports:
        - a dict of agents (id → config)
        - a tuple: (config, id)
        - a single config dict (id auto-generated)
        """
        if isinstance(agents, dict):
            # Case 1: multiple agents
            if all(isinstance(v, dict) for v in agents.values()):
                for agent_id, agent_config in agents.items():
                    self._add_single_agent(agent_config, agent_id)
            # Case 2: single agent config (ID auto-generated)
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
        # 1. Required fields
        required_fields = ['type', 'initial_state', 'dynamics_model', 'controller']
        missing = [k for k in required_fields if k not in agent_config]
        if missing:
            raise ValueError(f"Missing required fields in agent config: {missing}")

        # 2. Validate types (optional but good for debugging)
        if not isinstance(agent_config['initial_state'], dict):
            raise TypeError("initial_state must be a dictionary.")
        if not isinstance(agent_config['controller'], dict):
            raise TypeError("controller must be a dictionary.")

        controller_block = agent_config['controller']

        # 3. NEW: Support per-channel or legacy controllers
        if 'type' in controller_block and 'specs' in controller_block:
            # Single controller (legacy format)
            if not isinstance(controller_block['specs'], list):
                raise TypeError("Controller 'specs' must be a list.")
        else:
            # New format: one controller per control channel
            for ctrl_name, ctrl_conf in controller_block.items():
                if not isinstance(ctrl_conf, dict):
                    raise TypeError(f"Controller for '{ctrl_name}' must be a dict.")
                if 'type' not in ctrl_conf:
                    raise ValueError(f"Controller '{ctrl_name}' must include a 'type' field.")
                if 'specs' not in ctrl_conf:
                    raise ValueError(f"Controller '{ctrl_name}' must include a 'specs' field.")
                if not isinstance(ctrl_conf['specs'], list):
                    raise TypeError(f"Controller '{ctrl_name}' 'specs' must be a list.")

        # 4. Generate ID if needed
        if agent_id is None:
            base = "agent"
            i = 1
            while f"{base}_{i}" in self.agent_ids:
                i += 1
            agent_id = f"{base}_{i}"

        # 5. Check for ID reuse
        if agent_id in self.agent_ids:
            raise ValueError(f"Agent ID '{agent_id}' already exists.")

        # 6. Add agent to active agents
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
        Runs the simulation forward for all time steps.
        
        Args:
            on_step (Callable[[Simulator, int], None], optional):
                A user-supplied callback that runs before each step.
        """
        start_time = time()
        iter = tqdm(range(self.num_timesteps), desc="Simulating", unit="step") if self.verbose else range(self.num_timesteps)

        for step_idx in iter:
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
        Steps the simulation forward by simulating all agents.
        """
        for idx, agent in enumerate(self.active_agents):
            if agent.manual_control_input is not None:
                control = agent.manual_control_input
            else:
                control = agent.compute_control(self.sim_time)
            agent.step(self.sim_time, control)

    def clear_manual_control(self, agent_id: str) -> None:
        for agent in self.active_agents:
            if agent._id == agent_id:
                agent.manual_control_input = None
                return

    def set_manual_control(self, agent_id: str, control: np.ndarray) -> None:
        for agent in self.active_agents:
            if agent._id == agent_id:
                agent.manual_control_input = control
                if self.verbose:
                    print(f"[Simulator] Manual control set for agent '{agent_id}'.")
                return
        raise ValueError(f"Agent with ID '{agent_id}' not found.")

    def remove_agent(self, agent: Agent) -> bool:
        """
        Removes an Agent from the active Agents and moves it to the inactive Agents.
        If the agent does not exist in the active Agent list, an error will be thrown.

        Args:
            agent (Agent): The Agent to make inactive.

        Raises:
            ValueError: _description_
        """
        if agent not in self.active_agents:
            raise ValueError('Agent ' + str(agent.id) + ' is not an active Agent.')
        self.inactive_agents.append(self.active_agents.pop(self.active_agents.index(agent)))
