import os
import json
import h5py
import numpy as np

class SimulationLogger:
    def __init__(self, simulator, default_filename="simulation"):
        self.sim = simulator
        logging_cfg = getattr(simulator, "logging_params", {})

        self.enabled = logging_cfg.get("enabled", True)
        self.format = logging_cfg.get("format", "hdf5")
        self.output_dir = logging_cfg.get("output_directory", "sim_output")
        self.filename = logging_cfg.get("filename", default_filename)
        self.output_path = os.path.join(self.output_dir, self.filename)
        self.verbose = logging_cfg.get("verbose", True)

    def log_to_hdf5(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with h5py.File(self.output_path, "w") as f:
            # Global simulation metadata
            f.attrs["duration"] = self.sim.duration
            f.attrs["dt"] = self.sim.dt
            f.attrs["num_timesteps"] = self.sim.num_timesteps
            f.attrs["integrator"] = self.sim.integrator

            if hasattr(self.sim, "env_params") and isinstance(self.sim.env_params, dict):
                f.attrs["env_params"] = json.dumps(self.sim.env_params)
            if hasattr(self.sim, "logging_params") and isinstance(self.sim.logging_params, dict):
                f.attrs["logging_params"] = json.dumps(self.sim.logging_params)

            agents = self.sim.active_agents + self.sim.inactive_agents
            for agent in agents:
                agent_group = f.create_group(agent._id)

                # State log
                state_group = agent_group.create_group("state")
                for col in agent.state_log.columns:
                    state_group.create_dataset(col, data=np.array(agent.state_log[col]))

                # Control log
                control_group = agent_group.create_group("control")
                for col in agent.control_log.columns:
                    control_group.create_dataset(col, data=np.array(agent.control_log[col]))

        if self.verbose:
            print(f"[SimulationLogger] Data saved to {self.output_path}")
