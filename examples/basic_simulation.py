"""Basic simulation using YAML configuration for Kamaji."""

import yaml
from kamaji.simulation.simulator import Simulator

if __name__ == "__main__":
    # Load configuration from a YAML file
    config_path = "examples/configs/basic_simulation.yml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create and run the simulator using the configuration
    sim = Simulator(config)
    sim.simulate()
    sim.plot.animate_trajectories()