# Kamaji

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Kamaji is a multi-agent simulation framework for modeling, controlling, and training agent-based systems. It supports configurable dynamics models, per-channel controllers (PID, constant, RL), Control Barrier Function (CBF) safety filters, and Gymnasium-based reinforcement learning — all driven by YAML configuration files.

## Quick Example

```python
import yaml
from kamaji.simulation.simulator import Simulator

# Load a config and run
with open("examples/configs/basic_simulation.yml") as f:
    config = yaml.safe_load(f)

sim = Simulator(config)
sim.simulate()

# Plot results
sim.plot.trajectories_2d()
```

Or build a simulation programmatically:

```python
from kamaji.simulation.simulator import Simulator

sim = Simulator()
sim.set_sim_params({"time_step": 0.01, "duration": 10.0, "integrator": "RK4"})

sim.add_agents({
    "agent_1": {
        "type": "agent",
        "initial_state": {"position_x": 0.0, "position_y": 0.0},
        "dynamics_model": "SingleIntegrator2DOF",
        "controller": {
            "velocity_x": {"type": "PID", "specs": [{"state": "position_x", "goal": 5.0, "kp": 1.0, "ki": 0.0, "kd": 0.1}]},
            "velocity_y": {"type": "PID", "specs": [{"state": "position_y", "goal": 3.0, "kp": 1.0, "ki": 0.0, "kd": 0.1}]},
        }
    }
})

sim.simulate()
```

## Installation Guide

### Prerequisites

- **Conda**: You will need Conda (or Miniconda) installed on your machine. You can download Conda from [Anaconda's website](https://www.anaconda.com/products/distribution).

- **Python 3.11+**: This package requires Python 3.11 or above.

### Step 1: Clone the Repository

Start by cloning the Kamaji repository to your local machine.

```bash
git clone https://github.com/JCorbin406/Kamaji.git
cd Kamaji
```

### Step 2: Create a Conda Environment

The easiest way to create a Conda environment with all the required dependencies is to use the environment.yml file provided in this repository.

#### Step 2.1: Create the Conda Environment

Run the following command in the repository directory to create the Conda environment:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment named kamaji with all the dependencies listed in the environment.yml file.

#### Step 2.2: Activate the Environment

Activate the environment using the following command:

```bash
conda activate kamaji
```

### Step 3: Install the Kamaji Package

Creating the environment in step 2.1 should have collected and installed all required dependencies. If you need to install further packages you can use:

```bash
conda install <YOUR-PACKAGE-NAME>
```

or alternatively:

```bash
pip install <YOUR-PACKAGE-NAME>
```

### Step 4: Verify Installation

To verify that Kamaji has been installed correctly, you can run the following command to check the installed version:

```bash
python -c "import kamaji; print(kamaji.__version__)"
```

If Kamaji is installed correctly, you should see the version number printed in the terminal.

### Step 5: Running Kamaji

The simplest way to run a simulation is with one of the example configs:

```bash
python -c "
import yaml
from kamaji.simulation.simulator import Simulator

with open('examples/configs/basic_simulation.yml') as f:
    config = yaml.safe_load(f)

sim = Simulator(config)
sim.simulate()
"
```

See `examples/configs/` for more configuration examples, including CBF safety filtering and RL-based controllers. For the full YAML configuration reference, see the [documentation](docs/concepts/configuration.md).

### Step 6: Deactivate the Conda Environment

When you're done, you can deactivate the Conda environment:

```bash
conda deactivate
```

### Step 7: Updating Dependencies

To update the Conda environment and its dependencies, you can run:

```bash
conda env update -f environment.yml
```

This will update the environment with any new dependencies that have been added to the environment.yml file.

## Documentation

Build and serve the docs locally:

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser. Key pages:

- [Dynamics Models](docs/concepts/dynamics.md) — available dynamics and how to create custom ones
- [Controllers](docs/concepts/controls.md) — PID, CBF safety filters, RL policies
- [Environment & Simulation](docs/concepts/environment.md) — the simulation loop, collision detection, Gym integration
- [Configuration](docs/concepts/configuration.md) — YAML config reference

## Development

If you'd like to contribute to the Kamaji project, please fork the repository and create a pull request with your changes. Ensure that you follow the code style and include appropriate tests for new features or bug fixes.

## License

Kamaji is licensed under the MIT License. See the LICENSE file for more details.
