# Kamaji Simulation Environment

Kamaji is a flexible and extensible multi-agent simulation environment designed for research and development in autonomous systems, control theory, and coordination strategies. It enables the modeling of large-scale, heterogeneous agent systems with support for custom dynamics, control architectures, and interaction mechanisms like auctions or barrier functions.

Kamaji is built for:

- Simulating realistic multi-agent behavior in 2D/3D space
- Testing control strategies like PID, CBF, geometric, or learned controllers
- Implementing and evaluating fairness-based coordination and resource allocation
- Running large simulations for urban air mobility (UAM), swarming, or multi-robot planning

---

## Features

- **Modular Agent Architecture**: Each agent can be assigned custom dynamics, goals, and control models.
- **Controller Support**: Includes PID, Control Barrier Functions (CBFs), and auction-based control allocation mechanisms.
- **Simulation Core**: A flexible time-stepped engine that supports integration with advanced control loops and interaction constraints.
- **Real-time GUI (WIP)**: PyQt-based interface for visualizing and interacting with simulations.
- **Logging & Analysis**: Integrated support for logging states and control inputs, with HDF5 export and analysis tools.
- **Extensible Models**: Easily add new dynamics, control types, or interaction rules.

---

## Project Structure

```plaintext
kamaji/
├── agent/         # Core agent logic and update routines
├── dynamics/      # Definitions for single/double integrators, unicycle models, etc.
├── controllers/   # PID, CBF, and geometric control implementations
├── simulation/    # Time-stepping simulator logic
├── auctions/      # Auction mechanisms for resource allocation and fairness
├── plotting/      # Tools for trajectory visualization
├── gui/           # PyQt-based interface (in development)
├── tools/         # Logging, ODE solvers, and utility functions
```

---

## Documentation

Navigate the sidebar to learn about:

- Agent configuration and initialization
- Control schemes and switching logic
- Simulation loop and how to run batch experiments
- Auction-based fairness and coordination mechanisms
- API reference for customizing and extending the framework

---

## Getting Started

To launch your first simulation:

```bash
pip install -e .
python examples/basic_simulation.py
```

Or use the GUI:

```bash
python kamaji/gui/gui_main.py
```

---

## Authors & Acknowledgments

Created and maintained by [Johnathan (Jack) Corbin](https://github.com/JCorbin406)  
With guidance from Georgia Tech's robotics and control faculty.

Kamaji is inspired by the need for scalable, interpretable, and fair multi-agent coordination tools — and borrows its name from the many-armed boiler master in *Spirited Away*, who utilizes an army of sootballs to run a complex but beautifully organized machine.
