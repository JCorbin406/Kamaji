# Quickstart

Welcome to Kamaji! This guide will help you run your first multi-agent simulation in just a few steps.

---

## Run Your First Simulation

After installing Kamaji and activating your environment, try this:

```bash
python examples/basic_simulation.py
```

This runs a basic simulation with several agents navigating a 2D environment using default dynamics and controllers.

---

## What It Does

- Initializes a simulator instance
- Spawns agents with default dynamics and goals
- Runs a time-stepped simulation
- Logs trajectories and optionally visualizes them

---

## Relevant Files

- `examples/basic_simulation.py`: Script to launch the simulation
- `kamaji/simulation/simulator.py`: Main simulator engine
- `kamaji/agent/agent.py`: Defines agent behavior and integration
- `kamaji/controllers/`: Available control models (e.g. PID, CBF)

For more customization, see the [Configuration Guide](configuration.md).
