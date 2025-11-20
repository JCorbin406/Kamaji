# Configuration Guide

Kamaji supports YAML-based configuration files to define agents, environments, and simulation parameters.

---

## Structure

A basic configuration file (`config.yaml`) may include:

```yaml
simulation:
  total_time: 20.0
  dt: 0.1

agents:
  - id: agent1
    dynamics_model: "Unicycle"
    initial_state:
      position_x: 0.0
      position_y: 0.0
    controller:
      velocity_x:
        type: "PID"
        specs:
          - state: position_x
            goal: 10.0
            kp: 1.0
            ki: 0.0
            kd: 0.1
```

---

## Key Fields

- `dynamics_model`: Must match a class from `kamaji.dynamics`
- `controller.type`: `"Constant"` or `"PID"`
- `initial_state`: Keys must match what the model expects

You can load and run YAML configs with custom entry scripts or GUI tools.
