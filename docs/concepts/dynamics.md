# Dynamics Models

Dynamics models define the equations of motion for agents in Kamaji. Each agent is assigned exactly one dynamics model, which determines:

- **State variables** — the quantities that evolve over time (e.g. position, velocity, heading).
- **Control variables** — the inputs that influence the state evolution (e.g. force, velocity command, acceleration).
- **Equations of motion** — how the state derivative $\dot{x}$ depends on the current state $x$ and control $u$.

During simulation, Kamaji integrates the dynamics forward using a numerical ODE solver (RK4 by default).

## Available Models

| Model | State Variables | Control Variables | Description |
|-------|----------------|-------------------|-------------|
| `SingleIntegrator1DOF` | `position_x` | `velocity_x` | 1-D point mass, velocity input |
| `SingleIntegrator2DOF` | `position_x`, `position_y` | `velocity_x`, `velocity_y` | 2-D point mass, velocity input |
| `SingleIntegrator3DOF` | `position_x`, `position_y`, `position_z` | `velocity_x`, `velocity_y`, `velocity_z` | 3-D point mass, velocity input |
| `DoubleIntegrator1DOF` | `position_x`, `velocity_x` | `acceleration_x` | 1-D point mass, acceleration input |
| `DoubleIntegrator2DOF` | `position_x`, `position_y`, `velocity_x`, `velocity_y` | `acceleration_x`, `acceleration_y` | 2-D point mass, acceleration input |
| `DoubleIntegrator3DOF` | `position_x`, `position_y`, `position_z`, `velocity_x`, `velocity_y`, `velocity_z` | `acceleration_x`, `acceleration_y`, `acceleration_z` | 3-D point mass, acceleration input |
| `Unicycle` | `position_x`, `position_y`, `theta` | `velocity`, `angular_velocity` | Planar non-holonomic model |
| `CruiseControl` | `position`, `velocity`, `z` | `throttle_force` | 1-D vehicle with aerodynamic drag |

## Model Details

### Single Integrator

The simplest dynamics model. Position is directly driven by a velocity command:

$$\dot{p} = u$$

Available in 1-D, 2-D, and 3-D variants. This is the best starting point for prototyping controllers and CBF constraints because the relative degree between position and control is 1.

### Double Integrator

Adds a velocity state so that the control input is acceleration:

$$\dot{p} = v, \quad \dot{v} = u$$

Available in 1-D, 2-D, and 3-D variants. The relative degree between position and control is 2, which means standard CBF constraints on position require a higher-order CBF (HOCBF) formulation. Useful for modeling agents where force or thrust is the natural control input.

### Unicycle

A planar non-holonomic model commonly used in mobile robotics:

$$\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \omega$$

where $v$ is the forward speed and $\omega$ is the angular velocity. The unicycle cannot move sideways instantaneously, making it more realistic for differential-drive robots. Note that the control inputs are speed and turn rate, not left/right wheel velocities.

### Cruise Control

A longitudinal vehicle model with aerodynamic drag:

$$\dot{p} = v, \quad \dot{v} = -\frac{F_r(v)}{m} + \frac{u}{m}, \quad \dot{z} = v_0 - v$$

where:

- $F_r(v) = f_0 + f_1 v + f_2 v^2$ is the resistive (drag) force (defaults: $f_0 = 0.1$, $f_1 = 5$, $f_2 = 0.25$)
- $m$ is the vehicle mass (default: 1650 kg)
- $v_0$ is the reference cruise speed (default: 14 m/s)
- $z$ is the integral tracking error $\int (v_0 - v)\, dt$

The mass $m$ and reference speed $v_0$ can be overridden via `kwargs` in the dynamics call.

## Choosing a Model

| Use case | Recommended model |
|----------|-------------------|
| Quick prototyping, simple CBF demos | `SingleIntegrator2DOF` |
| Force/thrust-controlled agents, HOCBF research | `DoubleIntegrator2DOF` or `3DOF` |
| Differential-drive robots, Robotarium experiments | `Unicycle` |
| Adaptive cruise control, 1-D safety examples | `CruiseControl` |
| 3-D multi-agent scenarios (e.g. drones) | `SingleIntegrator3DOF` or `DoubleIntegrator3DOF` |

## Creating a Custom Model

To add a new dynamics model, subclass `Dynamics` and implement three methods:

```python
from kamaji.dynamics.dynamics import Dynamics
import numpy as np

class MyCustomDynamics(Dynamics):
    """Example: 2-D double integrator with damping."""

    @staticmethod
    def state_variables():
        return ["position_x", "position_y", "velocity_x", "velocity_y"]

    @staticmethod
    def control_variables():
        return ["force_x", "force_y"]

    def dynamics(self, t, state_dict, u, **kwargs):
        m = kwargs.get("mass", 1.0)
        b = kwargs.get("damping", 0.1)

        vx = state_dict["velocity_x"]
        vy = state_dict["velocity_y"]

        # dp/dt = v, dv/dt = (F - b*v) / m
        return np.array([
            vx,
            vy,
            (u[0] - b * vx) / m,
            (u[1] - b * vy) / m,
        ])
```

**Important conventions:**

1. The return array from `dynamics()` must be ordered to match `state_variables()`.
2. State variable names starting with `position_` are used by the simulator's collision detection.
3. Register the new class in the `model_map` dictionary inside `Agent.assign_dynamics()` so it can be referenced by name in YAML configs.
