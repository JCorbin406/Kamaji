# Controllers

Controllers in Kamaji compute control inputs for each agent at every time step. Kamaji uses a **per-channel** controller architecture: each control variable (e.g. `velocity_x`, `velocity_y`) is assigned its own controller in the YAML configuration. The outputs are concatenated into the full control vector before being passed to the dynamics model.

## Architecture Overview

```
Agent Config (YAML)
  └── controller:
        ├── velocity_x:  { type: PID, specs: [...] }
        └── velocity_y:  { type: PID, specs: [...] }
                ↓
        Per-channel outputs concatenated
                ↓
            u = [u_x, u_y]
                ↓
        (Optional) CBF safety filter
                ↓
            Dynamics model
```

Control channels are sorted alphabetically by name, so the ordering of the concatenated control vector is deterministic and matches the dynamics model's `control_variables()` list.

## Built-in Controllers

### Constant

Returns a fixed value every time step. Useful for open-loop testing or stationary agents.

**YAML example:**

```yaml
velocity_x:
  type: Constant
  specs:
    - value: 1.0
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | float | The constant control output |

### PID

A Proportional-Integral-Derivative controller with anti-windup and derivative smoothing.

$$u = K_p \, e + K_i \int e \, dt + K_d \, \dot{e}$$

where $e = \text{goal} - \text{state}$.

**YAML example:**

```yaml
velocity_x:
  type: PID
  specs:
    - state: position_x
      goal: 5.0
      kp: 1.0
      ki: 0.01
      kd: 0.1
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | string | Name of the state variable to track |
| `goal` | float | Target setpoint |
| `kp` | float | Proportional gain |
| `ki` | float | Integral gain |
| `kd` | float | Derivative gain |

**Implementation details:**

- **Anti-windup:** The integral term is clamped to $[-10, 10]$ to prevent integral windup during sustained error.
- **Derivative smoothing:** The raw derivative is passed through an exponential moving average filter with coefficient $\alpha = 0.1$ (configurable). This reduces noise amplification from the derivative term.
- **Output saturation:** Optional `lower_limit` and `upper_limit` can be set programmatically to clip the control output.

**Gain tuning tips:**

1. Start with only $K_p$ (set $K_i = K_d = 0$). Increase until the response is fast but not oscillating.
2. Add $K_d$ to dampen overshoot.
3. Add $K_i$ last if you need zero steady-state error. Keep it small — large $K_i$ causes windup and oscillation.

The goal can be updated at runtime by calling `set_goal()` on the PID controller instance.

### CBF Safety Filter

The CBF (Control Barrier Function) safety filter is not a standalone controller — it is a **post-processing layer** applied to the nominal control vector computed by the per-channel controllers. It solves a Quadratic Program that finds the closest safe control to the nominal:

$$u^* = \arg\min_u \| u - u_{\text{nom}} \|^2 \quad \text{s.t.} \quad L_f h + L_g h \, u + \alpha(h) \geq 0$$

for each registered CBF constraint.

**Setup (programmatic):**

```python
from sympy import symbols, sqrt, Matrix
from kamaji.controllers.CBF import CBFSystem

# Define symbolic state variables
x0, y0, x1, y1 = symbols('x0 y0 x1 y1')
state_vars = [x0, y0, x1, y1]

# Barrier function: inter-agent distance >= safety radius
h = sqrt((x0 - x1)**2 + (y0 - y1)**2) - 0.5

# Drift and actuation (single-integrator: f=0, g=I)
f = Matrix([0, 0, 0, 0])
g = Matrix.eye(4)

# Class-K function
alpha = lambda h_val: 2.0 * h_val

# Create system and add constraint
cbf = CBFSystem()
cbf.add_cbf("collision_01", agents=[0, 1],
            state_vars=state_vars, h_expr=h,
            f_expr=f, g_expr=g, alpha_func=alpha)

# Attach to simulator
sim.set_cbf_system(cbf)
```

The symbolic pipeline automatically computes Lie derivatives ($L_f h$, $L_g h$) via SymPy and lambdifies them for fast numerical evaluation.

**Infeasibility handling:**

If the QP has no feasible solution (constraints cannot all be satisfied simultaneously), the default behavior is to fall back to the nominal control with a warning. You can override this by passing an `on_infeasible` callback to `CBFSystem`:

```python
def my_fallback(state_values, u_nom):
    return CBFFilterResult(control=np.zeros_like(u_nom), feasible=False)

cbf = CBFSystem(on_infeasible=my_fallback)
```

### SARL (Reinforcement Learning Policy)

The `SARLPolicy` controller wraps a Stable-Baselines3 RL model. It observes a subset of state features and outputs actions for one or more control channels.

**YAML example:**

```yaml
velocity_x:
  type: SARLPolicy
  specs:
    - state_features: [position_x, goal_x]
      model: PPO
velocity_y:
  type: SARLPolicy
  specs:
    - state_features: [position_y, goal_y]
      model: PPO
```

**Supported algorithms:** PPO, SAC, DDPG, A2C, TD3

**Lifecycle:**

1. **Configuration** — The YAML specifies `state_features` and `model` for each RL-controlled channel.
2. **Aggregation** — All SARL channels on one agent are combined into a single `SARLPolicy` instance (one model per agent). State features are unioned; action outputs are the channel names.
3. **Gym environment** — Call `simulator.create_gym_envs()` with reward, termination, and truncation callbacks to create a `WrapperEnv` for the agent.
4. **Training** — Call `simulator.train_agent_controller(agent_id, training_steps)` to train via SB3's `.learn()`.
5. **Inference** — During simulation, the policy's `.predict()` is called each step to produce actions.

**Important:** Only one RL model name can be specified per agent. All SARL channels must use the same algorithm.

## Creating a Custom Controller

Subclass `Controls` and implement the `update()` method:

```python
from kamaji.controllers.controllers import Controls
import numpy as np

class BangBang(Controls):
    """Simple bang-bang controller that switches between two values."""

    def __init__(self, state_name, goal, u_high=1.0, u_low=-1.0):
        super().__init__()
        self.state_name = state_name
        self.goal = goal
        self.u_high = u_high
        self.u_low = u_low

    def update(self, t, state_dict):
        error = self.goal - state_dict[self.state_name]
        return np.array([self.u_high if error > 0 else self.u_low])
```

To use it in YAML configs, add an `elif` branch for your new type in `Agent.assign_controller()`.
