# Environment & Simulation Loop

The `Simulator` class is the central orchestrator in Kamaji. It manages agents, runs the simulation loop, handles CBF safety filtering, detects collisions, and optionally integrates with Gymnasium for reinforcement learning.

## Simulation Loop

Each call to `simulator.step()` advances the simulation by one time step ($\Delta t$). The full pipeline is:

```
┌─────────────────────────────────────┐
│  1. Gather state from all agents    │
│     → build global state_values     │
├─────────────────────────────────────┤
│  2. Compute nominal control         │
│     → each agent's controllers      │
│     → concatenate into u_nom        │
├─────────────────────────────────────┤
│  3. CBF safety filter (if active)   │
│     → solve QP: min ||u - u_nom||²  │
│       s.t. CBF constraints          │
│     → u_filtered                    │
├─────────────────────────────────────┤
│  4. Slice controls back to agents   │
│     → each agent gets its portion   │
│     → integrate dynamics (RK4)      │
│     → update agent state            │
├─────────────────────────────────────┤
│  5. Collision detection             │
│     → pairwise distance check       │
│     → remove colliding agents       │
├─────────────────────────────────────┤
│  6. Update Gym environments         │
│     → push new states to WrapperEnv │
└─────────────────────────────────────┘
```

### Step-by-Step Details

**1. Gather state.** The simulator reads each agent's current state dictionary and builds a flat `state_values` dictionary with indexed keys (e.g. `position_x_0`, `position_y_0` for agent 0, `position_x_1`, `position_y_1` for agent 1). This global view is needed by the CBF system, which operates on the joint state of all agents.

**2. Compute nominal control.** Each agent calls `compute_control(t)`, which iterates over its control channels in sorted order, queries each controller (PID, Constant, or SARL), and concatenates the results. All per-agent control vectors are then concatenated into a single `u_nom` vector.

**3. CBF safety filter.** If a `CBFSystem` has been attached via `simulator.set_cbf_system()`, the concatenated `u_nom` is filtered through the safety QP. The result is `u_filtered` — the minimally-modified control vector that satisfies all barrier function constraints. If no CBF system is set, `u_filtered = u_nom`.

**4. Slice and integrate.** The filtered control vector is sliced back into per-agent portions (based on each agent's control dimension). Each agent then integrates its dynamics forward by $\Delta t$ using the RK4 solver and updates its internal state.

**5. Collision detection.** The simulator checks all pairs of active agents for collisions. For each pair, it extracts position state variables (those starting with `position_`), computes the Euclidean distance between the two agents, and compares it against the sum of their collision radii (default 0.25 each). Colliding agents are moved to the inactive list.

**6. Gym update.** If Gymnasium environments have been created, the simulator pushes the updated state to each `WrapperEnv` so that RL agents can observe the new state.

## Running a Simulation

### From YAML config

```python
import yaml
from kamaji.simulation.simulator import Simulator

with open("my_config.yml") as f:
    config = yaml.safe_load(f)

sim = Simulator(config)
sim.simulate()
```

### Programmatic setup

```python
from kamaji.simulation.simulator import Simulator

sim = Simulator()
sim.set_sim_params({
    "time_step": 0.01,
    "duration": 10.0,
    "integrator": "RK4"
})

agent_config = {
    "type": "default",
    "initial_state": {"position_x": 0.0, "position_y": 0.0},
    "dynamics_model": "SingleIntegrator2DOF",
    "controller": {
        "velocity_x": {"type": "Constant", "specs": [{"value": 1.0}]},
        "velocity_y": {"type": "PID", "specs": [{
            "state": "position_y", "goal": 5.0,
            "kp": 1.0, "ki": 0.0, "kd": 0.1
        }]}
    }
}

sim.add_agents(agent_config)
sim.simulate()
```

### Step callback

You can inject custom logic at each time step using the `on_step` callback:

```python
def my_callback(simulator, step_idx):
    # Example: update PID goal halfway through
    if step_idx == simulator.num_timesteps // 2:
        agent = simulator.active_agents[0]
        agent.control_model["velocity_y"].set_goal([10.0])

sim.simulate(on_step=my_callback)
```

## Collision Detection

Collision detection runs automatically at the end of each time step. The algorithm:

1. For each pair of active agents $(i, j)$, find the common position axes (state variables whose names start with `position_`).
2. Compute the Euclidean distance: $d_{ij} = \| p_i - p_j \|_2$.
3. If $d_{ij} < r_i + r_j$ (where $r$ is the agent's `radius` attribute, default 0.25), both agents are flagged for removal.
4. Flagged agents are moved from the active list to the inactive list. Their state/control history is preserved.

If all agents are removed, the simulation ends early.

## Gymnasium Integration

Kamaji supports training RL agents via the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) library. The integration works through `WrapperEnv`, a Gymnasium-compatible environment that wraps the Kamaji simulator.

### Setup

```python
def reward_fn(prev_env_state, action, new_env_state):
    """Compute step reward from environment transition."""
    pos = new_env_state["agent_1"]["position_x"]
    return -abs(pos - 5.0)  # reward proximity to goal

def termination_fn(prev_env_state, new_env_state):
    """Check if episode should terminate. Returns (terminated, bonus_reward)."""
    if new_env_state["agent_1"] is None:  # agent was removed (collision)
        return True, -100.0
    return False, 0.0

def truncation_fn(sim_time):
    """Check if episode should be truncated (e.g. time limit)."""
    return sim_time >= 10.0

sim.create_gym_envs(
    agent_ids=["agent_1"],
    reward_fns=[reward_fn],
    termination_fns=[termination_fn],
    truncation_fns=[truncation_fn],
)

# Train
sim.train_agent_controller("agent_1", training_steps=50000)

# Run trained policy
sim.simulate()
```

### How WrapperEnv Works

Each `WrapperEnv` instance:

- **Observation space:** A `Box` with dimension equal to the number of `state_features` defined in the SARL controller config. Values are extracted from the agent's state dictionary.
- **Action space:** A `Box` with dimension equal to the number of SARL-controlled channels. Default bounds are $[-10, 10]$.
- **Step:** Calls `simulator.step((agent_id, action))`, which advances *all* agents by one time step (the provided action overrides the SARL prediction for this agent, while other agents use their normal controllers).
- **Reset:** Calls `simulator.reset()`, restoring all agents to their initial states.

### Key design points

- The simulator is **shared** — when one RL agent steps, all agents advance. This means multi-agent interactions are captured naturally.
- Reward, termination, and truncation are **callback-based**, giving you full flexibility to define the learning objective.
- An agent whose state becomes `None` in the environment state (due to collision removal) triggers automatic termination.

## Configuration

For details on the YAML configuration format (simulation parameters, agent specs, logging), see [Configuration](configuration.md).
