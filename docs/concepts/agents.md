# Agent

The `Agent` class represents a single autonomous entity in the Kamaji simulation framework.  
An agent combines dynamics, controllers (including support for reinforcement learning policies), and simulation bookkeeping into a unified abstraction.

---

## Responsibilities

An Agent is responsible for:

- Maintaining its internal state
- Computing control inputs from one or more controllers
- Propagating dynamics forward in time
- Logging state and control trajectories

---

## Initialization

Agent(agent_config, t=0.0, dt=0.01, **kwargs)

### Parameters

- agent_config (dict)  
  Configuration defining agent ID, initial state, dynamics model, and controllers; can be loaded from .yml file

- t (float)  
  Initial simulation time

- dt (float)  
  Simulation timestep

- **kwargs  
  Arbitrary attributes attached to the agent instance

### Initialization workflow

During initialization, the agent:

1. Initializes and logs the starting state
2. Instantiates the configured dynamics model
3. Builds per-channel controllers
4. Validates controller–dynamics compatibility

---

## State Representation

The agent state is stored internally as a dictionary:

    {
      "x": ...,
      "y": ...,
      "vx": ...
    }

Note: these are example state elements, and is not representative of a specific dynamics model.

---

## Dynamics Models

[Dynamics](https://jcorbin406.github.io/Kamaji/concepts/dynamics/) are assigned through the configuration and instantiated internally.

Supported models include:

- Unicycle
- CruiseControl
- SingleIntegrator1DOF
- SingleIntegrator2DOF
- SingleIntegrator3DOF
- DoubleIntegrator1DOF
- DoubleIntegrator2DOF
- DoubleIntegrator3DOF

Each dynamics model must implement:

- state_variables()
- control_variables()
- dynamics(t, state, control)

All physical evolution is delegated to the selected dynamics model.

---

## Controllers

Controllers are defined per control channel using a declarative configuration.

### Constant Controller

Outputs a fixed control value.

    type: Constant
    specs:
      - value: 1.0

---

Supported Controllers include:

- PID
- Single-Agent Reinforcement Learning Policies

## Control Computation

    agent.compute_control(t, rl_action=None)

Behavior:

- Evaluates each control channel independently
- Combines channel outputs into a single control vector
- Supports hybrid classical and learning-based control
- When SARL is denoted as the Controller for multiple channels, all inputs associated with these channels are gathered as the input for a single policy, and all the specified control channels are the outputs of this policy

## Simulation Step

    agent.step(t, control_input)

A single simulation step:

1. Integrates dynamics
2. Updates the internal state
3. Logs state and control trajectories

---

## Logging

The agent automatically logs:

- State history
- Control history

Logs are stored as Pandas DataFrames and can be accessed via:

    agent.state_log  
    agent.control_log

---

## Valuation Functions

Agents can be assigned user-defined valuation functions:

    agent.set_valuation(fn)  
    agent.set_marginal_valuation(fn)

These functions are application-specific and not enforced by the simulation core.

---

## Resetting

    agent.reset(t=0.0)

Resets the agent state to its initial configuration.

---