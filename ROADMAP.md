# Kamaji Improvement Roadmap

## Context
Kamaji is a multi-agent simulation framework built for research in control theory, CBFs, and auction-based coordination. The goal is to mature it into a publicly accessible, well-documented tool that researchers can install and use for their own multi-agent simulation work. The codebase has solid architecture but needs cleanup, better documentation, performance improvements, and packaging polish.

This roadmap is organized into phases that can be tackled incrementally.

---

## Phase 1: Foundation — Fix Critical Issues & Packaging
*Goal: Make Kamaji installable, correct, and safe to use.*

### 1.1 Fix `setup.py` / `pyproject.toml` dependencies — DONE
- [x] Added all missing runtime dependencies to `pyproject.toml`: `sympy`, `qpsolvers`, `cvxopt`, `pandas`, `h5py`, `networkx`, `tqdm`, `pyyaml`, `numpy`, `scipy`, `matplotlib`
- [x] Added optional dependency groups: `[rl]` for stable-baselines3/gymnasium, `[gui]` for dearpygui/pyqtgraph, `[doc]` for mkdocs, `[dev]` for pytest/numba
- [x] Removed dead `entry_points` (`kamaji.simulation:main` doesn't exist)
- [x] Consolidated `setup.py` into a thin shim — all config lives in `pyproject.toml`
- [x] Fixed `packages.find` include pattern to `kamaji*` (catches subpackages)

### 1.2 Fix CBF silent failure on infeasibility — DONE
- [x] Replaced `print()` with Python `logging` module (`logger.warning`)
- [x] Added `CBFFilterResult` return object with `.control` and `.feasible` fields so callers know when safety was violated
- [x] Added `QPInfeasibleError` exception class for strict-mode use
- [x] Added optional `on_infeasible` callback parameter to `CBFSystem.__init__`
- [x] Updated `simulator.py` to unwrap `CBFFilterResult` objects
- **Files modified:** `kamaji/controllers/CBF.py`, `kamaji/simulation/simulator.py`

### 1.3 Fix hardcoded state field names in Simulator — DONE
- [x] Refactored CBF state gathering to derive variable names from agents' dynamics models instead of hardcoding `position_x`, `position_y` (uses `f"{var_name}_{idx}"` pattern)
- [x] Refactored collision detection to discover position keys dynamically (any state var starting with `position_`)
- **File modified:** `kamaji/simulation/simulator.py`
- **Note:** CBF variable naming convention changed — CBF definitions must now use `{state_var}_{agent_idx}` format (e.g., `position_x_0` instead of `x0`)

### 1.4 Fix ODE solver issues — DONE
- [x] Fixed `solve_ode()` signature — now correctly passes `u` to all step functions
- [x] Fixed RK45 adaptive stepping — rejected steps now retry with smaller `h` in a loop instead of accepting stale state
- [x] Removed commented `@njit` decorators and `numba` import
- [x] Removed `__main__` benchmark block (was ~100 lines of test/benchmark code)
- [x] Removed stale debug `print(self.num_timesteps)` in `simulator.py:107`
- [x] Added `min_h` safety floor in RK45 to prevent infinite retry loops
- **Files modified:** `kamaji/tools/ode_solvers.py`, `kamaji/simulation/simulator.py`

### 1.5 Clean up wildcard imports — DONE
- [x] Replaced `from kamaji.dynamics.dynamics import *` with explicit imports of all 8 dynamics classes
- [x] Replaced `from kamaji.controllers.controllers import *` with explicit imports of `Controls, PID, Constant, SARLPolicy`
- [x] Added explicit `import gymnasium as gym` (was leaking through wildcard)
- **File modified:** `kamaji/agent/agent.py`

---

## Phase 2: Code Quality — Refactor & Clean Up
*Goal: Make the codebase readable, maintainable, and idiomatic.*

### 2.1 Deduplicate dynamics models
- [ ] Replace 3 SingleIntegrator classes with one parameterized `SingleIntegrator(dof=N)` class
- [ ] Replace 3 DoubleIntegrator classes with one parameterized `DoubleIntegrator(dof=N)` class
- [ ] Keep backward-compatible aliases (`SingleIntegrator2DOF = SingleIntegrator(dof=2)`) or update config parsing
- [ ] Remove unused state variable reads in DoubleIntegrator (lines 153-154, 172-173)
- **File:** `kamaji/dynamics/dynamics.py`

### 2.2 Clean up auction module
- [ ] Remove commented-out code blocks (multiple payment method variants)
- [ ] Choose canonical payment method and document it
- [ ] Rename the auction `Agent` class to `AuctionParticipant` or similar to avoid collision with `kamaji.agent.Agent`
- [ ] Move archived auction code to `Archive/` or remove entirely
- **File:** `kamaji/auctions/resource.py`

### 2.3 Add input validation
- [ ] Agent: validate state dict keys match dynamics model's `state_variables()` at init
- [ ] Agent: check for NaN/Inf after each `step()` call
- [ ] Dynamics: validate control vector length matches `control_variables()`
- [ ] CBF: validate symbolic variable names exist in state_values dict before evaluation
- [ ] Controllers: make PID windup limits configurable (currently hardcoded to +/-10)

### 2.4 Improve Agent control pipeline
- [ ] Cache SARL model reference during `assign_controller()` instead of searching on every `compute_control()` call
- [ ] Make ODE solver configurable per-agent (currently hardcoded to RK4 in `agent.step()`)

### 2.5 Replace print statements with logging
- [ ] Add `import logging` throughout
- [ ] Replace all `print()` calls with appropriate log levels
- [ ] Let users configure verbosity via standard Python logging config

---

## Phase 3: Documentation
*Goal: Make Kamaji's docs a clear, comprehensive reference for new users.*

### 3.1 Code-level documentation (docstrings)
Priority files for docstring improvement:
- [ ] `kamaji/dynamics/dynamics.py` — document each dynamics model: equations of motion, state meaning, units, when to use
- [ ] `kamaji/controllers/controllers.py` — document PID tuning, Constant usage, SARLPolicy lifecycle
- [ ] `kamaji/controllers/CBF.py` — document the symbolic pipeline, how to define custom CBFs, QP solver behavior
- [ ] `kamaji/simulation/simulator.py` — already has good docstrings, minor cleanup
- [ ] `kamaji/agent/agent.py` — document the control computation flow, state management

Style: Google-style docstrings (compatible with mkdocstrings)

### 3.2 Fill empty mkdocs pages
- [ ] `docs/concepts/dynamics.md` — overview of available dynamics, equations, table, custom model how-to
- [ ] `docs/concepts/controls.md` — PID, CBF, SARL docs, custom controller how-to
- [ ] `docs/concepts/environment.md` — sim loop, collision detection, Gym integration

### 3.3 Improve API reference
- [ ] Add narrative context around auto-generated docs
- [ ] Add usage examples for key methods
- [ ] Add pages for: CBFSystem, SimulationLogger, SimulationPlotter, ODE solvers

### 3.4 Add tutorials
- [ ] Tutorial 1: "Your first simulation" — YAML config to plotted trajectories
- [ ] Tutorial 2: "Adding CBF safety" — symbolic CBF setup
- [ ] Tutorial 3: "Training an RL agent" — reward design, training, evaluation
- [ ] Tutorial 4: "Creating a custom dynamics model"
- [ ] Tutorial 5: "Creating a custom controller"

### 3.5 Fix README
- [ ] Fill in "Step 5: Running Kamaji" (currently TBD)
- [ ] Add badges (license, Python version, docs link)
- [ ] Add a minimal code example that runs without a config file
- [ ] Link to full docs site

---

## Phase 4: Performance
*Goal: Make simulations fast enough for RL training and large-scale experiments.*

### 4.1 Profile current bottlenecks
- [ ] Profile a typical simulation (100 agents, 10k steps) to identify hotspots

### 4.2 Vectorize agent stepping
- [ ] Batch all agents into a single state matrix and vectorize the dynamics computation

### 4.3 Replace DataFrame logging with array-based logging
- [ ] Pre-allocate NumPy arrays for state/control history
- [ ] Convert to DataFrame only when accessed (lazy conversion)

### 4.4 Optimize CBF evaluation
- [ ] Consider batching constraint evaluation across all pairs
- [ ] For large agent counts, use spatial hashing to skip distant pairs

### 4.5 Enable Numba/JAX acceleration
- [ ] Refactor ODE step functions to work with pure NumPy arrays (no dicts, no closures)
- [ ] Gate behind a `use_jit=True` flag

### 4.6 Parallel simulation runs
- [ ] Add utility for running multiple independent simulations in parallel (multiprocessing)

---

## Phase 5: Testing
*Goal: Build confidence that changes don't break things.*

### 5.1 Set up pytest infrastructure
- [ ] Create `tests/` directory with `conftest.py` and common fixtures
- [ ] Add to CI pipeline (`.github/workflows/`)

### 5.2 Core unit tests
- [ ] `test_dynamics.py` — verify derivatives against analytical solutions
- [ ] `test_agent.py` — state init, control computation, stepping, logging
- [ ] `test_simulator.py` — config loading, agent management, sim loop, collision detection
- [ ] `test_cbf.py` — symbolic computation, constraint evaluation, QP filtering, infeasibility handling
- [ ] `test_controllers.py` — PID response, Constant output, SARL initialization
- [ ] `test_ode_solvers.py` — accuracy against analytical solutions

### 5.3 Integration tests
- [ ] End-to-end: load config -> simulate -> verify trajectories
- [ ] CBF: verify safety constraints are never violated
- [ ] RL: verify gym environment step/reset/reward contract

---

## Phase 6: Nice-to-Have Features
*Goal: Expand capabilities once the foundation is solid.*

### 6.1 HOCBF support in CBFSystem
- [ ] Add support for arbitrary relative degree with recursive auxiliary functions

### 6.2 GUI improvements
- [ ] Consider switching to web-based interface (Streamlit or Panel)

### 6.3 3D visualization
- [ ] Add optional 3D trajectory visualization (Plotly)

### 6.4 ROS2 bridge
- [ ] Publish/subscribe agent states and controls via ROS2 topics

### 6.5 Config validation schema
- [ ] Add JSON Schema or Pydantic model for YAML config validation

---

## Implementation Order (Suggested)

| Priority | Phase | Estimated Effort | Impact |
|----------|-------|-----------------|--------|
| ~~**Now**~~ | ~~Phase 1 (Foundation)~~ | ~~Small-Medium~~ | ~~Correctness, installability~~ |
| **Next** | Phase 2.1-2.3 (Key refactors) | Medium | Readability, maintainability |
| **Next** | Phase 3.1-3.2 (Core docs) | Medium | Usability for new users |
| **Then** | Phase 5.1-5.2 (Core tests) | Medium | Confidence in changes |
| **Then** | Phase 4.1-4.3 (Quick perf wins) | Medium | RL training speed |
| **Later** | Phase 3.3-3.5 (Tutorials, API docs) | Medium-Large | Onboarding experience |
| **Later** | Phase 4.4-4.6 (Advanced perf) | Large | Scale to 100+ agents |
| **Later** | Phase 2.4-2.5 (Polish) | Small | Code quality |
| **Eventually** | Phase 6 (Features) | Large | Capability expansion |
