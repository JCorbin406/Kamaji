# Examples

Kamaji includes several examples to showcase key features and use cases.

---

## Basic Simulation

**File**: `examples/basic_simulation.py`  
Simulates multiple agents with simple dynamics in a 2D space.

---

## PID-Controlled Navigation

**File**: `examples/pid_controller_demo.py`  
Shows how PID controllers can be used to follow a trajectory.

---

## Auction-Based Collision Avoidance

**File**: `examples/auction_based_cbf.py`  
Demonstrates multi-agent safety enforcement via auction-based allocation of control effort.

---

## Urban Air Mobility Scenario (3D)

**File**: `examples/uam_formation.py`  
Simulates a UAM scenario with 3D coordinated agents using geometric controllers.

---

## Symbolic CBF-Based Collision Avoidance

**File**: `examples/cbf_simulation.py`  
Uses a symbolic Control Barrier Function (CBF) system to enforce pairwise collision avoidance between agents based on a minimum separation distance.  
Agents are defined through a YAML configuration, and the CBF constraints are applied via a runtime-constructed symbolic filter.

This simulation uses the example YAML file `examples\configs\cbf_simulation.yml`.

---

### Key Steps

1. Define symbolic CBFs between all agent pairs using `make_cbf_system()`
2. Load the YAML configuration to initialize agents and their controllers
3. Inject the CBF system using `sim.set_cbf_system(...)`
4. Run the simulation and visualize trajectories with guaranteed minimum separation

---

### How `make_cbf_system()` Works

The `make_cbf_system()` function constructs a symbolic representation of **pairwise separation constraints** between agents using SymPy. The constraints are shaped like:


\\[ h_{ij}(x) = \lVert p_i - p_j \rVert^2 - r^2 \\]


This ensures each pair of agents maintains a distance of at least `r`.

#### Breakdown

| Step | Purpose |
|------|---------|
| **1. Symbol Assignment** | Creates symbolic position variables \\(x_i, y_i\\) for each agent |
| **2. Global Dynamics Setup** | Constructs zero drift `f` and identity matrix `g` assuming single-integrator agents |
| **3. Pairwise CBF Creation** | Defines one constraint \\(h_{ij}\\) for each agent pair using Euclidean distance |
| **4. Matrix Slicing** | Extracts the relevant rows from `f` and `g` for each CBF |
| **5. CBFSystem Population** | Adds each symbolic CBF to the system with `cbf_sys.add_cbf(...)` |

---

### Highlight: Symbolic CBF Definition

```python
h = (xi - xj)**2 + (yi - yj)**2 - radius**2
cbf_sys.add_cbf(
    cbf_id=f"cbf_{agent_i._id}_{agent_j._id}",
    agents=[agent_i._id, agent_j._id],
    state_vars=[xi, yi, xj, yj],
    h_expr=h,
    f_expr=f_sub,
    g_expr=g_sub,
    alpha_func=lambda h: 2.0 * h
)
```

---

## Visualization

Many examples support real-time visualization via matplotlib or the PyQt GUI:

```bash
python kamaji/gui/gui_main.py
```