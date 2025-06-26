"""Basic simulation using YAML configuration for Kamaji, 
with symbolic Control Barrier Functions (CBFs) for pairwise collision avoidance."""

import yaml
from kamaji.simulation.simulator import Simulator
from sympy import symbols, Matrix, sqrt
import numpy as np
from kamaji.controllers.CBF import CBFSystem  # Import your symbolic CBF system class


if __name__ == "__main__":
    def make_cbf_system(agents, radius=1.0):
        """
        Generate a CBFSystem that enforces pairwise safety constraints between agents.

        Args:
            agents (List[Agent]): List of active agents in the simulation.
            radius (float): Minimum separation distance between agents.

        Returns:
            CBFSystem: Configured control barrier function system.
        """
        cbf_sys = CBFSystem()
        all_vars = []             # Flat list of all state symbols (x0, y0, x1, y1, ...)
        agent_symbols = {}        # Mapping from agent ID -> (x, y) symbols

        # Step 1: Assign symbolic variables for each agent
        for idx, agent in enumerate(agents):
            x, y = symbols(f"x{idx} y{idx}")
            all_vars += [x, y]
            agent_symbols[agent._id] = (x, y)

        # Step 2: Create full system dynamics (0 drift, identity input)
        f = Matrix([0.0] * len(all_vars))     # No drift
        g = Matrix.eye(len(all_vars))         # Control input is directly applied to state

        # Step 3: Add pairwise separation constraints
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if j <= i:
                    continue  # Avoid duplicate or self-pairs

                xi, yi = agent_symbols[agent_i._id]
                xj, yj = agent_symbols[agent_j._id]

                # Define pairwise safety constraint: h(x) = ||p_i - p_j|| - r ≥ 0
                h = (xi - xj)**2 + (yi - yj)**2 - radius**2
                var_list = [xi, yi, xj, yj]  # variables used in this CBF

                # Get indices of these vars in full state vector
                idxs = [all_vars.index(v) for v in var_list]
                f_sub = f.extract(idxs, [0])                          # extract relevant rows of f
                g_sub = g.extract(idxs, list(range(g.shape[1])))     # extract relevant rows of g

                # Add the CBF to the system
                cbf_sys.add_cbf(
                    cbf_id=f"cbf_{agent_i._id}_{agent_j._id}",
                    agents=[agent_i._id, agent_j._id],
                    state_vars=var_list,
                    h_expr=h,
                    f_expr=f_sub,
                    g_expr=g_sub,
                    alpha_func=lambda h: 2.0 * h  # class K function
                )

        return cbf_sys

    # Load simulation configuration from a YAML file
    config_path = "examples/configs/cbf_simulation.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Initialize the simulator from the YAML config
    sim = Simulator(config)

    # Construct and inject a pairwise-CBF system
    cbf_sys = make_cbf_system(sim.active_agents, radius=1.0)
    sim.set_cbf_system(cbf_sys)

    # Run the simulation
    sim.simulate()

    # Animate the resulting trajectories
    sim.plot.animate_trajectories()
