import numpy as np

def inject_manual_control(sim, t):
    for agent in sim.active_agents:
        x = agent.state["position_x"]
        y = agent.state["position_y"]
        dx, dy = -x, -y
        u = np.array([dx, dy])
        norm = np.linalg.norm(u)
        if norm > 1e-6:
            u = u / norm
        sim.set_manual_control(agent._id, u)
