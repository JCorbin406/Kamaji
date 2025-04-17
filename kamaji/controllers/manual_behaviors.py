import numpy as np

class ManualBehavior:
    """Base class for all manual control strategies."""
    def update(self, sim, t):
        raise NotImplementedError("Override this method in subclasses.")


class GoToOrigin(ManualBehavior):
    def update(self, sim, t):
        for agent in sim.active_agents:
            x = agent.state.get("position_x", 0.0)
            y = agent.state.get("position_y", 0.0)
            dx = -x
            dy = -y
            u = np.array([dx, dy])
            norm = np.linalg.norm(u)
            if norm > 1e-6:
                u = u / norm * 1.0
            sim.set_manual_control(agent._id, u)


class ConstantControl(ManualBehavior):
    def __init__(self, control_vector):
        self.u = np.array(control_vector)

    def update(self, sim, t):
        for agent in sim.active_agents:
            sim.set_manual_control(agent._id, self.u)
