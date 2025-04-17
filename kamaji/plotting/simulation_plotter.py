import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import numpy as np

class SimulationPlotter:
    def __init__(self, simulator, radius=0.5):
        self.sim = simulator
        self.radius = radius
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    def trajectories(self, show=True, initial_conditions=False):
        agents = self.sim.inactive_agents if self.sim.inactive_agents else self.sim.active_agents
        fig, ax = plt.subplots()
        x_all, y_all = [], []

        for idx, agent in enumerate(agents):
            x = agent.state_log['position_x']
            y = agent.state_log['position_y']
            color = self.colors[idx % len(self.colors)]
            ax.plot(x, y, label=agent._id, color=color)
            if initial_conditions:
                circle = Circle((x.iloc[0], y.iloc[0]), radius=self.radius, alpha=0.2, color=color)
                ax.add_patch(circle)
            x_all += list(x)
            y_all += list(y)

        ax.set_xlim(min(x_all) - 1, max(x_all) + 1)
        ax.set_ylim(min(y_all) - 1, max(y_all) + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Agent Trajectories")
        ax.grid(True)
        ax.legend()

        if show:
            plt.show()

    def controls(self, agent_id, show=True):
        agent = next(a for a in self.sim.inactive_agents if a._id == agent_id)
        df = agent.control_log
        fig, ax = plt.subplots()
        for col in df.columns:
            if col.startswith('u'):
                ax.plot(df['time'], df[col], label=col)
        ax.set_title(f"Control Inputs for {agent_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Control Input")
        ax.grid(True)
        ax.legend()
        if show:
            plt.show()

    def rms_error(self, desired_y_by_agent, show=True):
        errors = []
        labels = []
        for agent in self.sim.inactive_agents:
            if agent._id in desired_y_by_agent:
                y = np.array(agent.state_log['position_y'])
                desired = desired_y_by_agent[agent._id]
                rms = np.sqrt(np.mean((y - desired)**2))
                errors.append(rms)
                labels.append(agent._id)

        fig, ax = plt.subplots()
        ax.bar(labels, errors, color='steelblue')
        ax.set_ylabel("RMS Error in y (m)")
        ax.set_title("RMS Y-Position Error")
        ax.grid(True, linestyle='--', alpha=0.6)
        if show:
            plt.show()

    def animate_trajectories(self):
        agents = self.sim.inactive_agents if self.sim.inactive_agents else self.sim.active_agents
        fig, ax = plt.subplots()
        lines, points, circles = [], [], []
        x_all, y_all = [], []

        for idx, agent in enumerate(agents):
            x = agent.state_log['position_x']
            y = agent.state_log['position_y']
            color = self.colors[idx % len(self.colors)]
            lines.append(ax.plot([], [], color=color, label=agent._id)[0])
            points.append(ax.plot([], [], marker='o', color=color)[0])
            circle = Circle((0, 0), radius=self.radius, color=color, alpha=0.2)
            ax.add_patch(circle)
            circles.append(circle)
            x_all += list(x)
            y_all += list(y)

        ax.set_xlim(min(x_all) - self.radius - 1, max(x_all) + self.radius + 1)
        ax.set_ylim(min(y_all) - self.radius - 1, max(y_all) + self.radius + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            for circle in circles:
                circle.set_center((0, 0))
            return lines + points + circles

        def update(frame):
            for idx, agent in enumerate(agents):
                x = agent.state_log['position_x']
                y = agent.state_log['position_y']
                lines[idx].set_data(x[:frame+1], y[:frame+1])
                points[idx].set_data([x[frame]], [y[frame]])
                circles[idx].set_center((x[frame], y[frame]))
            return lines + points + circles

        ani = FuncAnimation(fig, update, frames=len(agents[0].state_log), init_func=init,
                            blit=True, interval=100, repeat=True)
        plt.show()
