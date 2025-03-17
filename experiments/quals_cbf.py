import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import cvxpy as cp

class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class ConsensusSimulation:
    def __init__(self, num_agents=10, default_speed=1.0, region_speed=2.0, dt=0.1, spawn_rate=0.2, min_distance=1.5):
        self.agents = []
        self.default_speed = default_speed
        self.region_speed = region_speed
        self.dt = dt
        self.spawn_rate = spawn_rate
        self.min_distance = min_distance
        
        # Define region spanning full x-range
        self.region_x_min = -15
        self.region_x_max = 15
        self.region_y_min = -2
        self.region_y_max = 2
        
        # Initial spawn: half outside, half inside region
        initial_outside = num_agents // 2
        initial_inside = num_agents - initial_outside
        
        for _ in range(initial_outside):
            self.spawn_agent_outside()
        
        for _ in range(initial_inside):
            self.spawn_agent_inside()

    def spawn_agent_outside(self):
        spawn_above = np.random.choice([True, False])
        pos_x = np.random.uniform(self.region_x_min, -5)
        pos_y = np.random.uniform(5, 10) if spawn_above else np.random.uniform(-10, -5)
        pos = np.array([pos_x, pos_y])
        
        target_y = 0.0
        direction = np.array([0, target_y - pos_y])
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        vel = direction * self.default_speed * np.random.uniform(0.5, 1.5)
        self.agents.append(Agent(pos, vel))

    def spawn_agent_inside(self):
        pos_x = np.random.uniform(self.region_x_min, self.region_x_max)
        pos_y = np.random.uniform(self.region_y_min, self.region_y_max)
        pos = np.array([pos_x, pos_y])
        
        vel = np.array([self.region_speed * np.random.uniform(0.5, 1.0), 0.0])
        self.agents.append(Agent(pos, vel))

    def in_region(self, position):
        return (self.region_x_min <= position[0] <= self.region_x_max and 
                self.region_y_min <= position[1] <= self.region_y_max)

    def control_law(self):
        region_target_dir = np.array([1.0, 0.0])
        agents_to_remove = []
        alpha = 1.0  # CBF decay rate
        
        for i, agent in enumerate(self.agents):
            # Nominal control
            if self.in_region(agent.position):
                avg_velocity = np.zeros(2)
                region_agents = [a for a in self.agents if self.in_region(a.position)]
                if region_agents:
                    for a in region_agents:
                        avg_velocity += a.velocity / (np.linalg.norm(a.velocity) + 1e-6)
                    avg_velocity /= len(region_agents)
                u_nominal = (avg_velocity + region_target_dir) * 0.5 * self.region_speed
            else:
                target_pos = np.array([agent.position[0], 0.0])
                direction = target_pos - agent.position
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                u_nominal = direction * self.default_speed
            
            # QP setup
            u = cp.Variable(2)  # Control input [ux, uy]
            objective = cp.Minimize(cp.sum_squares(u - u_nominal))
            constraints = []
            
            # Speed limit
            max_speed = self.region_speed if self.in_region(agent.position) else self.default_speed
            constraints.append(cp.norm(u) <= max_speed)
            
            # Agent-agent collision avoidance
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    diff = agent.position - other_agent.position
                    h = np.dot(diff, diff) - self.min_distance**2
                    dh = 2 * diff @ (u - other_agent.velocity)  # Lie derivative
                    constraints.append(dh >= -alpha * h)
            
            # Wall constraints (inside region only)
            if self.in_region(agent.position):
                # Top wall: h = y_max - y
                h_top = self.region_y_max - agent.position[1]
                constraints.append(-u[1] >= -alpha * h_top)  # -uy = dh/dt
                
                # Bottom wall: h = y - y_min
                h_bottom = agent.position[1] - self.region_y_min
                constraints.append(u[1] >= -alpha * h_bottom)  # uy = dh/dt
            
            # Solve QP
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve()
                if prob.status == cp.OPTIMAL:
                    agent.velocity = u.value
                else:
                    agent.velocity = u_nominal  # Fallback to nominal if QP fails
            except:
                agent.velocity = u_nominal  # Fallback on solver error
            
            # Update position
            agent.position += agent.velocity * self.dt
            
            # Remove if at right edge
            if agent.position[0] >= self.region_x_max:
                agents_to_remove.append(agent)
        
        for agent in agents_to_remove:
            if agent in self.agents:
                self.agents.remove(agent)
        
        if np.random.random() < self.spawn_rate:
            self.spawn_agent_outside()

    def update(self, frame):
        self.control_law()
        plt.cla()
        
        plt.gca().add_patch(Rectangle((self.region_x_min, self.region_y_min),
                                    self.region_x_max - self.region_x_min,
                                    self.region_y_max - self.region_y_min,
                                    facecolor='green', alpha=0.3, edgecolor='g', linewidth=2))
        
        plt.grid(True)
        
        for agent in self.agents:
            color = 'r' if self.in_region(agent.position) else 'b'
            plt.plot(agent.position[0], agent.position[1], f'{color}o')
            plt.arrow(agent.position[0], agent.position[1], 
                     agent.velocity[0]*0.5, agent.velocity[1]*0.5, 
                     head_width=0.2, color=color)
        
        plt.xlim(-15, 15)
        plt.ylim(-10, 10)
        plt.title(f"Agents: {len(self.agents)} (Red = in region)")

def main():
    sim = ConsensusSimulation(num_agents=10, default_speed=1.0, region_speed=2.0, dt=0.1, spawn_rate=0.2, min_distance=1.5)
    fig = plt.figure(figsize=(10, 5))
    ani = FuncAnimation(fig, sim.update, frames=500, interval=50, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()