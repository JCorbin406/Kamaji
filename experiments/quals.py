
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Rectangle
# import sympy as sp
# from qpsolvers import solve_qp

# class Agent:
#     def __init__(self, position, velocity):
#         self.position = np.array(position, dtype=float)
#         self.velocity = np.array(velocity, dtype=float)

# class ConsensusSimulation:
#     def __init__(self, num_agents=10, default_speed=1.0, region_speed=2.0, dt=0.05, spawn_rate=0.2):
#         self.agents = []
#         self.default_speed = default_speed
#         self.region_speed = region_speed
#         self.dt = dt
#         self.spawn_rate = spawn_rate
        
#         # Define region
#         self.region_x_min = -15
#         self.region_x_max = 15
#         self.region_y_min = -3
#         self.region_y_max = 3
        
#         # Initial spawn
#         initial_outside = num_agents // 2
#         initial_inside = num_agents - initial_outside
        
#         for _ in range(initial_outside):
#             self.spawn_agent_outside()
#         for _ in range(initial_inside):
#             self.spawn_agent_inside()

#         # CBF setup
#         x1, x2, y1, y2, r = sp.symbols('x1 x2 y1 y2 r')
#         h = sp.sqrt((x1 - x2)**2 + (y1 - y2)**2) - r
#         f = sp.Matrix([0, 0])  # No drift term
#         g = sp.Matrix([[1, 0], [0, 1]])  # Direct velocity control
#         state_vars1 = sp.Matrix([x1, y1])
#         grad_h1 = h.diff(state_vars1)
#         Lfh = grad_h1.T * f  # Should be zero
#         Lgh = grad_h1.T * g
#         self.h = sp.lambdify((x1, y1, x2, y2, r), h)
#         self.Lfh = sp.lambdify((x1, y1, x2, y2, r), Lfh)
#         self.Lgh = sp.lambdify((x1, y1, x2, y2, r), Lgh)
#         self.min_distance = 1.0  # Safe distance

#     def spawn_agent_outside(self):
#         spawn_above = np.random.choice([True, False])
#         pos_x = np.random.uniform(self.region_x_min, -5)
#         pos_y = np.random.uniform(5, 10) if spawn_above else np.random.uniform(-10, -5)
#         pos = [pos_x, pos_y]
#         target_y = 0.0
#         direction = np.array([0, target_y - pos_y])
#         direction /= np.linalg.norm(direction) + 1e-6
#         vel = direction * self.default_speed * np.random.uniform(0.5, 1.5)
#         self.agents.append(Agent(pos, vel))

#     def spawn_agent_inside(self):
#         pos_x = np.random.uniform(self.region_x_min, self.region_x_max)
#         pos_y = np.random.uniform(self.region_y_min, self.region_y_max)
#         pos = [pos_x, pos_y]
#         vel = [self.region_speed * np.random.uniform(0.5, 1.0), 0.0]
#         self.agents.append(Agent(pos, vel))

#     def in_region(self, position):
#         return (self.region_x_min <= position[0] <= self.region_x_max and 
#                 self.region_y_min <= position[1] <= self.region_y_max)

#     def control_law(self):
#         region_target_dir = np.array([1.0, 0.0])
#         agents_to_remove = []
        
#         for i, agent in enumerate(self.agents):
#             if self.in_region(agent.position):
#                 avg_velocity = np.zeros(2)
#                 region_agents = [a for a in self.agents if self.in_region(a.position) and a is not agent]
                
#                 # Compute distances to other agents
#                 distances = [(a, np.linalg.norm(agent.position - a.position)) for a in region_agents]
#                 # Sort by distance and take the nearest 5 (or fewer if less than 5)
#                 nearest_agents = [a for a, _ in sorted(distances, key=lambda x: x[1])[:min(5, len(distances))]]
                
#                 num_constraints = len(nearest_agents)
#                 G = np.zeros((num_constraints, 2)) if num_constraints > 0 else None
#                 h = np.zeros(num_constraints) if num_constraints > 0 else None
                
#                 if region_agents:
#                     # Compute average velocity from all region agents
#                     for a in region_agents:
#                         avg_velocity += a.velocity / (np.linalg.norm(a.velocity) + 1e-6)
#                     avg_velocity /= len(region_agents)
                
#                 if nearest_agents:
#                     for idx, a in enumerate(nearest_agents):
#                         # CBF constraint: Lfh + Lgh @ u >= -alpha * h
#                         G[idx] = -self.Lgh(agent.position[0], agent.position[1], 
#                                           a.position[0], a.position[1], self.min_distance)
#                         h_val = self.h(agent.position[0], agent.position[1], 
#                                       a.position[0], a.position[1], self.min_distance)
#                         h[idx] = self.Lfh(agent.position[0], agent.position[1], 
#                                          a.position[0], a.position[1], self.min_distance) + 8.0 * h_val
                
#                 avg_dir = avg_velocity / (np.linalg.norm(avg_velocity) + 1e-6)
#                 blended_dir = (avg_dir + region_target_dir) * 0.5
#                 blended_dir /= np.linalg.norm(blended_dir) + 1e-6
#                 target_velocity = blended_dir * self.region_speed
#                 control = (target_velocity - agent.velocity) * 0.5
                
#                 # QP setup
#                 P = 2 * np.eye(2)
#                 q = -2 * control
#                 if G is not None and h is not None:
#                     try:
#                         u = solve_qp(P, q, G, h, solver="cvxopt")
#                         if u is None:
#                             # print(f"QP failed: G={G}, h={h}, control={control}")
#                             agent.velocity += control * self.dt
#                         else:
#                             agent.velocity += u * self.dt
#                     except Exception as e:
#                         # print(f"QP error: {e}, G={G}, h={h}")
#                         agent.velocity += control * self.dt
#                 else:
#                     agent.velocity += control * self.dt
                
#                 speed = np.linalg.norm(agent.velocity)
#                 if speed > self.region_speed:
#                     agent.velocity = agent.velocity / speed * self.region_speed
#             else:
#                 target_pos = np.array([agent.position[0], 0.0])
#                 direction = target_pos - agent.position
#                 direction /= np.linalg.norm(direction) + 1e-6
#                 target_velocity = direction * self.default_speed
#                 control = (target_velocity - agent.velocity) * 0.2
#                 agent.velocity += control * self.dt
                
#                 speed = np.linalg.norm(agent.velocity)
#                 if speed > self.default_speed:
#                     agent.velocity = agent.velocity / speed * self.default_speed
            
#             agent.position += agent.velocity * self.dt
            
#             if agent.position[0] >= self.region_x_max:
#                 agents_to_remove.append(agent)
        
#         # Collision check
#         # for i, a1 in enumerate(self.agents):
#         #     for a2 in self.agents[i+1:]:
#         #         dist = np.linalg.norm(a1.position - a2.position)
#         #         if dist < self.min_distance:
#         #             print(f"Collision detected: dist={dist}, pos1={a1.position}, pos2={a2.position}")
        
#         for agent in agents_to_remove:
#             if agent in self.agents:
#                 self.agents.remove(agent)
        
#         if np.random.random() < self.spawn_rate:
#             self.spawn_agent_outside()

#     def update(self, frame):
#         self.control_law()
#         plt.cla()
#         plt.gca().add_patch(Rectangle((self.region_x_min, self.region_y_min),
#                                       self.region_x_max - self.region_x_min,
#                                       self.region_y_max - self.region_y_min,
#                                       facecolor='green', alpha=0.3, edgecolor='g', linewidth=2))
#         plt.grid(True)
        
#         for agent in self.agents:
#             color = 'r' if self.in_region(agent.position) else 'b'
#             plt.plot(agent.position[0], agent.position[1], f'{color}o')
#             plt.arrow(agent.position[0], agent.position[1], 
#                       agent.velocity[0]*0.5, agent.velocity[1]*0.5, 
#                       head_width=0.2, color=color)
        
#         plt.xlim(-15, 15)
#         plt.ylim(-10, 10)
#         plt.title(f"Agents: {len(self.agents)} (Red = in region)")

# def main():
#     sim = ConsensusSimulation(num_agents=10, default_speed=1.0, region_speed=2.0, dt=0.05, spawn_rate=0.2)
#     fig = plt.figure(figsize=(10, 5))
#     ani = FuncAnimation(fig, sim.update, frames=500, interval=50, repeat=False)
#     # plt.show()
#     ani.save('agent_simulation.gif', writer='pillow', fps=20, dpi=100)

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import sympy as sp
from qpsolvers import solve_qp

class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class ConsensusSimulation:
    def __init__(self, num_agents=10, default_speed=1.0, region_speed=2.0, dt=0.05, spawn_rate=0.2):
        self.agents = []
        self.default_speed = default_speed
        self.region_speed = region_speed
        self.dt = dt
        self.spawn_rate = spawn_rate
        
        # Define region
        self.region_x_min = -15
        self.region_x_max = 15
        self.region_y_min = -3
        self.region_y_max = 3
        
        # Initial spawn
        initial_outside = num_agents // 2
        initial_inside = num_agents - initial_outside
        
        for _ in range(initial_outside):
            self.spawn_agent_outside()
        for _ in range(initial_inside):
            self.spawn_agent_inside()

        # CBF setup
        x1, x2, y1, y2, r = sp.symbols('x1 x2 y1 y2 r')
        h = sp.sqrt((x1 - x2)**2 + (y1 - y2)**2) - r
        f = sp.Matrix([0, 0])  # No drift term
        g = sp.Matrix([[1, 0], [0, 1]])  # Direct velocity control
        state_vars1 = sp.Matrix([x1, y1])
        grad_h1 = h.diff(state_vars1)
        Lfh = grad_h1.T * f  # Should be zero
        Lgh = grad_h1.T * g
        self.h = sp.lambdify((x1, y1, x2, y2, r), h)
        self.Lfh = sp.lambdify((x1, y1, x2, y2, r), Lfh)
        self.Lgh = sp.lambdify((x1, y1, x2, y2, r), Lgh)
        self.min_distance = 1.0  # Safe distance

    def spawn_agent_outside(self):
        spawn_above = np.random.choice([True, False])
        pos_x = np.random.uniform(self.region_x_min, -5)
        pos_y = np.random.uniform(5, 10) if spawn_above else np.random.uniform(-10, -5)
        pos = [pos_x, pos_y]
        target_y = 0.0
        direction = np.array([0, target_y - pos_y])
        direction /= np.linalg.norm(direction) + 1e-6
        vel = direction * self.default_speed * np.random.uniform(0.5, 1.5)
        self.agents.append(Agent(pos, vel))

    def spawn_agent_inside(self):
        pos_x = np.random.uniform(self.region_x_min, self.region_x_max)
        pos_y = np.random.uniform(self.region_y_min, self.region_y_max)
        pos = [pos_x, pos_y]
        vel = [self.region_speed * np.random.uniform(0.5, 1.0), 0.0]
        self.agents.append(Agent(pos, vel))

    def in_region(self, position):
        return (self.region_x_min <= position[0] <= self.region_x_max and 
                self.region_y_min <= position[1] <= self.region_y_max)

    def control_law(self):
        region_target_dir = np.array([1.0, 0.0])
        agents_to_remove = []
        
        for i, agent in enumerate(self.agents):
            # Compute distances to all other agents (not just in-region)
            other_agents = [a for a in self.agents if a is not agent]
            distances = [(a, np.linalg.norm(agent.position - a.position)) for a in other_agents]
            nearest_agents = [a for a, _ in sorted(distances, key=lambda x: x[1])[:min(5, len(distances))]]
            
            num_constraints = len(nearest_agents)
            G = np.zeros((num_constraints, 2)) if num_constraints > 0 else None
            h = np.zeros(num_constraints) if num_constraints > 0 else None
            
            if nearest_agents:
                for idx, a in enumerate(nearest_agents):
                    G[idx] = -self.Lgh(agent.position[0], agent.position[1], 
                                      a.position[0], a.position[1], self.min_distance)
                    h_val = self.h(agent.position[0], agent.position[1], 
                                  a.position[0], a.position[1], self.min_distance)
                    h[idx] = self.Lfh(agent.position[0], agent.position[1], 
                                     a.position[0], a.position[1], self.min_distance) + 8.0 * h_val
            
            # Nominal control depends on region
            if self.in_region(agent.position):
                avg_velocity = np.zeros(2)
                region_agents = [a for a in self.agents if self.in_region(a.position) and a is not agent]
                if region_agents:
                    for a in region_agents:
                        avg_velocity += a.velocity / (np.linalg.norm(a.velocity) + 1e-6)
                    avg_velocity /= len(region_agents)
                avg_dir = avg_velocity / (np.linalg.norm(avg_velocity) + 1e-6)
                blended_dir = (avg_dir + region_target_dir) * 0.5
                blended_dir /= np.linalg.norm(blended_dir) + 1e-6
                target_velocity = blended_dir * self.region_speed
                control = (target_velocity - agent.velocity) * 0.5
                max_speed = self.region_speed
            else:
                target_pos = np.array([agent.position[0], 0.0])
                direction = target_pos - agent.position
                direction /= np.linalg.norm(direction) + 1e-6
                target_velocity = direction * self.default_speed
                control = (target_velocity - agent.velocity) * 0.2
                max_speed = self.default_speed
            
            # QP optimization with CBF always applied
            P = 2 * np.eye(2)
            q = -2 * control
            if G is not None and h is not None:
                try:
                    u = solve_qp(P, q, G, h, solver="cvxopt")
                    if u is None:
                        # print(f"QP failed: G={G}, h={h}, control={control}")
                        agent.velocity += control * self.dt
                    else:
                        agent.velocity += u * self.dt
                except Exception as e:
                    # print(f"QP error: {e}, G={G}, h={h}")
                    agent.velocity += control * self.dt
            else:
                agent.velocity += control * self.dt
            
            # Enforce speed limit
            speed = np.linalg.norm(agent.velocity)
            if speed > max_speed:
                agent.velocity = agent.velocity / speed * max_speed
            
            agent.position += agent.velocity * self.dt
            
            if agent.position[0] >= self.region_x_max:
                agents_to_remove.append(agent)
        
        # Collision check (optional, uncomment if needed)
        # for i, a1 in enumerate(self.agents):
        #     for a2 in self.agents[i+1:]:
        #         dist = np.linalg.norm(a1.position - a2.position)
        #         if dist < self.min_distance:
        #             print(f"Collision detected: dist={dist}, pos1={a1.position}, pos2={a2.position}")
        
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
    sim = ConsensusSimulation(num_agents=10, default_speed=1.0, region_speed=2.0, dt=0.05, spawn_rate=0.2)
    fig = plt.figure(figsize=(10, 5))
    ani = FuncAnimation(fig, sim.update, frames=500, interval=50, repeat=False)
    # plt.show()
    ani.save('agent_simulation.gif', writer='pillow', fps=20, dpi=100)

if __name__ == "__main__":
    main()