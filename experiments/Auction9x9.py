from sympy import symbols, sqrt, Matrix, lambdify
import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm

# Define parameters
n_agents = 10
sep = 1.0  # Separation distance
alpha = 2.0
a = 0.00001
beta = 1.0
dt = 0.01
time_steps = 100  # Increased to allow agent 10 to reach goal

# Initial positions for 9 agents in 3x3 grid
grid_positions = [(x, y) for x in [-3, 0, 3] for y in [-3, 0, 3]]
spawn_positions_nominal = grid_positions + [(-5, 0)]  # Nominal spawn positions

# Add ±0.01 noise to spawn positions
np.random.seed(42)  # For reproducibility; remove for true randomness
noise = np.random.uniform(-0.01, 0.01, (n_agents, 2))  # Shape: (10, 2) for x and y
spawn_positions = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(spawn_positions_nominal, noise)]

# Goal positions (9 agents return to nominal spawn, agent 10 goes to (5, 0))
goal_positions = grid_positions + [(5, 0)]  # Goals remain the nominal positions

# Initial states and currency
states = [list(pos) for pos in spawn_positions]  # Current positions [x, y] with noise
state_history = [[pos[0]] for pos in spawn_positions]  # x history
state_history_y = [[pos[1]] for pos in spawn_positions]  # y history
time_history = [0.0]
base_currency = 1000.0
currency = [base_currency] * (n_agents - 1) + [base_currency * 10]  # Agent 10 gets 10x currency
currency_history = [[c] for c in currency]
gamma = [a * c**beta for c in currency]

# Nominal control laws
def get_nominal_control(agent_idx, current_pos):
    goal = goal_positions[agent_idx]
    dx = goal[0] - current_pos[0]
    dy = goal[1] - current_pos[1]
    k = 5.0  # Control gain
    return np.array([k * dx, k * dy])

# Single integrator dynamics: dx/dt = u_x, dy/dt = u_y
def dynamics(state, control):
    return control  # For single integrator, derivative of position is the control (velocity)

# RK4 integration
def rk4_step(state, control, dt):
    k1 = dynamics(state, control)
    k2 = dynamics(state + 0.5 * dt * k1, control)
    k3 = dynamics(state + 0.5 * dt * k2, control)
    k4 = dynamics(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Define symbolic variables for pairwise constraints
x_vars = symbols('x0:%d' % n_agents)
y_vars = symbols('y0:%d' % n_agents)
r = symbols('r')

# Dynamics and CBF setup
def get_cbf_constraints(positions):
    constraints = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            x_i, y_i = x_vars[i], y_vars[i]
            x_j, y_j = x_vars[j], y_vars[j]
            h = sqrt((x_i - x_j)**2 + (y_i - y_j)**2) - r
            state_i = Matrix([x_i, y_i])
            state_j = Matrix([x_j, y_j])
            g = Matrix([[1, 0], [0, 1]])
            grad_h_i = h.diff(state_i).T
            grad_h_j = h.diff(state_j).T
            Lg_h_i = grad_h_i * g
            Lg_h_j = grad_h_j * g
            Lg_h_i_func = lambdify((x_i, y_i, x_j, y_j, r), Lg_h_i, 'numpy')
            Lg_h_j_func = lambdify((x_i, y_i, x_j, y_j, r), Lg_h_j, 'numpy')
            h_func = lambdify((x_i, y_i, x_j, y_j, r), h, 'numpy')
            # Evaluate at current positions
            A_i = Lg_h_i_func(positions[i][0], positions[i][1], positions[j][0], positions[j][1], sep)
            A_j = Lg_h_j_func(positions[i][0], positions[i][1], positions[j][0], positions[j][1], sep)
            h_val = h_func(positions[i][0], positions[i][1], positions[j][0], positions[j][1], sep)
            constraints.append((i, j, A_i, A_j, -alpha * h_val))
    return constraints

# Simulation loop
for t in tqdm(range(time_steps)):
    current_positions = [[state_history[i][-1], state_history_y[i][-1]] for i in range(n_agents)]
    u_nom = [get_nominal_control(i, current_positions[i]) for i in range(n_agents)]
    u_nom_flat = np.hstack(u_nom)

    # CBF constraints
    constraints = get_cbf_constraints(current_positions)
    n_constraints = len(constraints)
    A = np.zeros((n_constraints, 2 * n_agents))
    b = np.zeros(n_constraints)
    for idx, (i, j, A_i, A_j, b_val) in enumerate(constraints):
        A[idx, i*2:i*2+2] = A_i
        A[idx, j*2:j*2+2] = A_j
        b[idx] = b_val

    # QP setup
    gamma_array = np.hstack([[g, g] for g in gamma])
    Q = 2 * np.diag(gamma_array)
    p = -Q @ u_nom_flat
    G = -A
    h = -b

    # Solve QP
    solution = solve_qp(Q, p, G=G, h=h, solver="cvxopt")
    if solution is not None:
        controls = np.split(solution, n_agents)
        # Update states
        for i in range(n_agents):
            new_state = rk4_step(np.array(current_positions[i]), controls[i], dt)
            state_history[i].append(float(new_state[0]))
            state_history_y[i].append(float(new_state[1]))
            # Update currency (simplified)
            currency[i] = max(currency[i] - 0.1 * np.sum(controls[i]**2), 0)
            currency_history[i].append(currency[i])
            gamma[i] = a * currency[i]**beta
    else:
        # Use nominal controls if optimization fails
        for i in range(n_agents):
            new_state = rk4_step(np.array(current_positions[i]), u_nom[i], dt)
            state_history[i].append(float(new_state[0]))
            state_history_y[i].append(float(new_state[1]))
            currency_history[i].append(currency[i])
    time_history.append(time_history[-1] + dt)

# Animation setup
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 4)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('10 Agent Trajectories with Spawn Noise')
ax.grid(True)
ax.set_aspect('equal')

# Plot elements
agents = [ax.plot([], [], 'o', markersize=8, label=f'Agent {i+1}')[0] for i in range(n_agents)]
traces = [ax.plot([], [], '-', alpha=0.3)[0] for i in range(n_agents)]
circles = [Circle(spawn_positions[i], sep/2, fill=True, alpha=0.2, linestyle='--', edgecolor='black') 
           for i in range(n_agents)]
for circle in circles:
    ax.add_patch(circle)
colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
for i in range(n_agents):
    agents[i].set_color(colors[i])
    traces[i].set_color(colors[i])
    circles[i].set_facecolor(colors[i])

# Plot nominal spawn positions for reference
for pos in spawn_positions_nominal:
    ax.plot(pos[0], pos[1], 'k+', alpha=0.5, markersize=10)

# Animation update
def update(frame):
    for i in range(n_agents):
        agents[i].set_data([state_history[i][frame]], [state_history_y[i][frame]])
        traces[i].set_data(state_history[i][:frame+1], state_history_y[i][:frame+1])
        circles[i].center = (state_history[i][frame], state_history_y[i][frame])
    return agents + traces + circles

ani = FuncAnimation(fig, update, frames=len(time_history), interval=50, blit=False, repeat=False)
plt.legend()
plt.show()

# Check if agent 10 reached (5, 0) and report final currencies
final_pos_10 = (state_history[9][-1], state_history_y[9][-1])
print(f"Agent 10 final position: {final_pos_10}")
print("Final currencies:")
for i, c in enumerate(currency):
    print(f"Agent {i+1}: {c:.2f}")