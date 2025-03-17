from sympy import symbols, sqrt, Matrix, simplify, diff, init_printing, lambdify
import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm

# Define variables
x1, x2, y1, y2, r, k = symbols('x1 x2 y1 y2 r k')
u1x, u1y, u2x, u2y = symbols('u1x u1y u2x u2y')

# Define h(x) function
h = sqrt((x1-x2)**2 + (y1-y2)**2) - r

# Define f(x) and g(x) as matrices
f = Matrix([0, 0])  # 2x1 zero dynamics (for CBF, not dynamics)
g = Matrix([[1, 0], [0, 1]])  # 2x2 identity
state_vars1 = Matrix([x1, y1])  # 2x1
state_vars2 = Matrix([x2, y2])  # 2x1

# Compute gradients
grad_h1 = h.diff(state_vars1).T  # 1x2
grad_h2 = h.diff(state_vars2).T  # 1x2

# Compute Lie derivatives
Lf1_h = grad_h1 * f  # 1x1 (should be zero)
Lf2_h = grad_h2 * f  # 1x1 (should be zero)
Lg1_h = grad_h1 * g  # 1x2
Lg2_h = grad_h2 * g  # 1x2

# Parameters
sep = 2.0
alpha = 2.0
a = 0.00001
beta = 1.0
c1 = 100.0  # Initial currency for agent 1
c2 = 10000.0  # Initial currency for agent 2
gamma1 = a * c1**beta
gamma2 = a * c2**beta
dt = 0.01  # Time step for RK4
time_steps = 100  # Number of time steps

# Lambdify functions
Lg1_h_func = lambdify((x1, y1, x2, y2, r), Lg1_h, 'numpy')
Lg2_h_func = lambdify((x1, y1, x2, y2, r), Lg2_h, 'numpy')
h_func = lambdify((x1, y1, x2, y2, r), h, 'numpy')

# Initial states and currency
x1_state = [-1.0]  # x1 history
y1_state = [0.01]  # y1 history
x2_state = [1.0]   # x2 history
y2_state = [-0.01] # y2 history
time_history = [0.0]  # Time history
c1_history = [c1]  # Currency history for agent 1
c2_history = [c2]  # Currency history for agent 2

# Nominal control inputs
u1_nom = np.array([10, 0])    # 2x1
u2_nom = np.array([-10, 0])   # 2x1

# Variables to store the timesteps when agents reach their thresholds
timestep_agent1_x5 = None  # Timestep when Agent 1 reaches x=5
timestep_agent2_xneg5 = None  # Timestep when Agent 2 reaches x=-5

# Function for single integrator dynamics
def dynamics(state, control):
    x, y = state
    u_x, u_y = control
    return np.array([u_x, u_y])  # dx/dt = u, dy/dt = v

# RK4 integration
def rk4_step(state, control, dt):
    k1 = dynamics(state, control)
    k2 = dynamics(state + 0.5 * dt * k1, control)
    k3 = dynamics(state + 0.5 * dt * k2, control)
    k4 = dynamics(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Simulation loop
for t in tqdm(range(time_steps)):
    # Current states and currency
    x1, y1, x2, y2 = x1_state[-1], y1_state[-1], x2_state[-1], y2_state[-1]
    c1_current, c2_current = c1_history[-1], c2_history[-1]
    # print(f"\nTime Step {t}: States - x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}")
    print(f"Currency - Agent 1: {c1_current:.2f}, Agent 2: {c2_current:.2f}")

    # Evaluate constraint at current state
    A1 = Lg1_h_func(x1, y1, x2, y2, sep)  # 1x2
    A2 = Lg2_h_func(x1, y1, x2, y2, sep)  # 1x2
    _A = np.hstack((A1, A2))  # 1x4

    # CBF constraint term
    h_val = h_func(x1, y1, x2, y2, sep)  # scalar
    b = -alpha * h_val  # scalar
    b = np.array([b])  # 1x1

    # Original QP formulation: min ½uᵀQu + pᵀu s.t. Gu ≤ h
    gamma_array = np.array([gamma1, gamma1, gamma2, gamma2])  # 4,
    _Q = 2 * np.diag(gamma_array)    # 4x4
    u_nom = np.hstack((u1_nom, u2_nom))  # 4,
    _p = -_Q @ u_nom                # 4x4 @ 4, = 4,
    _G = -_A                        # 1x4
    _h = -b                         # 1x1

    # Solve original QP
    solution = solve_qp(_Q, _p, G=_G, h=_h, solver="cvxopt")
    # print("Original Solution:", solution)

    if solution is not None:
        # Split the solution into u1 and u2
        u1 = solution[:2]  # [u1x, u1y]
        u2 = solution[2:]  # [u2x, u2y]
        # print("u1:", u1)
        # print("u2:", u2)

        # Original total welfare
        W_total = -gamma1 * np.sum((u1 - u1_nom)**2) - gamma2 * np.sum((u2 - u2_nom)**2)
        # print("Original Total Welfare:", W_total)

        # --- Calculate Externality for Agent 1 ---
        # Optimize only u1, fix u2 at u2_nom
        Q_u1 = 2 * np.diag([gamma1, gamma1])  # 2x2
        p_u1 = -Q_u1 @ u1_nom  # 2x1
        G_u1 = -_A[0, :2].reshape(1, 2)  # 1x2 (constraint for u1)
        h_u1 = -b + (_A[0, 2:] @ u2_nom)  # Adjust b for fixed u2
        h_u1 = np.array([h_u1])  # 1x1

        solution_u1_only = solve_qp(Q_u1, p_u1, G=G_u1, h=h_u1, solver="cvxopt")
        # print("Solution with u2 fixed (u1 only):", solution_u1_only)

        if solution_u1_only is not None:
            u1_new = solution_u1_only
            u2_new = u2_nom  # Fixed at nominal
            W_without_agent1 = -gamma1 * np.sum((u1_new - u1_nom)**2) - gamma2 * np.sum((u2_new - u2_nom)**2)
            externality_1 = W_total - W_without_agent1
            # print("Externality for Agent 1:", externality_1)

        # --- Calculate Externality for Agent 2 ---
        # Optimize only u2, fix u1 at u1_nom
        Q_u2 = 2 * np.diag([gamma2, gamma2])  # 2x2
        p_u2 = -Q_u2 @ u2_nom  # 2x1
        G_u2 = -_A[0, 2:].reshape(1, 2)  # 1x2 (constraint for u2)
        h_u2 = -b + (_A[0, :2] @ u1_nom)  # Adjust b for fixed u1
        h_u2 = np.array([h_u2])  # 1x1

        solution_u2_only = solve_qp(Q_u2, p_u2, G=G_u2, h=h_u2, solver="cvxopt")
        # print("Solution with u1 fixed (u2 only):", solution_u2_only)

        if solution_u2_only is not None:
            u2_new = solution_u2_only
            u1_new = u1_nom  # Fixed at nominal
            W_without_agent2 = -gamma1 * np.sum((u1_new - u1_nom)**2) - gamma2 * np.sum((u2_new - u2_nom)**2)
            externality_2 = W_total - W_without_agent2
            # print("Externality for Agent 2:", externality_2)

        # --- Adjust Currency Based on Externalities ---
        # Payment for Agent 1
        if externality_1 > 0:
            payment_1 = min(externality_1, c1_current)  # Pay up to available currency
        else:
            payment_1 = 0  # No reward for negative externality
        c1_new = max(c1_current - payment_1, 0)  # Ensure currency doesn't go negative

        # Payment for Agent 2
        if externality_2 > 0:
            payment_2 = min(externality_2, c2_current)  # Pay up to available currency
        else:
            payment_2 = 0  # No reward for negative externality
        c2_new = max(c2_current - payment_2, 0)  # Ensure currency doesn't go negative

        # print(f"Payment for Agent 1: {payment_1:.2f}, New Currency: {c1_new:.2f}")
        # print(f"Payment for Agent 2: {payment_2:.2f}, New Currency: {c2_new:.2f}")

        # Update currency history
        c1_history.append(c1_new)
        c2_history.append(c2_new)

        # Update states using RK4
        state1 = np.array([x1_state[-1], y1_state[-1]])
        state2 = np.array([x2_state[-1], y2_state[-1]])
        new_state1 = rk4_step(state1, u1, dt)
        new_state2 = rk4_step(state2, u2, dt)
        x1_state.append(float(new_state1[0]))
        y1_state.append(float(new_state1[1]))
        x2_state.append(float(new_state2[0]))
        y2_state.append(float(new_state2[1]))
        time_history.append(time_history[-1] + dt)

        # Check if Agent 1 has reached x=5 (only record the first occurrence)
        if timestep_agent1_x5 is None and x1_state[-1] >= 4.0:
            timestep_agent1_x5 = t
            print(f"Agent 1 reached x=4 at timestep {t} (x1={x1_state[-1]:.3f})")

        # Check if Agent 2 has reached x=-5 (only record the first occurrence)
        if timestep_agent2_xneg5 is None and x2_state[-1] <= -4.0:
            timestep_agent2_xneg5 = t
            print(f"Agent 2 reached x=-4 at timestep {t} (x2={x2_state[-1]:.3f})")

    else:
        print("Optimization failed at time step", t)
        # Use nominal controls if optimization fails
        state1 = np.array([x1_state[-1], y1_state[-1]])
        state2 = np.array([x2_state[-1], y2_state[-1]])
        new_state1 = rk4_step(state1, u1_nom, dt)
        new_state2 = rk4_step(state2, u2_nom, dt)
        x1_state.append(float(new_state1[0]))
        y1_state.append(float(new_state1[1]))
        x2_state.append(float(new_state2[0]))
        y2_state.append(float(new_state2[1]))
        time_history.append(time_history[-1] + dt)
        # No payment if optimization fails
        c1_history.append(c1_history[-1])
        c2_history.append(c2_history[-1])

        # Check thresholds even if optimization fails (since states are still updated)
        if timestep_agent1_x5 is None and x1_state[-1] >= 4.0:
            timestep_agent1_x5 = t
            print(f"Agent 1 reached x=4 at timestep {t} (x1={x1_state[-1]:.3f})")
        if timestep_agent2_xneg5 is None and x2_state[-1] <= -4.0:
            timestep_agent2_xneg5 = t
            print(f"Agent 2 reached x=-4 at timestep {t} (x2={x2_state[-1]:.3f})")

# Animation Setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Agent Trajectories Animation')
ax.grid(True)
ax.set_aspect('equal')

# Set axis limits based on the full trajectory
x_min = min(min(x1_state), min(x2_state)) - 0.5
x_max = max(max(x1_state), max(x2_state)) + 0.5
y_min = min(min(y1_state), min(y2_state)) - 0.5
y_max = max(max(y1_state), max(y2_state)) + 0.5
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Initialize plot elements
agent1, = ax.plot([], [], 'bo', label='Agent 1', markersize=10)
agent2, = ax.plot([], [], 'ro', label='Agent 2', markersize=10)
trace1, = ax.plot([], [], 'b-', alpha=0.3)  # Trace for Agent 1
trace2, = ax.plot([], [], 'r-', alpha=0.3)  # Trace for Agent 2
ax.plot(x1_state[0], y1_state[0], 'b*', label='Agent 1 Start', markersize=15)
ax.plot(x2_state[0], y2_state[0], 'r*', label='Agent 2 Start', markersize=15)
# ax.legend()

# Add initial circles with radius sep/2
circle1 = Circle((x1_state[0], y1_state[0]), sep/2, fill=True, color='blue', alpha=0.2, 
                 linestyle='--', edgecolor='black', linewidth=1)
circle2 = Circle((x2_state[0], y2_state[0]), sep/2, fill=True, color='red', alpha=0.2, 
                 linestyle='--', edgecolor='black', linewidth=1)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Animation update function
def update(frame):
    # Update agent positions
    agent1.set_data([x1_state[frame]], [y1_state[frame]])
    agent2.set_data([x2_state[frame]], [y2_state[frame]])
    # Update traces
    trace1.set_data(x1_state[:frame+1], y1_state[:frame+1])
    trace2.set_data(x2_state[:frame+1], y2_state[:frame+1])
    # Update circle centers
    circle1.center = (x1_state[frame], y1_state[frame])
    circle2.center = (x2_state[frame], y2_state[frame])
    return agent1, agent2, trace1, trace2, circle1, circle2

# Create animation
ani = FuncAnimation(fig, update, frames=len(time_history), interval=50, blit=False, repeat=False)
plt.show()
# Save animation as GIF
# ani.save('trajectory_animation.gif', writer='pillow', fps=20)

# Print the timesteps when agents reach their thresholds
print("\nThreshold Timesteps:")
if timestep_agent1_x5 is not None:
    print(f"Agent 1 reached x=5 at timestep {timestep_agent1_x5} (time={time_history[timestep_agent1_x5]:.2f}s)")
else:
    print("Agent 1 did not reach x=5 within the simulation.")
if timestep_agent2_xneg5 is not None:
    print(f"Agent 2 reached x=-5 at timestep {timestep_agent2_xneg5} (time={time_history[timestep_agent2_xneg5]:.2f}s)")
else:
    print("Agent 2 did not reach x=-5 within the simulation.")

# Print final states and currency
# print("\nFinal States:")
# print("x1:", x1_state)
# print("y1:", y1_state)
# print("x2:", x2_state)
# print("y2:", y2_state)
# print("Final Currency:")
# print("Agent 1:", c1_history[-1])
# print("Agent 2:", c2_history[-1])