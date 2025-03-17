import numpy as np
import uam.dynamics.dynamics as dyn  # Import the appropriate Dynamics class
from uam.agent.agent import Agent  # Import the Agent class
import matplotlib.pyplot as plt
from time import time
from uam.controllers import serret_frenet
from uam.tools import tools
from uam.tools.plotters import Plotter
import plotly.io as pio

# Example usage:
if __name__ == "__main__":
    initial_state = np.array([10, -0.1, 0.0, 0.0, 0.0, 0.0])  # Define the initial state
    helix = tools.generate_3d_helix(1000, 10, 5, 5)

    s = np.linspace(0, np.pi, 100)
    x = 10*np.cos(s)
    y = 10*np.sin(s)
    z = np.zeros(len(s))
    path = np.column_stack((x, y, z))

    """Toy trajectory."""
    # test_trajectory_x = np.array([0, 50, 50, 90, 90, 70, 70, 130, 130, 100, 160])
    # test_trajectory_y = np.array([0, 0, 30, 30, -10, -10, 40, 40, 0, 30, 30])
    # test_trajectory_z = np.array([0, 20, 0, 0, -20, -20, 20, 10, -10, 30, 30])
    # test_trajectory = np.column_stack((test_trajectory_x, test_trajectory_y, test_trajectory_z))
    # test_trajectory_interp = tools.interpolate_points(test_trajectory, 10)
    # test_trajectory_bspline = tools.fit_b_spline(test_trajectory_interp, 2000, smoothing=0)
    # test_trajectory_bspline_coarse = tools.fit_b_spline(test_trajectory, 2000, smoothing=0)

    """2-segment trajectory"""
    # tx = np.array([0, 0, 50])
    # ty = np.array([0, 50, 50])
    # tz = np.array([0, 0, 0])

    # traj = np.column_stack((tx, ty, tz))
    # traj_interp = tools.interpolate_points(traj, 10)

    total_time = 30.0  # Total time for the simulation
    dt = 0.01  # Time step size

    agent = Agent(initial_state, "DoubleIntegrator3DOF", dt, path=path)

    start_time = time()

    time_steps = int(total_time / dt)  # Calculate number of time steps
    t = 0.0  # Initialize time

    for _ in range(time_steps):
        agent.step(t, dt)  # Advance the agent's state by one time step
        t += dt  # Update time

    sim_time = time() - start_time

    print(f"Sim time: {sim_time:.6f}")

    plot = Plotter()

    # plt.plot(helix[:, 0], , color='k', label='Desired Path')
    # plt.plot(agent.state_log[:, 0], agent.state_log[:, 1], color='b', label='Actual Path')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()

    """Interactive plot."""
    plots = [agent.state_log[:, 0:3], path]
    labels = ['Actual Path', 'Desired Trajectory']
    bools = [False, True]
    markers= [False, False]
    widths = [4, 1]
    t =  np.arange(0, total_time+dt, dt)
    # interactive_plot = plot.interactive_3d_trajectories(plots, labels, bools, t, markers, widths)
    # interactive_plot.show()
    # pio.write_html(interactive_plot, 'testing_html.html')

    """Static 3D plot."""
    sphere_centers = [(0, 11, 0.0)]
    sphere_radii = [1.5]
    plot.plot_3d_trajectories(plots, labels, widths, [None, None], sphere_centers, sphere_radii)