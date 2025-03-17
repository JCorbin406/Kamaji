import numpy as np
import uam.dynamics.dynamics as dyn  # Import the appropriate Dynamics class
from uam.agent.agent import Agent  # Import the Agent class
import matplotlib.pyplot as plt
from time import time
from uam.controllers import serret_frenet
from uam.tools import tools
from uam.tools import plotters
import plotly.io as pio

# Example usage:
if __name__ == "__main__":
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Define the initial state
    helix = tools.generate_3d_helix(2000, 10, 5, 25)
    helix_sparse = tools.generate_3d_helix(10, 10, 5, 25)
    helix_bspline = tools.fit_b_spline(helix_sparse, 2000, smoothing=0)

    test_trajectory_x = np.array([0, 50, 50, 90, 90, 70, 70, 130, 130, 100, 160])
    test_trajectory_y = np.array([0, 0, 30, 30, -10, -10, 40, 40, 0, 30, 30])
    test_trajectory_z = np.array([0, 20, 0, 0, -20, -20, 20, 10, -10, 30, 30])
    test_trajectory = np.column_stack((test_trajectory_x, test_trajectory_y, test_trajectory_z))
    test_trajectory_interp = tools.interpolate_points(test_trajectory, 10)
    test_trajectory_bspline = tools.fit_b_spline(test_trajectory_interp, 2000, smoothing=0)
    test_trajectory_bspline_coarse = tools.fit_b_spline(test_trajectory, 2000, smoothing=0)

    total_time = 50.0  # Total time for the simulation
    dt = 0.01  # Time step size

    agent = Agent(initial_state, "DoubleIntegrator3DOF", dt, path=test_trajectory_bspline)

    start_time = time()

    time_steps = int(total_time / dt)  # Calculate number of time steps
    t = 0.0  # Initialize time

    for _ in range(time_steps):
        agent.step(t, dt)  # Advance the agent's state by one time step
        t += dt  # Update time

    sim_time = time() - start_time

    print(f"Sim time: {sim_time:.6f}")

    
    """B-spline Fitting Example"""
    # segment_x = np.array([0, 2.5, 5, 7.5, 10])
    # segment_y = np.array([0, 2.5, 2.5, -2.5, 0])
    # segment_z = np.array([0, 0, 0, 0, 0])
    # segment = np.column_stack((segment_x, segment_y, segment_z))
    # segment_interp = tools.interpolate_points(segment, 5)

    # bspline_coarse = tools.fit_b_spline(segment, 1000, smoothing=0)
    # bspline_dense = tools.fit_b_spline(segment_interp, 1000, smoothing=0)

    # plt.plot(segment_x, segment_y, color='r', label='Initial Segment')
    # plt.scatter(segment_x, segment_y, color='k')
    # # plt.plot(bspline_coarse[:, 0], bspline_coarse[:, 1], color='b', label='Coarse B-spline')
    # plt.scatter(segment_interp[:, 0], segment_interp[:, 1], color='k', label='Interpolated Points')
    # plt.plot(bspline_dense[:, 0], bspline_dense[:, 1], color='b', label='Dense B-spline')
    # plt.grid(True)
    # plt.show()
    
    """Matplotlib plotting stuff."""
    # Set up the 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the helix and agent path
    # # ax.plot(helix[:, 0], helix[:, 1], helix[:, 2], color='k', label='Desired Path')
    # # ax.plot(helix_bspline[:, 0], helix_bspline[:, 1], helix_bspline[:, 2], color='k', label='Desired')
    # ax.plot(test_trajectory_bspline[:, 0], test_trajectory_bspline[:, 1], test_trajectory_bspline[:, 2], color='g', label='B-Spline Dense')
    # ax.plot(test_trajectory_bspline_coarse[:, 0], test_trajectory_bspline_coarse[:, 1], test_trajectory_bspline_coarse[:, 2], color='r', label='B-Spline Coarse')
    # ax.plot(test_trajectory_x, test_trajectory_y, test_trajectory_z, color='k', label='Desired Trajectory')
    # ax.plot(agent.state_log[:, 0], agent.state_log[:, 1], agent.state_log[:, 2], color='b', label='Actual Trajectory')

    # # Set axis labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Helix')

    # # Calculate combined limits for both paths
    # # all_x = np.concatenate((helix[:, 0], agent.state_log[:, 0]))
    # # all_y = np.concatenate((helix[:, 1], agent.state_log[:, 1]))
    # # all_z = np.concatenate((helix[:, 2], agent.state_log[:, 2]))

    # # max_range = np.array([all_x.max() - all_x.min(),
    # #                     all_y.max() - all_y.min(),
    # #                     all_z.max() - all_z.min()]).max() / 2.0

    # # mid_x = (all_x.max() + all_x.min()) * 0.5
    # # mid_y = (all_y.max() + all_y.min()) * 0.5
    # # mid_z = (all_z.max() + all_z.min()) * 0.5

    # # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.set_aspect('equal', 'box')
    # ax.legend()

    # # Show the plot
    # plt.show()

    """Interactive plot."""
    plots = [agent.state_log[:, 0:3], test_trajectory, test_trajectory_bspline_coarse, test_trajectory_bspline]
    labels = ['Actual Path', 'Desired Trajectory', 'Coarse B-Spline', 'Dense B-Spline']
    bools = [False, True, True, True]
    markers= [False, True, False, False]
    widths = [6, 2, 2, 2]
    t =  np.arange(0, total_time+dt, dt)
    interactive_plot = plotters.interactive_3d_multi_trajectory_plot(plots, labels, bools, t, markers, widths)
    interactive_plot.show()
    pio.write_html(interactive_plot, 'testing_html.html')