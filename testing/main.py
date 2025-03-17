import numpy as np

from kamaji.agent.agent import Agent  # Import the Agent class
# from kamaji.plotting import plot_6DOFQuad, plot_3DOFPlane, plot_6DOFTilt  # Import the quadcopter/plane plotting function
from kamaji.simulation.simulator import Simulator  # Import the Simulation class
from kamaji.tools import tools
# from kamaji.Archive import plotter_test
from kamaji.tools import plotters
import plotly.io as pio
import matplotlib.pyplot as plt

# Example usage:
if __name__ == "__main__":
    test_trajectory_x = np.array([0, 50, 50, 90, 90, 70, 70, 130, 130, 100, 160])
    test_trajectory_y = np.array([0, 0, 30, 30, -10, -10, 40, 40, 0, 30, 30])
    test_trajectory_z = np.array([0, 20, 0, 0, -20, -20, 20, 10, -10, 30, 30])
    test_trajectory = np.column_stack((test_trajectory_x, test_trajectory_y, test_trajectory_z))
    test_trajectory_interp = tools.interpolate_points(test_trajectory, 10)
    test_trajectory_bspline = tools.fit_b_spline(test_trajectory_interp, 2000, smoothing=0)
    test_trajectory_bspline_coarse = tools.fit_b_spline(test_trajectory, 2000, smoothing=0)

    total_time = 30.0
    dt = 0.01
    sim = Simulator(total_time, dt)
    
    path_gain = 2.0
    # obstacles = [(25, 0.0, 7.5, 5.0),
    #              (100, 42, 16, 5.0),
    #             #  (50, 35, 0, 10.0),
    #              (110, 20, 7.5, 10.0)]
    
    """Deadlock Scenario"""
    obstacles = [(25, 0.0, 10.0, 5.0)]
   
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    setpoint_dict = {
        'path': test_trajectory_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': obstacles
    }

    setpoint_des = [setpoint_dict]
    time_steps = int(total_time / dt)
    t = 0.0
    for _ in range(time_steps):
        setpoint_dict = {
        'path': test_trajectory_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': obstacles
    }
        setpoint_des.append(setpoint_dict)
        t += dt

    agent = Agent(initial_state, "DoubleIntegrator3DOF", "PathFollowerDyn", setpoint_des)
    sim.add_agent(agent)
    sim.simulate()

    """Interactive plot."""
    plots = [sim.inactive_agents[0].state_log[:, 0:3], test_trajectory, test_trajectory_bspline_coarse, test_trajectory_bspline]
    labels = ['Actual Path', 'Desired Trajectory', 'Coarse B-Spline', 'Dense B-Spline']
    bools = [False, True, True, True]
    markers= [False, True, False, False]
    widths = [6, 2, 2, 2]
    t =  np.arange(0, total_time+dt, dt)
    interactive_plot = plotters.Plotter()
    plot = interactive_plot.interactive_3d_trajectories(plots, labels, bools, t, markers, widths, obstacles)
    # plot.show()
    pio.write_html(plot, 'viewer.html')
    
    x = sim.inactive_agents[0].state_log[:, 0]
    y = sim.inactive_agents[0].state_log[:, 1]
    z = sim.inactive_agents[0].state_log[:, 2]
    xc, yc, zc, r = 25, 0.0, 10.0, 5.0
    h = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2) - r
    plt.plot(t, h)
    # plt.xlim((0, 500))
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('CBF, h')
    # plt.hlines(0, 0, 500)
    plt.show()


    """Matlab plots."""
    # # Set up the 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot the helix and agent path
    # ax.plot(test_trajectory_bspline[:, 0], test_trajectory_bspline[:, 1], test_trajectory_bspline[:, 2], color='g', label='B-Spline Dense')
    # ax.plot(test_trajectory_bspline_coarse[:, 0], test_trajectory_bspline_coarse[:, 1], test_trajectory_bspline_coarse[:, 2], color='r', label='B-Spline Coarse')
    # ax.plot(test_trajectory_x, test_trajectory_y, test_trajectory_z, color='k', label='Desired Trajectory')
    # ax.plot(sim.inactive_agents[0].state_log[:, 0], sim.inactive_agents[0].state_log[:, 1], sim.inactive_agents[0].state_log[:, 2], color='b', label='Actual Trajectory')
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