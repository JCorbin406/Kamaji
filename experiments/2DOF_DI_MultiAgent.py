import numpy as np

from kamaji.agent.agent import Agent  # Import the Agent class
# from kamaji.plotting import plot_6DOFQuad, plot_3DOFPlane, plot_6DOFTilt  # Import the quadcopter/plane plotting function
from kamaji.simulation.simulator import Simulator  # Import the Simulation class
from kamaji.tools import tools
# from kamaji.Archive import plotter_test
from kamaji.tools import plotters
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# Example usage:
if __name__ == "__main__":
    a1_x = np.array([0, 0])
    a1_y = np.array([10, -10])
    a1_z = np.array([0, 0])
    a2_x = np.array([10, -10])
    a2_y = np.array([0, 0])
    a2_z = np.array([0, 0])
    a3_x = np.array([-10*np.sin(np.pi/4), 10*np.sin(np.pi/4)])
    a3_y = np.array([-10*np.sin(np.pi/4), 10*np.sin(np.pi/4)])
    a3_z = np.array([0, 0])
    a1_traj = np.column_stack((a1_x, a1_y, a1_z))
    a2_traj = np.column_stack((a2_x, a2_y, a2_z))
    a3_traj = np.column_stack((a3_x, a3_y, a3_z))
    a1_interp = tools.interpolate_points(a1_traj, 10)
    a2_interp = tools.interpolate_points(a2_traj, 10)
    a3_interp = tools.interpolate_points(a3_traj, 10)
    a1_bspline = tools.fit_b_spline(a1_interp)
    a2_bspline = tools.fit_b_spline(a2_interp)
    a3_bspline = tools.fit_b_spline(a3_interp)

    total_time = 30.0
    dt = 0.01
    sim = Simulator(total_time, dt)
    
    path_gain = 2.0
    # # obstacles = [(25, 0.0, 7.5, 5.0),
    # #              (100, 42, 16, 5.0),
    # #             #  (50, 35, 0, 10.0),
    # #              (110, 20, 7.5, 10.0)]
    
    # """Deadlock Scenario"""
    # obstacles = [(25, 0.0, 10.0, 5.0)]
   
    
    a1_setpoint = {
        'path': a1_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': []
    }
    a2_setpoint = {
        'path': a2_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': []
    }
    a3_setpoint = {
        'path': a3_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': []
    }

    a1_set_des = [a1_setpoint]
    a2_set_des = [a2_setpoint]
    a3_set_des = [a3_setpoint]

    time_steps = int(total_time / dt)
    for _ in range(time_steps):
        a1_setpoint = {
        'path': a1_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': []
        }
        a2_setpoint = {
        'path': a2_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': []
        }
        a3_setpoint = {
        'path': a3_bspline, 
        'gains': [path_gain, path_gain, path_gain],
        't_go': total_time,
        'obstacles': []
        }

        a1_set_des.append(a1_setpoint)
        a2_set_des.append(a2_set_des)
        a3_set_des.append(a3_set_des)

    a1_init = np.array([0.0, 10.1]) 
    a2_init = np.array([10.1, 0.0]) 
    a3_init = np.array([-10.1*np.sin(np.pi/4), -10.1*np.cos(np.pi/4)]) 

    agent1 = Agent(a1_init, "SingleIntegrator2DOF", "PathFollowerDyn2D", a1_set_des, id=1)
    agent2 = Agent(a2_init, "SingleIntegrator2DOF", "PathFollowerDyn2D", a2_set_des, id=2)
    agent3 = Agent(a3_init, "SingleIntegrator2DOF", "PathFollowerDyn2D", a3_set_des, id=3)
    sim.add_agent(agent1)
    sim.add_agent(agent2)
    sim.add_agent(agent3)
    sim.simulate()

    t = np.linspace(0.0, total_time, time_steps)

    """Static Matplotlib plots."""
    # fig, ax = plt.subplots()
    # # ax.plot(a1_bspline[:, 0], a1_bspline[:, 1], color='g', label='A1 - Desired')
    # # ax.plot(a2_bspline[:, 0], a2_bspline[:, 1], color='r', label='A2 - Desired')
    # # ax.plot(a3_bspline[:, 0], a3_bspline[:, 1], color='k', label='A3 - Desired')
    # # ax.plot(sim.inactive_agents[0].state_log[:, 0], sim.inactive_agents[0].state_log[:, 1], label='A1 - Actual')
    # # ax.plot(sim.inactive_agents[1].state_log[:, 0], sim.inactive_agents[1].state_log[:, 1], label='A2 - Actual')
    # # ax.plot(sim.inactive_agents[2].state_log[:, 0], sim.inactive_agents[2].state_log[:, 1], label='A3 - Actual')
    # ax.plot(sim.inactive_agents[0].state_log[:, 0], sim.inactive_agents[0].state_log[:, 1], label='A1 - Actual')
    # ax.plot(sim.inactive_agents[1].state_log[:, 0], sim.inactive_agents[1].state_log[:, 1], label='A2 - Actual')
    # ax.plot(sim.inactive_agents[2].state_log[:, 0], sim.inactive_agents[2].state_log[:, 1], label='A3 - Actual')
    # # Set axis labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_aspect('equal', 'box')
    # ax.legend()
    # # Show the plot
    # plt.show()

    """Dynamic Matplotlib plots."""
    # Time vector
    t = np.linspace(0.0, total_time, time_steps)

    # Extract state logs for each agent
    a1_traj = sim.inactive_agents[0].state_log
    a2_traj = sim.inactive_agents[1].state_log
    a3_traj = sim.inactive_agents[2].state_log

    # Create figure and axis
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax1 = axes[1]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", "box")
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel('Time (s)')
    
    # ax.set_xlim(np.min([a1_traj[:, 0], a2_traj[:, 0], a3_traj[:, 0]]) - 1, 
    #             np.max([a1_traj[:, 0], a2_traj[:, 0], a3_traj[:, 0]]) + 1)
    # ax.set_ylim(np.min([a1_traj[:, 1], a2_traj[:, 1], a3_traj[:, 1]]) - 1, 
    #             np.max([a1_traj[:, 1], a2_traj[:, 1], a3_traj[:, 1]]) + 1)

    # Initialize agent positions as points
    # a1_plot, = ax.plot([], [], 'go', label='A1')
    # a2_plot, = ax.plot([], [], 'ro', label='A2')
    # a3_plot, = ax.plot([], [], 'ko', label='A3')

    circle_center = (0, 0)
    circle_radius = 10
    circle = patches.Circle(circle_center, circle_radius, color='k', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.scatter(0, -10, color='k', marker='D', s=200)
    ax.scatter(-10, 0, color='r', marker='D', s=200)
    ax.scatter(10*np.cos(np.pi/4), 10*np.sin(np.pi/4), color='g', marker='D', s=200)

    a1_circle = patches.Circle(a1_traj[0, :2], radius = 0.2, color='g')
    a2_circle = patches.Circle(a2_traj[0, :2], radius = 0.2, color='r')
    a3_circle = patches.Circle(a3_traj[0, :2], radius = 0.2, color='k')
    a1_buff = patches.Circle(a1_traj[0, :2], radius = 1.0, color='k', fill=False, linestyle='--', linewidth=1)
    a2_buff = patches.Circle(a2_traj[0, :2], radius = 1.0, color='k', fill=False, linestyle='--', linewidth=1)
    a3_buff = patches.Circle(a3_traj[0, :2], radius = 1.0, color='k', fill=False, linestyle='--', linewidth=1)

    ax.add_patch(a1_circle)
    ax.add_patch(a2_circle)
    ax.add_patch(a3_circle)
    ax.add_patch(a1_buff)
    ax.add_patch(a2_buff)
    ax.add_patch(a3_buff)

    # Initialize trajectory lines
    a1_traj_plot, = ax.plot([], [], 'g-', label='A3')
    a2_traj_plot, = ax.plot([], [], 'r-', label='A2')
    a3_traj_plot, = ax.plot([], [], 'k-', label='A1')

    h1_plot, = ax1.plot([], [], label='A1-A2')
    h2_plot, = ax1.plot([], [], label='A1-A3')
    h3_plot, = ax1.plot([], [], label='A2-A3')

    ax1.set_xlim([0, total_time])

    # Compute the min and max values for h1, h2, h3
    h1_min, h1_max = np.min(sim.h1_hist), np.max(sim.h1_hist)
    h2_min, h2_max = np.min(sim.h2_hist), np.max(sim.h2_hist)
    h3_min, h3_max = np.min(sim.h3_hist), np.max(sim.h3_hist)

    # Find overall min and max
    y_min = min(h1_min, h2_min, h3_min)
    y_max = max(h1_max, h2_max, h3_max)

    # Set the ylim for ax1
    ax1.set_ylim(y_min, y_max)

    ax.legend(loc='upper right')
    ax1.legend(loc='lower right')

    # **Update function for animation**
    def update(frame):
        # Update agent positions
        # a1_plot.set_data(a1_traj[frame, 0], a1_traj[frame, 1])
        # a2_plot.set_data(a2_traj[frame, 0], a2_traj[frame, 1])
        # a3_plot.set_data(a3_traj[frame, 0], a3_traj[frame, 1])
        a1_circle.set_center(a1_traj[frame, :2])
        a2_circle.set_center(a2_traj[frame, :2])
        a3_circle.set_center(a3_traj[frame, :2])
        a1_buff.set_center(a1_traj[frame, :2])
        a2_buff.set_center(a2_traj[frame, :2])
        a3_buff.set_center(a3_traj[frame, :2])

        # Update trajectory lines
        a1_traj_plot.set_data(a1_traj[:frame+1, 0], a1_traj[:frame+1, 1])
        a2_traj_plot.set_data(a2_traj[:frame+1, 0], a2_traj[:frame+1, 1])
        a3_traj_plot.set_data(a3_traj[:frame+1, 0], a3_traj[:frame+1, 1])

        h1_plot.set_data(t[:frame+1], sim.h1_hist[:frame+1])
        h2_plot.set_data(t[:frame+1], sim.h2_hist[:frame+1])
        h3_plot.set_data(t[:frame+1], sim.h3_hist[:frame+1])

        return a1_buff, a2_buff, a3_buff, a1_circle, a2_circle, a3_circle, a1_traj_plot, a2_traj_plot, a3_traj_plot, h1_plot, h2_plot, h3_plot

    # ax.grid(True)

    ax.set_title('Trajectories')
    ax1.set_title('Control Barrier Functions (h)')
    ax1.grid(True)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=5/len(t)*1000, blit=True)
    # ani.save("animation.mp4", writer="ffmpeg")

    # ax1.plot(-np.array(sim.H_hist))
    # ax1.plot(t, sim.h1_hist, color='k', label='H1')
    # ax1.plot(t, sim.h2_hist, color='r', label='H2')
    # ax1.plot(t, sim.h3_hist, color='g', label='H3')
    # ax1.legend()
    # ax1.set_yscale('log')

    plt.show()
    
    print('done')