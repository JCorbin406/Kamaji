import numpy as np
import uam.dynamics.dynamics as dyn  # Import the appropriate Dynamics class
from uam.agent.agent import Agent  # Import the Agent class
import matplotlib.pyplot as plt
from time import time

# Example usage:
if __name__ == "__main__":
    initial_state = np.array([0.0, 20.0, 160.0])  # Define the initial state
    agent = Agent(initial_state, "cruise")

    # agent.assign_controller('ACC_CBF')

    total_time = 20.0  # Total time for the simulation
    dt = 0.01  # Time step size

    start_time = time()

    time_steps = int(total_time / dt)  # Calculate number of time steps
    t = 0.0  # Initialize time

    for _ in range(time_steps):
        agent.step(t, dt)  # Advance the agent's state by one time step
        t += dt  # Update time

    sim_time = time() - start_time

    t = np.linspace(0, total_time, time_steps+1)

    print(f"Sim time: {sim_time:.6f}")

    # plt.plot(agent.state_log[:, 0], agent.state_log[:, 1])
    # plt.show()
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(t, agent.state_log[:, 0])
    axs[0, 0].set_title('p')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Position (m)')
    axs[0, 0].grid(True)
    axs[0, 1].plot(t, agent.state_log[:, 1])
    axs[0, 1].set_title('v')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Velocity (m/s)')
    axs[0, 1].grid(True)
    axs[1, 0].plot(t, agent.state_log[:, 2])
    axs[1, 0].set_title('z')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Delta Position (m)')
    axs[1, 0].grid(True)

    # axs[0, 1].plot(t, cbf_t)
    # axs[0, 1].set_title('cbf cstr')
    # axs[1, 1].plot(t, h_t)
    # axs[1, 1].set_title('cbf')
    axs[1, 1].plot(t, agent.control_log[:, 0], label='Nominal')
    # axs[2, 1].plot(t, u_filtered_t, '--')
    axs[1, 1].set_title('Controls')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel('Time (s)')
    # axs[2, 1].legend()

    plt.tight_layout()
    plt.show()

    # np.savez('simulation_data_alpha=0.2.npz', state_log=agent.state_log, control_log=agent.control_log)
