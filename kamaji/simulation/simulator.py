# simulator.py
from time import time
from typing import Optional
from tqdm import tqdm
import numpy as np
from sympy import symbols, sqrt, Matrix, simplify, diff, init_printing, lambdify
import sympy as sp
from qpsolvers import solve_qp

from kamaji.agent.agent import Agent


# Add agent IDs, whatever number they were initialized in
class Simulator:
    def __init__(self, total_time: float, dt: float, agents: Optional[list[Agent]] = None) -> None:
        """
        Initializes a Simulation with a specified total time and step size.

        Args:
            total_time (float): The total time the simulation should run for, in seconds
            dt (float): The step size of the simulation, in seconds.
            agents (Optional[list[Agent]]): Initial agents to add to the sim, if desired.
        """
        if agents is None:
            self.active_agents = []
        else:
            self.active_agents = agents
        self.inactive_agents = []
        self.total_time = total_time
        self.t = 0.0
        self.dt = dt

        """CBF Stuff"""
        x1, x2, x3, y1, y2, y3, r, alpha, gamma = symbols('x1 x2 x3 y1 y2 y3 r alpha gamma')
        # Define h(x) functions
        h1 = sqrt((x1-x2)**2 + (y1-y2)**2) - r
        h2 = sqrt((x1-x3)**2 + (y1-y3)**2) - r
        h3 = sqrt((x2-x3)**2 + (y2-y3)**2) - r
        h_funcs = [h1, h2, h3]
        # Define f(x) and g(x) as matrices
        f = Matrix([0, 0])
        g = Matrix([
            [1, 0],  # Position derivatives (p1, p2, p3) do not depend on control inputs
            [0, 1]
        ])
        sum = 0
        for idx, cbf in enumerate(h_funcs):
            sum += sp.exp(-gamma*cbf)
        h = -(1/gamma)*sum
        state_vars1 = Matrix([x1, y1])
        state_vars2 = Matrix([x2, y2])
        state_vars3 = Matrix([x3, y3])
        grad_h1 = h.diff(state_vars1).T
        grad_h2 = h.diff(state_vars2).T
        grad_h3 = h.diff(state_vars3).T
        Lf1_h = grad_h1 * f
        Lg1_h = grad_h1 * g
        Lf2_h = grad_h2 * f
        Lg2_h = grad_h2 * g
        Lf3_h = grad_h3 * f
        Lg3_h = grad_h3 * g
        self.h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), h)
        self.f_func = lambdify((), f)
        self.g_func = lambdify((), g)
        self.Lf1_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lf1_h)
        self.Lf2_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lf2_h)
        self.Lf3_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lf3_h)
        self.Lg1_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lg1_h)
        self.Lg2_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lg2_h)
        self.Lg3_h_func = lambdify((x1, y1, x2, y2, x3, y3, r, gamma), Lg3_h)
        self.h1_func = lambdify((x1, y1, x2, y2, x3, y3, r), h1)
        self.h2_func = lambdify((x1, y1, x2, y2, x3, y3, r), h2)
        self.h3_func = lambdify((x1, y1, x2, y2, x3, y3, r), h3)
        self.H_hist = []
        self.h1_hist = []
        self.h2_hist = []
        self.h3_hist = []

    def simulate(self) -> None:
        """
        Runs the entire length of the simulation.
        """
        start_time = time()
        time_steps = int(self.total_time / self.dt)
        for _ in tqdm(range(time_steps)):
            self.step()
        sim_time = time() - start_time
        print(f"Sim time: {sim_time:.6f}")
        while len(self.active_agents) > 0:
            self.inactive_agents.append(self.active_agents.pop())

    def step(self) -> None:
        """
        Steps the simulation forward by simulating all agents.
        """
        """CBF Stuff"""
        if True:
            GAMMA = 100.0
            radius = 2.0
            controls = []
            for a in self.active_agents:
                controls += list(a.control_step(self.t, self.dt))
            controls = np.array(controls)
            controls.reshape(len(controls), 1)
            q = -2*controls
            P = 2*np.eye(6)
            ax1, ay1 = self.active_agents[0].state[0], self.active_agents[0].state[1]
            ax2, ay2 = self.active_agents[1].state[0], self.active_agents[1].state[1]
            ax3, ay3 = self.active_agents[2].state[0], self.active_agents[2].state[1]
            lg1h = self.Lg1_h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            lg2h = self.Lg2_h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            lg3h = self.Lg3_h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            lf1h = self.Lf1_h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            lf2h = self.Lf2_h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            lf3h = self.Lf3_h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            _G = -np.concatenate((lg1h, lg2h, lg3h), axis=1)
            """ Add noise. """
            # _G += 2 * np.random.rand(1, 6) - 1
            _H = lf1h + lf2h + lf3h + 1.0*self.h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA)
            u_filtered = solve_qp(P, q, _G, _H, solver="cvxopt")
            control_tuples = list(zip(u_filtered[::2], u_filtered[1::2]))
            self.H_hist.append(self.h_func(ax1, ay1, ax2, ay2, ax3, ay3, radius, GAMMA).item())
            self.h1_hist.append(self.h1_func(ax1, ay1, ax2, ay2, ax3, ay3, radius).item())
            self.h2_hist.append(self.h2_func(ax1, ay1, ax2, ay2, ax3, ay3, radius).item())
            self.h3_hist.append(self.h3_func(ax1, ay1, ax2, ay2, ax3, ay3, radius).item())

        for idx, a in enumerate(self.active_agents):
            # control_input = a.control_step(self.t, self.dt)
            control_input = control_tuples[idx]
            a.step(self.t, self.dt, control_input)
        self.t += self.dt

    def add_agent(self, agent: Agent) -> None:
        """
        Initializes a Simulation with a specified total time and step size.

        Args:
            t (float): The total time the simulation should run for, in seconds.
            dt (float): The step size of the simulation, in seconds.
        """
        self.active_agents.append(agent)

    def remove_agent(self, agent: Agent) -> bool:
        """
        Removes an Agent from the active Agents and moves it to the inactive Agents.
        If the agent does not exist in the active Agent list, an error will be thrown.

        Args:
            agent (Agent): The Agent to make inactive.

        Raises:
            ValueError: _description_
        """
        if agent not in self.active_agents:
            raise ValueError('Agent ' + str(agent) + ' is not an active Agent.')
        self.inactive_agents.append(self.active_agents.pop(self.active_agents.index(agent)))
        return True

    def get_active_agents(self) -> list[Agent]:
        """
        Gives the list of active Agents.

        Returns:
            list[Agent]: The list of active Agents.
        """
        return self.active_agents

    def get_inactive_agents(self) -> list[Agent]:
        """
        Gives the list of inactive Agents.

        Returns:
            list[Agent]: The list of inactive Agents.
        """
        return self.inactive_agents
