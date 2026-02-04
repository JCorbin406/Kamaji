from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
from kamaji.agent.agent import Agent


class WrapperEnv(Env):
    def __init__(self, simulator, agent, reward_fn, termination_fn, truncation_fn):
        super().__init__()
        self.agent = agent
        self.agent_id = agent._id
        self.state = None
        self.environment_state = None
        self.simulator = simulator
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.truncation_fn = truncation_fn

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(agent.sarl_info["state_features"]),), dtype=np.float32)
        self.action_space = Box(low=-10, high=10, shape=(len(agent.sarl_info["action_outputs"]),), dtype=np.float32)

    def update_state(self, new_environment_state: dict):
        self.environment_state = new_environment_state
        if new_environment_state[self.agent_id] is not None:
            self.state = np.array([new_environment_state[self.agent_id][feature] for feature in self.agent.sarl_info["state_features"]])
        else:
            self.state = None

    def step(self, action):
        orig_environment_state = self.environment_state.copy()
        orig_state = self.state.copy()
        self.simulator.step((self.agent_id, action))

        reward = self.reward_fn(orig_environment_state, action, self.environment_state) if self.state is not None else 0

        terminated, reward_term = self.termination_fn(orig_environment_state, self.environment_state)
        if self.state is None:
            terminated = True
        reward += reward_term
        truncated = self.truncation_fn(self.simulator.sim_time)

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.simulator.reset()
        return self.state, {}
    
    