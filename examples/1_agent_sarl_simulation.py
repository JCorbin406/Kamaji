"""RL simulation using YAML configuration for Kamaji."""

import yaml
from kamaji.simulation.simulator import Simulator

if __name__ == "__main__":
    # Load configuration from a YAML file
    config_path = "examples/configs/1agent_sarl_simulation.yml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    def reward_fn(orig_state, action, new_state):
        # Simple reward: negative distance to goal
        curr_x, curr_y, goal_x, goal_y = orig_state
        reward = -((curr_x - goal_x) ** 2 + (curr_y - goal_y) ** 2) ** 0.5
        return reward  # Reward is current distance to goal
    
    def termination_fn(orig_env_state, new_env_state):
        a1_x, a1_y, a1_goal_x, a1_goal_y = orig_env_state["agent_1"]["position_x"], orig_env_state["agent_1"]["position_y"], orig_env_state["agent_1"]["goal_x"], orig_env_state["agent_1"]["goal_y"]
        
        terminate = False
        reward_term = None
        if ((a1_x - a1_goal_x) ** 2 + (a1_y - a1_goal_y) ** 2) ** 0.5 < 0.1:
            terminate = True
        return terminate, reward_term
    
    def truncation_fn(time_step):
        if time_step >= 50:
            return True
        return False
    
    sim = Simulator(config)
    sim.create_gym_envs(agent_ids=["agent_1"], reward_fns=[reward_fn], termination_fns=[termination_fn], truncation_fns=[truncation_fn])
    sim.train_agent_controller(agent_id="agent_1", training_steps=60000)
    sim.simulate()
    sim.plot.animate_trajectories()