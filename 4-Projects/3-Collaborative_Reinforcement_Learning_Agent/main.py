import gym
from ma_ddgp_agent import maddpgagent
import numpy as np
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import interact

env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64", seed=0)
# reset env and extract state_dim and action_dim
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state_dim = env_info.vector_observations.shape[1]
action_dim = brain.vector_action_space_size
# number of agents
num_agents = len(env_info.agents)
agent = maddpgagent(state_dim=state_dim, action_dim=action_dim,num_agents = num_agents, seed=0)
scores = interact(env, state_dim, brain_name, agent,num_agents)
