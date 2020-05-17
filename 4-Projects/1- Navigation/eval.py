import gym
from nav_agent import Agent
import numpy as np
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import interact
import torch

env = UnityEnvironment(file_name="/home/mirshad7/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86")
# reset env and extract state_dim and action_dim
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state_dim = len(env_info.vector_observations[0])
action_dim = brain.vector_action_space_size

agent = Agent(state_dim=state_dim, action_dim=action_dim, seed=0)

#Watch a smart agent 
agent.qNetwork_local.load_state_dict(torch.load('./logging/nav_dqn_model_20200510-001226.pth'))
  
state = env_info.vector_observations[0]                # get the current state (for each agent)
score = 0                                             # initialize the score (for each agent)
while True:
    action = agent.act(state)                          # select an action (for each agent)
    env_info = env.step(action)[brain_name]            # send all actions to tne environment
    next_state = env_info.vector_observations[0]       # get next state (for each agent)
    reward = env_info.rewards[0]                       # get reward (for each agent)
    dones = env_info.local_done[0]                     # see if episode finished
    score += env_info.rewards[0]                         # update the score (for each agent)
    state = next_state                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(score))