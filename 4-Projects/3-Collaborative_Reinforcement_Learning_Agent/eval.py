import gym
import numpy as np
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import interact
import torch
from ma_ddgp_agent import maddpgagent
from ddpg_agent import Agent

env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
# reset env and extract state_dim and action_dim
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]    # reset the environment  
state_dim = env_info.vector_observations.shape[1]
action_dim = brain.vector_action_space_size
# number of agents
num_agents = len(env_info.agents)
agent = maddpgagent(state_dim=state_dim, action_dim=action_dim,num_agents = num_agents, seed=0)

#Watch a smart agent 
# Load the saved training parameters

agent_0 = Agent(state_dim, action_dim, 1, random_seed=0)
agent_1 = Agent(state_dim, action_dim, 1, random_seed=0)
agent_0.actor_local.load_state_dict(torch.load('logging/checkpoint_actor_0.pth', map_location='cpu'))
agent_0.critic_local.load_state_dict(torch.load('logging/checkpoint_critic_0.pth', map_location='cpu'))
agent_1.actor_local.load_state_dict(torch.load('logging/checkpoint_actor_1.pth', map_location='cpu'))
agent_1.critic_local.load_state_dict(torch.load('logging/checkpoint_critic_1.pth', map_location='cpu'))
  
states = env_info.vector_observations                  # get the current state (for each agent)
states = np.reshape(states, (1,48))
score = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    # actions = agent.ma_act(states)
    action_0 = agent_0.act(states, add_noise=False)         
    action_1 = agent_1.act(states, add_noise=False)        
    actions = np.concatenate((action_0, action_1), axis=0) 
    actions = np.reshape(actions, (1, 4))
    env_info = env.step(actions)[brain_name]
    next_states = np.reshape(env_info.vector_observations, (1,48)) # flatten states
    rewards = env_info.rewards
    dones = env_info.local_done
    states = next_states
    score += rewards
    if any(dones):
        break