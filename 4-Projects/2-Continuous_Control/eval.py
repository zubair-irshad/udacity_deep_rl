import gym
from ddpg_agent import Agent
import numpy as np
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import interact
import torch

env = UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")
# reset env and extract state_dim and action_dim
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]    # reset the environment  
state_dim = env_info.vector_observations.shape[1]
action_dim = brain.vector_action_space_size
# number of agents
num_agents = len(env_info.agents)
agent = Agent(state_dim=state_dim, action_dim=action_dim,num_agents = num_agents, seed=np.random.randint(100))

#Watch a smart agent 
agent.actor_local.load_state_dict(torch.load('./logging/checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('./logging/checkpoint_critic.pth'))
  
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = agent.act(states)                        # select an action (for each agent)
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))