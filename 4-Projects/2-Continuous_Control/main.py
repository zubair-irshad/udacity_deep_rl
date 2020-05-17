import gym
from ddpg_agent import Agent
import numpy as np
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from monitor import interact

env = UnityEnvironment(file_name="./Reacher_Linux_NoVis/Reacher.x86_64")
# reset env and extract state_dim and action_dim
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state_dim = env_info.vector_observations.shape[1]
action_dim = brain.vector_action_space_size
# number of agents
num_agents = len(env_info.agents)

agent = Agent(state_dim=state_dim, action_dim=action_dim,num_agents = num_agents, seed=np.random.randint(100))
scores = interact(env, brain_name, agent, num_agents)

# # plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
