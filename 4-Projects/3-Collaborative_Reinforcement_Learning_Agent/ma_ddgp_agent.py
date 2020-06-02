from ddpg_agent import Agent
import numpy as np

class maddpgagent:
    def __init__(self, state_dim, action_dim,num_agents, seed):

        # Initlaize each agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_unique_agents=1
        self.seed = seed
        self.agents = [Agent(self.state_dim, self.action_dim,self.num_unique_agents, self.seed) for _ in range(num_agents)]


    def reset(self):
        for agent in self.agents:
            agent.reset()

    def ma_act(self,states):
        action_0 = self.agents[0].act(states, True)
        action_1 = self.agents[1].act(states, True)
        actions = np.concatenate((action_0, action_1), axis=0) 
        actions = np.reshape(actions, (1, 4))
        return actions
        # return np.concatenate((action_0, action_1), axis=0).flatten()
        # actions = [agent.act(states) for agent in self.agents ]
        # return np.reshape(actions, (1, self.num_agents*self.action_dim))

    def ma_step(self, states, actions, rewards, next_states, dones):
        for i, agent in enumerate(self.agents):
            agent.step(states, actions, rewards[i], next_states, dones,i)