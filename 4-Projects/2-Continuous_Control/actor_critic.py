import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

def hidden_init(layer):
    d = layer.weight.data.size()[0]
    lim = 1./np.sqrt(d)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seed, fc1_dim=400, fc2_dim=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed, c_fc1_dim=400, c_fc2_dim=256, c_fc3_dim=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.c_fc1 = nn.Linear(state_dim, c_fc1_dim)
        self.c_fc2 = nn.Linear(c_fc1_dim+action_dim, c_fc2_dim)
        self.c_fc3 = nn.Linear(c_fc2_dim, c_fc3_dim)
        self.c_fc4 = nn.Linear(c_fc3_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.c_fc1.weight.data.uniform_(*hidden_init(self.c_fc1))
        self.c_fc2.weight.data.uniform_(*hidden_init(self.c_fc2))
        self.c_fc3.weight.data.uniform_(*hidden_init(self.c_fc3))
        self.c_fc4.weight.data.uniform_(-3e-3,3e-3)

    def forward(self, state, action):
        xs = F.relu(self.c_fc1(state))
        x = torch.cat((xs,action), dim=1)
        x = F.relu(self.c_fc2(x))
        x = F.relu(self.c_fc3(x))
        x = self.c_fc4(x)
        return x
