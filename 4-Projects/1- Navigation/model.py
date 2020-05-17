import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Network_FullyConnected(nn.Module):
    def __init__(self, state_dim, action_dim, seed, fc1_dim=128, fc2_dim=64):
        super(Q_Network_FullyConnected, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
