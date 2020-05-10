import random
import torch
import numpy as np
from model import Q_Network_FullyConnected
from collections import deque, namedtuple
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 100
learn_every_tstep = 5
gamma = 0.90
learning_rate = 5e-4
TAU = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_dim, action_dim, seed):
        """Initialize an agent object
        
        Params
        ==========
        state_dim(int)
        action_dim(int)
        seed(int)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)
        # Initialize Q network
        self.qNetwork_local = Q_Network_FullyConnected(state_dim, action_dim, seed).to(device)
        self.qNetwork_target = Q_Network_FullyConnected(state_dim, action_dim, seed).to(device)
        self.optimizer = optim.Adam(self.qNetwork_local.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, action_dim, seed)

        # initialize timestep to keep track of learning updates
        self.timestep = 0

    def act(self, state, eps=0.):
        '''
        Select epsilon-greedy action
        '''
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.qNetwork_local.eval()
        # get_actions
        with torch.no_grad():
            action_values = self.qNetwork_local(state)

        self.qNetwork_local.train()
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.action_dim))

        return action

    def step(self, state, action, reward, next_state, done):
        '''
        Stores state, action, reward, next_state, done tuple in replay memory
        Learn every LEARN_EVERY timestep
        '''
        self.memory.add(state, action, reward, next_state, done)
        # Learn every LEARN_EVERY timestep

        self.timestep = (self.timestep + 1) % learn_every_tstep
        if self.timestep == 0:
            # Check if memory is not underfull
            if len(self.memory) > BATCH_SIZE:
                sampled_exp = self.memory.sample()
                self.learn(sampled_exp, gamma)

    def learn(self, experiences, gamma):
        '''
        Training step from collected experiences
        '''

        states, actions, rewards, next_states, dones = experiences

        # Expected Q_value
        expected_Q = self.qNetwork_local(states).gather(1, actions)

        # Target Q_value
        next_Q = self.qNetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        target_Q = rewards + (gamma * next_Q * (1 - dones))
        loss = F.mse_loss(expected_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network#
        self.soft_update(self.qNetwork_local, self.qNetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1 - tau) * target_params.data)


class ReplayBuffer:
    """ Buffer to store and sample experiences"""

    def __init__(self, buffer_size, batch_size, action_size, seed):
        """Initialize a Replay Buffer Object
        Params
        ======
        batch_size(int) : size of sampled batch
        buffer_size(int): size of replay buffer
        action_size(int): action space size
        seed(int)       : random seed
        """
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        # create a memory queue
        self.memory = deque(maxlen=buffer_size)
        # Create a named tuple to store an experiences : (state,action,rewards, next_state, done)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        # Add an incoming (state,action,reward, next_state, done) tuple in memory
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        # Sample a batch from memory
        experience = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
