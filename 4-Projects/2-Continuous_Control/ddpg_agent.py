import random
import torch
import numpy as np
from actor_critic import Actor, Critic
from collections import deque, namedtuple
import torch.optim as optim
import torch.nn.functional as F
import copy

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
learn_every_tstep = 20
gamma = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
num_updates=10
Epsilon = 1.0
eps_decay = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_dim, action_dim,num_agents, seed):
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
        self.actor_local = Actor(state_dim, action_dim, seed).to(device)
        self.actor_target = Actor(state_dim, action_dim, seed).to(device)
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_dim, action_dim, seed).to(device)
        self.critic_target = Critic(state_dim, action_dim, seed).to(device)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Initialize replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, action_dim, seed)

        self.num_agents = num_agents
        
        # Noise process
        self.noise = OUNoise(self.action_dim,self.num_agents, seed)

        # initialize timestep to keep track of learning updates
        self.timestep = 0

        self.epsilon = Epsilon

    def act(self, state, add_noise=True):
        '''
        Select Deterministic policy at all times
        '''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        # get_actions
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon*self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        '''
        Stores state, action, reward, next_state, done tuple in replay memory
        Learn every LEARN_EVERY timestep
        '''
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        # Check if memory is not underfull
        self.timestep = (self.timestep + 1) % learn_every_tstep
        if self.timestep ==0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(num_updates):
                    sampled_exp = self.memory.sample()
                    self.learn(sampled_exp, gamma)

    def learn(self, experiences, gamma):
        '''
        Training step from collected experiences
        '''

        states, actions, rewards, next_states, dones = experiences

        #Update Critic loss and perform backprop to nudge grad values to predict Q better

        #Q_target
        next_actions = self.actor_target(next_states)
        next_Q       = self.critic_target(next_states, next_actions)
        Q_target     = rewards + (gamma* next_Q*(1-dones))
        # Expected Q_value & loss
        Q_expected = self.critic_local(states, actions.float())
        q_loss = F.mse_loss(Q_expected,Q_target)
        #backward pass
        self.optimizer_critic.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.optimizer_critic.step()

        #Update Actor loss and perform backprop 
        actor_actions = self.actor_local(states)
        actor_loss    = -self.critic_local(states, actor_actions).mean()
         #backward pass
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Update target network#
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        self.epsilon -= eps_decay
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)

    def reset(self):
        self.noise.reset()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed,num_agents, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
        self.state = x + dx
        return self.state

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
