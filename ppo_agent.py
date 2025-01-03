import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=3e-4):
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor-Critic Networks
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_size),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + 
                                  list(self.critic.parameters()), 
                                  lr=learning_rate)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        return action_probs.detach().numpy()

    def update(self, trajectories):
        states = torch.FloatTensor([t[0] for t in trajectories])
        actions = torch.FloatTensor([t[1] for t in trajectories])
        rewards = torch.FloatTensor([t[2] for t in trajectories])
        
        # Simple policy gradient update
        self.optimizer.zero_grad()
        values = self.critic(states)
        advantages = rewards - values.detach()
        
        action_probs = self.actor(states)
        actor_loss = -(action_probs * advantages).mean()
        critic_loss = nn.MSELoss()(values, rewards)
        
        loss = actor_loss + 0.5 * critic_loss
        loss.backward()
        self.optimizer.step() 