import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#  Hyperparameter Configuration
ACTOR_LR = 3e-4
CRITIC_LR = 9e-3
GAMMA = 0.99
TAU = 0.005
N_ATOMS = 51
V_MIN = 0
V_MAX = 1000.0

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.model(x)

class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_atoms=N_ATOMS):
        super().__init__()
        self.n_atoms = n_atoms
        self.linear1 = nn.Linear(state_dim + action_dim, 400)
        self.linear2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, n_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)

class D4PGAgent:
    def __init__(self, state_dim, action_dim, max_action,
                 actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR,
                 gamma=GAMMA,
                 tau=TAU,
                 device="cpu"):
        self.device = device
        self.n_atoms = N_ATOMS
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)
        self.support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(device)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = DistributionalCritic(state_dim, action_dim, N_ATOMS).to(device)
        self.critic_target = DistributionalCritic(state_dim, action_dim, N_ATOMS).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).detach().cpu().numpy().flatten()

    def projection_distribution(self, next_dist, rewards, dones):
        batch_size = rewards.size(0)
        projected = torch.zeros(batch_size, self.n_atoms, device=self.device)

        r = rewards.view(-1)
        d = dones.view(-1)

        for i in range(self.n_atoms):
            z_i = self.support[i]
            tz_j = r + self.gamma * z_i * (1 - d)
            tz_j = tz_j.clamp(self.v_min, self.v_max)
            b_j = (tz_j - self.v_min) / self.delta_z

            l = b_j.floor().long()
            u = b_j.ceil().long()
            eq = (u == l).float()

            batch_idx = torch.arange(batch_size, device=self.device)
            l_idx = batch_idx * self.n_atoms + l
            u_idx = batch_idx * self.n_atoms + u

            proj_flat = projected.view(-1)

            w_l = next_dist[:, i] * (u.float() - b_j + eq)
            proj_flat.index_add_(0, l_idx, w_l)

            w_u = next_dist[:, i] * (b_j - l.float())
            proj_flat.index_add_(0, u_idx, w_u)

        return projected

    def train(self, replay_buffer, batch_size):
        # Sample a batch from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_logits = self.critic_target(next_state, next_action)
            next_prob = F.softmax(next_logits, dim=1)
            target_prob = self.projection_distribution(next_prob, reward, done)

        logits = self.critic(state, action)
        log_prob = F.log_softmax(logits, dim=1)
        critic_loss = -torch.sum(target_prob * log_prob, dim=1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.sum(F.softmax(self.critic(state, self.actor(state)), dim=1) * self.support, dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
