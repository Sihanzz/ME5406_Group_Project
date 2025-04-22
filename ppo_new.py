import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
from collections import namedtuple
from datetime import datetime
from tqdm import tqdm

# Global parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device (GPU/CPU)
NUM_ENVS = 8  # Number of parallel environments
SAMPLE_STEPS = 2048  # Steps per sampling iteration
TOTAL_STEPS = 10_000_000  # Total training steps
MINI_BATCH_SIZE = 256  # Mini-batch size for training
EPOCHES = 10  # Number of epochs per update
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
CLIP_EPS = 0.2  # PPO clipping epsilon
ENV_ID = "Humanoid-v5"  # Environment

# Orthogonal initialization for network layers
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Agent network architecture(Actor-Critic network)
# We use a shared feature extractor and two separate heads for the Beta distribution parameters
# Beta distribution is used to model the action space, which is continuous and bounded
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Agent, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Alpha parameter head for Beta distribution
        self.actor_head_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Beta parameter head for Beta distribution
        self.actor_head_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Value function head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(orthogonal_init)

    def forward(self, x, action=None):
        feature = self.shared(x)

        # Generate Beta distribution parameters
        alpha = F.softplus(self.actor_head_alpha(feature)) + 1.0  # use softplus to ensure positive values(softplus(x) = ln(1 + exp(x)), alpha and beta must be positive as defined in the Beta distribution)
        beta = F.softplus(self.actor_head_beta(feature)) + 1.0
        dist = torch.distributions.Beta(alpha, beta)
        entropy = dist.entropy().sum(-1)
        if action is None:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)

        value = self.critic_head(feature)
        return action, log_prob, entropy, value




# Environment wrapper setup
def make_env(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
    return env

# Named tuple for storing transitions
Transitions = namedtuple('Transitions', ['states', 'actions', 'log_probs', 'values', 'advantages', 'returns'])

# Collect rollout data from environment
def rollout(env, agent, state_dim, action_dim, state=None):
    # Initialize storage tensors
    states = torch.zeros((SAMPLE_STEPS, NUM_ENVS, state_dim), dtype=torch.float32).to(DEVICE)  # make a tensor with shape (SAMPLE_STEPS, NUM_ENVS, state_dim)
    actions = torch.zeros((SAMPLE_STEPS, NUM_ENVS, action_dim), dtype=torch.float32).to(DEVICE)
    log_probs = torch.zeros((SAMPLE_STEPS, NUM_ENVS), dtype=torch.float32).to(DEVICE)
    values = torch.zeros_like(log_probs)
    rewards = torch.zeros_like(log_probs)
    dones = torch.zeros_like(log_probs)
    advantages = torch.zeros_like(log_probs)

    rs = []
    next_state = None
    # Collect steps
    for step in tqdm(range(SAMPLE_STEPS)):
        x = torch.from_numpy(state).float().to(DEVICE)
        action, log_prob, _, value = agent(x)
        states[step] = x
        actions[step] = action
        log_probs[step] = log_prob
        values[step] = value.squeeze(-1)

        # Scale actions to environment range
        action_ = (action.cpu().numpy() - 0.5) * 2 * env.action_space.high  # (0, 1) -> (-0.4, 0.4)
        next_state, reward, terminated, truncated, info = env.step(action_)

        # Record episode rewards
        if 'episode' in info:
            rr = info['episode']['r']  
            rr_ = info['episode']['_r']
            rs += rr[rr_].tolist()

        rewards[step] = torch.from_numpy(reward).float().to(DEVICE)  # convert numpy array to tensor to calculate the advantages
        dones[step] = torch.from_numpy(np.logical_or(terminated, truncated)).float().to(DEVICE)
        state = next_state

    # Calculate advantages using GAE
    # GAE(Generalized Advantage Estimation) is a technique to estimate the advantages of the actions
    next_value = agent.critic_head(agent.shared(torch.from_numpy(next_state).float().to(DEVICE))).squeeze(-1)
    next_advantage = 0
    for step in reversed(range(SAMPLE_STEPS)):
        delta = rewards[step] + GAMMA * (1 - dones[step]) * next_value - values[step]  
        advantages[step] = delta + GAMMA * GAE_LAMBDA * next_advantage * (1 - dones[step])  

        next_value = values[step]
        next_advantage = advantages[step]

    returns = values + advantages
    return rs, next_state, Transitions(
        states.flatten(0, 1),
        actions.flatten(0, 1),
        log_probs.flatten(0, 1),
        values.flatten(0, 1),
        advantages.flatten(0, 1),
        returns.flatten(0, 1))  # flatten the tensors to 1D tensor

# PPO training function
def ppo_train(transitions, agent, optimizer, writer, update_step=0):
    N = transitions.states.shape[0]  
    for epoch in range(EPOCHES):
        indices = torch.randperm(N)  # random permutation of the indices
        for chunk_idx in torch.split(indices, MINI_BATCH_SIZE):
            states = transitions.states[chunk_idx]
            actions = transitions.actions[chunk_idx]
            log_probs = transitions.log_probs[chunk_idx]
            returns = transitions.returns[chunk_idx]
            advantages = transitions.advantages[chunk_idx]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize the advantages

            # Update policy and value network
            _, new_log_probs, entropy, new_values = agent(states, actions)
            new_values = new_values.squeeze(-1)
            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            entropy_loss = -entropy.mean()
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + value_loss * 0.5 + entropy_loss * 0.0  # we find the total loss without the entropy loss is better, so we just ignore it

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)  
            optimizer.step()

            # Log metrics
            writer.add_scalar('Loss/Total', total_loss.item(), update_step)
            writer.add_scalar('Loss/Policy', policy_loss.item(), update_step)
            writer.add_scalar('Loss/Value', value_loss.item(), update_step)
            writer.add_scalar('Loss/Entropy', entropy_loss.item(), update_step)
            update_step += 1

    return update_step

# Main training loop
def train():
    # Setup environment and agent
    env = gym.make_vec(ENV_ID, num_envs=NUM_ENVS, wrappers=[make_env], vector_kwargs={'autoreset_mode': 'NextStep'})  
    state_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    agent = Agent(state_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4, eps=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS // (NUM_ENVS * SAMPLE_STEPS))  # cosine annealing learning rate scheduler

    # Setup logging
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/{ENV_ID}_{time_str}')
    update_step = 0
    total_timestep = 0
    state, _ = env.reset()
    
    # Training loop
    while True:
        with torch.no_grad():
            rss, state, transitions = rollout(env, agent, state_dim, act_dim, state)

        total_timestep += NUM_ENVS * SAMPLE_STEPS
        if len(rss) > 0:
            writer.add_scalar('Reward', np.mean(rss), total_timestep)
            print(f"Steps: {total_timestep}, Mean Reward: {np.mean(rss)}, Std Reward: {np.std(rss)}, Count: {len(rss)}")
        update_step = ppo_train(transitions, agent, optimizer, writer, update_step=update_step)
        lr_scheduler.step()
        if total_timestep > TOTAL_STEPS:
            break
    env.close()

if __name__ == '__main__':
    train()
