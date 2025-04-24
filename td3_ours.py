import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque
import random
import copy
from torch.utils.tensorboard import SummaryWriter
import os
import argparse

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Humanoid-v5', help='Environment name')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum training steps')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
parser.add_argument('--sigma', type=float, default=0.1, help='Exploration noise std')
parser.add_argument('--sigma_', type=float, default=0.2, help='Target policy smoothing noise std')
parser.add_argument('--c', type=float, default=0.5, help='Noise clip range')
parser.add_argument('--d', type=int, default=2, help='Policy update delay')
parser.add_argument('--save_interval', type=int, default=10000, help='Model save interval')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--start_steps', type=int, default=25000, help='Steps before training starts')
parser.add_argument('--play', action='store_true', help='Play best model')
parser.add_argument('--train', action='store_true', help='Train the model')

args = parser.parse_args()

# Actor network that maps states to actions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        # Define network architecture with two hidden layers
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Tanh activation to bound actions
        )
        self.max_action = max_action

    def forward(self, x):
        # Scale output to max action range
        return self.net(x) * self.max_action

# Critic network that estimates Q-values
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Two Q-networks for TD3's double Q-learning
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Concatenate state and action as input
        sa = torch.cat([state, action], dim=1)
        return self.net1(sa), self.net2(sa)

    def Q1(self, state, action):
        # Get Q-value from first network only
        sa = torch.cat([state, action], dim=1)
        return self.net1(sa)

# Training function implementing TD3 algorithm
def train():
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize environment and logging
    env = gym.make(args.env)
    writer = SummaryWriter('runs/TD3_training')

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize actor and critic networks
    actor = Actor(state_dim, action_dim, max_action)
    actor_target = copy.deepcopy(actor)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    critic = Critic(state_dim, action_dim)
    critic_target = copy.deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)

    # Initialize replay buffer and training variables
    replay = deque(maxlen=int(1e6))
    state, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_count = 0
    best_reward = float('-inf')

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Main training loop
    pbar = tqdm(range(args.max_steps), desc='Training')
    for step in pbar:
        # Initial exploration phase
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            # Select action according to policy with noise
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = (
                    actor(state_tensor).squeeze().numpy()
                    + np.random.normal(0, max_action * args.sigma, size=action_dim)
                ).clip(-max_action, max_action)

        # Execute action and store transition
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Store experience in replay buffer
        replay.append((state, action, reward, next_state, done))
        state = next_state

        # Handle episode termination
        if done:
            state, _ = env.reset()
            writer.add_scalar('Episode Reward', episode_reward, episode_count)

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'best_reward': best_reward
                }, 'models/best_model.pth')

            # Periodic model saving
            if step % args.save_interval == 0:
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'step': step,
                    'episode_reward': episode_reward
                }, f'models/model_step_{step}.pth')

            episode_count += 1
            episode_reward = 0

        # Training phase
        if len(replay) >= args.batch_size and step >= args.start_steps:
            # Sample mini-batch from replay buffer
            batch = random.sample(replay, args.batch_size)
            state_batch = torch.FloatTensor(np.array([t[0] for t in batch]))
            action_batch = torch.FloatTensor(np.array([t[1] for t in batch]))
            reward_batch = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1)
            next_state_batch = torch.FloatTensor(np.array([t[3] for t in batch]))
            done_batch = torch.FloatTensor(np.array([t[4] for t in batch])).unsqueeze(1)

            with torch.no_grad():
                # TD3's target policy smoothing
                noise = (torch.randn_like(action_batch) * args.sigma_).clamp(-args.c, args.c)
                next_action = (actor_target(next_state_batch) + noise).clamp(-max_action, max_action)

                # Compute target Q-values
                target_Q1, target_Q2 = critic_target(next_state_batch, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward_batch + (1 - done_batch) * args.gamma * target_Q

            # Update critic
            current_Q1, current_Q2 = critic(state_batch, action_batch)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            writer.add_scalar('Critic Loss', critic_loss.item(), step)

            # Delayed policy updates
            if step % args.d == 0:
                # Update actor
                actor_loss = -critic.Q1(state_batch, actor(state_batch)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                writer.add_scalar('Actor Loss', actor_loss.item(), step)

                # Soft update target networks
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # Update progress bar
        pbar.set_postfix({
            'Episode': episode_count,
            'Reward': episode_reward,
            'Best Reward': best_reward
        })

    # Save final model
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'step': args.max_steps,
        'episode_reward': episode_reward
    }, 'models/final_model.pth')

    env.close()
    writer.close()

# Testing function to evaluate trained model
def test():
    # Initialize environment and model
    env = gym.make(args.env, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create and load actor network
    actor = Actor(state_dim, action_dim, max_action)
    checkpoint = torch.load('models/best_model.pth', weights_only=False, map_location='cpu')
    # checkpoint = torch.load('humanoid/runs/td3-ours/models/best_model.pth', weights_only=False, map_location='cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    print(checkpoint['best_reward'])
    actor.eval()

    # Run episodes indefinitely
    while True:
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select action using policy
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action = actor(state_tensor).numpy().squeeze(0)

            # Execute action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        print(f"Episode Reward: {episode_reward}")


if __name__ == "__main__":
    if args.train:
        train()
    else:
        test()


