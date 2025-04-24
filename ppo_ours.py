# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
from collections import namedtuple
from datetime import datetime
from tqdm import tqdm
import sys
# Set device and hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 4  # Number of parallel environments
SAMPLE_STEPS = 2048  # Steps to sample per iteration
TOTAL_STEPS = 4_000_000  # Total training steps
MINI_BATCH_SIZE = 256  # Mini batch size for training
EPOCHES = 10  # Number of epochs per iteration
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
CLIP_EPS = 0.2  # PPO clipping epsilon
ENV_ID = "Humanoid-v5"  # Environment ID

# Initialize network weights using orthogonal initialization
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Define the Agent network architecture
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Agent, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor network heads for Beta distribution parameters
        self.actor_head_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.actor_head_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic network head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(orthogonal_init)

    # Forward pass through the network
    def forward(self, x, action=None, test=False):
        feature = self.shared(x)
        alpha = F.softplus(self.actor_head_alpha(feature)) + 1.0
        beta = F.softplus(self.actor_head_beta(feature)) + 1.0
        dist = torch.distributions.Beta(alpha, beta)
        entropy = dist.entropy().sum(-1)
        if action is None:
            if test:
                action = alpha / (alpha + beta)
            else:
                action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic_head(feature)
        return action, log_prob, entropy, value

# Environment wrapper function
def make_env(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
    return env

# Define namedtuple for storing transitions
Transitions = namedtuple(
    "Transitions", ["states", "actions", "log_probs", "values", "advantages", "returns"]
)

# Collect rollout data from environment
def rollout(env, agent, state_dim, action_dim, obs=None, test=False):
    states = torch.zeros((SAMPLE_STEPS, NUM_ENVS, state_dim), dtype=torch.float32).to(
        DEVICE
    )
    actions = torch.zeros((SAMPLE_STEPS, NUM_ENVS, action_dim), dtype=torch.float32).to(
        DEVICE
    )
    log_probs = torch.zeros((SAMPLE_STEPS, NUM_ENVS), dtype=torch.float32).to(DEVICE)
    values = torch.zeros_like(log_probs)
    rewards = torch.zeros_like(log_probs)
    dones = torch.zeros_like(log_probs)
    advantages = torch.zeros_like(log_probs)

    rs = []
    next_obs = None
    for step in range(SAMPLE_STEPS):
        x = torch.from_numpy(obs).float().to(DEVICE)
        action, log_prob, _, value = agent(x, test=test)
        states[step] = x
        actions[step] = action
        log_probs[step] = log_prob
        values[step] = value.squeeze(-1)

        action_ = (action.cpu().numpy() - 0.5) * 2 * env.action_space.high
        next_obs, reward, terminated, truncated, info = env.step(action_)

        if "final_info" in info:
            rr = info["final_info"]["episode"]["r"]
            rr_ = info["final_info"]["episode"]["_r"]
            rs += rr[rr_].tolist()

        rewards[step] = torch.from_numpy(reward).float().to(DEVICE)
        dones[step] = (
            torch.from_numpy(np.logical_or(terminated, truncated)).float().to(DEVICE)
        )
        obs = next_obs

    # Calculate advantages using GAE
    next_o = torch.from_numpy(next_obs).float().to(DEVICE)
    next_value = agent.critic_head(agent.shared(next_o)).squeeze(-1)
    next_advantage = 0
    for step in reversed(range(SAMPLE_STEPS)):
        delta = rewards[step] + GAMMA * (1 - dones[step]) * next_value - values[step]
        advantages[step] = delta + GAMMA * GAE_LAMBDA * next_advantage * (
            1 - dones[step]
        )

        next_value = values[step]
        next_advantage = advantages[step]

    returns = values + advantages
    return (
        rs,
        next_obs,
        Transitions(
            states.flatten(0, 1),
            actions.flatten(0, 1),
            log_probs.flatten(0, 1),
            values.flatten(0, 1),
            advantages.flatten(0, 1),
            returns.flatten(0, 1),
        ),
    )

# PPO training function
def ppo_train(transitions, agent, optimizer, writer, update_step=0, global_step=0):
    N = transitions.states.shape[0]
    for epoch in range(EPOCHES):
        indices = torch.randperm(N)
        for chunk_idx in torch.split(indices, MINI_BATCH_SIZE):
            states = transitions.states[chunk_idx]
            actions = transitions.actions[chunk_idx]
            log_probs = transitions.log_probs[chunk_idx]
            returns = transitions.returns[chunk_idx]
            advantages = transitions.advantages[chunk_idx]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update policy and value network
            _, new_log_probs, entropy, new_values = agent(states, actions)
            new_values = new_values.squeeze(-1)
            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
            entropy_loss = -entropy.mean()
            value_loss = F.mse_loss(new_values, returns)

            total_loss = policy_loss + value_loss * 0.5

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            update_step += 1
    writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    return update_step

# Main training loop
def train():
    # Initialize environment and agent
    env = gym.make_vec(
        ENV_ID,
        num_envs=NUM_ENVS,
        wrappers=[make_env],
        vector_kwargs={"autoreset_mode": "SameStep"},
    )
    state_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    agent = Agent(state_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4, eps=1e-5)

    # Load pretrained model if specified
    load_pretrained = False
    if load_pretrained:
        data = torch.load("model_best.pt", weights_only=False)
        agent.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        obs_rms = data["obs_rms"]
        for env_, rms in zip(env.envs, obs_rms):
            env_.env.obs_rms.mean = rms[0]
            env_.env.obs_rms.var = rms[1]
            env_.env.obs_rms.count = rms[2]
        print("Loaded weights, best episode reward:", data["mean_reward"])

    # Setup learning rate scheduler and tensorboard writer
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_STEPS // (NUM_ENVS * SAMPLE_STEPS)
    )
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/ppo_{ENV_ID}_{time_str}")
    update_step = 0
    total_timestep = 0
    obs, _ = env.reset()
    last_mean_reward = 0
    
    # Training iterations
    for _ in tqdm(range(TOTAL_STEPS // (NUM_ENVS * SAMPLE_STEPS) + 1)):
        with torch.no_grad():
            rss, obs, transitions = rollout(env, agent, state_dim, act_dim, obs)

        total_timestep += NUM_ENVS * SAMPLE_STEPS
        if len(rss) > 0:
            writer.add_scalar("charts/episodic_return", np.mean(rss), total_timestep)

            # Save best model
            if np.mean(rss) > last_mean_reward:
                last_mean_reward = np.mean(rss)
                obs_rms = []
                for env_ in env.envs:
                    rms = env_.env.obs_rms
                    obs_rms.append((rms.mean, rms.var, rms.count))
                torch.save(
                    {
                        "model": agent.state_dict(),
                        "mean_reward": last_mean_reward,
                        "obs_rms": obs_rms,
                        "optimizer": optimizer.state_dict(),
                    },
                    f"runs/ppo_{ENV_ID}_{time_str}/model_best.pt",
                )
        update_step = ppo_train(
            transitions, agent, optimizer, writer, update_step=update_step, global_step=total_timestep
        )
        lr_scheduler.step()
    env.close()

# Test function to visualize trained agent
def test():
    env = gym.make_vec(
        ENV_ID,
        num_envs=1,
        wrappers=[make_env],
        render_mode="human",
        vector_kwargs={"autoreset_mode": "NextStep"},
    )
    state_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    agent = Agent(state_dim, act_dim).to(DEVICE)

    # data = torch.load("runs/ppo_Humanoid-v5_2025-04-24_21-03-36/model_best.pt", weights_only=False)  # your load path needs to be changed when you want to train it again to visualize the model_best.pt since the name of the file is according to the exact time of training
    data = torch.load("model_best.pt", weights_only=False, map_location='cpu')
    agent.load_state_dict(data["model"])
    obs_rms = data["obs_rms"]
    for env_, rms in zip(env.envs, obs_rms):
        env_.env.obs_rms.mean = rms[0]
        env_.env.obs_rms.var = rms[1]
        env_.env.obs_rms.count = rms[2]
    print("Loaded weights, best episode reward:", data["mean_reward"])

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    obs, _ = env.reset()

    with torch.no_grad():
        rss, obs, transitions = rollout(env, agent, state_dim, act_dim, obs, test=True)
    print(rss)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        test()
