import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# PER buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition, td_error=1.0):
        p = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(p)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities)
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = (abs(td) + 1e-5) ** self.alpha

# Hyerparameters
env_id = "Humanoid-v5"
total_timesteps = 1000000
learning_starts = 10000
buffer_size = int(1e6)
batch_size = 256
gamma = 0.99
tau = 0.01
policy_lr = 3e-4
q_lr = 3e-4
policy_freq = 2
target_freq = 1
alpha = 0.2  # initial alpha value
auto_tune_alpha = True  # Enable auto-tuning of alpha
best_avg_reward = -float("inf")  # Initialize best average reward

def make_env():
    env = gym.make(env_id)
    return gym.wrappers.RecordEpisodeStatistics(env)

#Networks
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_space):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.register_buffer("scale", torch.tensor((act_space.high - act_space.low) / 2.0))
        self.register_buffer("bias", torch.tensor((act_space.high + act_space.low) / 2.0))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.tanh(self.log_std(x))
        log_std = -5 + 0.5 * (2 + 5) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale + self.bias
        log_prob = dist.log_prob(x_t) - torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), mean

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

#Training
def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

def train():
    best_avg_reward = -float("inf")  
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(obs_dim, act_dim, env.action_space).to(device)
    q1 = QNetwork(obs_dim, act_dim).to(device)
    q2 = QNetwork(obs_dim, act_dim).to(device)
    q1_target = QNetwork(obs_dim, act_dim).to(device)
    q2_target = QNetwork(obs_dim, act_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())


    actor_opt = optim.Adam(actor.parameters(), lr=policy_lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=q_lr)
    
    # Alpha auto-tuning setup
    target_entropy = -act_dim  # -dim(A)
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=q_lr)

    replay_buffer = PrioritizedReplayBuffer(buffer_size)
    writer = SummaryWriter()
    obs, _ = env.reset()

    for step in range(total_timesteps):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = actor.get_action(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = action.squeeze().detach().cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done and isinstance(info, dict) and "episode" in info:
            ep_info = info["episode"]
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], step)
            writer.add_scalar("charts/episodic_length", ep_info["l"], step)
            print(f"[Step {step}] Reward: {ep_info['r']:.2f}, Length: {ep_info['l']}")

        if "ep_info" in locals():
            if ep_info["r"] > best_avg_reward:
                best_avg_reward = ep_info["r"]
                save_dir = f"models_SAC/sac_{env_id}_best_finetune2"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(actor.state_dict(), os.path.join(save_dir, "actor.pth"))
                torch.save(q1.state_dict(), os.path.join(save_dir, "q1.pth"))
                torch.save(q2.state_dict(), os.path.join(save_dir, "q2.pth"))
                torch.save(log_alpha, os.path.join(save_dir, "log_alpha.pth"))
                print("new model saved")

        replay_buffer.push((obs, action, reward, next_obs, done), td_error=1.0)
        obs = next_obs if not done else env.reset()[0]

        if len(replay_buffer.buffer) > batch_size and step >= learning_starts:
            batch, indices, weights = replay_buffer.sample(batch_size)
            obs_b, act_b, rew_b, next_obs_b, done_b = map(np.array, zip(*batch))
            weights = weights.to(device).unsqueeze(1)

            obs_b = torch.tensor(obs_b, dtype=torch.float32, device=device)
            act_b = torch.tensor(act_b, dtype=torch.float32, device=device)
            rew_b = torch.tensor(rew_b, dtype=torch.float32, device=device).unsqueeze(-1)
            next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32, device=device)
            done_b = torch.tensor(done_b, dtype=torch.float32, device=device).unsqueeze(-1)

            with torch.no_grad():
                next_act, next_logp, _ = actor.get_action(next_obs_b)
                q1_target_val = q1_target(next_obs_b, next_act)
                q2_target_val = q2_target(next_obs_b, next_act)
                q_target = rew_b + gamma * (1 - done_b) * (torch.min(q1_target_val, q2_target_val) - log_alpha.exp() * next_logp)

            td_error1 = q1(obs_b, act_b) - q_target
            td_error2 = q2(obs_b, act_b) - q_target
            q1_loss = (td_error1.pow(2) * weights).mean()
            q2_loss = (td_error2.pow(2) * weights).mean()
            q_opt.zero_grad()
            (q1_loss + q2_loss).backward()
            q_opt.step()

            with torch.no_grad():
                td_errors = 0.5 * (td_error1 + td_error2).abs().squeeze().cpu().numpy()
            replay_buffer.update_priorities(indices, td_errors)

            if step % policy_freq == 0:
                new_act, logp, _ = actor.get_action(obs_b)
                min_q_val = torch.min(q1(obs_b, new_act), q2(obs_b, new_act))
                actor_loss = (log_alpha.exp() * logp - min_q_val).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # Alpha auto-tuning
                if auto_tune_alpha:
                    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    writer.add_scalar("Loss/alpha_loss", alpha_loss.item(), step)

            if step % target_freq == 0:
                soft_update(q1_target, q1, tau)
                soft_update(q2_target, q2, tau)

            writer.add_scalar("Loss/q1_loss", q1_loss.item(), step)
            writer.add_scalar("Loss/q2_loss", q2_loss.item(), step)
            writer.add_scalar("Loss/actor_loss", actor_loss.item(), step)
            writer.add_scalar("alpha", log_alpha.exp().item(), step)

     
    env.close()
    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    train()
