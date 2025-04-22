import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ly1 = nn.Linear(348, 512)
        self.ly2 = nn.Linear(512, 17)
        self.log_std = nn.Parameter(torch.zeros(1, 17))


    def forward(self, x, action=None): 
        x1 = F.tanh(self.ly1(x))
        mean = F.tanh(self.ly2(x1))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None: action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.ly1 = nn.Linear(348, 512)
        self.ly2 = nn.Linear(512, 128)
        self.value = nn.Linear(128, 1)


    def forward(self, x):
        x1 = F.tanh(self.ly1(x))
        x2 = F.tanh(self.ly2(x1))
        value = self.value(x2)
        return value   


def rollout(actor, critic, env, gamma=0.99, lamb=0.95):
    result = {
        "states": [],
        "actions": [],
        "rewards": [],
        "log_probs": [],
        "values": [],
        "advantages": [],
        "returns": []
    }

    state, _ = env.reset()
    count = 0
    while True:
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = actor(state)
            value = critic(state)
        action = action.squeeze(0)
        state_next, reward, done, truncation, info = env.step(action.numpy())
        reward = torch.tensor(reward, dtype = torch.float32)
        result["states"].append(state.squeeze(0))
        result["actions"].append(action)
        result["rewards"].append(reward)
        result["log_probs"].append(log_prob)
        result["values"].append(value.squeeze(0))
        count += 1
        state = state_next
        if done or truncation:
            episode_reward = info['episode']['r']
            break
    adv = 0

    if done:
        next_v = 0
    else:
        with torch.no_grad():
            next_v = torch.tensor(state_next, dtype = torch.float32).unsqueeze(0)
            next_v = critic(next_v).squeeze(0)
    
    for idx in reversed(range(count)):
        delta = result['rewards'][idx] + gamma * next_v - result['values'][idx]
        adv = delta + gamma * lamb * adv
        result['advantages'].append(adv)
        result['returns'].append(adv + result['values'][idx])
        next_v = result['values'][idx]
    
    result['advantages'] = reversed(result['advantages'])
    result['returns'] = reversed(result['returns'])
    return result, episode_reward, count



def ppo_loss(actor, critic, states, actions, log_probs, returns, advantages, epsilon = 0.2):
    _, new_log_probs = actor(states, actions)
    values = critic(states).squeeze(-1)
    ratio = torch.exp(new_log_probs - log_probs)
    clip_ratio =  torch.clip(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = - torch.min(ratio * advantages, clip_ratio * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    total_loss = policy_loss + value_loss
    return total_loss

def make_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda x: np.clip(x, -10, 10), env.observation_space)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda x: np.clip(x, -10, 10))
    return env



def main():
    env = make_env('Humanoid-v5')
    actor = Actor()
    critic = Critic()

    optimizer_actor = torch.optim.Adam(actor.parameters())
    optimizer_critic = torch.optim.Adam(critic.parameters())

    total_steps = 1e6
    total_count = 0
    while True:
        result, episode_reward, count = rollout(actor, critic, env)
        total_loss = ppo_loss(actor, critic, result['states'], result['actions'], result['log_probs'], result['returns'], result['advantages'])
        break
        


if __name__ == "__main__":
    main()
