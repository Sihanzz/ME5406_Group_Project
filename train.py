import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from ddpg_modules import DDPGAgent
from replay_buffer import ReplayBuffer

# Training Hyperparameters
ENV_NAME = "Humanoid-v5"
EPISODES = 4000
MAX_STEPS = 2000
EXPL_NOISE_START = 0.4
EXPL_NOISE_END = 0.01
EXPL_NOISE_DECAY = 0.9985
BATCH_SIZE = 256
TRAIN_EVERY = 2
PRETRAIN_STEPS = 15000
REWARD_PLOT_PATH = "reward_curve.png"
MODEL_SAVE_PATH = "actor_final.pth"

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create environment without rendering
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize agent and replay buffer
agent = DDPGAgent(state_dim, action_dim, max_action, device=DEVICE)
replay_buffer = ReplayBuffer()
rewards_log = []
best_reward = -float("inf")
expl_noise = EXPL_NOISE_START

# Pre-fill replay buffer with random actions
print("Pre-filling replay buffer...")
while len(replay_buffer.storage) < PRETRAIN_STEPS:
    state, _ = env.reset()
    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, float(done or truncated))
        state = next_state
        if done or truncated:
            break

# Training loop
print("Start training...")
for episode in trange(EPISODES, desc="Training"):
    state, _ = env.reset()
    episode_reward = 0

    for t in range(MAX_STEPS):
        action = agent.select_action(np.array(state))
        noise = np.random.normal(0, expl_noise, size=action_dim)
        action = np.clip(action + noise, env.action_space.low, env.action_space.high)

        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, float(done or truncated))
        state = next_state
        episode_reward += reward

        if len(replay_buffer.storage) > BATCH_SIZE and t % TRAIN_EVERY == 0:
            agent.train(replay_buffer, BATCH_SIZE)

        if done or truncated:
            break

    rewards_log.append(episode_reward)
    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Noise: {expl_noise:.3f}")

    # Save best model
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(agent.actor.state_dict(), "actor_best.pth")

    # Save every 500 episodes
    if (episode + 1) % 500 == 0:
        torch.save(agent.actor.state_dict(), f"actor_ep{episode + 1}.pth")

    # Decay exploration noise
    expl_noise = max(EXPL_NOISE_END, expl_noise * EXPL_NOISE_DECAY)

env.close()

# Plot reward curve with smoothing
def moving_average(x, window=50):
    return np.convolve(x, np.ones(window) / window, mode='valid')

plt.plot(rewards_log, label="Raw")
plt.plot(moving_average(rewards_log), label="Smoothed")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"DDPG on {ENV_NAME}")
plt.grid()
plt.legend()
plt.savefig(REWARD_PLOT_PATH)
plt.show()

# Save final model
torch.save(agent.actor.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved as {MODEL_SAVE_PATH}")
