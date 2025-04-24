import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
from d4pg_modules import D4PGAgent
from replay_buffer_d4pg import ReplayBuffer

#Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

#Hyperparameters
ENV_NAME       = "Humanoid-v5"
EPISODES       = 4000
MAX_STEPS      = 2000
EXPL_NOISE     = [0.3, 0.01, 0.999]
BATCH_SIZE     = 256
TRAIN_FREQ     = 2
WARMUP_STEPS   = 15000
REWARD_PLOT    = "reward_curve_d4pg.png"
FINAL_MODEL    = "actor_final_d4pg.pth"
CHECKPOINT_DIR = "checkpoints/"

#Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Environment and Agent Setup
env = gym.make(ENV_NAME)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = D4PGAgent(state_dim, action_dim, max_action, device=DEVICE)
replay_buffer = ReplayBuffer(n_step=5, gamma=0.99)

# Warm-up Replay Buffer
print("Warm-up: filling replay buffer...")
while len(replay_buffer) < WARMUP_STEPS:
    state, _ = env.reset(seed=SEED)
    for _ in range(MAX_STEPS):
        random_action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(random_action)
        done = bool(terminated or truncated)
        replay_buffer.add(state, random_action, reward, next_state, done)
        state = next_state
        if done:
            break

#Training Loop
reward_history = []
best_reward = -np.inf
current_noise = EXPL_NOISE[0]

print("Starting training...")
for episode in trange(1, EPISODES + 1, desc="Episode"):
    state, _ = env.reset()
    total_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        action = agent.select_action(np.array(state))
        noise = np.random.normal(0, current_noise, size=action_dim)
        noisy_action = np.clip(action + noise, env.action_space.low, env.action_space.high)

        next_state, reward, terminated, truncated, _ = env.step(noisy_action)
        done = bool(terminated or truncated)
        replay_buffer.add(state, noisy_action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(replay_buffer) >= BATCH_SIZE and step % TRAIN_FREQ == 0:
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            break

    reward_history.append(total_reward)
    current_noise = max(EXPL_NOISE[1], current_noise * EXPL_NOISE[2])

    #Save best model and periodic checkpoints
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.actor.state_dict(), CHECKPOINT_DIR + "actor_best.pth")
    if episode % 500 == 0:
        torch.save(agent.actor.state_dict(), CHECKPOINT_DIR + f"actor_ep{episode}.pth")

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Noise: {current_noise:.3f}")

env.close()

#Plotting Results
def moving_average(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode='valid')

plt.figure()
plt.plot(reward_history, label="Raw Rewards")
plt.plot(moving_average(reward_history), label="Smoothed Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title(f"D4PG Performance on {ENV_NAME}")
plt.legend()
plt.grid(True)
plt.savefig(REWARD_PLOT)
plt.show()

# Save Final Model
torch.save(agent.actor.state_dict(), FINAL_MODEL)
print(f"Final model saved as {FINAL_MODEL}")
