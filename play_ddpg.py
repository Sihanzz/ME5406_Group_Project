import gymnasium as gym
import torch
import numpy as np
import time
from ddpg_modules import DDPGAgent

# Create visual environment
env = gym.make("Humanoid-v5", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Load the agent and the trained model parameters
agent = DDPGAgent(state_dim, action_dim, max_action)
agent.actor.load_state_dict(torch.load("actor_best.pth"))

# Play several episodes
episodes = 200
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = agent.select_action(np.array(state))
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        time.sleep(0.001)  # Control playback speed

        if done or truncated:
            print(f"Episode {ep + 1} done. Total reward: {total_reward:.2f}, steps: {steps}")
            break

env.close()
