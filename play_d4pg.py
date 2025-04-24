import gymnasium as gym
import torch
import numpy as np
import time
from d4pg_modules import D4PGAgent

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create visual environment
env = gym.make("Humanoid-v5", render_mode="human")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

#Load agent and weights
agent = D4PGAgent(state_dim, action_dim, max_action, device=DEVICE)
checkpoint = torch.load("checkpoints/actor_best.pth", map_location=DEVICE)
agent.actor.load_state_dict(checkpoint)
agent.actor.eval()   # set to evaluation mode

#Run episodes
episodes = 200
for ep in range(1, episodes + 1):
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.select_action(np.array(state))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        time.sleep(0.001)  # control playback speed

        if terminated or truncated:
            print(f"Episode {ep} done. Total reward: {total_reward:.2f}, steps: {steps}")
            break

env.close()
