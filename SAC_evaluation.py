import gymnasium as gym
import torch
import numpy as np
import time
import argparse
from sac_customized_finetuned import Actor  # load model


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained actor model")
args = parser.parse_args()


env_id = "Humanoid-v5"
model_path = args.model_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_test_episodes = 100

# load env with rendering
env = gym.make(env_id, render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_space = env.action_space




# load actor model
model = Actor(obs_dim, act_dim, act_space).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()

#evaluation and visualization
rewards_per_episode = []

for ep in range(num_test_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model.get_action(obs_tensor)[0].squeeze().cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.01)

    rewards_per_episode.append(total_reward)
    print(f"[Episode {ep + 1}] Total Reward: {total_reward:.2f}")

env.close()

