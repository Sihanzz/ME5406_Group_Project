# Project Title: Deep Reinforcement Learning for Humanoid Locomotion

This repository contains the implementation of two deep reinforcement learning algorithms, DDPG and D4PG, applied to the MuJoCo Humanoid-v5 environment.

## Code Structure

- `ddpg_modules.py`: Implementation of the DDPG actor and critic networks and training logic.
- `d4pg_modules.py`: Implementation of the D4PG architecture with distributional critic and N-step returns.
- `train_ddpg.py`: Script for training the DDPG agent.
- `train_d4pg.py`: Script for training the D4PG agent.
- `play_ddpg.py`: Visualization and testing script for DDPG.
- `play_d4pg.py`: Visualization and testing script for D4PG.
- `replay_buffer.py`: Replay buffer used by DDPG.
- `replay_buffer_d4pg.py`: N-step replay buffer used by D4PG.


## Environment Setup

- Python 3.10.0
- PyTorch
- Gymnasium with MuJoCo support
- NumPy
- Matplotlib


This will run multiple episodes in visual mode and print the total reward per episode.

## Notes

- DDPG uses 1-step return and a simple MSE loss.
- D4PG incorporates N-step return and a distributional critic with categorical projection.
- All hyperparameters are defined at the top of the training scripts and can be easily adjusted.

## Author

Xu Chunnan  
ME5406 — Deep Learning for Robotics  
