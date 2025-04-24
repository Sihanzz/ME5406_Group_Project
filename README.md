# PPO and TD3 Implementation for Humanoid Control

This repository contains an implementation of Proximal Policy Optimization (PPO) for training a humanoid agent in the MuJoCo environment. The implementation features parallel environment training, generalized advantage estimation (GAE), and a Beta distribution-based policy network.

This repository also implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for controlling the Humanoid-v5 environment in Gymnasium. The implementation includes training and testing scripts with comprehensive visualization capabilities.

## Project Structure

- `ppo_ours.py`: Main training script containing the PPO implementation
- `td3_ours.py` Main TD3 implementation file
- `runs/`: Directory containing current tensorboard logs
- `data_best/`: Directory for storing the best model checkpoints for PPO
- `humanoid/`: Directory containing our pretrained tensorboard logs
─ `models/` Directory for saved models for TD3

## Dependencies

The project requires the following dependencies with their minimum versions:

- Python >= 3.10
- PyTorch >= 2.6.0
- Gymnasium[mujoco] >= 1.1.1
- NumPy >= 1.24.0
- TensorBoard >= 2.19.0
- tqdm >= 4.67.1

## Installation

1. Install UV for python package management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Clone Repo:
```
git clone https://github.com/Sihanzz/rl-repo.git
```

## Swith to different Branch

### Swith to DDPG and D4PG branch
```
git checkout D4PG-xcn
```
Check D4PG-xcn branch's README for instructions of training DDPG and D4PG

### Switch to SAC branch
```
git checkout ???
```
Check ??? branch's README for instructions of training SAC

## PPO Usage
### Switch to main branch
```
git checkout main
```

### PPO Training

Install dependencies:
```bash
uv sync
```

To start training PPO the agent:

```bash
uv run ppo_ours.py train
```

Or just test PPO
```bash
uv run ppo_ours.py
```

The training script will:
- Initialize parallel environments for training
- Use a Beta distribution-based policy network
- Implement PPO with GAE for advantage estimation
- Save the best model based on episode rewards
- Log training metrics to TensorBoard

### PPO Monitoring Training

To monitor training progress:

```bash
uv run tensorboard --logdir runs
```

This will start a TensorBoard server where you can visualize:
- Episode returns
- Policy and value losses
- Entropy
- Learning rate

### PPO Model Checkpoints

The best model is automatically saved in the `data_best/` directory. You can load a pretrained model by setting `load_pretrained = True` in the training script.

### PPO Implementation Details

The implementation includes:
- Parallel environment training for efficient sampling
- Generalized Advantage Estimation (GAE)
- Beta distribution-based policy network
- Orthogonal initialization of network weights
- Cosine annealing learning rate scheduler
- Normalized observations and rewards
- Gradient clipping

### PPO Hyperparameters

Key hyperparameters (defined in `ppo_ours.py`):
- `NUM_ENVS`: Number of parallel environments (default: 4)
- `SAMPLE_STEPS`: Steps to sample per iteration (default: 2048)
- `TOTAL_STEPS`: Total training steps (default: 4,000,000)
- `MINI_BATCH_SIZE`: Mini batch size for training (default: 256)
- `EPOCHES`: Number of epochs per iteration (default: 10)
- `GAMMA`: Discount factor (default: 0.99)
- `GAE_LAMBDA`: GAE lambda parameter (default: 0.95)
- `CLIP_EPS`: PPO clipping epsilon (default: 0.2)


## TD3 Usage
### TD3 Training
```bash
uv run td3_ours.py --train
```

Key training parameters:
- `--env`: Environment name (default: Humanoid-v5)
- `--max_steps`: Maximum training steps (default: 1,000,000)
- `--batch_size`: Batch size (default: 256)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.005)

### TD3 Testing/Visualization
```bash
uv run td3_ours.py
```

The testing script will:
1. Load the best saved model
2. Run forever evaluation episodes until you manually stop
3. Display performance metrics
4. Show real-time visualization

### TD3 Model Architecture
- Actor Network: 256-256 hidden layers with ReLU activation
- Critic Networks: Two independent 256-256 networks
- Target networks updated via Polyak averaging (τ = 0.005)

### TD3 Key Hyperparameters
- Policy update delay (d): 2
- Target policy smoothing noise (σ): 0.2
- Noise clip range (c): 0.5
- Initial random steps: 25,000
- Replay buffer size: 1,000,000

### TD3 Monitoring
Training progress can be monitored using TensorBoard:
```bash
uv run tensorboard --logdir runs/TD3_training
```

### TD3 Implementation Details
The implementation follows the original TD3 paper with several optimizations:
- Efficient network architecture
- Comprehensive logging
- Robust model saving/loading
- Real-time visualization

## References
- Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods"
- Original TD3 paper: https://arxiv.org/abs/1802.09477

## License
This project is open source and available under the MIT License.
