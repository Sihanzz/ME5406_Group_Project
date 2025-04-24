# PPO Implementation for Humanoid Control

This repository contains an implementation of Proximal Policy Optimization (PPO) for training a humanoid agent in the MuJoCo environment. The implementation features parallel environment training, generalized advantage estimation (GAE), and a Beta distribution-based policy network.

## Project Structure

- `ppo_ours.py`: Main training script containing the PPO implementation
- `runs/`: Directory containing current tensorboard logs
- `data_best/`: Directory for storing the best model checkpoints
- `humanoid/`: Directory containing our existing tensorboard training logs

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

2. Install dependencies:
```bash
uv sync
```


## Usage

### Training

To start training the agent:

```bash
uv run ppo_ours.py train
```

Or just test
```bash
uv run ppo_ours.py
```

The training script will:
- Initialize parallel environments for training
- Use a Beta distribution-based policy network
- Implement PPO with GAE for advantage estimation
- Save the best model based on episode rewards
- Log training metrics to TensorBoard

### Monitoring Training

To monitor training progress:

```bash
uv run tensorboard --logdir runs
```

This will start a TensorBoard server where you can visualize:
- Episode returns
- Policy and value losses
- Entropy
- Learning rate

### Model Checkpoints

The best model is automatically saved in the `data_best/` directory. You can load a pretrained model by setting `load_pretrained = True` in the training script.

## Implementation Details

The implementation includes:
- Parallel environment training for efficient sampling
- Generalized Advantage Estimation (GAE)
- Beta distribution-based policy network
- Orthogonal initialization of network weights
- Cosine annealing learning rate scheduler
- Normalized observations and rewards
- Gradient clipping

## Hyperparameters

Key hyperparameters (defined in `ppo_ours.py`):
- `NUM_ENVS`: Number of parallel environments (default: 4)
- `SAMPLE_STEPS`: Steps to sample per iteration (default: 2048)
- `TOTAL_STEPS`: Total training steps (default: 4,000,000)
- `MINI_BATCH_SIZE`: Mini batch size for training (default: 256)
- `EPOCHES`: Number of epochs per iteration (default: 10)
- `GAMMA`: Discount factor (default: 0.99)
- `GAE_LAMBDA`: GAE lambda parameter (default: 0.95)
- `CLIP_EPS`: PPO clipping epsilon (default: 0.2)