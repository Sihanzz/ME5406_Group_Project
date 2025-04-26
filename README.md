# Humanoid Robot Walking Using Reinforcement Learning in MuJoCo

This project implements a customized SAC agent for training a humanoid robot to walk using reinforcement learning.

## Project Structure
```plaintext
Behavior comparison/       # Pictures comparing baseline SAC, SAC with PER, and finetuned SAC models
runs/                      # TensorBoard log files for training visualization
SAC_best_model/            # Best saved SAC models
    ├── actor.pth          # Trained Actor network (policy)
    ├── q1.pth             # Trained Q-network 1 (critic)
    ├── q2.pth             # Trained Q-network 2 (critic)
    ├── log_alpha.pth      # Learned entropy coefficient log(α)
videos/                    # Evaluation videos of SAC walking results
SAC_best.py                # Finetuned SAC training script
SAC_evaluation.py          # Script to evaluate and visualize trained SAC model
requirements.txt           # Package dependencies
```



## Saved Model Components

The `SAC_best_model/` folder contains the following trained components:

- **`actor.pth`** — Trained Actor network for generating actions from observations.
- **`q1.pth`** — First Q-network estimating action values.
- **`q2.pth`** — Second Q-network to reduce overestimation bias.
- **`log_alpha.pth`** — Learned entropy coefficient controlling exploration.

> During evaluation (`SAC_evaluation.py`), only `actor.pth` is required.

## Dependencies
```plaintext
gymnasium==0.29.1
Gymnasium[mujoco]>=1.1.1
torch==2.1.0
numpy==1.26.4
tensorboard==2.16.2
tqdm>=4.67.1
```




## How to run:

### 1. Train the SAC agent
python SAC_best.py


### 2. Visualize training progress

Launch TensorBoard:

tensorboard --logdir runs/

### 3. Evaluate the trained model

python SAC_evaluation.py --model_path SAC_best_model/actor.pth




