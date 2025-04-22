# 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Clone Repo

```bash
git clone https://github.com/sihanzz/rl-repo.git
cd rl-repo
```


# Update env

```bash
uv sync
```

# Run example

```bash
uv run ppo_new.py
```

# Visualize training progress
In another terminal:
```bash
uv run tensorboard --logdir runs
```