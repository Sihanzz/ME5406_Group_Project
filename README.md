# 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# 2. Clone Repo

```bash
git clone https://github.com/sihanzz/rl-repo.git
cd rl-repo
```


# 3. Update env

```bash
uv sync
```

# 4. Run example

```bash
uv run ppo_new.py
```

# 5. Visualize training progress
In another terminal:
```bash
uv run tensorboard --logdir runs
```