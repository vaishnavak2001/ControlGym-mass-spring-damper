# ControlGym Linear Control RL Project

This project implements a Reinforcement Learning (RL) agent using PPO to control linear dynamical systems using ControlGym.

## Available Environments

ControlGym provides several linear control environments:
- `toy`: Simple 1-state, 1-action system
- `pas`: 3-state, 1-action system  
- `lah`: 1-state, 1-action system
- `rea`: 1-state, 1-action system
- `psm`: 3-state, 2-action system
- `he1-he6`, `je1-je2`, `umv`: Various higher-dimensional systems

## Quickstart

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Train the agent (default uses 'toy' environment):
    ```bash
    python src/train_ppo_msd.py --total_timesteps 10000
    ```

3.  Evaluate the agent:
    ```bash
    python src/eval_ppo_msd.py --model_path results/final_model.zip
    ```

## Advanced Usage

Train with different environment:
```bash
python src/train_ppo_msd.py --env_id pas --total_timesteps 50000
```

Customize hyperparameters:
```bash
python src/train_ppo_msd.py --learning_rate 0.0003 --seed 123
```
