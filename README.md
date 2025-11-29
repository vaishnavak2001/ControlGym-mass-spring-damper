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

## Using SAC (Soft Actor-Critic)

SAC is an off-policy algorithm particularly effective for continuous control tasks.

### Train with SAC

```bash
python src/train_sac_msd.py --total_timesteps 10000
```

### SAC with Custom Hyperparameters

```bash
python src/train_sac_msd.py --env_id toy --total_timesteps 10000 \
    --learning_rate 0.0003 --buffer_size 100000 --batch_size 256
```

### Evaluate SAC Model

```bash
python src/eval_ppo_msd.py --model_path results/final_model_sac.zip --env_id toy
```

## Algorithms Comparison

| Algorithm | Type | Best For | Key Hyperparameters |
|-----------|------|----------|--------------------|
| **PPO** | On-policy | Sample efficiency, stable training | learning_rate, n_steps, batch_size |
| **SAC** | Off-policy | Continuous control, exploration | learning_rate, buffer_size, tau |

### When to Use Each Algorithm

- **Use PPO** when:
  - You have limited computational resources
  - You want stable, reliable convergence
  - Sample efficiency is less critical

- **Use SAC** when:
  - You have access to significant replay buffer storage
  - Maximum sample efficiency is crucial
  - The task requires strong exploration

## Reward Shaping

Both algorithms support reward shaping to improve learning:

```bash
# Adjust penalties for state deviation and control effort
python src/train_ppo_msd.py --alpha 0.01 --beta 0.01
python src/train_sac_msd.py --alpha 0.01 --beta 0.01
```

- `--alpha`: Penalty for state deviation (overshoot)
- `--beta`: Penalty for control effort (aggressive actions)
