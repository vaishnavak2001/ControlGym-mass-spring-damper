# ControlGym Mass-Spring-Damper RL Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Reinforcement Learning framework for control systems using ControlGym. Features PPO and SAC algorithms, hybrid RL-PD control, reward shaping, and automated experiment tracking.

## 🚀 Features

- **Multiple RL Algorithms**: PPO (on-policy) and SAC (off-policy)
- **Hybrid Control**: Combine RL with classical PD controllers
- **Reward Shaping**: Penalize overshoot and excessive control effort
- **Experiment Logging**: Automated tracking with timestamped results
- **Hyperparameter Sweeps**: Automated grid search with result comparison
- **Comprehensive Visualization**: Reward curves, trajectories, and PDF reports

## 📋 Project Structure

```
ControlGym mass-spring-damper/
├── src/
│   ├── train_ppo_msd.py          # PPO training script
│   ├── train_sac_msd.py          # SAC training script
│   ├── eval_ppo_msd.py           # Evaluation script
│   ├── utils.py                  # Plotting utilities
│   ├── experiment_logger.py      # Experiment tracking
│   └── run_experiments.py        # Automated sweeps
├── results/                      # Training checkpoints and logs
├── plots/                        # Generated visualizations
├── experiments/                  # Experiment tracking directories
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "ControlGym mass-spring-damper"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies:**
   - `controlgym` - Control system environments
   - `stable-baselines3` - RL algorithms
   - `torch` - Deep learning backend
   - `matplotlib` - Visualization
   - `pandas` - Data analysis
   - `numpy` - Numerical computations

## 🎮 Quick Start

### Train PPO Agent

```bash
python src/train_ppo_msd.py --total_timesteps 10000
```

### Train SAC Agent

```bash
python src/train_sac_msd.py --total_timesteps 10000
```

### Evaluate Trained Model

```bash
python src/eval_ppo_msd.py --model_path results/final_model.zip
```

## 🎛️ Classical vs RL Control

### Classical Controllers

The project includes implementations of classical control strategies for benchmarking:

**PID Controller:**
- Proportional-Integral-Derivative control
- Control law: `u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt`
- Tunable gains: `kp`, `ki`, `kd`
- Simple, interpretable, widely used

**LQR Controller:**
- Linear Quadratic Regulator (optimal control)
- Control law: `u = -K*x` (K computed via Riccati equation)
- State and control cost matrices: `Q`, `R`
- Mathematically optimal for linear systems

### Benchmark Comparison

Compare all three approaches:

```bash
python controllers/benchmark.py --n_steps 500
```

**Sample Results (toy environment, 300 steps):**

| Controller | Total Reward | Mean |Position| |
|------------|--------------|----------------|
| **PID** | -13.88 | 0.1466 |
| **LQR** | -21.51 | 0.0546 |
| **PPO** | -26.30 | 0.2266 |

**Key Insights:**
- **PID**: Best total reward, good balance of performance and simplicity
- **LQR**: Lowest position error (0.0546), optimal for known linear systems
- **PPO**: Competitive performance, learns without system model

**When to Use Each:**
- **Classical (PID/LQR)**: Known system dynamics, fast deployment, interpretability
- **RL (PPO/SAC/TD3)**: Unknown dynamics, nonlinear systems, complex tasks

## 🔍 System Identification

Estimate system parameters (mass, damping, stiffness) from data using Least Squares:

```bash
python system_id/estimate_parameters.py
```

**Method:**
1. Collect trajectory data (acceleration, velocity, position, force)
2. Formulate linear regression: `m*a + c*v + k*x = u`
3. Solve for `[m, c, k]` using Least Squares

**Accuracy:**
- Typically achieves < 0.1% error on noise-free data
- Robust to moderate measurement noise



## 🔧 Available Environments

ControlGym provides several linear control environments:

| Environment | States | Actions | Description |
|-------------|--------|---------|-------------|
| `toy` | 1 | 1 | Simple linear system |
| `pas` | 3 | 1 | Passivation system |
| `lah` | 1 | 1 | Load-and-haul |
| `rea` | 1 | 1 | Reaction system |
| `psm` | 3 | 2 | Process system model |
| `he1-he6` | - | - | Heat exchangers |
| `je1-je2` | - | - | Jacket exchangers |
| `umv` | - | - | Unmanned vehicle |

## 📖 Usage Examples

### 1. Basic Training

```bash
# Train PPO on default 'toy' environment
python src/train_ppo_msd.py --total_timesteps 10000

# Train SAC on 'pas' environment
python src/train_sac_msd.py --env_id pas --total_timesteps 20000
```

### 2. Hyperparameter Tuning

```bash
# PPO with custom learning rate
python src/train_ppo_msd.py --learning_rate 0.001 --seed 42

# SAC with larger replay buffer
python src/train_sac_msd.py --buffer_size 200000 --batch_size 512
```

### 3. Reward Shaping

Penalize overshoot and control effort:

```bash
python src/train_ppo_msd.py --alpha 0.01 --beta 0.01
```

- `--alpha`: State deviation penalty (default: 0.01)
- `--beta`: Control effort penalty (default: 0.01)

### 4. Hybrid RL-PD Control

Combine RL with classical PD controller:

```bash
python src/train_ppo_msd.py --enable_hybrid \
    --lambda_pd 0.3 --kp 1.0 --kd 0.5
```

- `--lambda_pd`: PD weighting (0=RL-only, 1=PD-only)
- `--kp`: Proportional gain
- `--kd`: Derivative gain

**Hybrid Control Formula:**
```
u_hybrid = (1 - λ) * u_rl + λ * u_pd
```

### 5. Experiment Logging

Run single experiment with automated logging:

```bash
python src/run_experiments.py --single \
    --algorithm ppo --timesteps 5000 --lr 0.001
```

Generated files:
- `experiments/{name}_{timestamp}/config.json` - Hyperparameters
- `experiments/{name}_{timestamp}/episodes.csv` - Episode data
- `experiments/{name}_{timestamp}/summary.json` - Statistics
- `experiments/{name}_{timestamp}/plots.pdf` - Visualizations

### 6. Hyperparameter Sweeps

Automatically test multiple configurations:

```bash
# PPO sweep (3 learning rates × 2 seeds = 6 experiments)
python src/run_experiments.py --sweep --algorithm ppo

# SAC sweep (2 learning rates × 2 buffer sizes = 4 experiments)
python src/run_experiments.py --sweep --algorithm sac
```

Results saved to: `experiments/{algorithm}_sweep_results.csv`

## 📊 Algorithms Comparison

| Feature | PPO | SAC |
|---------|-----|-----|
| **Type** | On-policy | Off-policy |
| **Best For** | Stable training | Continuous control |
| **Sample Efficiency** | Moderate | High |
| **Memory Usage** | Low (~133 KB) | High (~3 MB) |
| **Hyperparameters** | learning_rate, n_steps | learning_rate, buffer_size, tau |

### When to Use Each

**Use PPO when:**
- Limited computational resources
- Want stable, reliable convergence
- Sample efficiency less critical

**Use SAC when:**
- Have replay buffer storage (100K+ transitions)
- Maximum sample efficiency needed
- Task requires strong exploration

## 🎯 Reward Shaping Details

The reward shaping wrapper modifies rewards to encourage better control:

```python
shaped_reward = original_reward - α * |state|² - β * |action|²
```

**Benefits:**
- Reduces overshoot by penalizing large state deviations
- Encourages smooth control by penalizing aggressive actions
- Improves learning convergence

**Example Results:**
- Without shaping: ep_rew = -3454
- With shaping (α=0.01, β=0.01): ep_rew = -1768 (48% improvement)

## 🤖 Hybrid RL-PD Controller

Combines RL policy with classical PD controller for safer control:

**PD Control Law:**
```python
u_pd = -Kp * position - Kd * velocity
```

**Hybrid Action:**
```python
u_hybrid = (1 - λ_pd) * u_rl + λ_pd * u_pd
```

**Benefits:**
- Prevents extreme RL actions
- Bootstraps learning with classical control
- Provides safety guardrails

**Recommended λ_pd values:**
- Start: 0.5 (50% PD for safety)
- During training: 0.3 (30% PD)
- After convergence: 0.0 (pure RL)

## 📈 Experiment Tracking

The `ExperimentLogger` provides comprehensive tracking:

**Logged Information:**
- Hyperparameters (JSON)
- Episode rewards and metrics (CSV)
- Summary statistics (JSON)
- Visualization plots (PDF)

**Example Experiment:**
```bash
python src/run_experiments.py --single --algorithm ppo --timesteps 2000
```

**Generated Directory:**
```
experiments/ppo_toy_single_20251129_122752/
├── config.json          # All hyperparameters
├── episodes.csv         # Per-episode data
├── summary.json         # Statistics (mean, std, etc.)
├── plots.pdf            # 3-page visualization
└── ppo_model.zip        # Trained model
```

**PDF Plots Include:**
1. Episode rewards with moving average
2. Cumulative reward over time
3. Reward distribution histogram

## 🔬 Running Hyperparameter Sweeps

Automatically search for best hyperparameters:

```bash
python src/run_experiments.py --sweep --algorithm ppo
```

**PPO Grid:**
- Learning rates: [1e-4, 3e-4, 1e-3]
- Seeds: [42, 123]
- Total: 6 experiments

**SAC Grid:**
- Learning rates: [1e-4, 3e-4]
- Buffer sizes: [50K, 100K]
- Total: 4 experiments

**Output:**
- Individual experiment directories
- Aggregated results: `experiments/{algorithm}_sweep_results.csv`
- Best configuration automatically identified

## 📝 Code Documentation

All source files include comprehensive docstrings:

```python
class ExperimentLogger:
    """Logger for tracking experiments with hyperparameters, metrics, and results.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for all experiments
    """
```

## 🗂️ Git Ignore

The `.gitignore` excludes:
- `results/` - Training checkpoints
- `plots/` - Generated visualizations
- `experiments/` - Experiment tracking
- `__pycache__/` - Python cache
- `*.pyc` - Compiled bytecode

## 🐛 Troubleshooting

**ImportError: No module named 'controlgym'**
```bash
pip install controlgym
```

**Environment not found**
- Ensure environment ID is in available list (toy, pas, lah, etc.)
- Use `--env_id toy` for testing

**CUDA out of memory (SAC)**
- Reduce `--buffer_size` (default: 100000)
- Use `--batch_size 128` instead of 256

## 📚 References

- [ControlGym](https://github.com/google-deepmind/control-gym) - Control environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Soft Actor-Critic

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Google DeepMind for ControlGym
- Stable-Baselines3 team for RL implementations
- Contributors and community members

---

**Happy Controlling! 🎛️**
