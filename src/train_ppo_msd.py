import argparse
import os
import csv
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import controlgym
from controlgym.envs.linear_control import LinearControlEnv
from utils import plot_reward_curves, plot_cumulative_rewards

class RewardShapingWrapper(gym.Wrapper):
    """Wrapper to add reward shaping that penalizes overshoot and high control effort."""
    def __init__(self, env, alpha=0.01, beta=0.01):
        super().__init__(env)
        self.alpha = alpha  # Penalty for state deviation
        self.beta = beta    # Penalty for control effort
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
    def reset(self, **kwargs):
        if self.current_episode_reward != 0.0:
            self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Original reward
        original_reward = reward
        
        # Add penalties
        # Penalize large state deviations (overshoot)
        state_penalty = self.alpha * np.sum(obs ** 2)
        
        # Penalize high control effort
        action_array = np.array(action) if not isinstance(action, np.ndarray) else action
        action_penalty = self.beta * np.sum(action_array ** 2)
        
        # Shaped reward
        shaped_reward = original_reward - state_penalty - action_penalty
        
        self.current_episode_reward += shaped_reward
        
        # Store original reward in info for logging
        if 'original_reward' not in info:
            info['original_reward'] = original_reward
        info['shaped_reward'] = shaped_reward
        
        return obs, shaped_reward, done, truncated, info

class OptimizedRewardWrapper(gym.Wrapper):
    """Advanced reward shaping for faster convergence.
    
    Features:
    - Distance-based exponential reward
    - Velocity-aware penalty scaling
    - Settling time bonus for staying near target
    - Progressive difficulty reduction
    """
    def __init__(self, env, target_threshold=0.1, settling_steps=50):
        super().__init__(env)
        self.target_threshold = target_threshold
        self.settling_steps = settling_steps
        self.steps_at_target = 0
        self.total_steps = 0
        
    def reset(self, **kwargs):
        self.steps_at_target = 0
        self.total_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        # Extract state information
        obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
        
        # Distance to target (assuming target is zero)
        if obs_array.ndim == 0 or len(obs_array) == 1:
            position = obs_array[0] if len(obs_array) > 0 else obs_array
            velocity = 0.0
        else:
            position = obs_array[0]
            velocity = obs_array[1] if len(obs_array) > 1 else 0.0
        
        distance = abs(position)
        
        # Distance-based exponential reward (closer to target = higher reward)
        distance_reward = np.exp(-5.0 * distance)
        
        # Velocity-aware penalty (penalize high velocity when far from target)
        velocity_penalty = abs(velocity) * distance
        
        # Settling time bonus (reward for staying near target)
        if distance < self.target_threshold:
            self.steps_at_target += 1
            settling_bonus = min(self.steps_at_target / self.settling_steps, 1.0)
        else:
            self.steps_at_target = 0
            settling_bonus = 0.0
        
        # Progressive difficulty (reduce shaping as agent improves)
        progress_factor = min(self.total_steps / 1000, 1.0)  # Full shaping for first 1000 steps
        
        # Combined optimized reward
        optimized_reward = (
            reward +  # Original reward
            distance_reward * (1.0 - 0.5 * progress_factor) +  # Distance shaping (reduces over time)
            settling_bonus * 2.0 +  # Settling bonus
            -0.1 * velocity_penalty  # Velocity penalty
        )
        
        # Store metrics
        info['distance'] = float(distance)
        info['settling_bonus'] = float(settling_bonus)
        info['optimized_reward'] = float(optimized_reward)
        
        return obs, optimized_reward, done, truncated, info

class PDController:
    """Classical PD controller for linear systems."""
    def __init__(self, kp=1.0, kd=0.5):
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
    
    def compute_action(self, state):
        """Compute PD control action.
        Assumes state is [position, velocity] or just [position].
        """
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        
        # Handle 1D or 2D states
        if state.ndim == 0 or len(state) == 1:
            # Only position available, assume velocity is 0
            position = state[0] if len(state) > 0 else state
            velocity = 0.0
        else:
            position = state[0]
            velocity = state[1] if len(state) > 1 else 0.0
        
        # PD control: u = -Kp * position - Kd * velocity
        action = -self.kp * position - self.kd * velocity
        return action

class HybridControllerWrapper(gym.Wrapper):
    """Wrapper that combines RL action with PD controller action."""
    def __init__(self, env, pd_controller, lambda_pd=0.3):
        super().__init__(env)
        self.pd_controller = pd_controller
        self.lambda_pd = lambda_pd  # Weight for PD action (0=RL-only, 1=PD-only)
        self.rl_actions = []
        self.pd_actions = []
        self.hybrid_actions = []
    
    def step(self, rl_action):
        # Compute PD action based on current state
        # We need to get the state before stepping
        # For this, we'll use the last observation stored
        if hasattr(self, '_last_obs'):
            pd_action = self.pd_controller.compute_action(self._last_obs)
        else:
            pd_action = 0.0
        
        # Combine RL and PD actions
        rl_action_val = np.array(rl_action) if not isinstance(rl_action, np.ndarray) else rl_action
        hybrid_action = (1 - self.lambda_pd) * rl_action_val + self.lambda_pd * pd_action
        
        # Log actions (extract scalar if array)
        self.rl_actions.append(float(rl_action_val.item() if hasattr(rl_action_val, 'item') else rl_action_val))
        self.pd_actions.append(float(pd_action))
        self.hybrid_actions.append(float(hybrid_action.item() if hasattr(hybrid_action, 'item') else hybrid_action))
        
        # Execute hybrid action
        obs, reward, done, truncated, info = self.env.step(hybrid_action)
        
        # Store observation for next step
        self._last_obs = obs
        
        # Store action info
        info['rl_action'] = float(rl_action_val.item() if hasattr(rl_action_val, 'item') else rl_action_val)
        info['pd_action'] = float(pd_action)
        info['hybrid_action'] = float(hybrid_action.item() if hasattr(hybrid_action, 'item') else hybrid_action)
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            self._last_obs = obs[0]
            return obs
        else:
            self._last_obs = obs
            return obs

class CustomLogCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomLogCallback, self).__init__(verbose)
        self.log_path = os.path.join(log_dir, 'training_log.csv')
        self.file = None
        self.writer = None

    def _on_training_start(self) -> None:
        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['step', 'episode_reward', 'loss'])

    def _on_step(self) -> bool:
        # Check for episode end
        for info in self.locals['infos']:
            if 'episode' in info:
                # Try to get loss from logger
                loss = 0.0 # Placeholder as PPO loss is not per-step
                if hasattr(self.model, 'logger'):
                    # This is tricky, SB3 logger doesn't expose values easily
                    pass
                
                self.writer.writerow([self.num_timesteps, info['episode']['r'], loss])
                self.file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.close()

def main():
    parser = argparse.ArgumentParser(description='Train PPO on Mass-Spring-Damper')
    parser.add_argument('--env_id', type=str, default='toy', help='ControlGym LinearControl environment ID (e.g., toy, pas, lah)')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='results', help='Log directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--alpha', type=float, default=0.01, help='Reward shaping: state deviation penalty')
    parser.add_argument('--beta', type=float, default=0.01, help='Reward shaping: control effort penalty')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Plot directory')
    parser.add_argument('--enable_hybrid', action='store_true', help='Enable hybrid RL-PD controller')
    parser.add_argument('--kp', type=float, default=1.0, help='PD controller proportional gain')
    parser.add_argument('--kd', type=float, default=0.5, help='PD controller derivative gain')
    parser.add_argument('--lambda_pd', type=float, default=0.3, help='PD weighting (0=RL-only, 1=PD-only)')
    parser.add_argument('--use_optimized_reward', action='store_true', help='Use optimized reward shaping for faster convergence')
    parser.add_argument('--policy_layers', type=str, default='64,64', help='Neural network hidden layers (comma-separated, e.g., "64,64" or "128,128,64")')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'elu'], help='Activation function')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Create environment using controlgym's LinearControlEnv
    try:
        env = LinearControlEnv(id=args.env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    except Exception as e:
        print(f"Error: Failed to create environment with ID '{args.env_id}': {e}")
        print("Available IDs: toy, pas, lah, rea, psm, he1-he6, je1-je2, umv")
        return

    # Wrap with reward shaping
    if args.use_optimized_reward:
        env = OptimizedRewardWrapper(env, target_threshold=0.1, settling_steps=50)
        print("Using optimized reward shaping for faster convergence")
    else:
        env = RewardShapingWrapper(env, alpha=args.alpha, beta=args.beta)
    
    # Optionally wrap with hybrid controller
    if args.enable_hybrid:
        pd_controller = PDController(kp=args.kp, kd=args.kd)
        env = HybridControllerWrapper(env, pd_controller, lambda_pd=args.lambda_pd)
        print(f"Hybrid controller enabled: lambda_pd={args.lambda_pd}, Kp={args.kp}, Kd={args.kd}")
    
    env = Monitor(env, args.log_dir)
    env = DummyVecEnv([lambda: env])

    # Custom policy network configuration
    net_arch = [int(x) for x in args.policy_layers.split(',')]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn={
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU
        }[args.activation]
    )
    print(f"Policy network: layers={net_arch}, activation={args.activation}")

    # Initialize PPO with custom network
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=args.log_dir,
        name_prefix='ppo_msd'
    )
    
    custom_callback = CustomLogCallback(args.log_dir)

    print(f"Starting training on {args.env_id} for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, custom_callback]
    )

    model.save(os.path.join(args.log_dir, "final_model"))
    print("Training finished. Model saved.")
    
    # Plot reward curves
    print("Generating reward plots...")
    # Get episode rewards from the wrapper
    # Note: We can't access the wrapper directly after DummyVecEnv
    # So we'll parse the CSV log instead
    try:
        import pandas as pd
        log_file = os.path.join(args.log_dir, 'training_log.csv')
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            rewards = df['episode_reward'].values
            
            os.makedirs(args.plot_dir, exist_ok=True)
            plot_reward_curves(rewards, args.plot_dir)
            plot_cumulative_rewards(rewards, args.plot_dir)
            print(f"Reward plots saved to {args.plot_dir}/")
        else:
            print("No training log found, skipping reward plots.")
    except Exception as e:
        print(f"Warning: Could not generate reward plots: {e}")

if __name__ == "__main__":
    main()
