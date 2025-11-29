import argparse
import os
import csv
import gymnasium as gym
import numpy as np
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
    env = RewardShapingWrapper(env, alpha=args.alpha, beta=args.beta)
    env = Monitor(env, args.log_dir)
    env = DummyVecEnv([lambda: env])

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
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
