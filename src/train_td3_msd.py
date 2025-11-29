import argparse
import os
import csv
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import controlgym
from controlgym.envs.linear_control import LinearControlEnv
from utils import plot_reward_curves, plot_cumulative_rewards

class RewardShapingWrapper(gym.Wrapper):
    """Wrapper to add reward shaping that penalizes overshoot and high control effort."""
    def __init__(self, env, alpha=0.01, beta=0.01):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
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
        state_penalty = self.alpha * np.sum(obs ** 2)
        action_array = np.array(action) if not isinstance(action, np.ndarray) else action
        action_penalty = self.beta * np.sum(action_array ** 2)
        
        # Shaped reward
        shaped_reward = original_reward - state_penalty - action_penalty
        self.current_episode_reward += shaped_reward
        
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
        for info in self.locals['infos']:
            if 'episode' in info:
                loss = 0.0  # Placeholder
                self.writer.writerow([self.num_timesteps, info['episode']['r'], loss])
                self.file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.close()

def main():
    parser = argparse.ArgumentParser(description='Train TD3 on Linear Control System')
    parser.add_argument('--env_id', type=str, default='toy', help='ControlGym LinearControl environment ID')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--learning_starts', type=int, default=100, help='Steps before learning starts')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--policy_delay', type=int, default=2, help='Policy update delay')
    parser.add_argument('--target_policy_noise', type=float, default=0.2, help='Target policy noise')
    parser.add_argument('--target_noise_clip', type=float, default=0.5, help='Target noise clip')
    parser.add_argument('--log_dir', type=str, default='results', help='Log directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--alpha', type=float, default=0.01, help='Reward shaping: state deviation penalty')
    parser.add_argument('--beta', type=float, default=0.01, help='Reward shaping: control effort penalty')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Plot directory')
    parser.add_argument('--policy_layers', type=str, default='256,256', help='Neural network hidden layers')
    parser.add_argument('--activation', type=str, default='relu', choices=['tanh', 'relu', 'elu'], help='Activation function')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Create environment
    try:
        env = LinearControlEnv(id=args.env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    except Exception as e:
        print(f"Error: Failed to create environment with ID '{args.env_id}': {e}")
        print("Available IDs: toy, pas, lah, rea, psm, he1-he6, je1-je2, umv")
        return

    # Wrap with reward shaping
    env = RewardShapingWrapper(env, alpha=args.alpha, beta=args.beta)
    env = Monitor(env, args.log_dir)

    # Action noise for exploration (TD3 uses Gaussian noise)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Custom policy network configuration
    net_arch = [int(x) for x in args.policy_layers.split(',')]
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn={
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'elu': torch.nn.ELU
        }[args.activation]
    )
    print(f"TD3 Policy network: layers={net_arch}, activation={args.activation}")

    # Initialize TD3
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        policy_delay=args.policy_delay,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=args.log_dir,
        name_prefix='td3_msd'
    )
    
    custom_callback = CustomLogCallback(args.log_dir)

    print(f"Starting TD3 training on {args.env_id} for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, custom_callback]
    )

    model.save(os.path.join(args.log_dir, "final_model_td3"))
    print("Training finished. Model saved.")
    
    # Plot reward curves
    print("Generating reward plots...")
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
