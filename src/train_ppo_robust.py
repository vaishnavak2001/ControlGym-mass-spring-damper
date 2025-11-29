import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from controlgym.envs.linear_control import LinearControlEnv
from robust_env import RobustControlWrapper
from train_ppo_msd import RewardShapingWrapper, OptimizedRewardWrapper

def main():
    parser = argparse.ArgumentParser(description='Train Robust PPO Agent')
    parser.add_argument('--env_id', type=str, default='toy', help='Environment ID')
    parser.add_argument('--total_timesteps', type=int, default=20000, help='Total timesteps')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Measurement noise std dev')
    parser.add_argument('--dist_prob', type=float, default=0.02, help='Disturbance probability')
    parser.add_argument('--dist_mag', type=float, default=2.0, help='Disturbance magnitude')
    parser.add_argument('--log_dir', type=str, default='results_robust', help='Log directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_optimized_reward', action='store_true', help='Use optimized reward')
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create base environment
    env = LinearControlEnv(id=args.env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    
    # Add Robustness (Noise + Disturbances)
    env = RobustControlWrapper(
        env, 
        noise_std=args.noise_std, 
        disturbance_prob=args.dist_prob, 
        disturbance_mag=args.dist_mag
    )
    print(f"Robust Environment Initialized:")
    print(f"  - Noise Std: {args.noise_std}")
    print(f"  - Disturbance Prob: {args.dist_prob}")
    print(f"  - Disturbance Mag: {args.dist_mag}")
    
    # Add Reward Shaping
    if args.use_optimized_reward:
        env = OptimizedRewardWrapper(env, target_threshold=0.1, settling_steps=50)
        print("  - Reward: Optimized")
    else:
        env = RewardShapingWrapper(env, alpha=0.01, beta=0.01)
        print("  - Reward: Standard Shaping")
        
    env = Monitor(env, args.log_dir)
    env = DummyVecEnv([lambda: env])
    
    # Train PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=args.log_dir,
        name_prefix='robust_ppo'
    )
    
    print(f"\nStarting Robust PPO training for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    
    # Save final model
    model_path = os.path.join(args.log_dir, "final_model_robust")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
