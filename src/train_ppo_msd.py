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
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Create environment using controlgym's LinearControlEnv
    try:
        env = LinearControlEnv(id=args.env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    except Exception as e:
        print(f"Error: Failed to create environment with ID '{args.env_id}': {e}")
        print("Available IDs: toy, pas, lah, rea, psm, he1-he6, je1-je2, umv")
        return

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

if __name__ == "__main__":
    main()
