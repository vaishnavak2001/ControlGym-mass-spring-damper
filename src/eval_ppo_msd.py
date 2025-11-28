import argparse
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import controlgym
from controlgym.envs.linear_control import LinearControlEnv
from utils import plot_trajectories

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO on Mass-Spring-Damper')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--env_id', type=str, default='toy', help='ControlGym LinearControl environment ID')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    model = PPO.load(args.model_path)

    # Create environment using controlgym's LinearControlEnv
    try:
        env = LinearControlEnv(id=args.env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    except Exception as e:
        print(f"Error: Failed to create environment with ID '{args.env_id}': {e}")
        return

    # Evaluate
    all_rewards = []
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset()  # Gym API returns (obs, info)
        done = False
        truncated = False
        episode_reward = 0
        
        states = []
        actions = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)  # Gym API returns 5 values
            
            states.append(obs)
            actions.append(action)
            episode_reward += reward
            
        all_rewards.append(episode_reward)
        
        # Plot last episode
        if episode == args.num_episodes - 1:
            plot_trajectories(np.array(states), np.array(actions), args.plot_dir)

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"Evaluation over {args.num_episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()
