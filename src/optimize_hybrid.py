"""
Optimize RL-PD hybrid controller weighting (lambda_pd) automatically.

This script evaluates a trained model with different lambda_pd values
and selects the best one based on performance metrics.
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from controlgym.envs.linear_control import LinearControlEnv
import matplotlib.pyplot as plt
import json


class PDController:
    """Classical PD controller for linear systems."""
    def __init__(self, kp=1.0, kd=0.5):
        self.kp = kp
        self.kd = kd
    
    def compute_action(self, state):
        """Compute PD control action."""
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        
        if state.ndim == 0 or len(state) == 1:
            position = state[0] if len(state) > 0 else state
            velocity = 0.0
        else:
            position = state[0]
            velocity = state[1] if len(state) > 1 else 0.0
        
        action = -self.kp * position - self.kd * velocity
        return action


def evaluate_with_lambda(model, env, pd_controller, lambda_pd, num_episodes=5):
    """Evaluate model with specific lambda_pd value.
    
    Returns:
        dict: Performance metrics (mean_reward, std_reward, mean_distance, overshoot)
    """
    episode_rewards = []
    episode_distances = []
    episode_overshoots = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        max_distance = 0
        distances = []
        
        while not (done or truncated):
            # Get RL action
            rl_action, _ = model.predict(obs, deterministic=True)
            
            # Get PD action
            pd_action = pd_controller.compute_action(obs)
            
            # Combine actions
            rl_action_val = np.array(rl_action) if not isinstance(rl_action, np.ndarray) else rl_action
            hybrid_action = (1 - lambda_pd) * rl_action_val + lambda_pd * pd_action
            
            # Step environment
            obs, reward, done, truncated, info = env.step(hybrid_action)
            episode_reward += reward
            
            # Track distance (assuming target is zero)
            obs_array = np.array(obs)
            distance = abs(obs_array[0] if len(obs_array) > 0 else obs_array)
            distances.append(distance)
            max_distance = max(max_distance, distance)
        
        episode_rewards.append(episode_reward)
        episode_distances.append(np.mean(distances))
        episode_overshoots.append(max_distance)
    
    return {
        'lambda_pd': lambda_pd,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_distance': np.mean(episode_distances),
        'mean_overshoot': np.mean(episode_overshoots),
        'rewards': episode_rewards
    }


def optimize_lambda_pd(model, env_id, kp=1.0, kd=0.5, 
                       lambda_range=(0.0, 1.0), num_points=11, 
                       num_episodes=5, save_dir='hybrid_optimization'):
    """Optimize lambda_pd parameter via grid search.
    
    Args:
        model: Trained RL model (PPO, SAC, or TD3)
        env_id: Environment ID
        kp: PD proportional gain
        kd: PD derivative gain
        lambda_range: (min, max) range for lambda_pd
        num_points: Number of lambda values to test
        num_episodes: Episodes per lambda value
        save_dir: Directory to save results
    
    Returns:
        dict: Best lambda_pd and performance metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = LinearControlEnv(id=env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    
    # Create PD controller
    pd_controller = PDController(kp=kp, kd=kd)
    
    # Generate lambda values to test
    lambda_values = np.linspace(lambda_range[0], lambda_range[1], num_points)
    
    # Evaluate each lambda
    results = []
    print(f"\nOptimizing lambda_pd over {num_points} values...")
    print(f"Range: {lambda_range[0]:.2f} to {lambda_range[1]:.2f}")
    print(f"Episodes per lambda: {num_episodes}\n")
    
    for i, lambda_pd in enumerate(lambda_values, 1):
        print(f"[{i}/{num_points}] Testing lambda_pd={lambda_pd:.3f}...", end=" ")
        metrics = evaluate_with_lambda(model, env, pd_controller, lambda_pd, num_episodes)
        results.append(metrics)
        print(f"mean_reward={metrics['mean_reward']:.2f}, overshoot={metrics['mean_overshoot']:.4f}")
    
    # Find best lambda based on reward
    best_idx = np.argmax([r['mean_reward'] for r in results])
    best_result = results[best_idx]
    
    # Save results
    results_path = os.path.join(save_dir, 'lambda_optimization.json')
    with open(results_path, 'w') as f:
        json.dump({
            'best_lambda': best_result['lambda_pd'],
            'best_metrics': best_result,
            'all_results': results,
            'kp': kp,
            'kd': kd
        }, f, indent=4)
    
    # Plot results
    plot_optimization_results(results, save_dir)
    
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"Best lambda_pd: {best_result['lambda_pd']:.3f}")
    print(f"Mean Reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
    print(f"Mean Distance: {best_result['mean_distance']:.4f}")
    print(f"Mean Overshoot: {best_result['mean_overshoot']:.4f}")
    print(f"Results saved to: {save_dir}/")
    print(f"{'='*60}\n")
    
    return best_result


def plot_optimization_results(results, save_dir):
    """Plot lambda_pd optimization results."""
    lambdas = [r['lambda_pd'] for r in results]
    rewards = [r['mean_reward'] for r in results]
    distances = [r['mean_distance'] for r in results]
    overshoots = [r['mean_overshoot'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reward vs lambda
    axes[0].plot(lambdas, rewards, 'b-o', linewidth=2)
    best_idx = np.argmax(rewards)
    axes[0].scatter([lambdas[best_idx]], [rewards[best_idx]], 
                    color='red', s=100, zorder=5, label='Best')
    axes[0].set_xlabel('Lambda_PD')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Reward vs Lambda')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Distance vs lambda
    axes[1].plot(lambdas, distances, 'g-o', linewidth=2)
    axes[1].set_xlabel('Lambda_PD')
    axes[1].set_ylabel('Mean Distance')
    axes[1].set_title('Distance vs Lambda')
    axes[1].grid(True, alpha=0.3)
    
    # Overshoot vs lambda
    axes[2].plot(lambdas, overshoots, 'r-o', linewidth=2)
    axes[2].set_xlabel('Lambda_PD')
    axes[2].set_ylabel('Mean Overshoot')
    axes[2].set_title('Overshoot vs Lambda')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lambda_optimization.png'), dpi=150)
    print(f"Plot saved: {save_dir}/lambda_optimization.png")


def main():
    parser = argparse.ArgumentParser(description='Optimize RL-PD hybrid lambda_pd')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac', 'td3'], 
                        help='Algorithm type')
    parser.add_argument('--env_id', type=str, default='toy', help='Environment ID')
    parser.add_argument('--kp', type=float, default=1.0, help='PD proportional gain')
    parser.add_argument('--kd', type=float, default=0.5, help='PD derivative gain')
    parser.add_argument('--lambda_min', type=float, default=0.0, help='Min lambda_pd')
    parser.add_argument('--lambda_max', type=float, default=1.0, help='Max lambda_pd')
    parser.add_argument('--num_points', type=int, default=11, help='Number of lambda values to test')
    parser.add_argument('--num_episodes', type=int, default=5, help='Episodes per lambda')
    parser.add_argument('--save_dir', type=str, default='hybrid_optimization', help='Save directory')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.algorithm.upper()} model from {args.model_path}...")
    if args.algorithm == 'ppo':
        model = PPO.load(args.model_path)
    elif args.algorithm == 'sac':
        model = SAC.load(args.model_path)
    elif args.algorithm == 'td3':
        model = TD3.load(args.model_path)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Optimize lambda_pd
    best_result = optimize_lambda_pd(
        model=model,
        env_id=args.env_id,
        kp=args.kp,
        kd=args.kd,
        lambda_range=(args.lambda_min, args.lambda_max),
        num_points=args.num_points,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir
    )
    
    print(f"\nRecommended lambda_pd for future training: {best_result['lambda_pd']:.3f}")


if __name__ == "__main__":
    main()
