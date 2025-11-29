"""
Benchmark classical controllers (PID, LQR) against RL methods (PPO).
"""

import argparse
import os
from stable_baselines3 import PPO
from controlgym.envs.linear_control import LinearControlEnv
import sys
sys.path.append('controllers')
from classical_controllers import (
    PIDController, LQRController,
    run_pid_simulation, run_lqr_simulation, run_rl_simulation,
    plot_comparison, get_toy_system_matrices
)
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Benchmark Classical vs RL Controllers')
    parser.add_argument('--env_id', type=str, default='toy', help='Environment ID')
    parser.add_argument('--n_steps', type=int, default=500, help='Simulation steps')
    parser.add_argument('--rl_model', type=str, default='results/final_model.zip', help='Path to RL model')
    parser.add_argument('--save_dir', type=str, default='controllers', help='Save directory')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    env = LinearControlEnv(id=args.env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Controllers on '{args.env_id}' environment")
    print(f"Simulation steps: {args.n_steps}")
    print(f"{'='*60}\n")
    
    # 1. PID Controller
    print("[1/3] Running PID controller...")
    pid = PIDController(kp=0.5, ki=0.01, kd=0.2, dt=0.1)
    pid_results = run_pid_simulation(env, pid, n_steps=args.n_steps, setpoint=0.0)
    print(f"  Total reward: {pid_results['total_reward']:.2f}")
    
    # 2. LQR Controller
    print("[2/3] Running LQR controller...")
    A, B = get_toy_system_matrices()
    Q = np.array([[1.0]])   # 1D state cost
    R = np.array([[0.1]])    # Control effort penalty
    lqr = LQRController(A, B, Q, R)
    print(f"  Optimal gain K = {lqr.K}")
    lqr_results = run_lqr_simulation(env, lqr, n_steps=args.n_steps)
    print(f"  Total reward: {lqr_results['total_reward']:.2f}")
    
    # 3. RL Controller (PPO)
    print("[3/3] Running PPO controller...")
    if os.path.exists(args.rl_model):
        ppo = PPO.load(args.rl_model)
        ppo_results = run_rl_simulation(env, ppo, n_steps=args.n_steps)
        print(f"  Total reward: {ppo_results['total_reward']:.2f}")
    else:
        print(f"  Warning: RL model not found at {args.rl_model}")
        print(f"  Train a model first: python src/train_ppo_msd.py")
        ppo_results = None
    
    # Generate comparison plot
    print("\nGenerating comparison plots...")
    results_dict = {
        'PID': pid_results,
        'LQR': lqr_results,
    }
    if ppo_results:
        results_dict['PPO'] = ppo_results
    
    plot_comparison(results_dict, save_path=os.path.join(args.save_dir, 'comparison.png'))
    
    # Summary
    print(f"\n{'='*60}")
    print("Performance Summary:")
    print(f"{'='*60}")
    print(f"{'Controller':<15} {'Total Reward':<15} {'Mean |Position|':<20}")
    print(f"{'-'*60}")
    
    for name, results in results_dict.items():
        states = results['states']
        positions = states[:, 0] if states.ndim > 1 else states
        mean_abs_pos = np.mean(np.abs(positions))
        print(f"{name:<15} {results['total_reward']:<15.2f} {mean_abs_pos:<20.4f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
