import argparse
import os
import sys
import itertools
import pandas as pd
from experiment_logger import ExperimentLogger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from controlgym.envs.linear_control import LinearControlEnv


def run_single_experiment(algorithm, env_id, hyperparams, experiment_name):
    """Run a single experiment with given hyperparameters.
    
    Args:
        algorithm: 'ppo' or 'sac'
        env_id: Environment ID
        hyperparams: Dictionary of hyperparameters
        experiment_name: Name for this experiment
        
    Returns:
        Dictionary with experiment results
    """
    # Create logger
    logger = ExperimentLogger(experiment_name)
    
    # Log hyperparameters
    config = {
        'algorithm': algorithm,
        'env_id': env_id,
        **hyperparams
    }
    logger.log_hyperparameters(config)
    
    # Create environment
    try:
        env = LinearControlEnv(id=env_id, n_steps=1000, sample_time=0.1, action_limit=10.0)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return None
    
    env = Monitor(env, logger.experiment_dir)
    env = DummyVecEnv([lambda: env])
    
    # Create model
    if algorithm.lower() == 'ppo':
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=hyperparams.get('learning_rate', 3e-4),
            verbose=0,
            seed=hyperparams.get('seed', 42)
        )
    elif algorithm.lower() == 'sac':
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=hyperparams.get('learning_rate', 3e-4),
            buffer_size=hyperparams.get('buffer_size', 100000),
            verbose=0,
            seed=hyperparams.get('seed', 42)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    total_timesteps = hyperparams.get('total_timesteps', 5000)
    print(f"\nTraining {algorithm.upper()} for {total_timesteps} timesteps...")
    
    # Custom callback to log episodes
    class EpisodeLogger:
        def __init__(self, logger):
            self.logger = logger
            self.episode_count = 0
        
        def __call__(self, locals_, globals_):
            # Log episode if completed
            for info in locals_['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    self.logger.log_episode(
                        step=locals_['self'].num_timesteps,
                        reward=info['episode']['r'],
                        metrics_dict={'episode_length': info['episode']['l']}
                    )
            return True
    
    episode_callback = EpisodeLogger(logger)
    
    # Train model
    model.learn(total_timesteps=total_timesteps, callback=episode_callback)
    
    # Save model
    model_path = os.path.join(logger.experiment_dir, f"{algorithm}_model.zip")
    model.save(model_path)
    
    # Finalize and generate reports
    summary = logger.finalize()
    
    return summary


def run_experiment_sweep(algorithm='ppo', env_id='toy'):
    """Run hyperparameter sweep.
    
    Args:
        algorithm: 'ppo' or 'sac'
        env_id: Environment ID
    """
    # Define hyperparameter grid
    if algorithm.lower() == 'ppo':
        param_grid = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'total_timesteps': [5000],
            'seed': [42, 123]
        }
    else:  # SAC
        param_grid = {
            'learning_rate': [1e-4, 3e-4],
            'buffer_size': [50000, 100000],
            'total_timesteps': [5000],
            'seed': [42]
        }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n{'='*70}")
    print(f"Running hyperparameter sweep for {algorithm.upper()}")
    print(f"Total experiments: {len(combinations)}")
    print(f"{'='*70}\n")
    
    # Run all experiments
    results = []
    for i, hyperparams in enumerate(combinations, 1):
        experiment_name = f"{algorithm}_{env_id}_exp{i}"
        print(f"\n[{i}/{len(combinations)}] Running: {experiment_name}")
        print(f"Hyperparameters: {hyperparams}")
        
        summary = run_single_experiment(algorithm, env_id, hyperparams, experiment_name)
        
        if summary:
            results.append({
                'experiment': experiment_name,
                'algorithm': algorithm,
                **hyperparams,
                'mean_reward': summary['mean_reward'],
                'std_reward': summary['std_reward'],
                'total_episodes': summary['total_episodes'],
                'experiment_dir': summary['experiment_dir']
            })
    
    # Save aggregate results
    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join('experiments', f'{algorithm}_sweep_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*70}")
        print(f"Sweep completed! Results saved to: {results_path}")
        print(f"{'='*70}\n")
        
        # Display summary
        print("\nTop 3 configurations by mean reward:")
        print(results_df.nlargest(3, 'mean_reward')[['experiment', 'learning_rate', 'mean_reward', 'std_reward']])
        
        # Find best
        best_idx = results_df['mean_reward'].idxmax()
        best = results_df.loc[best_idx]
        print(f"\nBest configuration:")
        print(f"  Experiment: {best['experiment']}")
        print(f"  Learning Rate: {best['learning_rate']}")
        print(f"  Mean Reward: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
        print(f"  Directory: {best['experiment_dir']}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run automated RL experiments')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'], help='RL algorithm')
    parser.add_argument('--env_id', type=str, default='toy', help='Environment ID')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--single', action='store_true', help='Run single experiment')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (for single experiment)')
    parser.add_argument('--timesteps', type=int, default=5000, help='Total timesteps (for single experiment)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for single experiment)')
    args = parser.parse_args()
    
    # Ensure experiments directory exists
    os.makedirs('experiments', exist_ok=True)
    
    if args.sweep:
        # Run hyperparameter sweep
        run_experiment_sweep(algorithm=args.algorithm, env_id=args.env_id)
    elif args.single:
        # Run single experiment
        hyperparams = {
            'learning_rate': args.lr,
            'total_timesteps': args.timesteps,
            'seed': args.seed
        }
        experiment_name = f"{args.algorithm}_{args.env_id}_single"
        run_single_experiment(args.algorithm, args.env_id, hyperparams, experiment_name)
    else:
        print("Please specify --sweep or --single")
        print("Example: python src/run_experiments.py --single --algorithm ppo")
        print("Example: python src/run_experiments.py --sweep --algorithm sac")


if __name__ == "__main__":
    main()
