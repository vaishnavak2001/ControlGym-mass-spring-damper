import os
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


class ExperimentLogger:
    """Logger for tracking experiments with hyperparameters, metrics, and results."""
    
    def __init__(self, experiment_name, base_dir='experiments'):
        """Initialize experiment logger with timestamped directory.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for all experiments
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # File paths
        self.config_path = os.path.join(self.experiment_dir, 'config.json')
        self.episodes_path = os.path.join(self.experiment_dir, 'episodes.csv')
        self.summary_path = os.path.join(self.experiment_dir, 'summary.json')
        self.plots_path = os.path.join(self.experiment_dir, 'plots.pdf')
        
        # Initialize episodes CSV
        self.episodes_data = []
        
        print(f"Experiment directory created: {self.experiment_dir}")
    
    def log_hyperparameters(self, params_dict):
        """Log hyperparameters to config.json.
        
        Args:
            params_dict: Dictionary of hyperparameters
        """
        with open(self.config_path, 'w') as f:
            json.dump(params_dict, f, indent=4)
        print(f"Hyperparameters logged to {self.config_path}")
    
    def log_episode(self, step, reward, metrics_dict=None):
        """Log episode data.
        
        Args:
            step: Current timestep
            reward: Episode reward
            metrics_dict: Optional dictionary of additional metrics
        """
        episode_data = {'step': step, 'reward': reward}
        if metrics_dict:
            episode_data.update(metrics_dict)
        self.episodes_data.append(episode_data)
    
    def save_episodes(self):
        """Save all logged episodes to CSV."""
        if not self.episodes_data:
            return
        
        with open(self.episodes_path, 'w', newline='') as f:
            fieldnames = self.episodes_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episodes_data)
        print(f"Episodes logged to {self.episodes_path}")
    
    def log_summary(self, summary_dict):
        """Log experiment summary to summary.json.
        
        Args:
            summary_dict: Dictionary containing summary statistics
        """
        with open(self.summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=4)
        print(f"Summary logged to {self.summary_path}")
    
    def generate_plots(self):
        """Generate plots and save to PDF."""
        if not self.episodes_data:
            print("No episode data to plot")
            return
        
        # Extract data
        steps = [ep['step'] for ep in self.episodes_data]
        rewards = [ep['reward'] for ep in self.episodes_data]
        
        # Create PDF with multiple pages
        with PdfPages(self.plots_path) as pdf:
            # Page 1: Reward curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, rewards, 'b-', alpha=0.3, label='Episode Reward')
            
            # Moving average
            if len(rewards) >= 10:
                window = min(10, len(rewards))
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Episode Reward')
            ax.set_title(f'Experiment: {self.experiment_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Cumulative reward
            fig, ax = plt.subplots(figsize=(10, 6))
            cumulative_rewards = np.cumsum(rewards)
            ax.plot(steps, cumulative_rewards, 'g-', linewidth=2)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Cumulative Reward - {self.experiment_name}')
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Reward distribution (if enough data)
            if len(rewards) >= 5:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(rewards, bins=min(30, len(rewards)//2), edgecolor='black', alpha=0.7)
                ax.axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
                ax.axvline(np.median(rewards), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
                ax.set_xlabel('Episode Reward')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Reward Distribution - {self.experiment_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        print(f"Plots saved to {self.plots_path}")
    
    def generate_report(self):
        """Generate comprehensive summary report."""
        if not self.episodes_data:
            print("No episode data for report")
            return
        
        rewards = [ep['reward'] for ep in self.episodes_data]
        
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_dir': self.experiment_dir,
            'total_episodes': len(self.episodes_data),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),
            'final_reward': float(rewards[-1]) if rewards else None
        }
        
        self.log_summary(summary)
        return summary
    
    def finalize(self):
        """Finalize experiment by saving all data and generating reports."""
        self.save_episodes()
        summary = self.generate_report()
        self.generate_plots()
        print(f"\n{'='*60}")
        print(f"Experiment '{self.experiment_name}' completed!")
        print(f"Directory: {self.experiment_dir}")
        if summary:
            print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
            print(f"Total Episodes: {summary['total_episodes']}")
        print(f"{'='*60}\n")
        return summary
