import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectories(states, actions, save_path):
    """
    Plot position, velocity, and actions.
    states: (T, state_dim) or (T,) for 1D states
    actions: (T, action_dim) or (T,) for 1D actions
    save_path: directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure states is 2D
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    
    # Ensure actions is 2D
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    
    time = np.arange(len(states))
    
    # Plot all state dimensions
    state_dim = states.shape[1]
    if state_dim >= 1:
        plt.figure(figsize=(10, 5))
        plt.plot(time, states[:, 0], label='State 0')
        plt.xlabel('Time Step')
        plt.ylabel('State 0')
        plt.title('State 0 Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'state_0.png'))
        plt.close()

    if state_dim >= 2:
        plt.figure(figsize=(10, 5))
        plt.plot(time, states[:, 1], label='State 1')
        plt.xlabel('Time Step')
        plt.ylabel('State 1')
        plt.title('State 1 Trajectory')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'state_1.png'))
        plt.close()

    # Plot all actions
    action_dim = actions.shape[1]
    for i in range(action_dim):
        plt.figure(figsize=(10, 5))
        plt.plot(time, actions[:, i], label=f'Action {i}')
        plt.xlabel('Time Step')
        plt.ylabel(f'Action {i}')
        plt.title(f'Control Action {i}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'action_{i}.png'))
        plt.close()

def plot_reward_curves(rewards, save_path, window_size=10):
    """
    Plot episode rewards over time with moving average.
    rewards: list or array of episode rewards
    save_path: directory to save plot
    window_size: window size for moving average
    """
    os.makedirs(save_path, exist_ok=True)
    
    rewards = np.array(rewards)
    episodes = np.arange(len(rewards))
    
    # Calculate moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_episodes = episodes[window_size-1:]
    else:
        moving_avg = rewards
        moving_avg_episodes = episodes
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.6, label='Episode Reward', linewidth=1)
    plt.plot(moving_avg_episodes, moving_avg, label=f'Moving Average (window={window_size})', linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'reward_curve.png'))
    plt.close()

def plot_cumulative_rewards(rewards, save_path):
    """
    Plot cumulative sum of rewards.
    rewards: list or array of episode rewards
    save_path: directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    rewards = np.array(rewards)
    cumulative = np.cumsum(rewards)
    episodes = np.arange(len(rewards))
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, cumulative, linewidth=2, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'cumulative_rewards.png'))
    plt.close()

def plot_action_profile(actions, save_path):
    """
    Plot action distribution and magnitude over time.
    actions: (T, action_dim) or (T,) array of actions
    save_path: directory to save plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure actions is 2D
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    
    action_dim = actions.shape[1]
    
    for i in range(action_dim):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Histogram
        ax1.hist(actions[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel(f'Action {i} Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Action {i} Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Magnitude over time
        time = np.arange(len(actions))
        ax2.plot(time, actions[:, i], alpha=0.6, linewidth=1)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel(f'Action {i} Value')
        ax2.set_title(f'Action {i} Magnitude Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'action_{i}_profile.png'))
        plt.close()
