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
