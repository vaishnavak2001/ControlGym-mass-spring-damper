"""
Classical Control Implementations: PID and LQR for Mass-Spring-Damper System

This module provides classical control strategies for comparison with RL approaches.
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from controlgym.envs.linear_control import LinearControlEnv


class PIDController:
    """
    PID (Proportional-Integral-Derivative) Controller.
    
    Control law: u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt
    where e(t) is the error (setpoint - current_position)
    """
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.5, dt=0.1):
        """
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Time step for integration/differentiation
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, error):
        """
        Compute PID control action.
        
        Args:
            error: Current error (setpoint - measurement)
            
        Returns:
            Control action
        """
        # Proportional term
        P = self.kp * error
        
        # Integral term (trapezoidal integration)
        self.integral += error * self.dt
        I = self.ki * self.integral
        
        # Derivative term (backward difference)
        derivative = (error - self.prev_error) / self.dt
        D = self.kd * derivative
        
        # Update state
        self.prev_error = error
        
        # PID output
        u = P + I + D
        return u


class LQRController:
    """
    LQR (Linear Quadratic Regulator) Controller.
    
    Optimal control for linear systems: u = -K*x
    where K is computed by solving the Algebraic Riccati Equation.
    """
    
    def __init__(self, A, B, Q, R):
        """
        Args:
            A: State matrix (n x n)
            B: Input matrix (n x m)
            Q: State cost matrix (n x n), positive semi-definite
            R: Control cost matrix (m x m), positive definite
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        
        # Solve Riccati equation to find optimal gain
        self.K = self._compute_lqr_gain()
        
    def _compute_lqr_gain(self):
        """
        Solve continuous-time algebraic Riccati equation.
        
        Returns:
            K: Optimal feedback gain matrix
        """
        # Solve CARE: A'P + PA - PBR^(-1)B'P + Q = 0
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # Compute optimal gain: K = R^(-1)B'P
        K = np.linalg.inv(self.R) @ self.B.T @ P
        
        return K
    
    def compute(self, state):
        """
        Compute LQR control action.
        
        Args:
            state: Current state vector
            
        Returns:
            Control action
        """
        state = np.array(state).flatten()
        u = -self.K @ state
        return u.item() if u.size == 1 else u


def run_pid_simulation(env, pid_controller, n_steps=1000, setpoint=0.0):
    """
    Run simulation with PID controller.
    
    Args:
        env: Gym environment
        pid_controller: PIDController instance
        n_steps: Number of simulation steps
        setpoint: Target position
        
    Returns:
        dict: Simulation results
    """
    pid_controller.reset()
    obs, _ = env.reset()
    
    states = []
    actions = []
    rewards = []
    
    for _ in range(n_steps):
        # Extract position (assuming first state is position)
        position = obs[0] if len(obs) > 0 else obs
        
        # Compute error
        error = setpoint - position
        
        # PID control
        action = pid_controller.compute(error)
        
        # Apply action
        obs, reward, done, truncated, _ = env.step(np.array([action]))
        
        # Store data
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        
        if done or truncated:
            obs, _ = env.reset()
            pid_controller.reset()
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'total_reward': np.sum(rewards)
    }


def run_lqr_simulation(env, lqr_controller, n_steps=1000):
    """
    Run simulation with LQR controller.
    
    Args:
        env: Gym environment
        lqr_controller: LQRController instance
        n_steps: Number of simulation steps
        
    Returns:
        dict: Simulation results
    """
    obs, _ = env.reset()
    
    states = []
    actions = []
    rewards = []
    
    for _ in range(n_steps):
        # LQR control
        action = lqr_controller.compute(obs)
        
        # Apply action
        obs, reward, done, truncated, _ = env.step(np.array([action]))
        
        # Store data
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        
        if done or truncated:
            obs, _ = env.reset()
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'total_reward': np.sum(rewards)
    }


def run_rl_simulation(env, model, n_steps=1000):
    """
    Run simulation with trained RL model.
    
    Args:
        env: Gym environment
        model: Trained RL model (PPO, SAC, TD3)
        n_steps: Number of simulation steps
        
    Returns:
        dict: Simulation results
    """
    obs, _ = env.reset()
    
    states = []
    actions = []
    rewards = []
    
    for _ in range(n_steps):
        # RL policy
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply action
        obs, reward, done, truncated, _ = env.step(action)
        
        # Store data
        states.append(obs.copy())
        actions.append(action.item() if hasattr(action, 'item') else action)
        rewards.append(reward)
        
        if done or truncated:
            obs, _ = env.reset()
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'total_reward': np.sum(rewards)
    }


def plot_comparison(results_dict, save_path='controllers/comparison.png'):
    """
    Plot comparison of different control strategies.
    
    Args:
        results_dict: Dictionary with controller names as keys and results as values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot positions
    for i, (name, results) in enumerate(results_dict.items()):
        states = results['states']
        positions = states[:, 0] if states.ndim > 1 else states
        axes[0].plot(positions, label=name, color=colors[i % len(colors)], linewidth=1.5)
    
    axes[0].set_ylabel('Position')
    axes[0].set_title('Position Tracking Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Target')
    
    # Plot actions
    for i, (name, results) in enumerate(results_dict.items()):
        actions = results['actions']
        axes[1].plot(actions, label=name, color=colors[i % len(colors)], linewidth=1.5)
    
    axes[1].set_ylabel('Control Action')
    axes[1].set_title('Control Effort Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot cumulative rewards
    for i, (name, results) in enumerate(results_dict.items()):
        rewards = results['rewards']
        cumulative_rewards = np.cumsum(rewards)
        axes[2].plot(cumulative_rewards, label=name, color=colors[i % len(colors)], linewidth=1.5)
    
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Cumulative Reward')
    axes[2].set_title('Performance Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def get_toy_system_matrices():
    """
    Get state-space matrices for the 'toy' linear control system.
    
    Returns:
        A, B: State and input matrices
    """
    # Toy system is 1D (single integrator)
    # State: [position]
    # x_dot = Ax + Bu
    
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    
    return A, B

