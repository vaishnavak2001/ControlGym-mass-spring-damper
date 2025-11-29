"""
Model Predictive Control (MPC) Implementation.

This module implements a Sampling-based MPC (Model Predictive Path Integral / Random Shooting)
for the mass-spring-damper system. It does not require external solvers like CVXPY.

Algorithm:
1. Sample N random action sequences over horizon H.
2. Simulate system dynamics forward for each sequence.
3. Compute cost for each trajectory.
4. Select the optimal first action (or weighted average).
5. Recede horizon.
"""

import numpy as np

class MPCController:
    """
    Sampling-based Model Predictive Controller.
    """
    
    def __init__(self, A, B, Q, R, horizon=10, n_samples=1000, action_limit=10.0):
        """
        Args:
            A, B: System matrices (x_next = Ax + Bu)
            Q: State cost matrix
            R: Control cost matrix
            horizon: Prediction horizon steps
            n_samples: Number of random trajectories to sample
            action_limit: Max control action magnitude
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.horizon = horizon
        self.n_samples = n_samples
        self.action_limit = action_limit
        
    def compute(self, state):
        """
        Compute optimal control action using random shooting.
        
        Args:
            state: Current state vector
            
        Returns:
            Optimal action (scalar or vector)
        """
        state = np.array(state).flatten()
        
        # 1. Sample random action sequences: (n_samples, horizon, action_dim)
        # Assuming 1D action for simplicity, shape (n_samples, horizon, 1)
        action_seqs = np.random.uniform(
            -self.action_limit, 
            self.action_limit, 
            size=(self.n_samples, self.horizon, 1)
        )
        
        costs = np.zeros(self.n_samples)
        
        # 2. Simulate forward
        # We can vectorize this simulation
        # Current states: (n_samples, state_dim)
        current_states = np.tile(state, (self.n_samples, 1))
        
        for t in range(self.horizon):
            actions = action_seqs[:, t, :] # (n_samples, 1)
            
            # Cost: x'Qx + u'Ru
            # Vectorized cost calculation
            # x'Qx
            state_cost = np.sum((current_states @ self.Q) * current_states, axis=1)
            # u'Ru
            control_cost = np.sum((actions @ self.R) * actions, axis=1)
            
            costs += state_cost + control_cost
            
            # Dynamics: x = Ax + Bu
            # A is (state_dim, state_dim), B is (state_dim, action_dim)
            # current_states is (n_samples, state_dim)
            # actions is (n_samples, action_dim)
            
            next_states = (current_states @ self.A.T) + (actions @ self.B.T)
            current_states = next_states
            
        # 3. Select best sequence
        best_idx = np.argmin(costs)
        best_action = action_seqs[best_idx, 0, :]
        
        return best_action.item() if best_action.size == 1 else best_action
