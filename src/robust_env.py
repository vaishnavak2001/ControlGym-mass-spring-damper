"""
Robust Control Environment Wrapper.

Adds Gaussian measurement noise and external impulse disturbances
to simulate real-world conditions and enable domain randomization.
"""

import gymnasium as gym
import numpy as np


class RobustControlWrapper(gym.Wrapper):
    """
    Gym wrapper to add robustness features:
    1. Gaussian measurement noise on observations
    2. Random impulse disturbances (external forces)
    """
    
    def __init__(self, env, noise_std=0.01, disturbance_prob=0.01, disturbance_mag=1.0):
        """
        Args:
            env: The environment to wrap
            noise_std: Standard deviation of Gaussian measurement noise
            disturbance_prob: Probability of impulse disturbance per step
            disturbance_mag: Magnitude of impulse disturbance
        """
        super().__init__(env)
        self.noise_std = noise_std
        self.disturbance_prob = disturbance_prob
        self.disturbance_mag = disturbance_mag
        
    def reset(self, **kwargs):
        # Handle Gymnasium API compatibility
        if 'options' in kwargs:
            kwargs.pop('options')
        return self.env.reset(**kwargs)

    def step(self, action):
        # 1. Apply Action (Standard Step)
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 2. Add External Disturbance (Impulse)
        # We modify the internal state directly if possible, or just the observation 
        # if we treat it as a sensor disturbance. For physical disturbance, 
        # we ideally want to affect the next state. 
        # Since we can't easily access the internal physics engine here without 
        # knowing the specific env structure, we'll simulate it by perturbing the 
        # observation (state) which feeds back into the loop.
        # For a more physics-accurate disturbance, we'd need to modify the env's step logic.
        # Assuming obs is [position, velocity, ...], perturbing velocity is like an impulse.
        
        if np.random.random() < self.disturbance_prob:
            disturbance = np.random.uniform(-self.disturbance_mag, self.disturbance_mag, size=obs.shape)
            obs = obs + disturbance
            info['disturbance'] = True
        else:
            info['disturbance'] = False
            
        # 3. Add Measurement Noise (Gaussian)
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        
        # Store true observation in info for debugging/logging
        info['true_obs'] = obs
        info['noise'] = noise
        
        return noisy_obs, reward, done, truncated, info
