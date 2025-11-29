import controlgym
from controlgym.envs.linear_control import LinearControlEnv
import os

ids = ['ac', 'bdt', 'cd', 'fs', 'he1', 'he2', 'he3', 'he4', 'he5', 'he6', 'he7', 'ih', 'isa', 'je1', 'je2', 'lah', 'pas', 'psm', 'rea', 'toy', 'umv']

for env_id in ids:
    try:
        env = LinearControlEnv(id=env_id)
        print(f"ID: {env_id}, State Dim: {env.observation_space.shape}, Action Dim: {env.action_space.shape}")
    except Exception as e:
        print(f"ID: {env_id} failed: {e}")
