"""
Smoke test to verify basic functionality of the project.
Runs a short training session to ensure no immediate crashes.
"""

import unittest
import os
import sys
import shutil
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.classical_controllers import PIDController, LQRController
import numpy as np

class TestProject(unittest.TestCase):
    def setUp(self):
        # Create a temp dir for test results
        self.test_dir = "test_results"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_classical_controllers(self):
        print("\nTesting Classical Controllers...")
        # PID
        pid = PIDController(kp=1.0, ki=0.1, kd=0.1)
        # PID expects error
        action = pid.compute(error=1.0)
        self.assertIsInstance(action, (float, np.floating))
        
        # LQR
        A = np.array([[0, 1], [-1, -1]])
        B = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.eye(1)
        lqr = LQRController(A, B, Q, R)
        action = lqr.compute(np.array([1.0, 0.0]))
        self.assertIsInstance(action, (float, np.floating, np.ndarray))

    def test_ppo_training_smoke(self):
        print("\nTesting PPO Training (Smoke Test)...")
        cmd = [
            sys.executable, "src/train_ppo_msd.py",
            "--total_timesteps", "100",
            "--log_dir", self.test_dir,
            "--env_id", "toy"
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            self.fail(f"PPO training failed: {e}")
            
    def test_sac_training_smoke(self):
        print("\nTesting SAC Training (Smoke Test)...")
        cmd = [
            sys.executable, "src/train_sac_msd.py",
            "--total_timesteps", "100",
            "--log_dir", self.test_dir,
            "--env_id", "toy",
            "--batch_size", "64",
            "--buffer_size", "1000"
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            self.fail(f"SAC training failed: {e}")

if __name__ == "__main__":
    unittest.main()
