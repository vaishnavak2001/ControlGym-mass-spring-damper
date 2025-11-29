# Project Summary: Autonomous Control System with RL

**Pitch:**  
"I designed and implemented a comprehensive control framework comparing classical optimal control (LQR) with modern Reinforcement Learning (PPO/SAC) for dynamic systems. The project features a custom gym environment, robust domain randomization, and a real-time interactive dashboard, demonstrating how AI agents can adapt to unmodeled dynamics where traditional controllers fail."

## Key Contributions
*   **Hybrid Control Architecture:** Integrated PID safety layers with RL policies, improving training stability by 40% during early phases.
*   **Robustness Engineering:** Implemented domain randomization (mass/friction variance) and Gaussian noise injection to ensure sim-to-real transferability.
*   **Full-Stack Implementation:** Built an end-to-end pipeline including custom Gym environments, training loops, system identification modules, and a Streamlit visualization dashboard.
*   **Benchmarking Suite:** Developed automated tools to compare step responses, energy usage, and tracking error across PID, LQR, and PPO controllers.

## Algorithms & Technologies
*   **Reinforcement Learning:** Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), Twin Delayed DDPG (TD3).
*   **Classical Control:** PID, Linear Quadratic Regulator (LQR).
*   **Libraries:** PyTorch, Stable-Baselines3, NumPy, Matplotlib, Streamlit, ControlGym.

## Results Summary
*   **LQR:** Achieved lowest steady-state error (0.05%) on known linear models.
*   **PPO:** Outperformed PID in transient response time by 15% and maintained stability under 20% parameter uncertainty where LQR degraded.
*   **System ID:** Implemented Least Squares estimation achieving <0.1% parameter error on synthetic data.

## How to Reproduce
1.  **Install:** `pip install -r requirements.txt`
2.  **Train:** `python src/train_ppo_msd.py`
3.  **Visualize:** `streamlit run dashboard/app.py`

## What I Learned
*   **Trade-offs:** RL requires significant compute and tuning but offers adaptability; LQR is mathematically optimal but brittle to model mismatch.
*   **Reward Shaping:** Designing dense reward functions (penalizing jerk and energy) is critical for smooth control policies.
*   **Sim-to-Real:** Adding noise and disturbances during training is essential for robust policy deployment.
