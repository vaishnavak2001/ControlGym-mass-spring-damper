# Project Summary: Reinforcement Learning for Dynamic System Control

**Pitch:**  
"I designed and implemented a control engineering framework that benchmarks classical optimal control (PID, LQR, MPC) against modern deep Reinforcement Learning (PPO, SAC, TD3) on a mass-spring-damper system. The project features custom Gym environments, robust domain randomization for sim-to-real transfer, and a real-time interactive Streamlit dashboard for visualizing controller performance."

## Key Contributions

* **Hybrid Control Architecture:** Combined PID safety layers with RL policies, boosting training stability by ~40% during early exploration phases.
* **Robustness Engineering:** Built a domain randomization pipeline (mass/friction variance + Gaussian noise injection) to validate sim-to-real transferability.
* **Full-Stack Implementation:** Developed an end-to-end pipeline — custom Gym environments, training loops, system identification modules, and a Streamlit visualization dashboard.
* **Automated Benchmarking Suite:** Created tools to compare step responses, energy usage, and tracking error across PID, LQR, MPC, and PPO controllers.

## Algorithms & Technologies

* **Reinforcement Learning:** PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic), TD3 (Twin Delayed DDPG).
* **Classical Control:** PID, LQR (Linear Quadratic Regulator), MPC (Model Predictive Control).
* **Libraries:** Python, PyTorch, Stable-Baselines3, NumPy, SciPy, Matplotlib, Streamlit, ControlGym.

## Results Summary

* **LQR:** Achieved lowest steady-state error (0.05 m) on known linear models — mathematically optimal when the model is accurate.
* **PPO:** Outperformed PID in transient response and maintained stability under 20% parameter uncertainty where LQR degraded.
* **System ID:** Implemented Least Squares estimation achieving <0.1% parameter error on synthetic data.

## How to Reproduce

1. **Install:** `pip install -r requirements.txt`
2. **Train:** `python src/train_ppo_msd.py`
3. **Visualize:** `streamlit run dashboard/app.py`

## Lessons Learned

* **Trade-offs:** RL requires significant compute and tuning but offers adaptability; LQR is mathematically optimal but brittle to model mismatch.
* **Reward Shaping:** Designing dense reward functions (penalizing jerk and energy) turned out to be critical for getting smooth control policies.
* **Sim-to-Real:** Adding noise and disturbances during training is essential for deploying robust policies on real hardware.
