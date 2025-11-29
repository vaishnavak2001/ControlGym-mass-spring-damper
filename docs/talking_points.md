# Interview Talking Points

## 1. Why did you choose PPO over DQN?
*   **Continuous Action Spaces:** The mass-spring-damper system requires continuous force control. DQN output is discrete, which would require discretization (loss of precision) or complex architecture. PPO handles continuous actions naturally via Gaussian policy heads.
*   **Stability:** PPO's clipped objective function prevents destructive policy updates, making training more stable than off-policy methods like DDPG in early stages.

## 2. How did you handle the "Sim-to-Real" gap?
*   **Domain Randomization:** I randomized physical parameters (mass, friction) during training so the agent learns to be robust to model mismatches.
*   **Noise Injection:** I added Gaussian noise to observations and random impulse disturbances to forces, simulating real-world sensor/actuator imperfections.

## 3. What was the hardest part?
*   **Reward Shaping:** Getting the agent to settle smoothly without oscillation was tough. I moved from a simple distance penalty to a comprehensive function including velocity penalties, control effort costs, and a "settling bonus" for staying within the target zone.

## 4. Why use Hybrid RL-PD?
*   **Safety & Speed:** Pure RL starts with random exploration, which can be dangerous or slow. The PD controller provides a "baseline" safe behavior, allowing the RL agent to learn residual corrections. This bootstraps performance and acts as a safety guardrail.

## 5. How does LQR compare to RL?
*   **LQR is Optimal (for Linear):** If we know the exact $A$ and $B$ matrices, LQR is mathematically guaranteed to be optimal.
*   **RL is Adaptive:** RL shines when we *don't* know the exact parameters or if the system becomes non-linear (e.g., friction, saturation). My benchmarks showed LQR wins on the perfect model, but RL is competitive and more robust to changes.

## 6. How did you validate the system?
*   **System ID:** I implemented a Least Squares estimator to verify I could recover the true physical parameters from trajectory data.
*   **Benchmarking:** I ran side-by-side simulations of PID, LQR, and PPO on the same random seeds to compare tracking error and energy usage fairly.

## 7. What tools did you use?
*   **Stable-Baselines3:** For reliable RL implementations.
*   **ControlGym:** For the environment interface.
*   **Streamlit:** To build the interactive demo tool.
*   **GitHub Actions:** For CI/CD smoke testing.

## 8. Future Improvements?
*   **Real Hardware:** Deploying the policy to a physical servo or cart-pole system.
*   **Model-Based RL:** Exploring algorithms like MBPO that learn the dynamics model explicitly to improve sample efficiency.
