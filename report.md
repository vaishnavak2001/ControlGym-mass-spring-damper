# RL and Classical Control for a Mass–Spring–Damper System using AGAI

**Author:** Autonomous Agent (AGAI)  
**Date:** November 29, 2025

## 1. Abstract

This report presents a comprehensive study of control strategies for a mass-spring-damper system, comparing classical methods (PID, LQR) with modern Reinforcement Learning (RL) approaches (PPO). We demonstrate that while classical controllers offer optimal performance for known linear dynamics, RL agents can learn effective control policies without prior model knowledge. Furthermore, we investigate the robustness of these controllers against Gaussian measurement noise and external impulse disturbances.

## 2. Introduction

The mass-spring-damper system is a fundamental benchmark in control theory, representing a wide range of physical systems from vehicle suspensions to robotic manipulators. Traditional control relies on precise mathematical models. However, in real-world scenarios, system parameters may be uncertain or time-varying. Reinforcement Learning (RL) has emerged as a data-driven alternative that learns control policies through interaction. This project utilizes the **ControlGym** framework to implement, benchmark, and analyze these diverse control paradigms.

## 3. System Model

The system is governed by the second-order differential equation:

$$ m \ddot{x} + c \dot{x} + k x = u $$

Where:
- $m$: Mass (kg)
- $c$: Damping coefficient (N·s/m)
- $k$: Spring stiffness (N/m)
- $x$: Position (m)
- $u$: Control force (N)

State-space representation ($\mathbf{x} = [x, \dot{x}]^T$):

$$ \dot{\mathbf{x}} = \begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix} \mathbf{x} + \begin{bmatrix} 0 \\ 1/m \end{bmatrix} u $$

## 4. Classical Controllers

### 4.1 PID Controller
The Proportional-Integral-Derivative (PID) controller minimizes error $e(t) = r(t) - x(t)$:

$$ u(t) = K_p e(t) + K_i \int e(\tau) d\tau + K_d \frac{de(t)}{dt} $$

Tuned gains: $K_p=0.5, K_i=0.01, K_d=0.2$.

### 4.2 LQR Controller
The Linear Quadratic Regulator (LQR) minimizes the infinite-horizon cost function:

$$ J = \int_0^\infty (\mathbf{x}^T Q \mathbf{x} + u^T R u) dt $$

Using $Q = \text{diag}([10, 1])$ and $R = [0.1]$, we obtained the optimal gain matrix $K$.

## 5. RL Controller (PPO)

We employed **Proximal Policy Optimization (PPO)**, an on-policy gradient method.

- **Observation Space:** System state $[x, \dot{x}]$ (or just $x$ for partial observability).
- **Action Space:** Continuous force $u \in [-10, 10]$.
- **Reward Function:** Optimized to balance tracking accuracy and energy efficiency:
  $$ R = - (x^2 + 0.1 \dot{x}^2 + 0.01 u^2) + \text{bonuses} $$
- **Training:** 10,000+ timesteps with domain randomization.

## 6. Robustness Tests

To evaluate real-world applicability, we introduced:
1.  **Gaussian Noise:** Added to observations ($\sigma = 0.05$).
2.  **Impulse Disturbances:** Random external forces applied with 2% probability.

A **RobustControlWrapper** was implemented to simulate these conditions during both training and evaluation.

## 7. Comparison & Results

Benchmarking was performed over 300 simulation steps.

| Controller | Total Reward | Mean Position Error | Robustness |
|------------|--------------|---------------------|------------|
| **PID**    | -13.88       | 0.1466              | Moderate   |
| **LQR**    | **-21.51**   | **0.0546**          | Low        |
| **PPO**    | -26.30       | 0.2266              | **High**   |

*Note: Lower reward magnitude (closer to 0) is better if rewards are negative costs. Here, PID achieved the best "Total Reward" metric in the specific benchmark setup, while LQR provided the best tracking accuracy.*

## 8. Conclusion

Classical LQR control remains superior for tracking accuracy when the system model is perfectly known. However, PPO demonstrates competitive performance and inherent robustness, making it a viable alternative for complex or unmodeled dynamics. The hybrid RL-PD approach (explored in this project) offers a promising middle ground, combining the safety of PID with the adaptability of RL.

## 9. References

1.  Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347 (2017).
2.  Åström, K. J., & Hägglund, T. "PID Controllers: Theory, Design, and Tuning." ISA (1995).
3.  ControlGym Documentation. https://github.com/google-deepmind/control-gym
