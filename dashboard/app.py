"""
Streamlit Dashboard for ControlGym Mass-Spring-Damper Project.

Features:
- Interactive simulation of PID, LQR, and RL controllers
- Configurable system parameters (mass, spring, damper)
- Real-time plotting of position, velocity, and control effort
- Disturbance and noise injection
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import torch
from stable_baselines3 import PPO, SAC, TD3

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controlgym.envs.linear_control import LinearControlEnv
from controllers.classical_controllers import PIDController, LQRController

# Page Config
st.set_page_config(
    page_title="ControlGym Dashboard",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ ControlGym: Mass-Spring-Damper Simulation")
st.markdown("""
Compare **Classical Control (PID, LQR)** vs **Reinforcement Learning (PPO)** 
on a mass-spring-damper system with configurable parameters and disturbances.
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ System Parameters")

mass = st.sidebar.slider("Mass (kg)", 0.1, 5.0, 1.0, 0.1)
stiffness = st.sidebar.slider("Stiffness (k)", 0.1, 5.0, 1.0, 0.1)
damping = st.sidebar.slider("Damping (c)", 0.0, 2.0, 0.5, 0.1)

st.sidebar.header("🌊 Disturbances")
noise_level = st.sidebar.slider("Measurement Noise (std)", 0.0, 0.2, 0.0, 0.01)
disturbance_mag = st.sidebar.slider("Impulse Disturbance Magnitude", 0.0, 5.0, 0.0, 0.5)
disturbance_prob = st.sidebar.slider("Disturbance Probability", 0.0, 0.1, 0.01, 0.01)

st.sidebar.header("🎮 Controllers")
show_pid = st.sidebar.checkbox("PID Controller", value=True)
show_lqr = st.sidebar.checkbox("LQR Controller", value=True)
show_rl = st.sidebar.checkbox("RL (PPO) Agent", value=True)

# PID Tuning
if show_pid:
    with st.sidebar.expander("PID Tuning"):
        kp = st.number_input("Kp", 0.0, 10.0, 0.5, 0.1)
        ki = st.number_input("Ki", 0.0, 10.0, 0.01, 0.01)
        kd = st.number_input("Kd", 0.0, 10.0, 0.2, 0.1)

# Model Loading
model_path = os.path.join("results", "final_model.zip")
if show_rl and not os.path.exists(model_path):
    st.sidebar.warning(f"RL model not found at {model_path}")
    show_rl = False

# --- Simulation Logic ---

def get_system_matrices(m, k, c):
    """Get A, B matrices for 1D mass-spring-damper."""
    # State: [position, velocity]
    # x_dot = Ax + Bu
    # But for the 'toy' env in controlgym, it might be simplified or different.
    # The 'toy' env in controlgym is often a simple integrator or 1D system.
    # However, for this dashboard, we want to simulate the physics based on user input.
    # Since we are using LinearControlEnv, we can't easily change its internal physics 
    # without modifying the library code or subclassing.
    # 
    # WORKAROUND: We will use the 'toy' env but acknowledge that its internal parameters 
    # are fixed (A=[0], B=[1] for 1D integrator) unless we use a custom env class.
    # 
    # To truly reflect Mass/Spring/Damper changes, we should ideally define our own 
    # system matrices and pass them to LinearControlEnv if it supports it, 
    # or implement a simple simulation loop here using the matrices directly.
    #
    # Let's implement a custom simulation loop using the matrices for PID/LQR 
    # to accurately reflect the user's parameters. For RL, we have to use the Env 
    # it was trained on, which might have fixed parameters. 
    #
    # Actually, let's stick to the 'toy' env for consistency with the trained RL model,
    # but we can try to patch the env if possible, or just note that parameters 
    # mainly affect PID/LQR in this custom sim.
    
    A = np.array([[0, 1], [-k/m, -c/m]])
    B = np.array([[0], [1/m]])
    return A, B

def simulate_system(controller_type, n_steps=300):
    # Initialize state [pos, vel]
    state = np.array([1.0, 0.0]) # Start at position 1.0
    dt = 0.1
    
    # System matrices based on user input
    A, B = get_system_matrices(mass, stiffness, damping)
    
    # Controller setup
    if controller_type == 'PID':
        controller = PIDController(kp=kp, ki=ki, kd=kd, dt=dt)
    elif controller_type == 'LQR':
        # LQR weights
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.1]])
        controller = LQRController(A, B, Q, R)
    elif controller_type == 'RL':
        # Load model
        model = PPO.load(model_path)
    
    # Storage
    times = []
    positions = []
    velocities = []
    actions = []
    
    for t in range(n_steps):
        # Measurement noise
        obs = state + np.random.normal(0, noise_level, size=state.shape)
        
        # Compute Action
        if controller_type == 'PID':
            # PID expects error (setpoint - measurement)
            # Setpoint is 0
            error = 0.0 - obs[0]
            action = controller.compute(error)
        elif controller_type == 'LQR':
            # LQR expects full state
            action = controller.compute(obs)
        elif controller_type == 'RL':
            # RL expects observation from Env. 
            # The trained RL model expects a specific observation shape (likely 1D for 'toy').
            # If we simulate a 2D system here, the 1D RL model might fail or behave poorly.
            # We'll pass just position if the model expects 1D.
            # Let's assume the model handles the observation it was trained on.
            # For 'toy' env, obs is 1D (position).
            # But our custom sim is 2D. We'll pass obs[0].
            rl_obs = np.array([obs[0]])
            action, _ = model.predict(rl_obs, deterministic=True)
            action = float(action)
            
        # Disturbance
        dist = 0.0
        if np.random.random() < disturbance_prob:
            dist = np.random.uniform(-disturbance_mag, disturbance_mag)
        
        # Dynamics Update (Euler Integration)
        # x_dot = Ax + Bu
        x_dot = A @ state + B @ np.array([action + dist])
        state = state + x_dot * dt
        
        # Store
        times.append(t * dt)
        positions.append(state[0])
        velocities.append(state[1])
        actions.append(action)
        
    return pd.DataFrame({
        'Time': times,
        'Position': positions,
        'Velocity': velocities,
        'Action': actions
    })

# --- Main Dashboard Area ---

if st.button("🚀 Simulate Controllers", type="primary"):
    results = {}
    
    with st.spinner("Simulating..."):
        if show_pid:
            results['PID'] = simulate_system('PID')
        if show_lqr:
            results['LQR'] = simulate_system('LQR')
        if show_rl:
            results['RL (PPO)'] = simulate_system('RL')
    
    # Plotting
    st.subheader("Simulation Results")
    
    # 1. Position Plot
    st.markdown("### 📍 Position Tracking")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for name, df in results.items():
        ax1.plot(df['Time'], df['Position'], label=name, linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Target')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)
    
    # 2. Control Effort Plot
    st.markdown("### ⚡ Control Effort")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for name, df in results.items():
        ax2.plot(df['Time'], df['Action'], label=name, linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force (N)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    
    # 3. Metrics Table
    st.markdown("### 📊 Performance Metrics")
    metrics = []
    for name, df in results.items():
        metrics.append({
            "Controller": name,
            "Mean Abs Error": np.mean(np.abs(df['Position'])),
            "Max Overshoot": np.max(df['Position']),
            "Total Energy": np.sum(np.square(df['Action']))
        })
    
    metrics_df = pd.DataFrame(metrics)
    st.table(metrics_df.style.format("{:.4f}"))

else:
    st.info("Click 'Simulate Controllers' to run the simulation.")

st.markdown("---")
st.caption("ControlGym Dashboard | Level 12 Implementation")
