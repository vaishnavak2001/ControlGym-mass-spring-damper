# Setup Guide

## Prerequisites
*   **Python 3.8+**
*   **Git**

## Windows Setup

1.  **Clone the repository:**
    ```cmd
    git clone https://github.com/yourusername/controlgym-rl.git
    cd controlgym-rl
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Run Setup Script:**
    ```cmd
    python src/setup_env.py
    ```
    *Alternatively:* `pip install -r requirements.txt`

4.  **Run Smoke Test:**
    ```cmd
    python tests/smoke_test.py
    ```

## Linux / macOS Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/controlgym-rl.git
    cd controlgym-rl
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Run Setup Script:**
    ```bash
    python src/setup_env.py
    ```

4.  **Run Smoke Test:**
    ```bash
    python tests/smoke_test.py
    ```

## Running the Project

*   **Train PPO:** `python src/train_ppo_msd.py`
*   **Run Dashboard:** `streamlit run dashboard/app.py`
*   **Benchmark:** `python controllers/benchmark.py`
