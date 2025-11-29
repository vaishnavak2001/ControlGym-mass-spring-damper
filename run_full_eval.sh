#!/bin/bash
echo "Running Full Evaluation..."

echo "1. Benchmarking Classical vs RL..."
python controllers/benchmark.py --n_steps 500

echo "2. Running System Identification..."
python system_id/estimate_parameters.py

echo "3. Evaluating PPO Model..."
if [ -f "results/final_model.zip" ]; then
    python src/eval_ppo_msd.py --model_path results/final_model.zip
else
    echo "Warning: results/final_model.zip not found. Skipping PPO eval."
fi

echo "Evaluation complete. Check plots/ and controllers/ folders."
