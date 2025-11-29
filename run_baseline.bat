@echo off
echo Starting PPO Baseline Training...
python src/train_ppo_msd.py --total_timesteps 5000 --env_id toy --log_dir results/baseline_ppo
echo Training complete.
pause
