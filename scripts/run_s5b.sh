#!/bin/bash
cd "$HOME/ScoliCMF" || exit 1
export PATH=/opt/slurm/bin:$PATH
echo "[launch $(date)] CLEAN s5b_scpga_v2 train (fixed code) on i64m1tga800u (A800)"
srun --partition=i64m1tga800u --qos=i64m1tga800u --gres=gpu:1 --time=12:00:00 --cpus-per-task=8 \
  "$HOME/.conda/envs/AgentOCR/bin/python" -u train_sa.py \
    --config configs/s5b_scpga_v2.yaml --max_steps 40000
echo "[exit $(date)] rc=$?"
