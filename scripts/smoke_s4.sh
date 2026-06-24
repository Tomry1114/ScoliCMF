#!/bin/bash
cd "$HOME/ScoliCMF" || exit 1
export PATH=/opt/slurm/bin:$PATH
echo "[launch $(date)] S4 (L_comp+L_roll+EMA) srun debug --qos=debug ..."
srun --partition=debug --qos=debug --gres=gpu:1 --time=00:30:00 --cpus-per-task=4 \
  "$HOME/.conda/envs/AgentOCR/bin/python" -u train_sa.py \
    --config configs/s4_comp.yaml --max_steps 400
echo "[done $(date)] rc=$?"
