#!/bin/bash
cd "$HOME/ScoliCMF" || exit 1
export PATH=/opt/slurm/bin:$PATH
echo "[launch $(date)] S5a (SC-PGA v2 + all losses) srun debug --qos=debug ..."
srun --partition=debug --qos=debug --gres=gpu:1 --time=00:30:00 --cpus-per-task=4 \
  "$HOME/.conda/envs/AgentOCR/bin/python" -u train_sa.py \
    --config configs/s5b_scpga_v2.yaml --max_steps 300
echo "[done $(date)] rc=$?"
