#!/bin/bash
export PATH=/opt/slurm/bin:$PATH
echo "[launch $(date)] matched smokes s2_base/s3_st/s4_comp/s5b_scpga_v2"
srun --partition=debug --qos=debug --gres=gpu:1 --time=00:25:00 --cpus-per-task=4 \
  bash "$HOME/ScoliCMF/scripts/_matched_inner.sh"
echo "[done $(date)] rc=$?"
