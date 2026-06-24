#!/bin/bash
#SBATCH --job-name=scpga_long
#SBATCH --partition=i64m1tga800u
#SBATCH --qos=i64m1tga800u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/hpc2hdd/home/ce483/ScoliCMF/sa_long_run.log
cd "$HOME/ScoliCMF"
echo "[sbatch start $(date)] node=$(hostname)"
~/.conda/envs/AgentOCR/bin/python -u train_sa.py --config configs/sc_pixel_long.yaml
echo "[sbatch end $(date)] rc=$?"
