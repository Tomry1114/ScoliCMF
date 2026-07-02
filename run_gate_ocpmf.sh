#!/bin/bash
cd ~/ScoliCMF
srun -p i64m1tga40u --gres=gpu:1 --time=00:30:00 --pty bash -lc "~/.conda/envs/AgentOCR/bin/python gate_ocpmf.py 2>&1 | tee gate_ocpmf.out"
