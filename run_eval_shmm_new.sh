#!/bin/bash
cd ~/ScoliCMF
PY=~/.conda/envs/AgentOCR/bin/python
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
$PY eval_ablation.py --configs shmm_dct,shmm_v1,shmm_v2 2>&1
echo "DRIVER_DONE EXIT=$?"
