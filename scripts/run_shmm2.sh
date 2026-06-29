#!/bin/bash
cd ~/ScoliCMF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/.conda/envs/AgentOCR/bin/python
for cfg in shmm_v1 shmm_v2; do
  echo "==== TRAIN $cfg $(date) ===="
  $PY train_sa.py --config configs/$cfg.yaml > runs/ablate2_${cfg}.out 2>&1
  echo "==== DONE $cfg exit=$? $(date) ===="
done
echo SHMM2_DONE
