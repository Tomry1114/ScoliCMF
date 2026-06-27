#!/bin/bash
cd ~/ScoliCMF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/.conda/envs/AgentOCR/bin/python
for cfg in s2_base s3_st s4_comp s5_scpga_identity s5b_scpga_v2; do
  echo "==== TRAIN $cfg $(date) ===="
  $PY train_sa.py --config configs/$cfg.yaml > runs/ablate_${cfg}.out 2>&1
  echo "==== DONE $cfg exit=$? $(date) ===="
done
echo ALL_ABLATE_DONE
