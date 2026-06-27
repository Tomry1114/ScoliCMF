#!/bin/bash
cd ~/ScoliCMF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/.conda/envs/AgentOCR/bin/python
for cfg in s2_base s5b_scpga_v2; do
  echo "==== EXTEND $cfg -> 15000  $(date) ===="
  $PY train_sa.py --config configs/$cfg.yaml --max_steps 15000 --resume runs/$cfg/ckpts/step_5000.pt > runs/extend_${cfg}.out 2>&1
  echo "==== DONE $cfg exit=$? $(date) ===="
done
echo ALL_EXTEND_DONE
