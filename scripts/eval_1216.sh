#!/bin/bash
cd ~/ScoliCMF
PY=~/.conda/envs/AgentOCR/bin/python
for s in 12000 16000; do
  echo "=========== step_${s} ==========="
  $PY eval_img.py --ckpt runs/s5b_scpga_v2/ckpts/step_${s}.pt --splits train,val --n 80
done
echo "ALL_DONE"
