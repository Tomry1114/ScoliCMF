#!/bin/bash
cd "$HOME/ScoliCMF"
for cfg in s2_base s3_st s4_comp s5b_scpga_v2; do
  echo "===== $cfg ====="
  "$HOME/.conda/envs/AgentOCR/bin/python" -u train_sa.py --config configs/$cfg.yaml --max_steps 150 2>&1 \
    | grep -E "model\]|Step: 150|Error|Traceback|nan|NaN|\[diag" | tail -4
done
