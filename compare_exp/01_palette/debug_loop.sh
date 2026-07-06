#!/bin/bash
cd ~/ScoliCMF/compare_exp/01_palette/repo
N=${1:-5}
for i in $(seq 1 $N); do
  LATEST=$(ls experiments/*/checkpoint/*_Network.pth 2>/dev/null | while read f; do e=$(basename "$f" | sed "s/_Network.pth//"); echo "$e $f"; done | sort -n | tail -1 | cut -d" " -f2 | sed "s/_Network.pth//")
  if [ -z "$LATEST" ]; then RESUME="pretrained/celebahq_inpaint/200"; else RESUME="$LATEST"; fi
  ~/.conda/envs/AgentOCR/bin/python -c "import json; d=json.load(open(\"config/finetune_scoli.json\")); d[\"path\"][\"resume_state\"]=\"$RESUME\"; json.dump(d,open(\"config/finetune_scoli.json\",\"w\"),indent=2)"
  echo "===== PAL CHUNK $i/$N (resume=$RESUME) $(date +%H:%M:%S) ====="
  /opt/slurm/bin/srun -p debug --gres=gpu:1 --time=00:28:00 bash -lc "~/.conda/envs/AgentOCR/bin/python run.py -c config/finetune_scoli.json -p train -b 4 --gpu_ids 0" 2>&1 | tr "\r" "\n" | grep -iE "Saving the self|epoch .*[0-9]|error|traceback|Training state" | tail -5
  echo "----- chunk $i done $(date +%H:%M:%S); latest ckpt epoch: $(ls experiments/*/checkpoint/*_Network.pth 2>/dev/null | while read f; do basename \"$f\" | sed s/_Network.pth//; done | sort -n | tail -1) -----"
done
echo "PAL_CHUNKS_DONE"
