#!/bin/bash
cd ~/ScoliCMF/compare_exp/02_i2sb/repo
N=${1:-3}
for i in $(seq 1 $N); do
  CK=""
  [ -f results/scoli/latest.pt ] && CK="--ckpt scoli"
  echo "===== CHUNK $i/$N (resume=$CK) $(date +%H:%M:%S) ====="
  /opt/slurm/bin/srun -p debug --gres=gpu:1 --time=00:28:00 bash -lc "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ~/.conda/envs/scoliagent/bin/python train_scoli.py --name scoli --corrupt mixture --cond-x1 --image-size 256 --batch-size 4 --microbatch 4 --num-itr 100000 $CK --n-gpu-per-node 1" 2>&1 | grep -viE "warn|deprecat|B/s|%\|" | grep -iE "train_it|Saved|error|traceback|Loaded pretrained" | tail -8
  echo "----- chunk $i done $(date +%H:%M:%S); ckpt: $(ls -la results/scoli/latest.pt 2>/dev/null | awk "{print \$5,\$6,\$7,\$8}") -----"
done
echo "I2SB_CHUNKS_DONE"
