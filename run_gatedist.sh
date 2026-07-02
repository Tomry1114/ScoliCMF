cd ~/ScoliCMF
srun -p debug --gres=gpu:1 --time=00:28:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/gate_dist.py 2>&1 | tee ~/ScoliCMF/gate_dist.out'
