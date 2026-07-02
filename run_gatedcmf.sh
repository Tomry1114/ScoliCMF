cd ~/ScoliCMF
srun -p debug --gres=gpu:1 --time=00:25:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/gate_dcmf.py 2>&1 | tee ~/ScoliCMF/gate_dcmf.out'
