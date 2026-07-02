cd ~/ScoliCMF
srun -p i64m1tga40u --gres=gpu:1 --time=00:25:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/gate_icmf.py 2>&1 | tee ~/ScoliCMF/gate_icmf.out'
