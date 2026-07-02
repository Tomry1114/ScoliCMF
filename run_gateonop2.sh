cd ~/ScoliCMF
srun -p i64m1tga40u --gres=gpu:1 --time=01:00:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/gate_onop2.py 2>&1 | tee ~/ScoliCMF/gate_onop2.out'
