cd ~/ScoliCMF
srun -p debug --gres=gpu:1 --time=00:20:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/boot_independence.py 2>&1 | tee ~/ScoliCMF/boot_ind.out'
