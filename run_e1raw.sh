cd ~/ScoliCMF
srun -p i64m1tga40u --gres=gpu:1 --time=04:00:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/train_2x2.py --mode direct --target raw --out ind_direct_raw --steps 5000 --save_step 1000 2>&1 | tee ~/ScoliCMF/ind_direct_raw.out'
