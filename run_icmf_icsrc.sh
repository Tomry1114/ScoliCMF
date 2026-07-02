cd ~/ScoliCMF
srun -p i64m1tga40u --gres=gpu:1 --time=02:00:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/train_icmf.py --path ic --src 1 --out icmf_icsrc --steps 3000 --save_step 1000 2>&1 | tee ~/ScoliCMF/icmf_icsrc.out'
