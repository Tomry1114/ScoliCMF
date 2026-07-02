cd ~/ScoliCMF
srun -p i64m1tga40u --gres=gpu:1 --time=02:00:00 --pty bash -lc '~/.conda/envs/AgentOCR/bin/python ~/ScoliCMF/train_icmf.py --path current --src 0 --endpoint_only 1 --out icmf_endpoint --steps 3000 --save_step 1000 2>&1 | tee ~/ScoliCMF/icmf_endpoint.out'
