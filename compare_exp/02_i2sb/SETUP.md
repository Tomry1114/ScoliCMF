# I2SB (02) Setup — FINETUNE from ADM ImageNet-256 (repo/ gitignored)

**Repo**: NVlabs/I2SB (Image-to-Image Schrodinger Bridge). **Env**: scoliagent (torch 2.5.1) — NOT the repo's i2sb env (conda env create failed: user .condarc tsinghua conda-forge URL is broken 'https//' missing colon). Installed pip: torch-ema prefetch_generator colored-traceback easydict ipdb termcolor rich lmdb wandb.
**Pretrained**: ADM `256x256_diffusion_uncond.pt` (2.2GB, auto-downloaded); with --cond-x1 it uses `256x256_diffusion_cond_fixedsigma`.

## Adaptation (paired preop->postop, NO corruption operator)
- KEY: Runner.sample_batch with `opt.corrupt=="mixture"` takes the 3-tuple path `(clean, corrupt, y)=next(loader)` -> uses our preop DIRECTLY as x1 (never calls corrupt_method). So a paired dataset + --corrupt mixture = real paired bridge.
- `dataset/scoli_paired.py` PairedScoli: returns (postop=clean/x0, preop=corrupt/x1, label=0), 256x256, [-1,1], 3ch.
- `train_scoli.py`: reuses train.py option parser + Runner; forces corrupt=mixture, corrupt_method=None, PairedScoli; --cond-x1 conditions on preop.

## Patches to repo (not in git)
1. train_scoli.py: `torch.backends.cudnn.enabled=False` (cu121 cuDNN on A800).
2. runner.py L153: skip DDP when global_size==1 (NCCL 'CUDA driver insufficient' on single GPU).
3. runner.py L160: `torchmetrics.Accuracy()` -> `Accuracy(task='multiclass',num_classes=1000)` (torchmetrics API).

## Train / eval
- Finetune: `train_scoli.py --name scoli --corrupt mixture --cond-x1 --image-size 256 --batch-size 32 --microbatch 8 --num-itr 10000`. save@5000, eval(DDPM 999-step)@500/3000/6000/9000. results/scoli/.
- Sample val (TODO): use sample.py on the trained ckpt -> generated postop -> preds/<stem>.png -> eval_common.py 02_i2sb.

## Status
Smoke PASSED (train_it 1/20 loss 1.36, ADM loaded, ckpt saved, DDPM eval ran). Full finetune (10000 itr) launched on 40u.
