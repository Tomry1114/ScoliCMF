# MOTFM (03) Setup — reproducibility (repo/ is gitignored)

**Repo**: milad1378yz/MOTFM (Medical OT Flow Matching, MICCAI'25) -> compare_exp/03_motfm/repo
**Env**: scoliagent (torch 2.5.1+cu121, matches MOTFM pin). Installed via `~/.conda/envs/scoliagent/bin/pip install -e .` (adds flow_matching==1.0.10, monai_generative==0.2.3, pytorch-lightning==2.5.6; numpy->1.26.4; torch untouched).

## 3 env fixes applied to repo (not in git)
1. `use_flash_attention: false` in configs/scoli*.yaml (xformers not installed).
2. `torch.backends.cudnn.enabled = False` at top of trainer.py AND inferer.py — cu121 cuDNN 9.1.9 throws CUDNN_STATUS_NOT_INITIALIZED on A800 nodes; native conv works (proven by bare-conv sanity).
3. (Task fit) mask_conditioning mode = ControlNet with preop as continuous `mask` cond, postop as `image` target. 2D, in/out 1ch. No class (cross_attention_dim: null).

## Data
`build_motfm_pkl.py` -> compare_exp/03_motfm/data/scoli.pkl = {train:432, valid:54, test:54}, each {image=postop(1,480,240 float[0,1]), mask=preop, class:0, metadata:{stem}}.

## Config (configs/scoli.yaml, from mask_conditioning.yaml)
pickle_path=scoli.pkl, batch_size=8, num_epochs=300, checkpoint_dir=compare_exp/03_motfm/ckpts. UNet fully-conv (5 levels, div-by-16 ok for 480x240). ~1 epoch=45s on A800 (cudnn off) -> 300ep ~3.75h.

## Run
- Train: `srun -p i64m1tga800u --gres=gpu:1 ~/.conda/envs/scoliagent/bin/python trainer.py --config_path configs/scoli.yaml`
- Infer (TODO after train): `inferer.py --config_path configs/scoli.yaml --model_path <ckpt> --num_samples 54` -> .pkl of generated postop; convert to preds/<stem>.png (map via metadata.stem) -> `python eval_common.py 03_motfm`.

## Status
Smoke (1 epoch) PASSED: train/loss 1.19, val/loss 1.20, val samples exported cleanly. Full train (300ep) launched.
