# Palette (01) Setup — FINETUNE from CelebA-HQ inpainting checkpoint (repo/ gitignored)

**Repo**: Janspiry/Palette-Image-to-Image-Diffusion-Models. **Env**: AgentOCR (torch 2.8) + pip tensorboardX clean-fid.
**Pretrained**: CelebA-HQ inpainting `200_Network.pth` (718MB, gdrive). in_channel=6/out=3/256x256 == matches our translation I/O (cond 3ch + y_t 3ch).

## Finetune approach (clean)
- Moved `200.state` -> `200.state.bak` so resume_training() finds no training-state -> starts epoch 0, fresh optimizer, loads ONLY network weights (load_network strict=False). = clean finetune from CelebA net init.
- ColorizationDataset REUSED (no new code): mask=None => pure conditional translation (network.forward else-branch). gray/=preop(cond), color/=postop(gt).

## Fixes
- `np.str/np.int/np.float` -> `str/int/float` in data/dataset.py (numpy 2.x removed np.str).

## Data (build_palette_data.py)
datasets/scoli/{gray,color}/{idx}.png (256x256 RGB), train.flist(0..431), val.flist(100000..100053), idx2stem.json. 486 imgs.

## Config (config/finetune_scoli.json, from inpainting_celebahq.json)
train=ColorizationDataset(train.flist), test=ColorizationDataset(val.flist), resume_state=pretrained/celebahq_inpaint/200, n_epoch=500, batch=4, beta_schedule train2000/test1000 steps, mse_loss. ~34s/epoch on A800 -> 500ep ~4.8h.

## Run
- Finetune: `run.py -c config/finetune_scoli.json -p train -b 4 --gpu_ids 0`
- Test (TODO): `run.py -c config/finetune_scoli.json -p test` -> generates val outputs; convert to preds/<stem>.png via idx2stem -> eval_common.py 01_palette.

## Status
Smoke PASSED (loads CelebA weights, trains 3.13 it/s, epoch 1). Full finetune (500ep) launched.
