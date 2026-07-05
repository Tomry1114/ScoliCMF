# Comparison Experiments — Preop → Postop Scoliosis X-ray Synthesis

**Task**: pre-op X-ray (cond) → post-op X-ray (target). Paired, 480×240 grayscale.
**Data**: `~/ScoliCMF/data/Spine.../train/{preop,postop}_standardized`, splits/{train(432),val(54),test(54)}.txt.
**Protocol (IDENTICAL for all)**: train on train.txt, eval on val.txt; metrics SSIM↑ / PSNR↑ / LPIPS↓ (eval_common.py, same as ours). Each method writes `<method>/preds/<stem>.png` → scored by `python eval_common.py <method_dir>`.

## Baselines (4) + Ours

| # | Method | Venue | Repo | Status |
|---|---|---|---|---|
| 01 | **Palette** | image-to-image diffusion | (Saharia'22) e.g. Janspiry/Palette-Image-to-Image-Diffusion-Models | TODO |
| 02 | **I2SB** | Image-to-Image Schrödinger Bridge | NVlabs/I2SB (arxiv 2302.05872) | TODO |
| 03 | **BBDM** | Brownian Bridge Diffusion | xuekt98/BBDM | TODO |
| 04 | **MOTFM** | Medical OT Flow Matching (MICCAI'25) | milad1378yz/MOTFM | TODO |
| — | **Ours (ScoliCMF)** | Cond. MeanFlow + ViT3 TTT mixer | ~/ScoliCMF_cmf (cmf/ --attn ttt) | val ~0.182/0.432 |

All 4 baselines are conditional generative image-to-image methods (diffusion / Schrödinger-bridge / OT-flow) — the right same-family comparison for a paired X-ray translation task.

## Workflow per baseline
1. `git clone <repo>` into `<method>/repo`.
2. adapt data loader → our paired preop/postop 480×240 (grayscale, 1-ch) + train/val split.
3. train on train.txt (ce483 srun+tmux, i64m1tga40u/800u).
4. generate val predictions → `<method>/preds/<stem>.png`.
5. `python eval_common.py <method_dir>` → SSIM/PSNR/LPIPS row.

## Order
- First: validate eval_common on OURS (dump our val preds, confirm ~0.182 matches training eval).
- Then baselines lightest→heaviest: 04 MOTFM (flow, few-step) & 02 I2SB (bridge) → 03 BBDM → 01 Palette (many-step diffusion, heaviest).
