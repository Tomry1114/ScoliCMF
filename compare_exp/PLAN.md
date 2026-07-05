# Comparison Experiments — Preoperative → Postoperative Scoliosis X-ray Synthesis

**Task**: given a pre-op frontal whole-spine X-ray (cond), predict the post-op X-ray (target). Paired.
**Data**: `~/ScoliCMF/data/Spine.../train/{preop,postop}_standardized`, 480×240 grayscale, splits/{train(432),val(54),test(54)}.txt.
**Protocol (identical for ALL methods)**:
- train on `train.txt`, evaluate on `val.txt` (test held out for final).
- metrics: **SSIM ↑, PSNR ↑, LPIPS ↓** (metrics_img.py; same as our method).
- report best-checkpoint numbers; for generative few-step methods report NFE used.
- a method's predictions are written to `compare_exp/<method>/preds/<stem>.png` → scored by `eval_common.py`.

## Method list (rows of the comparison table)

| # | Method | Type | Source | Status |
|---|---|---|---|---|
| 01 | Pix2Pix | paired GAN | junyanz/pytorch-CycleGAN-and-pix2pix | TODO |
| 02 | CycleGAN | unpaired GAN | junyanz/pytorch-CycleGAN-and-pix2pix | TODO |
| 03 | CUT | contrastive unpaired | taesungp/contrastive-unpaired-translation | TODO |
| 04 | RegGAN | registration GAN (medical) | Kid-Liet/Reg-GAN | TODO |
| 05 | ResViT | transformer medical synth | icon-lab/ResViT | TODO |
| 06 | SynDiff | diffusion medical translation | icon-lab/SynDiff | TODO |
| 07 | BBDM | Brownian-bridge diffusion | xuekt98/BBDM | TODO |
| 08 | FlowMatching | conditional flow matching | in-framework (cmf/, flow-matching loss) | TODO |
| 09 | MeanFlow | conditional MeanFlow (= our base, vanilla attn) | cmf/ --attn vanilla | DONE (0.183/0.434) |
| 10 | **Ours (ScoliCMF)** | Cond. MeanFlow + ViT3 TTT mixer | cmf/ --attn ttt | DONE (0.182/0.432, faster conv) |

## Ablations (our method, separate table)
- attn: vanilla / ttt(mixer) / ttt+cpe(full DiT3) — CPE HURTS (0.161 < 0.176), keep mixer only.
- inner_lr (eta): 0 (capacity) / 0.25 — eta control shows gain is the TTT online-update mechanism, not capacity.
- text: on / off / shuffle (factorized region+direction embeddings) — text ~= no gain (phenotype subset of x_pre).

## Priority / GPU note
- All methods on same 432/54 split, 480×240. GPU: ce483 i64m1tga40u/800u via srun+tmux.
- Suggested first wave (cheapest, strongest baselines): 01 Pix2Pix, 09 MeanFlow(done), 08 FlowMatching, 07 BBDM.
- Second wave (medical-specific): 04 RegGAN, 05 ResViT, 06 SynDiff.
- 02/03 (unpaired) are weaker baselines for a paired task — include for completeness.
