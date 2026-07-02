# ScoliCMF — Comparison / Baseline Experiments Plan

Goal: a single fair table comparing **9 baselines + our full ScoliCMF** on post-op spine
X-ray prediction (pre-op -> post-op), same data / split / preprocessing / metrics.

## 0. Methods & classification

| # | Method | Category | Pairing | Source | Where it runs |
|---|--------|----------|---------|--------|---------------|
| 01 | Pix2Pix  | classic paired GAN        | paired   | junyanz/pytorch-CycleGAN-and-pix2pix | official repo |
| 02 | CycleGAN | classic unpaired GAN      | unpaired | junyanz/pytorch-CycleGAN-and-pix2pix | official repo |
| 03 | CUT      | contrastive unpaired      | unpaired | taesungp/contrastive-unpaired-translation | official repo |
| 04 | RegGAN   | misalignment-robust (med) | paired*  | Kid-Liet/Reg-GAN | official repo (MOST related to ADOC) |
| 05 | ResViT   | medical Transformer       | paired   | icon-lab/ResViT | official repo |
| 06 | SynDiff  | medical diffusion         | paired/unpaired | icon-lab/SynDiff | official repo |
| 07 | BBDM     | Brownian-bridge diffusion | paired   | xuekt98/BBDM | official repo |
| 08 | FM       | flow matching (inst. v)   | paired   | OUR framework | in-framework (same DiT backbone) |
| 09 | MeanFlow | mean-velocity base        | paired   | OUR framework (= s2_base Bridge) | ALREADY TRAINED |
| 10 | ScoliCMF | MeanFlow + APTD + ADOC     | paired   | OUR framework | ALREADY TRAINED (aptd_long_fs015 / aptd_adoc) |

*RegGAN explicitly models paired misalignment via a registration net — the closest prior
work to ADOC; a key head-to-head.

Fairness tiers:
- **Tier A (in-framework, fairest):** 08 FM, 09 MeanFlow, 10 ScoliCMF share the SAME DiT
  backbone / data loader / training budget -> isolates the generative-formulation contribution.
- **Tier B (official repos):** 01-07 run in their own conda envs, adapted to our data
  (1-channel grayscale, 480x240, our 432/54 split). Compared on identical exported predictions.

## 1. Shared infrastructure (build ONCE in 00_common/)

1. **Data export** `00_common/export_data.py`:
   - Load our canonicalized 432/54 split (dataset_sa, 480x240 grayscale, [0,1]).
   - Emit THREE views from the same pixels:
     - `aligned/{train,val}/{A,B}/*.png`  (A=pre-op, B=post-op; pix2pix/ResViT/BBDM combined or A|B).
     - `unaligned/{trainA,trainB,testA,testB}/*.png`  (cycle/CUT/SynDiff unpaired loaders).
     - `cache/{xp,xq}_{train,val}.pt`  (tensors for Tier A + unified eval).
   - Fixed filenames = sorted-stem order so predictions stay case-aligned.
2. **Unified eval** `00_common/eval_compare.py`:
   - Input: a method prediction tensor `pred_val.pt` [54,1,480,240] in [0,1], RAW frame.
   - Computes SSIM / PSNR / LPIPS (reuse ../../metrics_img.py) + (optional) Cobb/CR.
   - Paired bootstrap (2000) vs ScoliCMF -> per-metric diff CI, flags EXCLUDES 0.
   - Writes `00_common/results/<method>.json` and rebuilds the master table.
3. **Master table** `00_common/make_table.py`: reads all results/*.json -> one markdown table.

Each method folder must produce exactly one artifact: `pred_val.pt` (+ its config/log).

## 2. CRITICAL decision — evaluation frame (BLOCKS the table design)

ADOC predicts the CANONICAL (acquisition-normalized) frame; baselines target the RAW
observed post-op. Two options:
- **(R) Raw-frame primary** (fairest to baselines; honest clinical target). On raw val,
  ADOC does NOT help (R70: ADOC-APTD 0.287 ~= APTD 0.294) -> headline ScoliCMF = MeanFlow+APTD;
  ADOC shown as a separate acquisition-normalization analysis/table.
- **(C) Canonical-frame primary** (ADOC home turf; needs protocol justification). All methods
  also evaluated after the SAME acquisition normalization -> ScoliCMF full shows its +0.033.
- **(Both)** report R as main + C as secondary "acquisition-normalized" table.
DECISION NEEDED before make_table.

## 3. Clinical metric (decision)

Image metrics (SSIM/PSNR/LPIPS) are ready. MIA journal wants Cobb angle / curve metrics.
Cobb needs a vertebral-landmark/seg extractor on generated images -> non-trivial extra build.
DECISION: include Cobb/CR now, or image-metrics-first + Cobb in a later round?

## 4. Per-method setup notes (Tier B)

- 01/02/03: one clone of pytorch-CycleGAN-and-pix2pix (covers pix2pix+cyclegan) + CUT repo.
  Flags: --input_nc 1 --output_nc 1 ; preprocess to 480x240 (pad H/W to /4) ; --direction AtoB.
- 04 RegGAN: paired mode with registration branch ON (the ablation that matches ADOC).
- 05 ResViT: 2-stage (pretrain + ViT) ; 1-channel ; our split.
- 06 SynDiff: adversarial diffusion ; 1-channel ; needs longest train.
- 07 BBDM: latent-or-pixel Brownian bridge ; pixel mode at 480x240 to match our pixel setup.
- Each gets its own conda env (envs likely conflict) under ~/.conda/envs/<method>.

## 5. Tier A (in-framework) — cheap, mostly done

- 09 MeanFlow = s2_base/ckpts/step_5000.pt (DONE) -> just export pred_val.pt.
- 10 ScoliCMF = aptd_long_fs015 (APTD) + aptd_adoc (APTD+ADOC) (DONE) -> export pred_val.pt
  for chosen frame.
- 08 FM = train one run: same backbone, instantaneous-velocity (t==r) flow-matching objective
  on the source-anchored path. 1 training run (~5000 steps, i64m1tga40u).

## 6. Compute plan

- Tier A: 08 FM = 1 short run; 09/10 already trained. ~0.5 GPU-day.
- Tier B: 7 methods x (env setup + 1 training each). GANs (01-04) fast (~hours each on 1 A40);
  ResViT/SynDiff/BBDM heavier (diffusion/2-stage, ~0.5-1 day each).
  Run in parallel across i64m1tga40u / i64m1tga800u cards, one tmux per method.
- Eval is shared + cheap (debug card).
- Rough total: ~4-6 GPU-days wall-clock if parallelized across ~4 cards.

## 7. Phased schedule

- **P0 infra:** build 00_common export + eval + master table; export data 3 views. (CPU/debug)
- **P1 Tier A:** export 09/10 preds, train+export 08 FM, populate table with 3 in-framework rows. (fast, validates pipeline end-to-end)
- **P2 GAN group:** clone + env + adapt + train 01 Pix2Pix, 02 CycleGAN, 03 CUT, 04 RegGAN; export preds. (parallel)
- **P3 heavy med group:** 05 ResViT, 06 SynDiff, 07 BBDM. (parallel, longest)
- **P4 finalize:** master table + paired CIs vs ScoliCMF + (optional) Cobb + per-method montages.

## 8. Fairness invariants (enforced)

- Identical 432/54 split, 480x240 grayscale, same canonicalized input pixels.
- Identical metric code + identical bootstrap resample indices across methods.
- Same eval frame for ALL methods (per section-2 decision).
- best-checkpoint selection by the SAME val criterion for every method (declare it).
- Each method: report the FULL metric set, not a flattering subset.
