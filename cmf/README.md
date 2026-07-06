# ScoliCMF: Conditional Mean Flow Generation for Medical Image Synthesis 


## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ and PyTorch installed. The training pipeline uses Hugging Face `accelerate` for distributed Multi-GPU training.

```bash
# Clone the repository
git clone https://github.com/anonymized/ScoliCMF.git
cd ScoliCMF

# Install dependencies
pip install torch torchvision
pip install accelerate einops tqdm scikit-image pyyaml
```

*Optional: Configure `accelerate` for your specific multi-GPU environment:*

```bash
accelerate config
```

## 📂 Dataset Preparation

Organize your paired datasets (Condition images and Target/GT images) into separate directories for training and testing. Ensure that paired images share the exact same filename.

```text
dataset_root/
├── train/
│   ├── cond/      # Preoperative images or Source Contrast
│   └── target/    # Postoperative images or Target Contrast
└── test/
    ├── cond/
    └── target/
```

Update the dataset paths in the `config.yaml` file accordingly.

## ⚙️ Configuration (`config.yaml`)

All training, inference, and model hyperparameters are centralized in `config.yaml`. Before running any scripts, ensure your data paths and parameters are correctly set:

```yaml
data:
  train_cond_root: "/path/to/train/cond"
  train_target_root: "/path/to/train/target"
  test_cond_root: "/path/to/test/cond"
  test_target_root: "/path/to/test/target"
  image_size: 256
```

## 🚀 Training

ScoliCMF is trained from scratch by directly optimizing the average-velocity network without any teacher model or distillation.

To start distributed training across multiple GPUs (e.g., 4 GPUs), run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train.py --config config.yaml
```

*During training, checkpoints and sampling grids will be saved periodically to the paths defined in `config.yaml`.*

## 🧪 Inference and Evaluation

The testing script performs low-NFE inference (default: 20 steps, easily adjustable in config) and calculates structural fidelity metrics such as SSIM and PSNR.

**Standard Evaluation (using config defaults):**

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config config.yaml
```

**Evaluate a specific checkpoint:**

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config config.yaml --ckpt checkpoints/step_500000.pt
```

Results, including predictions, ground truths, and combined grids, will be saved to the `outdir` specified in your configuration.
```
## Note on ViT3/TTT (honesty)
This uses a **ViT3-style TTT mixer**, NOT the full official ViT3/DiT3. Deviations: RMSNorm (vs official LayerNorm), inner_lr=0.25 (vs ~1.0), CPE zero-init, fixed (frozen) sincos pos. Phenotype-text options: --text_emb {factorized,joint,both}, --inject {global,spatial,both}.
