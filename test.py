import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models.dit import MFDiT
from meanflow import MeanFlow
from utils import load_config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class PairedCondGT(Dataset):
    def __init__(self, cond_root: str, target_root: str, image_size: int):
        super().__init__()
        self.cond_root = cond_root
        self.target_root = target_root
        
        self.filenames = [
            fname for fname in sorted(os.listdir(cond_root))
            if any(fname.lower().endswith(e) for e in IMG_EXTS)
            and os.path.exists(os.path.join(target_root, fname))
        ]
        
        if not self.filenames:
            raise RuntimeError(f"No paired images found in {cond_root} and {target_root}")

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        cond = Image.open(os.path.join(self.cond_root, fname)).convert("L")
        gt = Image.open(os.path.join(self.target_root, fname)).convert("L")
        return self.tf(cond), self.tf(gt), fname


@torch.no_grad()
def run_inference(cfg, model, meanflow, dataloader, device):
    outdir = cfg['inference']['outdir']
    max_eval = cfg['inference']['max_eval']
    steps = cfg['inference']['sample_steps']
    
    dirs = {k: os.path.join(outdir, k) for k in ["pred", "gt", "cond", "grid"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    total_ssim, total_psnr, n_eval = 0.0, 0.0, 0

    for cond_img, gt_img, filenames in tqdm(dataloader, desc="Inference"):
        cond_img = cond_img.to(device)
        gt_img = gt_img.to(device)

        pred = meanflow.sample_given_cond(
            model,
            cond_img=cond_img,
            sample_steps=steps,
            device=device,
            show_progress=False,
        ).detach().cpu()

        gt_cpu = gt_img.cpu()
        cond_cpu = cond_img.cpu()

        for c, g, p, fname in zip(cond_cpu, gt_cpu, pred, filenames):
            save_image(c, os.path.join(dirs["cond"], fname))
            save_image(g, os.path.join(dirs["gt"], fname))
            save_image(p, os.path.join(dirs["pred"], fname))

            grid = make_grid(torch.stack([c, g, p], dim=0), nrow=3)
            save_image(grid, os.path.join(dirs["grid"], fname))

            g_np = g.squeeze(0).numpy()
            p_np = p.squeeze(0).numpy()

            total_ssim += ssim(g_np, p_np, data_range=1.0)
            total_psnr += psnr(g_np, p_np, data_range=1.0)
            n_eval += 1

            if n_eval >= max_eval:
                break
        
        if n_eval >= max_eval:
            break

    if n_eval > 0:
        print(f"\n[Eval finished on {n_eval} images]")
        print(f"Average SSIM: {total_ssim / n_eval:.4f}")
        print(f"Average PSNR: {total_psnr / n_eval:.4f} dB")
    else:
        print("[Warn] No images evaluated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MFDiT(
        input_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        data_channels=cfg['data']['data_channels'],
        cond_channels=cfg['data']['cond_channels'],
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model']['mlp_ratio'],
    ).to(device)
    model.eval()

    ckpt_path = args.ckpt if args.ckpt else cfg['inference']['checkpoint']
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)

    meanflow = MeanFlow(
        channels=cfg['data']['data_channels'],
        image_size=cfg['data']['image_size'],
        normalizer=cfg['meanflow']['normalizer'],
        flow_ratio=cfg['meanflow']['flow_ratio'],
        time_dist=cfg['meanflow']['time_dist'],
        jvp_api=cfg['meanflow']['jvp_api'],
    )

    dataset = PairedCondGT(
        cond_root=cfg['data']['test_cond_root'],
        target_root=cfg['data']['test_target_root'],
        image_size=cfg['data']['image_size'],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['inference']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
    )

    run_inference(cfg, model, meanflow, dataloader, device)


if __name__ == "__main__":
    main()