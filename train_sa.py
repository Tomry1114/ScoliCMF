"""Train source-anchored MeanFlow (S2: L_span+L_end ; S3: +L_ST). Reuses train.py skeleton.
Usage: python train_sa.py --config configs/sc_pixel.yaml [--max_steps N]
"""
import os
import argparse
import copy
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from accelerate import Accelerator

from utils import cycle, count_parameters, load_config, log_to_file
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from models.sc_dit import SCDiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sc_pixel.yaml")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_steps = args.max_steps or cfg["training"]["n_steps"]
    seed = cfg["training"].get("seed", 0)
    torch.manual_seed(seed)

    acc = Accelerator(mixed_precision=cfg["training"]["mixed_precision"])
    device = acc.device
    if acc.is_main_process:
        os.makedirs(cfg["project"]["image_save_path"], exist_ok=True)
        os.makedirs(cfg["project"]["checkpoint_path"], exist_ok=True)

    H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    ds = PairedSpineDataset(root=os.path.expanduser(cfg["data"]["root"]),
                            split=cfg["data"]["split"], size=(H, W),
                            canon_dir=cfg["data"]["canon_dir"],
                            split_file=cfg["data"].get("split_file"),
                            augment=cfg["data"].get("augment", False))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True)
    loader = cycle(acc.prepare(loader))

    cond_module = None
    if cfg["model"].get("cond", "base") == "scpga":
        from sc_pga import SCPGA
        cond_module = SCPGA(img_size=(H, W), dim=cfg["model"]["dim"], patch_size=cfg["model"]["patch_size"],
                            J=cfg["model"].get("J", 12), Kg=cfg["model"].get("Kg", 4),
                            Kt=cfg["model"].get("Kt", 2), beta=cfg["model"].get("beta", 40.0),
                            eta=cfg["model"].get("eta", 4.0), proj=cfg["model"].get("proj", "v2"),
                            dyn_off=cfg["model"].get("dyn_off", False))
    model = SCDiT(img_size=(H, W), patch_size=cfg["model"]["patch_size"],
                  data_channels=cfg["data"]["data_channels"], cond_channels=cfg["data"]["cond_channels"],
                  dim=cfg["model"]["dim"], depth=cfg["model"]["depth"],
                  num_heads=cfg["model"]["num_heads"], mlp_ratio=cfg["model"]["mlp_ratio"],
                  cond_module=cond_module, decode_head=cfg["model"].get("decode_head", "conv"),
                  xpre_mode=cfg["model"].get("xpre_mode", "full"))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                            weight_decay=cfg["training"]["weight_decay"])
    mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"], sigma_m=cfg["meanflow"]["sigma_m"],
                                lambda_end=cfg["meanflow"]["lambda_end"], rho_end=cfg["meanflow"]["rho_end"],
                                lambda_st=cfg["meanflow"].get("lambda_st", 0.0),
                                st_mode=cfg["meanflow"].get("st_mode", "detach"),
                                jvp_api=cfg["meanflow"].get("jvp_api", "autograd"),
                                lambda_comp=cfg["meanflow"].get("lambda_comp", 0.0),
                                lambda_roll=cfg["meanflow"].get("lambda_roll", 0.0),
                                comp_ramp_steps=cfg["meanflow"].get("comp_ramp_steps", 2000),
                                lambda_time=cfg["meanflow"].get("lambda_time", 0.0),
                                lambda_end_roll=cfg["meanflow"].get("lambda_end_roll", 0.0),
                                end_roll_steps=cfg["meanflow"].get("end_roll_steps", 2))
    model, opt = acc.prepare(model, opt)
    ema = copy.deepcopy(acc.unwrap_model(model)).to(device).eval()
    for _p in ema.parameters():
        _p.requires_grad_(False)
    ema_decay = cfg["meanflow"].get("ema_decay", 0.999)
    if acc.is_main_process:
        print(f"[model] {count_parameters(model)/1e6:.2f}M params; steps={n_steps}; "
              f"data={len(ds)} pairs {H}x{W}; lambda_st={mf.lambda_st} st_mode={mf.st_mode}")

    run, cnt = defaultdict(float), defaultdict(int)
    pbar = tqdm(range(n_steps), dynamic_ncols=True, disable=not acc.is_local_main_process)
    model.train()
    for step in pbar:
        x_pre, x_post = next(loader)
        loss, logs = mf.loss(model, x_pre, x_post, teacher=ema, step=step)  # wrapped model -> DDP syncs
        opt.zero_grad(); acc.backward(loss); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), acc.unwrap_model(model).parameters()):
                pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
        for k, v in logs.items():
            run[k] += v.item(); cnt[k] += 1
        gs = step + 1
        if acc.is_main_process and gs % cfg["training"]["log_step"] == 0:
            avg = {k: run[k] / max(1, cnt[k]) for k in run}
            pbar.set_postfix(**{k: round(x, 4) for k, x in avg.items()})
            if "m_dyn_rms" in avg:
                print("[diag step %d] m_dyn/m_static rms=%.4f l_span=%.4f" % (
                    gs, avg["m_dyn_rms"]/max(1e-9, avg["m_static_rms"]), avg.get("l_span", 0)), flush=True)
            log_to_file(cfg["project"]["log_file"], gs, avg.get("l_span", 0.0),
                        avg.get("l_st", avg.get("l_end", 0.0)), opt.param_groups[0]["lr"])
            run, cnt = defaultdict(float), defaultdict(int)
        if acc.is_main_process and gs % cfg["training"]["sample_step"] == 0:
            model.eval()
            with torch.no_grad():
                xc, xg = x_pre[:4], x_post[:4]
                pred = mf.sample(acc.unwrap_model(model), xc, steps=cfg["training"]["sample_steps"])
                grid = make_grid(torch.cat([xc, xg, pred], dim=0), nrow=4)
                save_image(grid, f"{cfg['project']['image_save_path']}/step_{gs}.png")
            model.train()
        if acc.is_main_process and gs % cfg["training"]["save_step"] == 0:
            ckpt = {"model": acc.unwrap_model(model).state_dict(), "ema": ema.state_dict(),
                    "optimizer": opt.state_dict(), "step": gs, "config": cfg, "seed": seed}
            torch.save(ckpt, os.path.join(cfg["project"]["checkpoint_path"], f"step_{gs}.pt"))


if __name__ == "__main__":
    main()
