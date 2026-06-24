import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from accelerate import Accelerator
from torchvision import transforms as T
from mydataset import PairedImageDataset
from utils import cycle, count_parameters, load_config, log_to_file
from meanflow import MeanFlow
from models.dit import MFDiT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision=cfg['training']['mixed_precision'])
    device = accelerator.device
    
    if accelerator.is_main_process:
        os.makedirs(cfg['project']['image_save_path'], exist_ok=True)
        os.makedirs(cfg['project']['checkpoint_path'], exist_ok=True)

    # Data Setup
    transform = T.Compose([
        T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
        T.ToTensor(),
    ])

    dataset = PairedImageDataset(
        cond_root=cfg['data']['train_cond_root'],
        target_root=cfg['data']['train_target_root'],
        transform_cond=transform,
        transform_target=transform
    )

    train_loader = DataLoader(
        dataset, 
        batch_size=cfg['training']['batch_size'],
        shuffle=True, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    # Accelerator handles device placement for the loader
    train_loader = accelerator.prepare(train_loader)
    train_loader = cycle(train_loader)

    # Model Setup
    model = MFDiT(
        input_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        data_channels=cfg['data']['data_channels'],
        cond_channels=cfg['data']['cond_channels'],
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model']['mlp_ratio'],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg['training']['lr']), 
        weight_decay=cfg['training']['weight_decay']
    )

    # MeanFlow Setup
    meanflow = MeanFlow(
        channels=cfg['data']['data_channels'],
        image_size=cfg['data']['image_size'],
        normalizer=cfg['meanflow']['normalizer'],
        flow_ratio=cfg['meanflow']['flow_ratio'],
        time_dist=cfg['meanflow']['time_dist'],
        jvp_api=cfg['meanflow']['jvp_api'],
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        print(f"[Model] Parameters: {count_parameters(model)/1e6:.2f}M")

    global_step = 0
    running_loss, running_mse = 0.0, 0.0
    pbar = tqdm(range(cfg['training']['n_steps']), dynamic_ncols=True, disable=not accelerator.is_local_main_process)
    
    model.train()
    for _ in pbar:
        cond_img, target_img = next(train_loader)
        
        # Calculate loss
        loss, mse_val = meanflow.loss(model, target_img, cond_img=cond_img)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        running_mse += mse_val.item()

        # Periodic Logging
        if accelerator.is_main_process and global_step % cfg['training']['log_step'] == 0:
            avg_loss = running_loss / cfg['training']['log_step']
            avg_mse = running_mse / cfg['training']['log_step']
            log_to_file(cfg['project']['log_file'], global_step, avg_loss, avg_mse, optimizer.param_groups[0]["lr"])
            running_loss, running_mse = 0.0, 0.0

        # Periodic Sampling/Visualization
        if global_step % cfg['training']['sample_step'] == 0:
            if accelerator.is_main_process:
                model.eval()
                with torch.no_grad():
                    # Sampling 4 pairs
                    c_vis, g_vis = cond_img[:4], target_img[:4]
                    y_hat = meanflow.sample_given_cond(
                        accelerator.unwrap_model(model),
                        cond_img=c_vis,
                        sample_steps=20,
                        device=device
                    )
                    # Grid layout: Condition | Ground Truth | Prediction
                    grid = make_grid(torch.cat([c_vis, g_vis, y_hat], dim=0), nrow=4)
                    save_image(grid, f"{cfg['project']['image_save_path']}/step_{global_step}.png")
                model.train()
            accelerator.wait_for_everyone()

        # Save Checkpoint
        if global_step % cfg['training']['save_step'] == 0:
            if accelerator.is_main_process:
                ckpt_path = os.path.join(cfg['project']['checkpoint_path'], f"step_{global_step}.pt")
                accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()