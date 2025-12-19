import math
import random
from dataclasses import dataclass
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from LeJEPA import LeJEPA, MultiCropTransform

@dataclass
class LeJEPAConfig:
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 20
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    latent_dim: int = 64  
    projector_hidden_dim: int = 256
    predictor_hidden_dim: int = 128
    global_crop_scale: tuple = (0.4, 1.0)
    local_crop_scale: tuple = (0.05, 0.4)
    num_local_crops: int = 4 
    mask_ratio: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def build_dataloader(cfg: LeJEPAConfig, data_root: str, use_ir: bool):
    transform = MultiCropTransform(cfg, use_ir=use_ir, dataset_path=data_root)
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    return loader

def block_mask(x, mask_ratio=0.5, min_num_blocks=1, max_num_blocks=4):
    """
    Applies random block-wise masking to spatial feature maps.
    Forces model to infer missing structure from global context
    """
    B, C, H, W = x.shape
    device = x.device
    mask = torch.ones((B, 1, H, W), device=device)

    for b in range(B):
        num_blocks = random.randint(min_num_blocks, max_num_blocks)
        for _ in range(num_blocks):
            block_area = mask_ratio * H * W / num_blocks

            aspect = random.uniform(0.5, 2.0)
            h = int(round(math.sqrt(block_area / aspect)))
            w = int(round(math.sqrt(block_area * aspect)))

            h = max(1, min(h, H))
            w = max(1, min(w, W))

            top = random.randint(0, H - h)
            left = random.randint(0, W - w)

            mask[b, :, top:top+h, left:left+w] = 0.0

    return x * mask

def cosine_lr(step, max_steps, base_lr, final_lr_ratio=0.0001):
    if step >= max_steps:
        return base_lr * final_lr_ratio
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return base_lr * (final_lr_ratio + (1 - final_lr_ratio) * q)

def train_lejepa(data_root: str, cfg: LeJEPAConfig, use_ir: bool):
    device = cfg.device

    loader = build_dataloader(cfg, data_root, use_ir=use_ir)
    model = LeJEPA(cfg, use_ir=use_ir).to(device)

    parameters = list(model.context_encoder.parameters()) \
                 + list(model.context_projector.parameters()) \
                 + list(model.predictor.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    max_steps = cfg.epochs * len(loader)
    step = 0

    model.train()
    for epoch in range(cfg.epochs):
        ### loader yields idx, views, labels
        ### views include (global1, global2, locals)
        for batch_idx, (views, _) in enumerate(loader):
            global1, global2, locals_ = views

            global1 = global1.to(device, non_blocking=True)
            global2 = global2.to(device, non_blocking=True)
            
            masked_ctx = block_mask(
                global1,
                mask_ratio=cfg.mask_ratio,
            )

            loss = model.forward_jepa(masked_ctx, global2)

            lr = cosine_lr(step, max_steps, cfg.base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target(m=cfg.ema_momentum)

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{cfg.epochs}] "
                    f"Step [{batch_idx}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {lr:.6f}"
                )

            step += 1
            if use_ir:
                save_ext = "ir"
            else:
                save_ext = "rgb"

        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "step": step,
            },
            f"lejepa_{save_ext}.pth",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true", help="Use IR mode (1-channel instead of 3).")
    parser.add_argument("--data_root", default="/home/megrad/Documents/Github/lejepa/ds/train/ir_images")
    args = parser.parse_args()
    
    cfg = LeJEPAConfig()
    data_root = args.data_root
    
    if (args.use_ir):
        print("Using IR")
    train_lejepa(data_root, cfg, use_ir=args.use_ir)
    