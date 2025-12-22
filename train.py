import subprocess
import sys
import math
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from LeJEPA import LeJEPA, MultiCropTransform
from config import LeJEPAConfig
from block_mask import block_mask

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
        
        
    """ integrate when ready 
    def sigreg_loss(z, eps=1e-4):
    # z: (B, D) normalized embeddings

    B, D = z.shape

    # Variance loss
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std))

    # Covariance loss
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    return var_loss + cov_loss

    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true", help="Use IR mode (1-channel instead of 3).")
    parser.add_argument("--data_root", default="/home/megrad/Documents/Github/lejepa/ds/train/ir_images")
    parser.add_argument("--run_eval", action="store_true")
    args = parser.parse_args()
    
    cfg = LeJEPAConfig()
    data_root = args.data_root
    
    if (args.use_ir):
        print("Using IR")
        
    ### train
    train_lejepa(data_root, cfg, use_ir=args.use_ir)
    
    ### eval
    if args.run_eval:
        make_target = "ir_all" if args.use_ir else "rgb_all"

        print(f"\nRunning eval pipeline: make {make_target}\n")

        result = subprocess.run(
            ["make", make_target],
            cwd="/home/megrad/Documents/Github/lejepa",
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        if result.returncode != 0:
            raise RuntimeError(f"`make {make_target}` failed")
    