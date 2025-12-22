import subprocess
import sys
import math
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from LeJEPA import LeJEPA, MultiCropTransform, forward_jepa
from config import LeJEPAConfig

def build_dataloader(cfg: LeJEPAConfig, data_root: str, use_ir: bool):
    transform = MultiCropTransform(cfg, use_ir=use_ir, dataset_path=data_root)
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )

def train_lejepa(data_root: str, cfg: LeJEPAConfig, use_ir: bool):
    device = cfg.device

    loader = build_dataloader(cfg, data_root, use_ir=use_ir)
    model = LeJEPA(cfg, use_ir=use_ir).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
    )
    
    step = 0
    save_ext = "ir" if use_ir else "rgb"
    
    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0

        print(f"\n=== Epoch [{epoch + 1}/{cfg.epochs}] ===")

        for batch_idx, (views, _) in enumerate(loader):
            global1, global2, _ = views
            global1 = global1.to(device)
            global2 = global2.to(device)

            loss = forward_jepa(model, global1, global2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if batch_idx % 25 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {avg_loss:.6f}"
                )

        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1} finished | Avg Loss: {epoch_loss:.6f}")

        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "step": step,
            },
            f"/home/megrad/Documents/Github/lejepa/ckpts/lejepa_{save_ext}.pth",
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_eval", action="store_true")
    args = parser.parse_args()

    cfg = LeJEPAConfig()

    train_lejepa(args.data_root, cfg, use_ir=args.use_ir)

    if args.run_eval:
        target = "ir_all" if args.use_ir else "rgb_all"
        result = subprocess.run(
            ["make", target],
            cwd=".",
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            raise RuntimeError(f"`make {target}` failed")