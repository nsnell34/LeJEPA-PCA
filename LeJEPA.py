import math
import random
from dataclasses import dataclass
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from transforms import Transforms

@dataclass
class LeJEPAConfig:
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 100
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    latent_dim: int = 8  
    projector_hidden_dim: int = 64
    predictor_hidden_dim: int = 32
    global_crop_scale: tuple = (0.4, 1.0)
    local_crop_scale: tuple = (0.05, 0.4)
    num_local_crops: int = 0 
    mask_ratio: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultiCropTransform:

    def __init__(self, cfg: LeJEPAConfig, use_ir: bool, dataset_path: str = None):
        self.cfg = cfg
        
        transforms = Transforms(use_ir, cfg, dataset_path)
        self.global_transform = transforms.global_transform
        self.local_transform = transforms.local_transform

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)

        locals_ = []
        for _ in range(self.cfg.num_local_crops):
            locals_.append(self.local_transform(img))

        return global1, global2, locals_


def block_mask(x, mask_ratio=0.5, min_num_blocks=1, max_num_blocks=4):

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

def get_resnet_backbone(use_ir: bool):
    backbone = models.resnet18(weights=None)

    if use_ir:
        channels = 1
    else:
        channels = 3
        
    old_conv = backbone.conv1
    backbone.conv1 = nn.Conv2d(
        in_channels=channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    backbone.fc = nn.Identity()
    return backbone

class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x_gate, x_value = x.chunk(2, dim=-1)  
        hidden = F.silu(x_gate) * x_value
        out = self.fc2(hidden)
        return out

class ProjectorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.core = SwiGLUBlock(in_dim, hidden_dim, out_dim)
        
        ## consider if training is unstable 
        #self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.core(x)
        return x


class PredictorMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.core = SwiGLUBlock(dim, hidden_dim, dim)

    def forward(self, x):
        x = self.core(x)
        return x

class LeJEPA(nn.Module):
    def __init__(self, cfg: LeJEPAConfig, use_ir: bool):
        super().__init__()
        self.cfg = cfg

        self.context_encoder = get_resnet_backbone(use_ir=use_ir)
        self.target_encoder = get_resnet_backbone(use_ir=use_ir)
        
        feat_dim = 512

        self.context_projector = ProjectorMLP(
            in_dim=feat_dim,
            hidden_dim=cfg.projector_hidden_dim,
            out_dim=cfg.latent_dim,
        )
        self.target_projector = ProjectorMLP(
            in_dim=feat_dim,
            hidden_dim=cfg.projector_hidden_dim,
            out_dim=cfg.latent_dim,
        )

        self.predictor = PredictorMLP(
            dim=cfg.latent_dim,
            hidden_dim=cfg.predictor_hidden_dim,
        )
        
        self._init_target()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _init_target(self):
        for p, q in zip(self.context_encoder.parameters(),
                        self.target_encoder.parameters()):
            q.data.copy_(p.data)

        for p, q in zip(self.context_projector.parameters(),
                        self.target_projector.parameters()):
            q.data.copy_(p.data)

    @torch.no_grad()
    def update_target(self, m: float):
        for p, q in zip(self.context_encoder.parameters(),
                        self.target_encoder.parameters()):
            q.data.mul_(m).add_(p.data, alpha=(1.0 - m))

        for p, q in zip(self.context_projector.parameters(),
                        self.target_projector.parameters()):
            q.data.mul_(m).add_(p.data, alpha=(1.0 - m))

    def forward_jepa(self, global_ctx, global_tgt):
        ctx_feat = self.context_encoder(global_ctx)     
        z_c = self.context_projector(ctx_feat)        
        z_pred = self.predictor(z_c)                  

        with torch.no_grad():
            tgt_feat = self.target_encoder(global_tgt)
            z_t = self.target_projector(tgt_feat)

        z_pred = F.normalize(z_pred, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        loss = F.mse_loss(z_pred, z_t)
        return loss

def cosine_lr(step, max_steps, base_lr, final_lr_ratio=0.01):
    if step >= max_steps:
        return base_lr * final_lr_ratio
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return base_lr * (final_lr_ratio + (1 - final_lr_ratio) * q)

def build_dataloader(cfg: LeJEPAConfig, data_root: str, use_ir: bool):
    transform = MultiCropTransform(cfg, use_ir=use_ir, dataset_path=data_root)
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader

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
    parser.add_argument("--data_root", default="/home/megrad/Documents/Github/lejepa/ds/train/rgb_images")
    args = parser.parse_args()
    
    cfg = LeJEPAConfig()
    data_root = args.data_root
    
    if (args.use_ir):
        print("Using IR")
    train_lejepa(data_root, cfg, use_ir=args.use_ir)