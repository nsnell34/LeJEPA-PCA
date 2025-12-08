import torch
import os
from glob import glob
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from LeJEPA import LeJEPA, LeJEPAConfig, MultiCropTransform, block_mask
import argparse

def load_model(ckpt_path: str, device: str = None, use_ir: bool = False):
    cfg = LeJEPAConfig(image_size=224, batch_size=128) 
    if device is None:
        device = cfg.device

    model = LeJEPA(cfg, use_ir=use_ir).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, device

def build_eval_transform(cfg: LeJEPAConfig, use_ir: bool):
     
    if (use_ir):
        normalize = transforms.Normalize(
            mean = [0.4226], 
            std = [0.1795],   
        )
        channels = 1
    else:
        normalize = transforms.Normalize(
            mean = [0.4807, 0.4986, 0.4881], 
            std = [0.2233, 0.2059, 0.1738],   
        )
        channels = 3

    eval_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.Grayscale(num_output_channels=channels),  
        transforms.ToTensor(),                      
        normalize,
    ])

    return eval_transform

def extract_embedding(model: LeJEPA, cfg: LeJEPAConfig, device: str, image_path: str, use_ir: bool):
    eval_transform = build_eval_transform(cfg, use_ir=use_ir)
    if use_ir:
        img = Image.open(image_path).convert("L")
    else:
        img = Image.open(image_path).convert("RGB")
        
    x = eval_transform(img).unsqueeze(0).to(device, non_blocking=True)
    
    with torch.no_grad():
        feat = model.context_encoder(x)
        z = model.context_projector(feat)
        z = F.normalize(z, dim=-1)
        
    return z.cpu()

def jepa_single_image_loss(model: LeJEPA, cfg: LeJEPAConfig, device: str, img_path: str, use_ir: bool):
    transform = MultiCropTransform(cfg, use_ir=use_ir)
    if use_ir:
        img = Image.open(img_path).convert("L")
    else:
        img = Image.open(img_path).convert("RGB")

    global1, global2, locals_ = transform(img)

    global1 = global1.unsqueeze(0).to(device)
    global2 = global2.unsqueeze(0).to(device)

    masked_ctx = block_mask(global1, mask_ratio=cfg.mask_ratio)

    with torch.no_grad():
        ctx_feat = model.context_encoder(masked_ctx)
        z_c = model.context_projector(ctx_feat)
        z_pred = model.predictor(z_c)

        tgt_feat = model.target_encoder(global2)
        z_t = model.target_projector(tgt_feat)

        z_pred = F.normalize(z_pred, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        loss = F.mse_loss(z_pred, z_t)

        sqdist = loss.item() * cfg.latent_dim
        cosθ = 1.0 - sqdist / 2.0

    return loss.item(), sqdist, cosθ

def evaluate_dataset(model: LeJEPA, cfg: LeJEPAConfig, device: str, root: str, use_ir: bool):
    
    exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".tif", ".tiff"]
    img_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.endswith(e) for e in exts):
                img_paths.append(os.path.join(dirpath, fname))

    if not img_paths:
        raise ValueError(f"No images found in {root}")

    total_loss = 0.0
    total_sq = 0.0
    total_cos = 0.0

    print(f"\nEvaluating {len(img_paths)} images in {root} ...")

    for p in img_paths:
        loss, sqdist, cosθ = jepa_single_image_loss(model, cfg, device, p, use_ir=use_ir)
        total_loss += loss
        total_sq += sqdist
        total_cos += cosθ

    N = len(img_paths)
    return total_loss / N, total_sq / N, total_cos / N


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true")
    parser.add_argument("--val_root", type=str)
    args = parser.parse_args()
    
    if args.use_ir:
        model_ckpt = "/home/megrad/Documents/Github/lejepa/lejepa_ir.pth"
    else:
        model_ckpt = "/home/megrad/Documents/Github/lejepa/lejepa_rgb.pth"

    model, cfg, device = load_model(model_ckpt, device=None, use_ir=args.use_ir)

    if args.val_root:
        avg_loss, avg_sqdist, avg_cos = evaluate_dataset(
            model, cfg, device, args.val_root, use_ir=args.use_ir
        )
        print("\n==== Validation Metrics ====")
        print(f"Val root: {args.val_root}")
        print(f"Avg Loss: {avg_loss:.6f}")
        print(f"Avg Squared Distance: {avg_sqdist:.6f}")
        print(f"Avg Cosine Similarity: {avg_cos:.6f}")
        print("============================\n")