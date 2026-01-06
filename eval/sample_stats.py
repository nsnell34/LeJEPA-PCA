from PIL import Image
import argparse
import torch
import os
import random

from LeJEPA import LeJEPA, LeJEPAConfig
from pca.transforms import build_transform

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
N_IMAGES = 64

def sample_images(root):
    paths = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dp, f))

    if len(paths) < N_IMAGES:
        raise ValueError(f"Found {len(paths)} images, need at least {N_IMAGES}")

    return random.sample(paths, N_IMAGES)

@torch.no_grad
def emb_stats(image_paths, use_ir, ckpt):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    cfg = LeJEPAConfig(image_size=224, batch_size=128)
    model = LeJEPA(cfg, use_ir=use_ir).to(device)

    ckpt = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = build_transform(cfg, use_ir=use_ir)
    
    images = []
    for p in image_paths:
        img = Image.open(p).convert("L" if use_ir else "RGB")
        images.append(transform(img))
        
    images = torch.stack(images).to(device)
    
    z = model.encode(images)
    
    var_per_dim = z.var(dim=0)
    mean_var = var_per_dim.mean().item()
    min_var = var_per_dim.min().item()
    max_var = var_per_dim.max().item()
 
    return {
        "mean_variance": mean_var,
        "min_variance": min_var,
        "max_variance": max_var
    }
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--use_ir", action="store_true")
    args = parser.parse_args()
    
    if (args.use_ir):
        ckpt = "/home/megrad/Documents/Github/lejepa/ckpts/lejepa_ir.pth"
    else:
        ckpt = "/home/megrad/Documents/Github/lejepa/ckpts/lejepa_rgb.pth"
    
    image_paths = sample_images(args.img_dir)
    stats = emb_stats(image_paths, args.use_ir, ckpt)

    print("=== stats ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")
        
if __name__ == "__main__":
    main()