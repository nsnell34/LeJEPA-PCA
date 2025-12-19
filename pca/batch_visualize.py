import os
import torch
import joblib
from PIL import Image
from LeJEPA import LeJEPA, LeJEPAConfig
from pca.PCA import colorize_image_patchwise_jepa
from torchvision import transforms

def collect_images(root, limit):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths = []
    for dp, _, fs in os.walk(root):
        fs.sort()
        for f in fs:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dp, f))
                if len(paths) >= limit:
                    return paths
    return paths

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = LeJEPAConfig(image_size=224, batch_size=128)
    model = LeJEPA(cfg, use_ir=args.use_ir).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pca = joblib.load(args.pca)
    transform = build_vis_transform(cfg, use_ir=args.use_ir)
    viz_pca_dir = os.path.join(args.out, "pca")
    viz_orig_dir = os.path.join(args.out, "original")

    os.makedirs(viz_pca_dir, exist_ok=True)
    os.makedirs(viz_orig_dir, exist_ok=True)

    img_paths = collect_images(args.root, args.num)

    for p in img_paths:
        fname = os.path.basename(p)
        raw_img = Image.open(p).convert("RGB" if args.use_ir else "L")

        model_img = raw_img.convert("L") if args.use_ir else raw_img
        x = transform(model_img).unsqueeze(0)

        vis = colorize_image_patchwise_jepa(
            model=model,
            pca=pca,
            img_tensor=x,
            device=device
        )

        raw_img.save(os.path.join(viz_orig_dir, fname))
        vis.save(os.path.join(viz_pca_dir, fname))
        
def build_vis_transform(cfg, use_ir):
    if use_ir:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4226], std=[0.1795]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4807, 0.4986, 0.4881],
                std=[0.2233, 0.2059, 0.1738],
            ),
        ])    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--use_ir", action="store_true")
    args = parser.parse_args()

    main(args)
