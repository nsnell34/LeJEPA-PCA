import os
import torch
import joblib
from PIL import Image
from LeJEPA import LeJEPA, LeJEPAConfig
from eval.lejepa_eval import build_eval_transform
from pca.PCA import colorize_image_patchwise_jepa


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
    transform = build_eval_transform(cfg, use_ir=args.use_ir)

    os.makedirs(args.out, exist_ok=True)

    img_paths = collect_images(args.root, args.num)

    for p in img_paths:
        img = Image.open(p).convert("L" if args.use_ir else "RGB")
        x = transform(img).unsqueeze(0)

        vis = colorize_image_patchwise_jepa(
            model=model,
            pca=pca,
            img_tensor=x,
            device=device
        )

        vis.save(os.path.join(args.out, os.path.basename(p)))


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
