import torch
import os
from PIL import Image
import torch.nn.functional as F
from LeJEPA import LeJEPA, MultiCropTransform
from config import LeJEPAConfig
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

@torch.no_grad()
def evaluate_dataset(model, cfg, device, root, use_ir):

    transform = MultiCropTransform(cfg, use_ir=use_ir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    img_paths = []
    for dp, _, fs in os.walk(root):
        fs.sort()
        for f in fs:
            if f.lower().endswith(exts):
                img_paths.append(os.path.join(dp, f))

    if not img_paths:
        raise ValueError(f"No images found in {root}")

    total_cos = 0.0

    for p in img_paths:
        img = Image.open(p).convert("L" if use_ir else "RGB")

        global1, global2, _ = transform(img)

        x1 = global1.unsqueeze(0).to(device)
        x2 = global2.unsqueeze(0).to(device)

        z1 = model.encode(x1)
        z2 = model.encode(x2)

        assert not torch.allclose(z1, z2), "Identical embeddings — eval bug"

        cos_sim = F.cosine_similarity(z1, z2, dim=-1).item()
        total_cos += cos_sim

    return total_cos / len(img_paths)


if __name__ == "__main__":
    
    ### if running directly in command line:
    #   run as a module for paths to work 
    #   i.e python3 -m eval.lejepa_eval --val_root <root> etc... 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true")
    parser.add_argument("--val_root", required=True)
    args = parser.parse_args()

    if args.use_ir:
        ckpt = "ckpts/lejepa_ir.pth"
    else:
        ckpt = "ckpts/lejepa_rgb.pth"

    model, cfg, device = load_model(ckpt, use_ir=args.use_ir)

    avg_cos = evaluate_dataset(
        model,
        cfg,
        device,
        args.val_root,
        use_ir=args.use_ir,
    )

    print("\n==== EVAL ====")
    print(f"Val root: {args.val_root}")
    print(f"Avg cosine similarity: {avg_cos:.6f}")
    print("================================\n")
