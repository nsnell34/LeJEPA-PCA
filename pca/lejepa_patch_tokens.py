import os
import torch
import numpy as np
import torch.nn.functional as F
from LeJEPA import LeJEPA, LeJEPAConfig
import argparse
from eval.lejepa_eval import build_eval_transform
from PIL import Image
import argparse

@torch.no_grad()
def dump_jepa_patch_tokens(model, cfg, device, root, use_ir, out_path, max_images=None):
    
    ### we use lejepa, not transformers, so we need patch tokens for pca fitting
    
    transform = build_eval_transform(cfg, use_ir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    tokens_all = []
    count = 0

    for dp, _, fs in os.walk(root):
        fs.sort()
        for f in fs:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue

            img = Image.open(os.path.join(dp, f)).convert("L" if use_ir else "RGB")

            x = transform(img).unsqueeze(0).to(device)

            fmap = model.context_encoder(x)
            _, C, Hf, Wf = fmap.shape

            # ------------------------------------------------------------------
            # Spatial feature map → patch tokens
            #
            # permute dimensions
            #   (B, C, Hf, Wf) → (B, Hf, Wf, C)
            #
            # reshape
            #   (B, Hf, Wf, C) → (B * Hf * Wf, C)
            #
            # This is the CNN-equivalent of ViT patch tokens.
            # ------------------------------------------------------------------
            tokens = (
                fmap
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
            )

            tokens = model.context_projector(tokens)

            tokens = F.normalize(tokens, dim=-1)

            tokens_all.append(tokens.cpu().numpy())

            count += 1
            if max_images and count >= max_images:
                break

    tokens_all = np.concatenate(tokens_all, axis=0)
    np.save(out_path, tokens_all)

    print(f"Saved JEPA tokens: {tokens_all.shape} → {out_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_ir", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LeJEPAConfig(image_size=224, batch_size=128)

    model = LeJEPA(cfg, use_ir=args.use_ir).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dump_jepa_patch_tokens(
        model, cfg, device,
        args.root,
        args.use_ir,
        args.out,
    )