import argparse
import torch
import numpy as np
from PIL import Image
import joblib
import torch.nn.functional as F
from LeJEPA import LeJEPA, LeJEPAConfig
from eval.lejepa_eval import build_eval_transform

@torch.no_grad()
def colorize_image_patchwise_jepa(model, pca, img_tensor, device="cuda", normalize=True):

    model.eval()
    img_tensor = img_tensor.to(device)
    fmap = model.context_encoder(img_tensor)    # (1, C, Hf, Wf)
    _, C, Hf, Wf = fmap.shape

    tokens = (fmap.permute(0, 2, 3, 1).reshape(-1, C))

    tokens = model.context_projector(tokens)    # (Hf*Wf, latent_dim)

    if normalize:
        tokens = F.normalize(tokens, dim=-1)

    tokens_np = tokens.cpu().numpy()
    mapped = pca.transform(tokens_np)           # (Hf*Wf, 3)

    mapped -= mapped.min(axis=0, keepdims=True)
    mapped /= mapped.max(axis=0, keepdims=True) + 1e-9
    mapped = (mapped * 255).astype(np.uint8)

    mapped_grid = mapped.reshape(Hf, Wf, 3)
    
    out = Image.fromarray(mapped_grid, mode="RGB")
    
    # smoothing
    # out = out.resize((img_tensor.shape[-1], img_tensor.shape[-2]), Image.BILINEAR)

    return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_ir", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = LeJEPAConfig(image_size=224, batch_size=128)
    model = LeJEPA(cfg, use_ir=args.use_ir).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pca = joblib.load(args.pca)

    transform = build_eval_transform(cfg, use_ir=args.use_ir)

    img = Image.open(args.img).convert("L" if args.use_ir else "RGB")
    img_tensor = transform(img).unsqueeze(0)

    vis = colorize_image_patchwise_jepa(
        model=model,
        pca=pca,
        img_tensor=img_tensor,
        device=device
    )

    vis.save(args.out)