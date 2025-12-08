import torch
import os
from glob import glob
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from LeJEPA import LeJEPA, LeJEPAConfig, MultiCropTransform, block_mask
import argparse
from PCA import colorize_image_resnet


# ----------------------------
#  Load Model
# ----------------------------
def load_model(ckpt_path: str, device: str = None, use_ir: bool = False):
    cfg = LeJEPAConfig(image_size=224, batch_size=128)

    if device is None:
        device = cfg.device

    model = LeJEPA(cfg, use_ir=use_ir).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, cfg, device


# ----------------------------
#  Eval Transform
# ----------------------------
def build_eval_transform(cfg: LeJEPAConfig, use_ir: bool):

    if use_ir:
        normalize = transforms.Normalize(mean=[0.4226], std=[0.1795])
        channels = 1
    else:
        normalize = transforms.Normalize(
            mean=[0.4807, 0.4986, 0.4881],
            std=[0.2233, 0.2059, 0.1738]
        )
        channels = 3

    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.Grayscale(num_output_channels=channels),
        transforms.ToTensor(),
        normalize,
    ])


# ----------------------------
#  Extract Embedding (fixed)
# ----------------------------
def extract_embedding(model: LeJEPA, cfg, device, image_path, use_ir):
    transform = build_eval_transform(cfg, use_ir)

    img = Image.open(image_path).convert("L" if use_ir else "RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.context_encoder(x)
        feat = feat.mean(dim=(2, 3))  # <<< FIXED
        z = model.context_projector(feat)
        z = F.normalize(z, dim=-1)

    return z.cpu()


# ----------------------------
#  JEPA Loss (fixed)
# ----------------------------
def jepa_single_image_loss(model: LeJEPA, cfg, device, img_path, use_ir):

    transform = MultiCropTransform(cfg, use_ir=use_ir)
    img = Image.open(img_path).convert("L" if use_ir else "RGB")

    global1, global2, locals_ = transform(img)

    global1 = global1.unsqueeze(0).to(device)
    global2 = global2.unsqueeze(0).to(device)

    masked_ctx = block_mask(global1, mask_ratio=cfg.mask_ratio)

    with torch.no_grad():

        # ---- context ----
        ctx_feat = model.context_encoder(masked_ctx)
        ctx_feat = ctx_feat.mean(dim=(2, 3))     # <<< FIXED
        z_c = model.context_projector(ctx_feat)
        z_pred = model.predictor(z_c)

        # ---- target ----
        tgt_feat = model.target_encoder(global2)
        tgt_feat = tgt_feat.mean(dim=(2, 3))     # <<< FIXED
        z_t = model.target_projector(tgt_feat)

        # ---- normalize ----
        z_pred = F.normalize(z_pred, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        # ---- mse ----
        loss = F.mse_loss(z_pred, z_t)

        sqdist = loss.item() * cfg.latent_dim
        cosθ = 1.0 - (sqdist / 2.0)

    return loss.item(), sqdist, cosθ


# ----------------------------
#  Evaluate Dataset
# ----------------------------
def evaluate_dataset(model, cfg, device, root, use_ir, viz_out=None):

    exts = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP", ".tif", ".tiff")

    img_paths = [
        os.path.join(dp, f)
        for dp, _, fs in os.walk(root)
        for f in fs
        if f.endswith(exts)
    ]

    if not img_paths:
        raise ValueError(f"No images found in {root}")

    total_loss = total_sq = total_cos = 0.0
    pca_global = None

    print(f"\nEvaluating {len(img_paths)} images in {root} ...")

    for p in img_paths:
        loss, sqdist, cosθ = jepa_single_image_loss(model, cfg, device, p, use_ir)
        total_loss += loss
        total_sq += sqdist
        total_cos += cosθ

        # PCA Visualization
        if viz_out:
            filename = os.path.basename(p).replace(".", "_")
            out_path = os.path.join(viz_out, filename + "_pca.png")

            vis_img, pca_global = visualize_single_image(
                model, cfg, device, p,
                use_ir=use_ir,
                save_path=out_path,
                pca=pca_global
            )

    N = len(img_paths)
    return total_loss / N, total_sq / N, total_cos / N


# ----------------------------
#  PCA Visualization Wrapper
# ----------------------------
def visualize_single_image(model, cfg, device, img_path, use_ir, save_path=None, pca=None):

    transform = build_eval_transform(cfg, use_ir)
    img = Image.open(img_path).convert("L" if use_ir else "RGB")
    x = transform(img).unsqueeze(0).to(device)

    vis_img, pca = colorize_image_resnet(
        model=model,
        img_tensor=x,
        pca=pca,
        device=device
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis_img.save(save_path)

    return vis_img, pca


# ----------------------------
#  Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true")
    parser.add_argument("--val_root", type=str)
    parser.add_argument("--viz_out", action="store_true")
    args = parser.parse_args()

    if args.use_ir:
        model_ckpt = "/home/megrad/Documents/Github/lejepa/ckpts/lejepa_ir.pth"
        save_path = "/home/megrad/Documents/Github/lejepa/ds/test_images/output/ir"
    else:
        model_ckpt = "/home/megrad/Documents/Github/lejepa/ckpts/lejepa_rgb.pth"
        save_path = "/home/megrad/Documents/Github/lejepa/ds/test_images/output/rgb"

    model, cfg, device = load_model(model_ckpt, use_ir=args.use_ir)

    if args.val_root:
        avg_loss, avg_sqdist, avg_cos = evaluate_dataset(
            model, cfg, device, args.val_root,
            use_ir=args.use_ir,
            viz_out=save_path if args.viz_out else None
        )

        print("\n==== Validation Metrics ====")
        print(f"Val root: {args.val_root}")
        print(f"Avg Loss: {avg_loss:.6f}")
        print(f"Avg Squared Distance: {avg_sqdist:.6f}")
        print(f"Avg Cosine Similarity: {avg_cos:.6f}")
        print("============================\n")
