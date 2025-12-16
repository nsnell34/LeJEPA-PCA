import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

from PCA import extract_feature_map
from LeJEPA import LeJEPA, LeJEPAConfig


def dataset_patch_embeddings(model, dataset, number, layer="layer3", device="cuda"):
    """
    Extracts patch-level feature embeddings from a dataset using a specified
    ResNet layer and saves them as a single NumPy array.

    Features are extracted per spatial location, flattened across the dataset,
    and written to disk for offline analysis (e.g. global PCA fitting).
    """
    all_patches = []
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device, non_blocking=True)

            tokens, _ = extract_feature_map(model, imgs, layer=layer)
            B, N, C = tokens.shape
            patches = tokens.reshape(B * N, C)
            all_patches.append(patches.cpu().numpy())

    combined_patches = np.concatenate(all_patches, axis=0)
    filename = f"combined_resnet_patches_{number}.npy"
    np.save(filename, combined_patches)
    print(f"Saved: {filename}  shape={combined_patches.shape}")
    return combined_patches


def build_dataset(data_root, img_size=224, use_ir=False):
    if use_ir:
        tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(data_root, transform=tfm)
    print(f"Loaded dataset: {len(dataset)} images from {data_root}")
    return dataset


def load_model(ckpt_path, device="cuda", use_ir=False):
    cfg = LeJEPAConfig(image_size=224, batch_size=1)
    model = LeJEPA(cfg, use_ir=use_ir).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ir", action="store_true")
    parser.add_argument("--output_id", type=int, default=0)
    args = parser.parse_args()

    USE_IR = args.use_ir
    OUTPUT_ID = args.output_id

    if USE_IR:
        CHECKPOINT = "/home/megrad/Documents/Github/lejepa/ckpts/lejepa_ir.pth"
        DATA_ROOT = "/home/megrad/Documents/Github/lejepa/ds/train/ir_images"
    else:
        CHECKPOINT = "/home/megrad/Documents/Github/lejepa/ckpts/lejepa_rgb.pth"
        DATA_ROOT = "/home/megrad/Documents/Github/lejepa/ds/train/rgb_images"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LAYER = "layer3"

    dataset = build_dataset(DATA_ROOT, use_ir=USE_IR)
    model = load_model(CHECKPOINT, device=DEVICE, use_ir=USE_IR)

    combined = dataset_patch_embeddings(
        model=model,
        dataset=dataset,
        number=OUTPUT_ID,
        layer=LAYER,
        device=DEVICE,
    )

    print("Combined embedding matrix:", combined.shape)


if __name__ == "__main__":
    main()
