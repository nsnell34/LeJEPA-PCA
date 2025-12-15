from sklearn.decomposition import PCA
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

def extract_feature_map(model, x, layer="layer3"):
    fmap = model.context_encoder(x, return_layer=layer)
    B, C, H, W = fmap.shape

    tokens = fmap.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
    return tokens, (H, W)

def colorize_image_resnet(model, img_tensor, pca=None, layer="layer3", device="cuda"):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        tokens, (Hf, Wf) = extract_feature_map(model, img_tensor, layer)
        B, N, C = tokens.shape

        feats = tokens.reshape(B * N, C).cpu().numpy()

        if pca is None:
            pca = PCA(n_components=3)
            pca.fit(feats)

        mapped = pca.transform(feats)

        mapped -= mapped.min(axis=0, keepdims=True)
        mapped /= (mapped.max(axis=0, keepdims=True) + 1e-9)
        mapped = (mapped * 255).astype(np.uint8)

        mapped_img = mapped.reshape(Hf, Wf, 3)
        mapped_img = Image.fromarray(mapped_img, "RGB")
        mapped_img = mapped_img.resize(
            (img_tensor.shape[3], img_tensor.shape[2]),
            Image.BILINEAR
        )

    return mapped_img

def compute_global_pca(npy_paths, n_components=3):
    arrays = [np.load(p) for p in npy_paths]
    feats = np.concatenate(arrays, axis=0)
    print(f"Global PCA fitting on {feats.shape[0]} tokens with dimension {feats.shape[1]}")
    pca = PCA(n_components=n_components)
    pca.fit(feats)
    return pca

def dataset_patch_embeddings(model, dataset, number, layer="layer3", device="cuda"):
    
    all_patches = []
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )
    
    with torch.no_grad():
        for imgs in loader: 
            imgs = imgs.to(device, non_blocking=True)
            
            tokens = extract_feature_map(model, imgs, layer=layer)
            
            B, N, C = tokens.shape
            patches = tokens.reshape(B * N, C) 

            all_patches.append(patches.cpu().numpy())
            
    combined_patches = np.concatenate(all_patches, axis=0)
    filename = "combined_patches_"+str(number)+"_.npy"
    np.save(filename, combined_patches)
    
    return combined_patches