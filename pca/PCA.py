from sklearn.decomposition import PCA
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

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

    return Image.fromarray(mapped_grid, mode="RGB")
