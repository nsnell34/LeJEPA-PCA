import torch
import math
import random

def block_mask(x, mask_ratio=0.5, min_num_blocks=1, max_num_blocks=4):
    """
    Applies random block-wise masking to spatial feature maps.
    Forces model to infer missing structure from global context
    """
    B, C, H, W = x.shape
    device = x.device
    mask = torch.ones((B, 1, H, W), device=device)

    for b in range(B):
        num_blocks = random.randint(min_num_blocks, max_num_blocks)
        for _ in range(num_blocks):
            block_area = mask_ratio * H * W / num_blocks

            aspect = random.uniform(0.5, 2.0)
            h = int(round(math.sqrt(block_area / aspect)))
            w = int(round(math.sqrt(block_area * aspect)))

            h = max(1, min(h, H))
            w = max(1, min(w, W))

            top = random.randint(0, H - h)
            left = random.randint(0, W - w)

            mask[b, :, top:top+h, left:left+w] = 0.0

    return x * mask