from dataclasses import dataclass
import torch
@dataclass
class LeJEPAConfig:
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    epochs: int = 20
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    latent_dim: int = 64  
    projector_hidden_dim: int = 256
    predictor_hidden_dim: int = 128
    global_crop_scale: tuple = (0.4, 1.0)
    local_crop_scale: tuple = (0.05, 0.4)
    num_local_crops: int = 4 
    mask_ratio: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"