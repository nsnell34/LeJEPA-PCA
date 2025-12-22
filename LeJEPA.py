import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transforms import Transforms
from config import LeJEPAConfig

class MultiCropTransform:

    def __init__(self, cfg: LeJEPAConfig, use_ir: bool, dataset_path: str = None):
        self.cfg = cfg
        
        transforms = Transforms(use_ir, cfg, dataset_path)
        self.global_transform = transforms.global_transform
        self.local_transform = transforms.local_transform

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)

        locals_ = []
        for _ in range(self.cfg.num_local_crops):
            locals_.append(self.local_transform(img))

        return global1, global2, locals_

class ResNetBackbone(nn.Module):
    """
    ResNet-based convolutional backbone for feature extraction.

    Wraps a torchvision ResNet (18 or 50) as a fully convolutional encoder,
    removing the classification head and returning a spatial feature map.

    Forward order:
        (B, C, H, W)
            → conv1 (stride 2)
            → bn1 → relu → maxpool (stride 2)
            → layer1 → layer2 → layer3 → layer4
            → output: (B, out_dim, H/32, W/32)

    Supports RGB (3-channel) or IR/grayscale (1-channel) input.
    """
    def __init__(self, use_ir: bool, resnet_type="resnet18"):
        super().__init__()

        if resnet_type == "resnet18":
            backbone = models.resnet18(weights=None)
            self.out_dim = 512
        elif resnet_type == "resnet50":
            backbone = models.resnet50(weights=None)
            self.out_dim = 2048
        else:
            raise ValueError("Unsupported ResNet type")
        
        in_channels = 1 if use_ir else 3
        old_conv = backbone.conv1
        
        # Adapt ResNet's first conv to 1- or 3-channel (IR vs RGB)
        backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Output spatial sizes are input-dependent. for 224×224 input:
        self.layer1 = backbone.layer1  # ≈ H/4  × W/4   (56×56)
        self.layer2 = backbone.layer2  # ≈ H/8  × W/8   (28×28)
        self.layer3 = backbone.layer3  # ≈ H/16 × W/16  (14×14)
        self.layer4 = backbone.layer4  # ≈ H/32 × W/32  (7×7)

        self.avgpool = backbone.avgpool
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class SwiGLUBlock(nn.Module):
    """
    Feedforward block using SwiGLU activation.

    Applies a gated linear transformation:
        in_dim → 2 * hidden_dim → hidden_dim → out_dim
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        gate, value = x.chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)

class ProjectorMLP(nn.Module):
    """
    Projection head mapping backbone features into the latent space.

    Uses a SwiGLU block followed by LayerNorm 
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.core = SwiGLUBlock(in_dim, hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.core(x))
    

class LeJEPA(nn.Module):
    """
    SIGReg-based Latent Joint-Embedding Predictive Architecture (LeJEPA).
    """
    def __init__(self, cfg: LeJEPAConfig, use_ir: bool):
        super().__init__()
        self.cfg = cfg

        self.encoder = ResNetBackbone(use_ir=use_ir)

        self.projector = ProjectorMLP(
            in_dim=self.encoder.out_dim,
            hidden_dim=cfg.projector_hidden_dim,
            out_dim=cfg.latent_dim,
        )

    def encode(self, x):
        feat = self.encoder(x)          # [batch_size, feat_dim, H, W]
        feat = feat.mean(dim=(2,3))     # global avg pool
        z = self.projector(feat)
        return F.normalize(z, dim=-1)

def forward_jepa(model, x1, x2, lambda_sigreg=1.0):
    """
    Computes the JEPA + SIGReg training loss for two global views.
    """
    z1 = model.encode(x1)
    z2 = model.encode(x2)

    # JEPA alignment loss
    loss_jepa = F.mse_loss(z1, z2)

    # SIGReg on both views
    loss_sigreg = sigreg_loss(z1) + sigreg_loss(z2)

    return loss_jepa + lambda_sigreg * loss_sigreg

def sigreg_loss(z, eps=1e-4):
    B, D = z.shape

    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std))

    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    return var_loss + cov_loss
