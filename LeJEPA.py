import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transforms import Transforms
from train import LeJEPAConfig

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

    Supports RGB (3-channel) or IR/grayscale (1-channel) input by replacing
    the first convolution layer.

    ResNet variants:
        - resnet18 → out_dim = 512
        - resnet50 → out_dim = 2048
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

    def forward(self, x, return_layer="layer4"):
    
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if return_layer == "layer1":
            return x

        x = self.layer2(x)
        if return_layer == "layer2":
            return x

        x = self.layer3(x)
        if return_layer == "layer3":
            return x

        x = self.layer4(x)
        return x
    
class SwiGLUBlock(nn.Module):
    """
    Feedforward block using SwiGLU activation.

    Applies a gated linear transformation:
        in_dim → 2 * hidden_dim → hidden_dim → out_dim

    The input is split into gate and value tensors; the gate is activated
    with SiLU and multiplied elementwise with the value projection.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x_gate, x_value = x.chunk(2, dim=-1)  
        hidden = F.silu(x_gate) * x_value
        out = self.fc2(hidden)
        return out

class ProjectorMLP(nn.Module):
    """
    Projection head mapping backbone features into the latent space.

    Uses a SwiGLU block followed by LayerNorm to transform per-sample
    features from backbone dimensionality into the JEPA latent dimension.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.core = SwiGLUBlock(in_dim, hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.core(x)
        return x


class PredictorMLP(nn.Module):
    """
    Predictor network operating in latent space.

    Maps latent representations to predicted latent targets using a
    SwiGLU block. Input and output dimensionality are the same.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.core = SwiGLUBlock(dim, hidden_dim, dim)

    def forward(self, x):
        x = self.core(x)
        return x

class LeJEPA(nn.Module):
    """
    Latent Joint-Embedding Predictive Architecture (LeJEPA).

    Consists of:
    - A context encoder and projector (trainable)
    - A target encoder and projector (EMA-updated, frozen)
    - A predictor operating in latent space

    The model predicts target latent representations from context views
    and optimizes a regression loss between normalized latent embeddings.
    """
    def __init__(self, cfg: LeJEPAConfig, use_ir: bool):
        super().__init__()
        self.cfg = cfg

        ### use one backbone for both 
        self.context_encoder = ResNetBackbone(use_ir=use_ir)
        self.target_encoder = ResNetBackbone(use_ir=use_ir)

        feat_dim = self.context_encoder.out_dim

        self.context_projector = ProjectorMLP(
            in_dim=feat_dim,
            hidden_dim=cfg.projector_hidden_dim,
            out_dim=cfg.latent_dim,
        )
        self.target_projector = ProjectorMLP(
            in_dim=feat_dim,
            hidden_dim=cfg.projector_hidden_dim,
            out_dim=cfg.latent_dim,
        )

        self.predictor = PredictorMLP(
            dim=cfg.latent_dim,
            hidden_dim=cfg.predictor_hidden_dim,
        )
        
        self._init_target()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _init_target(self):
        for p, q in zip(self.context_encoder.parameters(),
                        self.target_encoder.parameters()):
            q.data.copy_(p.data)

        for p, q in zip(self.context_projector.parameters(),
                        self.target_projector.parameters()):
            q.data.copy_(p.data)

    @torch.no_grad()
    def update_target(self, m: float):
        for p, q in zip(self.context_encoder.parameters(),
                        self.target_encoder.parameters()):
            q.data.mul_(m).add_(p.data, alpha=(1.0 - m))

        for p, q in zip(self.context_projector.parameters(),
                        self.target_projector.parameters()):
            q.data.mul_(m).add_(p.data, alpha=(1.0 - m))
            
    @torch.no_grad()
    def forward_embed(self, x):
        feat = self.context_encoder(x)
        feat = feat.mean(dim=(2,3))
        z = self.context_projector(feat)
        return F.normalize(z, dim=-1)

    def forward_jepa(self, global_ctx, global_tgt):
        ## normalize scalar to have result shape (B, C) instead of (B, C, H, W)  
        ## essentially implements global average pooling
        ctx_feat = self.context_encoder(global_ctx)   
        ctx_feat = ctx_feat.mean(dim=(2, 3)) 
        z_c = self.context_projector(ctx_feat)        
        z_pred = self.predictor(z_c)                  

        with torch.no_grad():
            tgt_feat = self.target_encoder(global_tgt)
            tgt_feat = tgt_feat.mean(dim=(2, 3)) 
            z_t = self.target_projector(tgt_feat)

        z_pred = F.normalize(z_pred, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        loss = F.mse_loss(z_pred, z_t)
        return loss
