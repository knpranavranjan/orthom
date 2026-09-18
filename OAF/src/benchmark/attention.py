"""PHASE 3 - PART D : spatial-attention experimental architecture.

Architecture (matches the literature design we are studying):

    input 3x224x224
        |
    CNN backbone (pretrained, classifier removed)   -> feature maps  F  [B, C, h, w]
        |
    SpatialAttention(F)                              -> attn map     A  [B, 1, h, w]  in (0,1)
        |
    F_att = F * A            (broadcast over channels; residual: F_att = F*(1+A))
        |
    Global Average Pool                              -> vector       [B, C]
        |
    Dropout(p) -> Linear(C -> 5)                     -> logits       [B, 5]   (KL0..KL4)

No Softmax here: CrossEntropyLoss consumes raw logits. softmax(logits) is applied
only for probability reporting / inference (see engine.evaluate).

SpatialAttention block
----------------------
* input  : F  [B, C, h, w]   (C, h, w depend on the backbone; for ResNet18 @224 -> 512 x 7 x 7)
* compute: channel-wise avg-pool and max-pool over C  -> [B, 2, h, w]
           7x7 conv (2 -> 1, padding 3)               -> [B, 1, h, w]
           Sigmoid                                    -> A in (0, 1)
* output : A  [B, 1, h, w]  ; applied as  F_att = F * (1 + A)  (residual gating)
* params : 2*1*7*7 + 1 (bias) = 99   (conv2d(2,1,7))
* integration point: immediately after the final backbone feature map, before
  global pooling and the classifier.

This is the CBAM spatial sub-module (Woo et al., 2018) in isolation - a single
learnable 7x7 conv over the [avg;max] channel descriptor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import BuiltModel, _last_conv_name


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention: A = sigmoid(conv7x7([avgpool_c(F); maxpool_c(F)]))."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_c = x.mean(dim=1, keepdim=True)                  # [B,1,h,w]
        max_c = x.amax(dim=1, keepdim=True)                  # [B,1,h,w]
        a = self.act(self.conv(torch.cat([avg_c, max_c], dim=1)))
        return a                                             # [B,1,h,w] in (0,1)


class SpatialAttentionClassifier(nn.Module):
    """backbone (features only) -> spatial attention -> GAP -> dropout -> Linear(5)."""

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int = 5,
                 dropout: float = 0.2, residual: bool = True):
        super().__init__()
        self.backbone = backbone                     # returns [B, C, h, w]
        self.attention = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim
        self.residual = residual
        self._last_attn = None                       # cached for Grad-CAM / QA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)                                 # [B,C,h,w]
        if f.dim() != 4:
            raise RuntimeError(f"backbone must return 4D feature maps, got {tuple(f.shape)}")
        a = self.attention(f)                                # [B,1,h,w]
        self._last_attn = a.detach()
        f = f * (1.0 + a) if self.residual else f * a
        v = torch.flatten(self.pool(f), 1)                   # [B,C]
        return self.classifier(self.dropout(v))              # [B,5] logits


# --------------------------------------------------------------------------- #
def _timm_feature_backbone(timm_id: str, in_chans: int, pretrained: bool):
    """timm model with classifier + global pool removed -> returns [B,C,h,w]."""
    import timm
    m = timm.create_model(timm_id, pretrained=pretrained, in_chans=in_chans,
                          num_classes=0, global_pool="")
    feat_dim = m.num_features
    return m, feat_dim


_ATTENTION_MODELS = {
    # name -> (base benchmark model name, timm id used for the feature backbone)
    "resnet18_spatial_attention": ("resnet18", "resnet18"),
    "convnext_tiny_spatial_attention": ("convnext_tiny", "convnext_tiny"),
}


def build_attention_model(name: str, num_classes: int = 5, in_chans: int = 3,
                          pretrained: bool = True, dropout: float = 0.2) -> BuiltModel:
    if name not in _ATTENTION_MODELS:
        raise KeyError(f"unknown attention model {name!r}; choices: {list(_ATTENTION_MODELS)}")
    base_name, timm_id = _ATTENTION_MODELS[name]
    backbone, feat_dim = _timm_feature_backbone(timm_id, in_chans, pretrained)
    model = SpatialAttentionClassifier(backbone, feat_dim, num_classes=num_classes,
                                       dropout=dropout, residual=True)

    n_attn = sum(p.numel() for p in model.attention.parameters())
    notes = (f"{base_name} feature backbone (pretrained) + CBAM spatial-attention block "
             f"({n_attn} params, 7x7 conv on [avg;max] channel descriptor, sigmoid, "
             f"residual gate F*(1+A)) + GAP + Dropout({dropout}) + Linear({feat_dim}->{num_classes}). "
             f"No softmax before CrossEntropyLoss.")
    lc = _last_conv_name(model)
    return BuiltModel(
        name=name, timm_name=f"{timm_id}+spatial_attention", model=model, backend="timm+custom",
        classifier_attr="classifier",
        gradcam_layers=[g for g in ["backbone." + (_last_conv_name(backbone) or ""),
                                    "attention.conv", lc or ""] if g and not g.endswith(".")],
        native_input_size=224,
        notes=notes,
        # Stage A: train attention block + classifier, keep pretrained backbone frozen.
        stage_a_trainable_prefixes=("attention.", "classifier."),
    )
