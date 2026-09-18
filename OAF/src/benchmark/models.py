"""Uniform model factory for the CNN benchmark (Phase 2 = 10 models, Phase 3 = +5).

Backends
--------
* ``timm``        - ResNet/DenseNet/EfficientNet/MobileNet/Inception/Xception/
                    ConvNeXt/VGG. Head replacement + pretrained weights handled by
                    ``timm.create_model(num_classes=..., in_chans=...)``.
* ``torchvision`` - GoogLeNet, SqueezeNet, ShuffleNet-V2 (no timm equivalents in
                    the installed timm 1.0.28). Head replaced manually.

Every model: pretrained ImageNet weights, 5-class head, 3-channel input,
224x224 (adaptive pooling where native != 224 - documented per model).
The evaluation / training protocol is identical for all backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# ---- Phase 2 : 10 timm models (unchanged) -------------------------------- #
_TIMM_ZOO_P2: dict[str, str] = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenet_v2": "mobilenetv2_100",
    "mobilenet_v3_large": "mobilenetv3_large_100",
    "efficientnet_b1": "efficientnet_b1",
    "inception_v3": "inception_v3",
    "xception": "legacy_xception",
    "convnext_tiny": "convnext_tiny",
}

# ---- Phase 3 : +5 architectures ---------------------------------------- #
#  name -> (backend, backend_id)
_ZOO_P3: dict[str, tuple[str, str]] = {
    "vgg16":       ("timm", "vgg16"),            # vgg16.tv_in1k  (classic, no BN)
    "vgg19":       ("timm", "vgg19"),            # vgg19.tv_in1k
    "googlenet":   ("torchvision", "googlenet"),         # Inception-v1
    "squeezenet":  ("torchvision", "squeezenet1_1"),
    "shufflenet":  ("torchvision", "shufflenet_v2_x1_0"),  # ShuffleNet-V2 1.0x
}

# Full registry: name -> (backend, backend_id)
MODEL_ZOO: dict[str, tuple[str, str]] = {
    **{k: ("timm", v) for k, v in _TIMM_ZOO_P2.items()},
    **_ZOO_P3,
}
BENCHMARK_MODELS = list(_TIMM_ZOO_P2)          # Phase-2 set (do not retrain)
PHASE3_NEW_MODELS = list(_ZOO_P3)             # Phase-3 additions
ALL_MODELS = list(MODEL_ZOO)


@dataclass
class BuiltModel:
    name: str
    timm_name: str                 # kept for backwards compat; = backend id
    model: "object"
    classifier_attr: str
    backend: str = "timm"
    gradcam_layers: list = field(default_factory=list)
    native_input_size: int = 224
    notes: str = ""
    stage_a_trainable_prefixes: tuple = ()   # for wrapper models (attention): freeze only backbone.*


def _last_conv_name(model) -> str | None:
    import torch.nn as nn
    last = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last = n
    return last


# --------------------------------------------------------------------------- #
# torchvision head replacement
# --------------------------------------------------------------------------- #
def _build_torchvision(backend_id: str, num_classes: int, pretrained: bool):
    """Return (model, classifier_attr, native_input, notes)."""
    import torch.nn as nn
    from torchvision import models as M

    if backend_id == "googlenet":
        w = M.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
        # torchvision forces aux_logits=True when loading pretrained weights;
        # build with aux, then hard-disable so forward() returns only the logits.
        model = M.googlenet(weights=w) if pretrained else M.googlenet(aux_logits=False)
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model, "fc", 224, "torchvision GoogLeNet (Inception-v1); aux classifiers disabled."
    if backend_id == "squeezenet1_1":
        w = M.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        model = M.squeezenet1_1(weights=w)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model, "classifier.1", 224, "torchvision SqueezeNet 1.1; 1x1-conv classifier."
    if backend_id == "shufflenet_v2_x1_0":
        w = M.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = M.shufflenet_v2_x1_0(weights=w)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model, "fc", 224, "torchvision ShuffleNet-V2 1.0x."
    raise KeyError(f"unhandled torchvision id {backend_id!r}")


# --------------------------------------------------------------------------- #
# public factory
# --------------------------------------------------------------------------- #
def build_model(name: str, num_classes: int = 5, in_chans: int = 3,
                pretrained: bool = True) -> BuiltModel:
    """Create one benchmark model with a fresh ``num_classes`` head."""
    # Attention wrappers live in a separate module to keep this one dependency-light.
    if name.endswith("_spatial_attention"):
        from .attention import build_attention_model
        return build_attention_model(name, num_classes=num_classes,
                                     in_chans=in_chans, pretrained=pretrained)

    if name not in MODEL_ZOO:
        raise KeyError(f"unknown model {name!r}; choices: {ALL_MODELS}")
    backend, backend_id = MODEL_ZOO[name]

    if backend == "timm":
        import timm
        model = timm.create_model(backend_id, pretrained=pretrained,
                                  num_classes=num_classes, in_chans=in_chans)
        cfg = getattr(model, "default_cfg", {}) or {}
        classifier_attr = cfg.get("classifier", "")
        native = cfg.get("input_size", (3, 224, 224))[-1]
        notes = ""
    elif backend == "torchvision":
        if in_chans != 3:
            raise ValueError("torchvision backend expects 3-channel input (grayscale is replicated).")
        model, classifier_attr, native, notes = _build_torchvision(backend_id, num_classes, pretrained)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # Grad-CAM target-layer candidates (Phase 5 prep - not used for training).
    gradcam = []
    try:
        fi = model.feature_info.info if hasattr(model, "feature_info") else []
        if fi:
            gradcam.append(fi[-1].get("module", ""))
    except Exception:  # noqa: BLE001
        pass
    lc = _last_conv_name(model)
    if lc and lc not in gradcam:
        gradcam.append(lc)

    if native != 224 and not notes:
        notes = (f"{backend} '{backend_id}' native input {native}px; run at 224px via "
                 f"adaptive pooling for benchmark fairness.")
    return BuiltModel(name=name, timm_name=backend_id, model=model, backend=backend,
                      classifier_attr=classifier_attr,
                      gradcam_layers=[g for g in gradcam if g],
                      native_input_size=native, notes=notes)


# --------------------------------------------------------------------------- #
# Transfer-learning stage control (identical schedule for every model).
# --------------------------------------------------------------------------- #
def _classifier_param_ids(model, classifier_attr: str) -> set[int]:
    """Param ids belonging to the (freshly initialised) classification head."""
    mods = []
    if hasattr(model, "get_classifier"):
        try:
            mods.append(model.get_classifier())
        except Exception:  # noqa: BLE001
            pass
    if classifier_attr:
        obj = model
        try:
            for part in classifier_attr.split("."):
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            mods.append(obj)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
    ids: set[int] = set()
    for m in mods:
        if hasattr(m, "parameters"):
            for p in m.parameters():
                ids.add(id(p))
    return ids


_HEAD_LEAF_NAMES = {
    "fc.weight", "fc.bias", "classifier.weight", "classifier.bias",
    "head.weight", "head.bias", "head.fc.weight", "head.fc.bias",
    "last_linear.weight", "last_linear.bias",
}


def _head_param_names(model, classifier_attr: str) -> set[str]:
    """Precise: names on the classifier attr path (prefix match), never substring
    matches (which would wrongly catch ConvNeXt ``mlp.fc*``, SE ``fc*``,
    EfficientNet/MobileNet ``conv_head``, VGG ``classifier.0/3`` hidden FCs...)."""
    names: set[str] = set()
    prefix = (classifier_attr + ".") if classifier_attr else ""
    for n, _ in model.named_parameters():
        if prefix and (n == classifier_attr or n.startswith(prefix)):
            names.add(n)
        elif not classifier_attr and n in _HEAD_LEAF_NAMES:
            names.add(n)
    return names


def set_stage(built: BuiltModel, stage: str) -> dict:
    """stage 'A' -> train the classification head ONLY (backbone frozen).
    stage 'B' -> unfreeze the whole network.

    Head params are found exactly via timm ``get_classifier()`` ids + the
    ``default_cfg['classifier']`` attr path. Wrapper models (spatial attention)
    additionally keep any param whose name starts with one of
    ``built.stage_a_trainable_prefixes`` trainable in Stage A (so the new
    attention block trains alongside the head, backbone stays frozen).
    """
    model = built.model
    head_ids = _classifier_param_ids(model, built.classifier_attr)
    head_names = _head_param_names(model, built.classifier_attr)
    prefixes = tuple(built.stage_a_trainable_prefixes or ())
    fell_back = False
    n_train = n_frozen = 0
    trainable_names: list[str] = []
    for name, p in model.named_parameters():
        if stage.upper() == "A":
            is_head = (id(p) in head_ids) or (name in head_names) \
                or any(name.startswith(pf) for pf in prefixes)
            p.requires_grad = bool(is_head)
        else:
            p.requires_grad = True
        if p.requires_grad:
            n_train += 1
            trainable_names.append(name)
        else:
            n_frozen += 1
    if stage.upper() == "A" and n_train == 0:
        fell_back = True
        for p in model.parameters():
            p.requires_grad = True
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        n_frozen = 0
    return {"stage": stage.upper(), "trainable_tensors": int(n_train),
            "frozen_tensors": int(n_frozen),
            "head_param_tensors": len(head_ids) if head_ids else len(head_names),
            "stage_a_trainable_names": trainable_names if stage.upper() == "A" else "ALL",
            "stage_a_fallback_unfreeze_all": fell_back}


def split_param_groups(built: BuiltModel):
    """(backbone_params, head_params) for differential-LR fine-tuning (Exp 3C).
    'head' = classifier head (+ attention block for wrapper models)."""
    model = built.model
    head_ids = _classifier_param_ids(model, built.classifier_attr)
    head_names = _head_param_names(model, built.classifier_attr)
    prefixes = tuple(built.stage_a_trainable_prefixes or ())
    backbone, head = [], []
    for name, p in model.named_parameters():
        if (id(p) in head_ids) or (name in head_names) or any(name.startswith(pf) for pf in prefixes):
            head.append(p)
        else:
            backbone.append(p)
    return backbone, head


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(total - trainable)}


def trainable_named_groups(model) -> Iterable[str]:
    return (n for n, p in model.named_parameters() if p.requires_grad)
