"""Reusable training / evaluation engine (spec sections 10-16).

One code path for every architecture: same loss, optimizer family, scheduler,
two-stage transfer-learning schedule, early stopping and selection metric.
Nothing here ever looks at the test split.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import compute_all

_CW_JSON = Path("metadata/class_weights.json")


def load_class_weights(key: str = "weights_list_balanced"):
    """Return a length-5 python list of TRAIN-ONLY class weights.

    Keys: 'weights_list_balanced' (default, Phase-2 baseline),
          'balanced', 'inverse_frequency_normalised', 'class_balanced_effective_num'.
    """
    from src.common import PROJECT_ROOT
    d = json.loads((PROJECT_ROOT / _CW_JSON).read_text())
    if key in ("weights_list_balanced", "balanced"):
        return list(d["weights_list_balanced"])
    sub = d["weights"].get(key)
    if sub is None:
        raise KeyError(f"unknown class_weights key {key!r}; have {list(d['weights'])}")
    return [float(sub[str(i)]) for i in range(5)]


# --------------------------------------------------------------------------- #
# Loss (configurable; Phase-2 default = weighted cross-entropy)
# --------------------------------------------------------------------------- #
def build_loss(cfg: dict, class_weights):
    """class_weights: tensor already on the target device (or None).

    cfg['name'] in:
      weighted_cross_entropy | cross_entropy       (Phase-2 baseline)
      focal                                        (weighted focal, gamma cfg['focal_gamma'])
      class_balanced                               (CE with effective-number weights;
                                                    caller must pass those weights)
    """
    import torch
    import torch.nn as nn

    name = cfg.get("name", "weighted_cross_entropy")
    smoothing = float(cfg.get("label_smoothing", 0.0))
    if name in ("weighted_cross_entropy", "wce", "class_balanced"):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=None, label_smoothing=smoothing)
    if name == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))

        class Focal(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", class_weights if class_weights is not None
                                     else torch.ones(5))
                self.g = gamma

            def forward(self, logits, target):
                ce = nn.functional.cross_entropy(logits, target, weight=self.w, reduction="none")
                pt = torch.exp(-ce)
                return ((1 - pt) ** self.g * ce).mean()

        return Focal()
    if name == "corn":                                   # Phase-7 opt-in ordinal loss
        from .ordinal import corn_loss

        class CORN(nn.Module):
            def forward(self, logits, target):
                return corn_loss(logits, target)

        return CORN()
    raise ValueError(f"unknown loss {name!r}")


def build_optimizer(params, cfg: dict):
    import torch

    name = cfg.get("name", "adamw").lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd,
                                 betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=float(cfg.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"unknown optimizer {name!r}")


def build_scheduler(optimizer, cfg: dict, epochs: int):
    import torch

    cfg = cfg or {}
    name = cfg.get("name", "none")
    name = (name or "none").lower()
    if name in ("none",):
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2,
            min_lr=float(cfg.get("min_lr", 1e-6)))
    if name == "cosine":
        epochs = max(1, epochs)
        warmup = min(int(cfg.get("warmup_epochs", 0)), max(0, epochs - 1))
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup), eta_min=float(cfg.get("min_lr", 1e-6)))
        if warmup <= 0:
            return cos
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cos], milestones=[warmup])
    raise ValueError(f"unknown scheduler {name!r}")


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopping:
    patience: int = 5
    mode: str = "max"
    min_delta: float = 0.0
    best: float = field(default=None)
    best_epoch: int = -1
    num_bad: int = 0
    should_stop: bool = False

    def update(self, value: float, epoch: int) -> bool:
        """Return True if this epoch is a new best."""
        if self.best is None:
            self.best, self.best_epoch = value, epoch
            return True
        improved = ((value > self.best + self.min_delta) if self.mode == "max"
                    else (value < self.best - self.min_delta))
        if improved:
            self.best, self.best_epoch, self.num_bad = value, epoch, 0
            return True
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True
        return False


# --------------------------------------------------------------------------- #
# Weight EMA (Phase-7 opt-in; nothing in Phases 1-6 constructs this)
# --------------------------------------------------------------------------- #
class ModelEMA:
    """Exponential moving average of model weights, kept in fp32.

    Buffers / integer tensors are copied verbatim (not averaged). Use `copy_to`
    to swap the EMA weights into a model for evaluation, restoring afterwards.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        import torch
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                s = self.shadow[k]
                if v.dtype.is_floating_point:
                    s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    s.copy_(v)

    def state_dict(self, ref_model) -> dict:
        ref = ref_model.state_dict()
        return {k: self.shadow[k].to(ref[k].dtype).clone() for k in ref}

    def copy_to(self, model):
        model.load_state_dict(self.state_dict(model), strict=True)


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, loss_fn, optimizer, device, *,
                    amp: bool = True, grad_clip: float = 0.0,
                    scaler=None, max_batches: Optional[int] = None,
                    ema=None, ordinal: Optional[str] = None) -> dict:
    import torch

    model.train()
    running = 0.0
    n = 0
    correct = 0
    t0 = time.perf_counter()
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
            if isinstance(logits, (tuple, list)):   # e.g. inception aux
                logits = logits[0]
            loss = loss_fn(logits, y)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = y.size(0)
        running += loss.item() * bs
        n += bs
        if ordinal == "corn":
            from .ordinal import corn_logits_to_label
            pred = torch.as_tensor(corn_logits_to_label(logits.detach().float().cpu().numpy()),
                                   device=y.device)
            correct += (pred == y).sum().item()
        else:
            correct += (logits.argmax(1) == y).sum().item()
    return {"loss": running / max(n, 1), "acc": correct / max(n, 1),
            "seconds": round(time.perf_counter() - t0, 2), "batches": bi + 1, "samples": n}


def evaluate(model, loader, loss_fn, device, *, amp: bool = True,
             max_batches: Optional[int] = None, return_predictions: bool = False,
             ordinal: Optional[str] = None) -> dict:
    import torch

    model.eval()
    running = 0.0
    n = 0
    all_true, all_pred, all_prob, all_meta = [], [], [], []
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
                meta = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = loss_fn(logits, y)
            if ordinal == "corn":
                from .ordinal import corn_logits_to_label, corn_logits_to_probs
                lg = logits.float().cpu().numpy()
                prob = torch.as_tensor(corn_logits_to_probs(lg))
                pred = torch.as_tensor(corn_logits_to_label(lg))
            else:
                prob = torch.softmax(logits.float(), dim=1)
                pred = prob.argmax(1)
            running += loss.item() * y.size(0)
            n += y.size(0)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_prob.append(prob.cpu().numpy())
            if meta is not None:
                keys = list(meta.keys())
                for i in range(len(y)):
                    all_meta.append({k: (meta[k][i].item() if hasattr(meta[k][i], "item")
                                         else meta[k][i]) for k in keys})

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    m = compute_all(y_true, y_pred, y_prob)
    m["loss"] = running / max(n, 1)
    if return_predictions:
        m["_predictions"] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
                             "meta": all_meta}
    return m


# --------------------------------------------------------------------------- #
# Full fit: 2-stage transfer learning + early stopping
# --------------------------------------------------------------------------- #
def fit(built, loaders, device, cfg: dict, out_dir: Path, *,
        logger=None, max_epochs: Optional[int] = None,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None) -> dict:
    """Train one model. Returns a summary dict; writes best.pt / last.pt / history."""
    import torch
    from .models import set_stage, count_parameters, split_param_groups

    log = logger.info if logger else print
    model = built.model.to(device)

    # class weights: Phase-2 baseline uses 'balanced'; loss 'class_balanced' uses
    # the effective-number weights. Always TRAIN-ONLY (metadata/class_weights.json).
    _loss_cfg = cfg["loss"]
    _cw_key = _loss_cfg.get("class_weights_key", "weights_list_balanced")
    if _loss_cfg.get("name") == "class_balanced":
        _cw_key = "class_balanced_effective_num"
    if _loss_cfg.get("name") in ("cross_entropy", "corn"):
        class_weights = None
    else:
        class_weights = torch.tensor(load_class_weights(_cw_key), dtype=torch.float32, device=device)
    loss_fn = build_loss(_loss_cfg, class_weights)
    ordinal = "corn" if _loss_cfg.get("name") == "corn" else None
    log(f"[{built.name}] loss={_loss_cfg.get('name')} class_weights_key={_cw_key if class_weights is not None else 'none'}"
        + (" | ORDINAL=corn (head=4 logits)" if ordinal else ""))

    tl = cfg["transfer_learning"]
    tcfg = cfg["train"]
    diff_lr = tl.get("differential_lr")  # {'backbone_lr':1e-5,'head_lr':1e-4} or None
    es_cfg = tcfg["early_stopping"]
    total_epochs = int(max_epochs or tcfg["max_epochs"])
    stage_a_epochs = min(int(tl.get("stage_a_epochs", 0)), total_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(tcfg.get("amp", True)) and device.type == "cuda")
    early = EarlyStopping(patience=int(es_cfg.get("patience", 5)),
                          mode=es_cfg.get("mode", "max"),
                          min_delta=float(es_cfg.get("min_delta", 0.0)))

    # ---- optional weight EMA (Phase-7 opt-in via cfg["train"]["ema"]) ----
    ema = None
    _ema_cfg = tcfg.get("ema")
    if _ema_cfg:
        _decay = float(_ema_cfg.get("decay", 0.999) if isinstance(_ema_cfg, dict) else _ema_cfg)
        ema = ModelEMA(model, decay=_decay)
        log(f"[{built.name}] weight EMA enabled (decay={_decay})")

    history: list[dict] = []
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _nominal_lr(cur_stage: str) -> str:
        if cur_stage == "A":
            return f"{float(tl['stage_a_lr']):.2e}"
        if diff_lr:
            return f"bb={float(diff_lr['backbone_lr']):.1e}/head={float(diff_lr['head_lr']):.1e}"
        return f"{float(tl['stage_b_lr']):.2e}"

    def _make_optimizer(cur_stage: str):
        oc = dict(cfg["optimizer"])
        if cur_stage == "A":
            oc["lr"] = float(tl["stage_a_lr"])
            return build_optimizer([p for p in model.parameters() if p.requires_grad], oc)
        # Stage B
        if diff_lr:
            bb, hd = split_param_groups(built)
            groups = [
                {"params": [p for p in bb if p.requires_grad], "lr": float(diff_lr["backbone_lr"])},
                {"params": [p for p in hd if p.requires_grad], "lr": float(diff_lr["head_lr"])},
            ]
            oc["lr"] = float(diff_lr["head_lr"])
            log(f"[{built.name}] differential LR: backbone={diff_lr['backbone_lr']:.1e} "
                f"head={diff_lr['head_lr']:.1e}")
            return build_optimizer(groups, oc)
        oc["lr"] = float(tl["stage_b_lr"])
        return build_optimizer([p for p in model.parameters() if p.requires_grad], oc)

    # ---- Stage A: frozen backbone, head only ----
    stage = "A" if stage_a_epochs > 0 else "B"
    stage_info = set_stage(built, stage)
    optimizer = _make_optimizer(stage)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_epochs)
    log(f"[{built.name}] stage {stage} | {stage_info} | base_lr={_nominal_lr(stage)} "
        f"(warmup {cfg.get('scheduler', {}).get('warmup_epochs', 0)} ep) "
        f"| trainable params={count_parameters(model)['trainable_parameters']:,}")

    wall0 = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        if stage == "A" and epoch == stage_a_epochs + 1:
            stage = "B"
            stage_info = set_stage(built, "B")
            optimizer = _make_optimizer("B")
            scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_epochs - stage_a_epochs)
            log(f"[{built.name}] -> stage B (unfreeze all) | base_lr={_nominal_lr('B')} "
                f"| trainable params={count_parameters(model)['trainable_parameters']:,}")

        _ep_t0 = time.perf_counter()
        tr = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device,
                             amp=bool(tcfg.get("amp", True)),
                             grad_clip=float(tcfg.get("grad_clip_norm", 0.0)),
                             scaler=scaler, max_batches=max_train_batches,
                             ema=ema, ordinal=ordinal)
        _val_t0 = time.perf_counter()
        if ema is not None:                        # evaluate on the EMA weights
            _raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        va = evaluate(model, loaders["val"], loss_fn, device,
                      amp=bool(tcfg.get("amp", True)), max_batches=max_val_batches,
                      ordinal=ordinal)
        if ema is not None:
            model.load_state_dict(_raw_sd, strict=True)
        _val_secs = round(time.perf_counter() - _val_t0, 2)
        _epoch_secs = round(time.perf_counter() - _ep_t0, 2)
        _cum_secs = round(time.perf_counter() - wall0, 1)

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(va["macro_f1"])
            else:
                scheduler.step()

        sel = va[cfg["eval"]["selection_metric"].replace("val_", "")]
        is_best = early.update(sel, epoch)
        rec = {
            "model": built.name, "epoch": epoch, "stage": stage,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"], "train_acc": tr["acc"], "train_seconds": tr["seconds"],
            "val_loss": va["loss"], "val_acc": va["accuracy"],
            "val_macro_f1": va["macro_f1"], "val_balanced_acc": va["balanced_accuracy"],
            "val_qwk": va["quadratic_weighted_kappa"], "val_macro_recall": va["macro_recall"],
            "val_macro_precision": va["macro_precision"],
            "val_kl4_f1": va["kl4_f1"], "val_kl3_f1": va["kl3_f1"],
            "epoch_seconds": _epoch_secs, "val_seconds": _val_secs,
            "cumulative_seconds": _cum_secs,
            "is_best": bool(is_best),
        }
        history.append(rec)
        log(f"[{built.name}] ep{epoch:02d}/{total_epochs} {stage} "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_macroF1={va['macro_f1']:.4f} val_QWK={va['quadratic_weighted_kappa']:.4f} "
            f"val_bAcc={va['balanced_accuracy']:.4f} "
            f"| ep_time={_epoch_secs:.1f}s (train {tr['seconds']:.1f}s / val {_val_secs:.1f}s) "
            f"cum={_cum_secs:.0f}s{'  * best' if is_best else ''}")

        ckpt = {
            "model_name": built.name, "timm_name": built.timm_name,
            "epoch": epoch,
            "state_dict": ema.state_dict(model) if ema is not None else model.state_dict(),
            "val_metrics": {k: v for k, v in va.items() if not k.startswith("_")},
            "selection_metric": cfg["eval"]["selection_metric"], "selection_value": sel,
            "ordinal": ordinal, "ema": bool(ema is not None),
        }
        torch.save(ckpt, last_path)
        if is_best:
            torch.save(ckpt, best_path)

        if early.should_stop:
            log(f"[{built.name}] early stop at epoch {epoch} "
                f"(best {cfg['eval']['selection_metric']}={early.best:.4f} @ epoch {early.best_epoch})")
            break

    wall = round(time.perf_counter() - wall0, 1)
    _best = max(history, key=lambda h: h["val_macro_f1"]) if history else {}
    log(f"[{built.name}] FINISHED | epochs_run={len(history)} "
        f"best_epoch={early.best_epoch} "
        f"best_val_macroF1={_best.get('val_macro_f1', float('nan')):.4f} "
        f"best_val_QWK={_best.get('val_qwk', float('nan')):.4f} "
        f"best_val_bAcc={_best.get('val_balanced_acc', float('nan')):.4f} "
        f"train_time={wall:.1f}s ({wall/60:.1f} min) "
        f"early_stop={early.should_stop} ckpt={best_path}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "model": built.name,
        "best_epoch": early.best_epoch,
        "best_val_selection": early.best,
        "epochs_run": len(history),
        "training_time_sec": wall,
        "history": history,
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "stopped_early": early.should_stop,
    }
