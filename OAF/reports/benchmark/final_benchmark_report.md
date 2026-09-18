# AETHER-OA - X-ray Module | PHASE 2: 10-CNN Benchmark Report

_Generated 2026-08-27T13:37:37.190695+00:00. Seed 42. Test set untouched during model selection._

## 1. Objective

Train and objectively compare 10 ImageNet-pretrained CNN backbones for 5-class Kellgren-Lawrence (KL 0-4) knee-OA severity classification, and select ONE model balancing diagnostic performance and deployment efficiency. Standard 5-class classification only (no attention / ordinal / Grad-CAM in this phase).

## 2. Dataset

- Source: Kaggle *Knee Osteoarthritis Dataset with Severity Grading* = verified OAI baseline (00m) knee ROIs. Kaggle-derived data only; OAI repo NOT merged, `auto_test` NOT used.
- Working set: 8,260 knee images / 4,130 patients. Preprocessing variant: **basic** (224x224 grayscale PNG -> float[0,1] -> 3-ch replicated grayscale -> imagenet normalization).

## 3. Patient-level split (frozen)

| split | knees | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|--:|--:|--:|--:|--:|--:|
| train | 5782 | 2293 | 1048 | 1509 | 756 | 176 |
| val | 1238 | 484 | 220 | 338 | 159 | 37 |
| test | 1240 | 476 | 227 | 328 | 171 | 38 |

Patient overlap train/val/test = 0 / 0 / 0 (re-verified at startup).

## 4. Preprocessing & 5. Augmentation

Deterministic preprocessing baked in Phase 1. Train-only augmentation (spec section 9): rotation +/-8deg, translation +/-5%, scale 0.95-1.05, shear +/-4deg, horizontal flip p=0.5, brightness/contrast 0.90-1.10, Gaussian noise p=0.2 (sigma<=0.02). Disabled: vertical_flip, arbitrary_rotation, heavy_color_jitter, elastic_warp, random_erasing, large_random_crop. Val/test deterministic.

## 6. Training configuration (identical for all models)

- **loss**: weighted cross-entropy (train-only class weights)
- **optimizer**: adamw lr(stageB)=0.0001 wd=0.0001
- **scheduler**: cosine
- **transfer_learning**: Stage A 3 ep head-only @lr 0.001, then Stage B full fine-tune @lr 0.0001
- **batch_size**: 64
- **max_epochs**: 30
- **early_stopping**: monitor val_macro_f1, patience 5
- **amp**: True
- **seed**: 42
- **input**: 3x224x224 (3-ch replicated grayscale, ImageNet norm)

## 7-10. Results

### Performance (validation)

| model | accuracy | macro_f1 | weighted_f1 | balanced_accuracy | quadratic_weighted_kappa | cohen_kappa | macro_precision | macro_recall |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | 0.6704 | 0.7055 | 0.6786 | 0.7160 | 0.8398 | 0.5491 | 0.7083 | 0.7160 |
| densenet121 | 0.6656 | 0.6932 | 0.6653 | 0.6934 | 0.8136 | 0.5363 | 0.6970 | 0.6934 |
| inception_v3 | 0.6462 | 0.6872 | 0.6485 | 0.6948 | 0.7849 | 0.5149 | 0.6824 | 0.6948 |
| xception | 0.6341 | 0.6870 | 0.6378 | 0.6864 | 0.7778 | 0.4972 | 0.6928 | 0.6864 |
| efficientnet_b1 | 0.6405 | 0.6776 | 0.6395 | 0.6866 | 0.7901 | 0.5023 | 0.6757 | 0.6866 |
| resnet50 | 0.6147 | 0.6625 | 0.6174 | 0.6782 | 0.7767 | 0.4751 | 0.6626 | 0.6782 |
| mobilenet_v3_large | 0.6357 | 0.6592 | 0.6246 | 0.6677 | 0.7559 | 0.4901 | 0.6577 | 0.6677 |
| resnet18 | 0.6018 | 0.6479 | 0.5992 | 0.6545 | 0.7688 | 0.4490 | 0.6589 | 0.6545 |
| mobilenet_v2 | 0.6018 | 0.6440 | 0.6036 | 0.6650 | 0.7644 | 0.4558 | 0.6351 | 0.6650 |
| efficientnet_b0 | 0.6236 | 0.6403 | 0.6148 | 0.6643 | 0.7713 | 0.4779 | 0.6235 | 0.6643 |

### Per-class F1 (validation)

| model | kl0_f1 | kl1_f1 | kl2_f1 | kl3_f1 | kl4_f1 |
|---|---|---|---|---|---|
| convnext_tiny | 0.7789 | 0.4053 | 0.6212 | 0.8220 | 0.9000 |
| densenet121 | 0.7688 | 0.3672 | 0.6197 | 0.8038 | 0.9067 |
| inception_v3 | 0.7223 | 0.3747 | 0.6120 | 0.8204 | 0.9067 |
| xception | 0.7234 | 0.3710 | 0.5733 | 0.8086 | 0.9589 |
| efficientnet_b1 | 0.7328 | 0.3355 | 0.5980 | 0.7988 | 0.9231 |
| resnet50 | 0.7184 | 0.3776 | 0.5143 | 0.7929 | 0.9091 |
| mobilenet_v3_large | 0.7067 | 0.2602 | 0.6317 | 0.8000 | 0.8974 |
| resnet18 | 0.7141 | 0.2892 | 0.4991 | 0.8160 | 0.9211 |
| mobilenet_v2 | 0.7050 | 0.3562 | 0.5190 | 0.7508 | 0.8889 |
| efficientnet_b0 | 0.7369 | 0.3130 | 0.5619 | 0.7118 | 0.8780 |

### Efficiency

| model | parameters | model_size_mb | gmacs | mean_latency_ms | median_latency_ms | throughput_images_per_sec | peak_memory_mb | cpu_mean_latency_ms |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | 27823973 | 106.198 | 4.470 | 3.329 | 3.275 | 300.360 | 421.280 | 22.927 |
| densenet121 | 6958981 | 27.095 | 2.865 | 7.921 | 7.870 | 126.250 | 168.700 | 27.676 |
| inception_v3 | 21795813 | 83.444 | 2.845 | 6.165 | 6.146 | 162.210 | 345.400 | 23.489 |
| xception | 20817197 | 79.701 | 4.574 | 4.187 | 4.162 | 238.810 | 341.170 | 23.923 |
| efficientnet_b1 | 6519589 | 25.254 | 0.587 | 5.816 | 5.780 | 171.920 | 159.810 | 15.803 |
| resnet50 | 23518277 | 90.006 | 4.109 | 2.992 | 2.900 | 334.260 | 372.880 | 18.142 |
| mobilenet_v3_large | 4208437 | 16.240 | 0.224 | 3.388 | 3.337 | 295.130 | 127.790 | 8.800 |
| resnet18 | 11179077 | 42.717 | 1.819 | 2.197 | 2.160 | 455.120 | 222.660 | 7.630 |
| mobilenet_v2 | 2230277 | 8.728 | 0.313 | 2.968 | 2.897 | 336.920 | 108.030 | 9.300 |
| efficientnet_b0 | 4013953 | 15.577 | 0.398 | 6.452 | 6.519 | 154.980 | 129.530 | 11.048 |

## 11. Confusion matrices

See `reports/benchmark/confusion_matrices/` (raw + normalized per model).

## 12. Per-class performance & adjacent-grade confusion

Per-model adjacent-grade confusion counts (KL1<->KL2, KL2<->KL3, KL3<->KL4) are in `classification_reports/<model>.json`.

## 13. Training curves

See `reports/benchmark/training_curves/`.

## 14-17. Latency / params / size / MACs

GMACs via `fvcore.FlopCountAnalysis` (one fused multiply-add = 1; FLOPs ~= 2x GMACs). Latency: batch=1, warmup then timed iterations, CUDA-synchronised. **GPU latency is development-only; Raspberry-Pi performance is NOT inferred here** (Phase 8).

## 18. Ranking

| model | macro_f1 | quadratic_weighted_kappa | balanced_accuracy | rank_macro_f1 | rank_qwk | rank_latency | rank_params | overall_score |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | 0.7055 | 0.8398 | 0.7160 | 1 | 1 | 4 | 10 | 2.8000 |
| densenet121 | 0.6932 | 0.8136 | 0.6934 | 2 | 2 | 10 | 5 | 3.8833 |
| inception_v3 | 0.6872 | 0.7849 | 0.6948 | 3 | 4 | 8 | 8 | 4.5000 |
| xception | 0.6870 | 0.7778 | 0.6864 | 4 | 5 | 6 | 7 | 5.2167 |
| efficientnet_b1 | 0.6776 | 0.7901 | 0.6866 | 5 | 3 | 7 | 4 | 4.4500 |
| resnet50 | 0.6625 | 0.7767 | 0.6782 | 6 | 6 | 3 | 9 | 6.0000 |
| mobilenet_v3_large | 0.6592 | 0.7559 | 0.6677 | 7 | 10 | 5 | 3 | 6.8000 |
| resnet18 | 0.6479 | 0.7688 | 0.6545 | 8 | 8 | 1 | 6 | 7.1167 |
| mobilenet_v2 | 0.6440 | 0.7644 | 0.6650 | 9 | 9 | 2 | 1 | 6.5167 |
| efficientnet_b0 | 0.6403 | 0.7713 | 0.6643 | 10 | 7 | 9 | 2 | 7.7167 |

- **Performance ranking** (Macro-F1 -> QWK -> balanced acc): convnext_tiny, densenet121, inception_v3, xception, efficientnet_b1, resnet50, mobilenet_v3_large, resnet18, mobilenet_v2, efficientnet_b0
- **Efficiency ranking** (latency -> params): resnet18, mobilenet_v2, resnet50, convnext_tiny, mobilenet_v3_large, xception, efficientnet_b1, inception_v3, efficientnet_b0, densenet121

## 19. Recommended model

**Primary (highest validation Macro-F1): `convnext_tiny`** - Macro-F1=0.7055, QWK=0.8398, balanced acc=0.7160, KL4-F1=0.9000, 27.8M params, 3.33 ms/img.

Apply the decision framework (spec section 35): if a much smaller/faster model is within a small Macro-F1 margin and matches on QWK / KL3-KL4 F1, prefer it. See the Pareto plots (`reports/benchmark/`). Final call stated below with evidence.

## 20. Limitations

- Single-source (OAI-derived) data; no independent external test yet.
- KL4 minority (176 train knees) -> KL4 metrics have wide confidence intervals.
- Inception/Xception run at 224px (native 299) for fairness -> not their optimal setting.
- Validation used for model selection; broader hyper-parameter tuning deferred to avoid val overf'g.
- Mixed precision + some CUDA ops are non-deterministic (documented in reproducibility.json).
- Research prototype for AI-based KL-grade classification - NOT a clinical diagnostic claim.

## 21. Next phase

PHASE 3 - best model + preprocessing ablation (basic vs CLAHE vs histeq). Test set remains frozen until Phase 6.


---
## Publication-style comparison table

| Model | Accuracy | Macro F1 | Balanced Acc. | QWK | KL4 F1 | Params(M) | Size(MB) | Latency(ms) |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | 0.6704 | 0.7055 | 0.7160 | 0.8398 | 0.9000 | 27.8200 | 106.1980 | 3.3294 |
| densenet121 | 0.6656 | 0.6932 | 0.6934 | 0.8136 | 0.9067 | 6.9600 | 27.0950 | 7.9209 |
| inception_v3 | 0.6462 | 0.6872 | 0.6948 | 0.7849 | 0.9067 | 21.8000 | 83.4440 | 6.1649 |
| xception | 0.6341 | 0.6870 | 0.6864 | 0.7778 | 0.9589 | 20.8200 | 79.7010 | 4.1874 |
| efficientnet_b1 | 0.6405 | 0.6776 | 0.6866 | 0.7901 | 0.9231 | 6.5200 | 25.2540 | 5.8165 |
| resnet50 | 0.6147 | 0.6625 | 0.6782 | 0.7767 | 0.9091 | 23.5200 | 90.0060 | 2.9917 |
| mobilenet_v3_large | 0.6357 | 0.6592 | 0.6677 | 0.7559 | 0.8974 | 4.2100 | 16.2400 | 3.3883 |
| resnet18 | 0.6018 | 0.6479 | 0.6545 | 0.7688 | 0.9211 | 11.1800 | 42.7170 | 2.1972 |
| mobilenet_v2 | 0.6018 | 0.6440 | 0.6650 | 0.7644 | 0.8889 | 2.2300 | 8.7280 | 2.9680 |
| efficientnet_b0 | 0.6236 | 0.6403 | 0.6643 | 0.7713 | 0.8780 | 4.0100 | 15.5770 | 6.4523 |