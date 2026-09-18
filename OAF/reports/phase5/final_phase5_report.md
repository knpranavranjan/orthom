# Phase 5 - Final 15-Model Test Benchmark

_Generated 2026-08-27T20:07:23.039557+00:00. Evaluation only - no training, no retraining, no tuning. Checkpoints = clean Phase-4 validation-selected best.pt._

## 1. Executive Summary

- Evaluated 15/15 clean Phase-4 checkpoints on the untouched TEST split (1240 samples). Failures: none.
- Best test Macro-F1: **vgg16** = 0.7382 (QWK 0.8753, balanced acc 0.7392, KL4-F1 0.8800, MAE 0.3008, 134.3M params).
- Top-3 by test Macro-F1: vgg16 0.7382, convnext_tiny 0.7169, vgg19 0.7110
- Mean generalization gap (test - Phase-4 val) Macro-F1: +0.0113

_Tiny numerical differences (e.g. < ~0.005 Macro-F1 on 1240 samples) are not claimed as meaningful; no significance test was run._

## 2. Research Objective

Produce the final, held-out TEST-set performance of all 15 architectures using the checkpoints already selected on VALIDATION Macro-F1 in the clean Phase-4 run, and pick a final model on the full evidence (not Macro-F1 alone).

## 3. Dataset

Kaggle Knee-OA severity, patient- and sample-disjoint splits. train 5782 / val 1238 / test 1240. Preprocessing variant `basic`, 224x224, 3-ch replicated grayscale, imagenet normalization (identical to Phase 3/4).

## 4. Dataset Integrity

| split | n | patients | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| train | 5782 | 2891 | 2293 | 1048 | 1509 | 756 | 176 |
| val | 1238 | 619 | 484 | 220 | 338 | 159 | 37 |
| test | 1240 | 620 | 476 | 227 | 328 | 171 | 38 |

Patient overlap train/val=0, train/test=0, val/test=0. Sample overlap=0. All 5 classes present.
Test class distribution vs expected {0: 476, 1: 227, 2: 328, 3: 171, 4: 38}: MATCH.

## 5. Phase 3 Summary

Phase 3 = 15-model benchmark, max_epochs 30, early stopping patience 5 on val Macro-F1. Phase-3 validation numbers are read from `reports/phase3/extended_benchmark_results.csv` (untouched) for the consolidated comparison.

## 6. Clean Phase 4 Verification

All 15 `models/phase4/<m>/best.pt` present and strict-loadable; each checkpoint's stored `val_metrics.macro_f1` equals its `result.json` macro_f1 (self-consistent). Phase 4 = same 15 models, max_epochs 50 (ceiling), same early stopping. Checkpoint selection metric = `val_macro_f1` (validation only).

## 7. Phase 3 vs Phase 4 (validation)

See `reports/phase5/phase3_phase4_phase5_consolidated.csv` (p3val_* vs p4val_* columns). Note: 50 epochs is only a CEILING - with early stopping enabled, models that stopped at epoch ~20 did not train for 50 epochs. Phrase any conclusion as *'allowing up to 50 epochs under the same early-stopping configuration'*, not *'training for 50 epochs'*.

## 8. Phase 5 Evaluation Methodology

- Load `models/phase4/<m>/best.pt` -> `load_state_dict(strict=True)` into the same `build_model()` architecture (pretrained=False; weights come only from the checkpoint).
- Forward the full TEST split via the existing `src.datasets.build_dataloaders(... return_meta=True, imbalance='none')` test loader (no augmentation, `shuffle=False`, evaluates all 1240 samples) and `src.benchmark.engine.evaluate(...)`.
- AMP for the forward pass = True (same regime as the Phase-4 validation numbers, so val<->test is comparable). Re-evaluation is deterministic for fixed weights.
- Metrics: `metrics.compute_all` (accuracy, macro/weighted P-R-F1, per-class, balanced accuracy, Cohen kappa, QWK, confusion matrix) + `phase5_metrics.ordinal_and_confidence` (MAE, exact / within-+/-1 accuracy, |pred-true| histogram, under/over-prediction, weighted precision & recall, micro P/R/F1, confidence stats, KL4->pred breakdown).
- Latency: `latency.measure_latency` - batch 1, synthetic 1x3x224x224, fp32 forward, 20 warmup + 100 timed iters, `torch.cuda.synchronize()` per iter. **Preprocessing and post-processing (softmax/argmax) are EXCLUDED** - pure model forward. Single process (Phase-5 lock enforces this).

## 9. Hardware and Software

- GPU NVIDIA GeForce RTX 5080 (17.09 GB, cc 12.0) | device CUDA
- torch 2.11.0+cu128 | CUDA 12.8 | cuDNN 91900 | torchvision 0.26.0+cu128 | timm 1.0.28 | Python 3.14.6
- seed 42 (evaluation is deterministic given fixed weights; seed only affects loader worker RNG which is irrelevant with shuffle=False) | config `configs/benchmark_phase4.yaml`

## 10. Complete Test-Set Results

| model | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 | balanced_accuracy | qwk | kl4_f1 | mae | exact_accuracy | within_1_accuracy | parameters | mean_latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 | 0.7258 | 0.7461 | 0.7392 | 0.7382 | 0.7385 | 0.7258 | 0.7267 | 0.7392 | 0.8753 | 0.8800 | 0.3008 | 0.7258 | 0.9750 | 134281029 | 3.0295 |
| convnext_tiny | 0.6968 | 0.7098 | 0.7306 | 0.7169 | 0.7122 | 0.6968 | 0.7023 | 0.7306 | 0.8618 | 0.8571 | 0.3323 | 0.6968 | 0.9718 | 27823973 | 4.2581 |
| vgg19 | 0.6895 | 0.7191 | 0.7124 | 0.7110 | 0.7060 | 0.6895 | 0.6918 | 0.7124 | 0.8637 | 0.8947 | 0.3363 | 0.6895 | 0.9750 | 139590725 | 3.3487 |
| densenet121 | 0.6863 | 0.7039 | 0.7124 | 0.7061 | 0.6893 | 0.6863 | 0.6855 | 0.7124 | 0.8440 | 0.8831 | 0.3581 | 0.6863 | 0.9581 | 6958981 | 8.1832 |
| inception_v3 | 0.6863 | 0.7138 | 0.7053 | 0.7050 | 0.6904 | 0.6863 | 0.6820 | 0.7053 | 0.8442 | 0.8947 | 0.3597 | 0.6863 | 0.9556 | 21795813 | 9.3890 |
| resnet50 | 0.6524 | 0.7042 | 0.6990 | 0.6967 | 0.6724 | 0.6524 | 0.6564 | 0.6990 | 0.8264 | 0.9487 | 0.3960 | 0.6524 | 0.9524 | 23518277 | 3.0760 |
| efficientnet_b1 | 0.6315 | 0.7026 | 0.6660 | 0.6742 | 0.6823 | 0.6315 | 0.6458 | 0.6660 | 0.8154 | 0.8493 | 0.4121 | 0.6315 | 0.9581 | 6519589 | 5.7207 |
| googlenet | 0.6468 | 0.6665 | 0.6770 | 0.6715 | 0.6470 | 0.6468 | 0.6467 | 0.6770 | 0.8168 | 0.8571 | 0.4089 | 0.6468 | 0.9452 | 5605029 | 3.4735 |
| efficientnet_b0 | 0.6516 | 0.6714 | 0.6753 | 0.6705 | 0.6389 | 0.6516 | 0.6415 | 0.6753 | 0.8044 | 0.8947 | 0.4226 | 0.6516 | 0.9282 | 4013953 | 4.1820 |
| resnet18 | 0.6282 | 0.6666 | 0.6708 | 0.6595 | 0.6430 | 0.6282 | 0.6237 | 0.6708 | 0.8160 | 0.9000 | 0.4274 | 0.6282 | 0.9460 | 11179077 | 3.0162 |
| shufflenet | 0.6226 | 0.6817 | 0.6531 | 0.6583 | 0.6392 | 0.6226 | 0.6179 | 0.6531 | 0.7986 | 0.9067 | 0.4468 | 0.6226 | 0.9315 | 1258729 | 5.4990 |
| mobilenet_v2 | 0.6363 | 0.6376 | 0.6780 | 0.6547 | 0.6237 | 0.6363 | 0.6284 | 0.6780 | 0.7958 | 0.8706 | 0.4419 | 0.6363 | 0.9242 | 2230277 | 4.6835 |
| mobilenet_v3_large | 0.6460 | 0.6445 | 0.6633 | 0.6518 | 0.6280 | 0.6460 | 0.6351 | 0.6633 | 0.7929 | 0.8250 | 0.4355 | 0.6460 | 0.9210 | 4208437 | 3.2768 |
| squeezenet | 0.5960 | 0.6563 | 0.6455 | 0.6474 | 0.6197 | 0.5960 | 0.6046 | 0.6455 | 0.7730 | 0.9000 | 0.4798 | 0.5960 | 0.9250 | 725061 | 1.2547 |
| xception | 0.6290 | 0.6590 | 0.6246 | 0.6350 | 0.6117 | 0.6290 | 0.6138 | 0.6246 | 0.7777 | 0.8235 | 0.4629 | 0.6290 | 0.9105 | 20817197 | 2.9777 |

## 11. Accuracy Analysis

| model | accuracy | exact_accuracy | within_1_accuracy | within_2_accuracy | balanced_accuracy |
|---|---|---|---|---|---|
| vgg16 | 0.7258 | 0.7258 | 0.9750 | 0.9984 | 0.7392 |
| convnext_tiny | 0.6968 | 0.6968 | 0.9718 | 0.9992 | 0.7306 |
| vgg19 | 0.6895 | 0.6895 | 0.9750 | 0.9992 | 0.7124 |
| densenet121 | 0.6863 | 0.6863 | 0.9581 | 0.9976 | 0.7124 |
| inception_v3 | 0.6863 | 0.6863 | 0.9556 | 0.9984 | 0.7053 |
| resnet50 | 0.6524 | 0.6524 | 0.9524 | 0.9992 | 0.6990 |
| efficientnet_b0 | 0.6516 | 0.6516 | 0.9282 | 0.9976 | 0.6753 |
| googlenet | 0.6468 | 0.6468 | 0.9452 | 0.9992 | 0.6770 |
| mobilenet_v3_large | 0.6460 | 0.6460 | 0.9210 | 0.9976 | 0.6633 |
| mobilenet_v2 | 0.6363 | 0.6363 | 0.9242 | 0.9976 | 0.6780 |
| efficientnet_b1 | 0.6315 | 0.6315 | 0.9581 | 0.9984 | 0.6660 |
| xception | 0.6290 | 0.6290 | 0.9105 | 0.9976 | 0.6246 |
| resnet18 | 0.6282 | 0.6282 | 0.9460 | 0.9984 | 0.6708 |
| shufflenet | 0.6226 | 0.6226 | 0.9315 | 0.9992 | 0.6531 |
| squeezenet | 0.5960 | 0.5960 | 0.9250 | 0.9992 | 0.6455 |

Exact accuracy == overall accuracy (single-label). Balanced accuracy = mean per-class recall - lower than accuracy indicates weak minority-class recall.

## 12. Precision Analysis

Per-class precision in `per_class_metrics.csv`. Macro (equal class weight) vs weighted (support weight):

| model | macro_precision | weighted_precision | kl4_precision |
|---|---|---|---|
| vgg16 | 0.7461 | 0.7385 | 0.8919 |
| vgg19 | 0.7191 | 0.7060 | 0.8947 |
| inception_v3 | 0.7138 | 0.6904 | 0.8947 |
| convnext_tiny | 0.7098 | 0.7122 | 0.7826 |
| resnet50 | 0.7042 | 0.6724 | 0.9250 |
| densenet121 | 0.7039 | 0.6893 | 0.8718 |
| efficientnet_b1 | 0.7026 | 0.6823 | 0.8857 |
| shufflenet | 0.6817 | 0.6392 | 0.9189 |
| efficientnet_b0 | 0.6714 | 0.6389 | 0.8947 |
| resnet18 | 0.6666 | 0.6430 | 0.8571 |
| googlenet | 0.6665 | 0.6470 | 0.8462 |
| xception | 0.6590 | 0.6117 | 0.9333 |
| squeezenet | 0.6563 | 0.6197 | 0.8571 |
| mobilenet_v3_large | 0.6445 | 0.6280 | 0.7857 |
| mobilenet_v2 | 0.6376 | 0.6237 | 0.7872 |

## 13. Recall Analysis

| model | macro_recall | weighted_recall | kl4_recall | balanced_accuracy |
|---|---|---|---|---|
| vgg16 | 0.7392 | 0.7258 | 0.8684 | 0.7392 |
| convnext_tiny | 0.7306 | 0.6968 | 0.9474 | 0.7306 |
| densenet121 | 0.7124 | 0.6863 | 0.8947 | 0.7124 |
| vgg19 | 0.7124 | 0.6895 | 0.8947 | 0.7124 |
| inception_v3 | 0.7053 | 0.6863 | 0.8947 | 0.7053 |
| resnet50 | 0.6990 | 0.6524 | 0.9737 | 0.6990 |
| mobilenet_v2 | 0.6780 | 0.6363 | 0.9737 | 0.6780 |
| googlenet | 0.6770 | 0.6468 | 0.8684 | 0.6770 |
| efficientnet_b0 | 0.6753 | 0.6516 | 0.8947 | 0.6753 |
| resnet18 | 0.6708 | 0.6282 | 0.9474 | 0.6708 |
| efficientnet_b1 | 0.6660 | 0.6315 | 0.8158 | 0.6660 |
| mobilenet_v3_large | 0.6633 | 0.6460 | 0.8684 | 0.6633 |
| shufflenet | 0.6531 | 0.6226 | 0.8947 | 0.6531 |
| squeezenet | 0.6455 | 0.5960 | 0.9474 | 0.6455 |
| xception | 0.6246 | 0.6290 | 0.7368 | 0.6246 |

## 14. F1 Analysis

**Macro F1** weights every KL grade equally - the key indicator of *balanced* performance across all five grades. **Weighted F1** weights by support - performance on the *actual* (imbalanced) test distribution. They answer different questions; neither is inherently 'better'.

| model | macro_f1 | weighted_f1 |
|---|---|---|
| vgg16 | 0.7382 | 0.7267 |
| convnext_tiny | 0.7169 | 0.7023 |
| vgg19 | 0.7110 | 0.6918 |
| densenet121 | 0.7061 | 0.6855 |
| inception_v3 | 0.7050 | 0.6820 |
| resnet50 | 0.6967 | 0.6564 |
| efficientnet_b1 | 0.6742 | 0.6458 |
| googlenet | 0.6715 | 0.6467 |
| efficientnet_b0 | 0.6705 | 0.6415 |
| resnet18 | 0.6595 | 0.6237 |
| shufflenet | 0.6583 | 0.6179 |
| mobilenet_v2 | 0.6547 | 0.6284 |
| mobilenet_v3_large | 0.6518 | 0.6351 |
| squeezenet | 0.6474 | 0.6046 |
| xception | 0.6350 | 0.6138 |

## 15. Per-Class Performance

Full 15x5 table in `reports/phase5/per_class_metrics.csv`. Per-class F1 (test):

| model | KL0_f1 | KL1_f1 | KL2_f1 | KL3_f1 | KL4_f1 |
|---|---|---|---|---|---|
| vgg16 | 0.8211 | 0.4909 | 0.6947 | 0.8045 | 0.8800 |
| convnext_tiny | 0.7772 | 0.4286 | 0.7061 | 0.8155 | 0.8571 |
| vgg19 | 0.7915 | 0.4294 | 0.6596 | 0.7797 | 0.8947 |
| densenet121 | 0.7677 | 0.4009 | 0.6823 | 0.7967 | 0.8831 |
| inception_v3 | 0.7721 | 0.3769 | 0.6702 | 0.8113 | 0.8947 |
| resnet50 | 0.7444 | 0.3562 | 0.6203 | 0.8140 | 0.9487 |
| efficientnet_b1 | 0.7100 | 0.4027 | 0.6255 | 0.7834 | 0.8493 |
| googlenet | 0.7197 | 0.3282 | 0.6615 | 0.7910 | 0.8571 |
| efficientnet_b0 | 0.7412 | 0.3051 | 0.6246 | 0.7867 | 0.8947 |
| resnet18 | 0.7311 | 0.2725 | 0.5795 | 0.8144 | 0.9000 |
| shufflenet | 0.7167 | 0.2527 | 0.5806 | 0.8348 | 0.9067 |
| mobilenet_v2 | 0.7273 | 0.3055 | 0.6137 | 0.7563 | 0.8706 |
| mobilenet_v3_large | 0.7427 | 0.2799 | 0.6230 | 0.7884 | 0.8250 |
| squeezenet | 0.6798 | 0.2890 | 0.5882 | 0.7799 | 0.9000 |
| xception | 0.7216 | 0.2404 | 0.5997 | 0.7899 | 0.8235 |

## 16. Confusion Matrix Analysis

Raw + row-normalized PNG per model in `reports/phase5/confusion_matrices/`; numerical arrays in `<model>_confusion.json` and `<model>_confusion_raw.csv`. Most-frequent off-diagonal confusions (top model shown; per-model in the JSON `_top_confusions`):

| true->pred | count | ordinal_distance |
|---|---|---|
| KL1->KL0 | 85 | 1 |
| KL2->KL1 | 78 | 1 |
| KL0->KL1 | 65 | 1 |
| KL2->KL3 | 34 | 1 |
| KL3->KL2 | 20 | 1 |
| KL1->KL2 | 18 | 1 |

## 17. Ordinal Error Analysis

Classes are ordered (KL0<KL1<KL2<KL3<KL4). `ordinal_error_analysis.csv` has the full |pred-true| histogram + under/over-prediction per model.

| model | mae | qwk | exact_accuracy | within_1_accuracy | error_0 | error_1 | error_2 | error_3 | error_4 | underprediction_count | overprediction_count | adjacent_error_fraction_of_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 | 0.3008 | 0.8753 | 0.7258 | 0.9750 | 900 | 309 | 29 | 2 | 0 | 209 | 131 | 0.9088 |
| convnext_tiny | 0.3323 | 0.8618 | 0.6968 | 0.9718 | 864 | 341 | 34 | 1 | 0 | 200 | 176 | 0.9069 |
| vgg19 | 0.3363 | 0.8637 | 0.6895 | 0.9750 | 855 | 354 | 30 | 1 | 0 | 234 | 151 | 0.9195 |
| densenet121 | 0.3581 | 0.8440 | 0.6863 | 0.9581 | 851 | 337 | 49 | 3 | 0 | 210 | 179 | 0.8663 |
| inception_v3 | 0.3597 | 0.8442 | 0.6863 | 0.9556 | 851 | 334 | 53 | 2 | 0 | 250 | 139 | 0.8586 |
| resnet50 | 0.3960 | 0.8264 | 0.6524 | 0.9524 | 809 | 372 | 58 | 1 | 0 | 262 | 169 | 0.8631 |
| googlenet | 0.4089 | 0.8168 | 0.6468 | 0.9452 | 802 | 370 | 67 | 1 | 0 | 206 | 232 | 0.8447 |
| efficientnet_b1 | 0.4121 | 0.8154 | 0.6315 | 0.9581 | 783 | 405 | 50 | 2 | 0 | 265 | 192 | 0.8862 |
| efficientnet_b0 | 0.4226 | 0.8044 | 0.6516 | 0.9282 | 808 | 343 | 86 | 3 | 0 | 251 | 181 | 0.7940 |
| resnet18 | 0.4274 | 0.8160 | 0.6282 | 0.9460 | 779 | 394 | 65 | 2 | 0 | 292 | 169 | 0.8547 |
| mobilenet_v3_large | 0.4355 | 0.7929 | 0.6460 | 0.9210 | 801 | 341 | 95 | 3 | 0 | 225 | 214 | 0.7768 |
| mobilenet_v2 | 0.4419 | 0.7958 | 0.6363 | 0.9242 | 789 | 357 | 91 | 3 | 0 | 222 | 229 | 0.7916 |
| shufflenet | 0.4468 | 0.7986 | 0.6226 | 0.9315 | 772 | 383 | 84 | 1 | 0 | 325 | 143 | 0.8184 |
| xception | 0.4629 | 0.7777 | 0.6290 | 0.9105 | 780 | 349 | 108 | 3 | 0 | 296 | 164 | 0.7587 |
| squeezenet | 0.4798 | 0.7730 | 0.5960 | 0.9250 | 739 | 408 | 92 | 1 | 0 | 292 | 209 | 0.8144 |

High `adjacent_error_fraction_of_errors` (close to 1.0) => errors are mostly neighbouring-grade, not large ordinal jumps.

## 18. Validation vs Test Generalization

delta = test - clean-Phase-4-validation. NEGATIVE = test lower than validation (a generalization gap, NOT an improvement). Full table: `reports/phase5/validation_vs_test_generalization.csv`.

| model | validation_macro_f1 | test_macro_f1 | delta_macro_f1 | validation_qwk | test_qwk | delta_qwk | validation_balanced_accuracy | test_balanced_accuracy | delta_balanced_accuracy | validation_kl4_f1 | test_kl4_f1 | delta_kl4_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 | 0.7077 | 0.7382 | 0.0305 | 0.8441 | 0.8753 | 0.0312 | 0.7058 | 0.7392 | 0.0334 | 0.9189 | 0.8800 | -0.0389 |
| convnext_tiny | 0.7115 | 0.7169 | 0.0054 | 0.8452 | 0.8618 | 0.0167 | 0.7366 | 0.7306 | -0.0060 | 0.8706 | 0.8571 | -0.0134 |
| vgg19 | 0.7079 | 0.7110 | 0.0030 | 0.8422 | 0.8637 | 0.0215 | 0.7133 | 0.7124 | -0.0010 | 0.9211 | 0.8947 | -0.0263 |
| densenet121 | 0.7017 | 0.7061 | 0.0044 | 0.8095 | 0.8440 | 0.0344 | 0.7108 | 0.7124 | 0.0016 | 0.9189 | 0.8831 | -0.0358 |
| inception_v3 | 0.6840 | 0.7050 | 0.0211 | 0.8097 | 0.8442 | 0.0345 | 0.6908 | 0.7053 | 0.0146 | 0.9211 | 0.8947 | -0.0263 |
| resnet50 | 0.6686 | 0.6967 | 0.0281 | 0.8008 | 0.8264 | 0.0256 | 0.6725 | 0.6990 | 0.0265 | 0.9067 | 0.9487 | 0.0421 |
| efficientnet_b1 | 0.6387 | 0.6742 | 0.0355 | 0.7724 | 0.8154 | 0.0430 | 0.6349 | 0.6660 | 0.0311 | 0.8571 | 0.8493 | -0.0078 |
| googlenet | 0.6731 | 0.6715 | -0.0015 | 0.7823 | 0.8168 | 0.0346 | 0.6801 | 0.6770 | -0.0030 | 0.9211 | 0.8571 | -0.0639 |
| efficientnet_b0 | 0.6498 | 0.6705 | 0.0206 | 0.7718 | 0.8044 | 0.0326 | 0.6673 | 0.6753 | 0.0079 | 0.8750 | 0.8947 | 0.0197 |
| resnet18 | 0.6457 | 0.6595 | 0.0139 | 0.7812 | 0.8160 | 0.0348 | 0.6590 | 0.6708 | 0.0118 | 0.9091 | 0.9000 | -0.0091 |
| shufflenet | 0.6452 | 0.6583 | 0.0131 | 0.7671 | 0.7986 | 0.0315 | 0.6498 | 0.6531 | 0.0033 | 0.9211 | 0.9067 | -0.0144 |
| mobilenet_v2 | 0.6469 | 0.6547 | 0.0078 | 0.7559 | 0.7958 | 0.0399 | 0.6633 | 0.6780 | 0.0146 | 0.8974 | 0.8706 | -0.0268 |
| mobilenet_v3_large | 0.6596 | 0.6518 | -0.0078 | 0.7594 | 0.7929 | 0.0335 | 0.6706 | 0.6633 | -0.0074 | 0.8861 | 0.8250 | -0.0611 |
| squeezenet | 0.6345 | 0.6474 | 0.0129 | 0.7514 | 0.7730 | 0.0216 | 0.6393 | 0.6455 | 0.0062 | 0.9231 | 0.9000 | -0.0231 |
| xception | 0.6523 | 0.6350 | -0.0173 | 0.7595 | 0.7777 | 0.0182 | 0.6534 | 0.6246 | -0.0288 | 0.9167 | 0.8235 | -0.0931 |

## 19. Model Efficiency

| model | macro_f1 | qwk | parameters | trainable_parameters | mean_latency_ms | throughput_img_per_s |
|---|---|---|---|---|---|---|
| vgg16 | 0.7382 | 0.8753 | 134281029 | 134281029 | 3.0295 | 330.0900 |
| convnext_tiny | 0.7169 | 0.8618 | 27823973 | 27823973 | 4.2581 | 234.8500 |
| vgg19 | 0.7110 | 0.8637 | 139590725 | 139590725 | 3.3487 | 298.6300 |
| densenet121 | 0.7061 | 0.8440 | 6958981 | 6958981 | 8.1832 | 122.2000 |
| inception_v3 | 0.7050 | 0.8442 | 21795813 | 21795813 | 9.3890 | 106.5100 |
| resnet50 | 0.6967 | 0.8264 | 23518277 | 23518277 | 3.0760 | 325.0900 |
| efficientnet_b1 | 0.6742 | 0.8154 | 6519589 | 6519589 | 5.7207 | 174.8000 |
| googlenet | 0.6715 | 0.8168 | 5605029 | 5605029 | 3.4735 | 287.9000 |
| efficientnet_b0 | 0.6705 | 0.8044 | 4013953 | 4013953 | 4.1820 | 239.1200 |
| resnet18 | 0.6595 | 0.8160 | 11179077 | 11179077 | 3.0162 | 331.5400 |
| shufflenet | 0.6583 | 0.7986 | 1258729 | 1258729 | 5.4990 | 181.8500 |
| mobilenet_v2 | 0.6547 | 0.7958 | 2230277 | 2230277 | 4.6835 | 213.5100 |
| mobilenet_v3_large | 0.6518 | 0.7929 | 4208437 | 4208437 | 3.2768 | 305.1800 |
| squeezenet | 0.6474 | 0.7730 | 725061 | 725061 | 1.2547 | 796.9800 |
| xception | 0.6350 | 0.7777 | 20817197 | 20817197 | 2.9777 | 335.8300 |

`model_efficiency.csv` adds macro_f1_per_Mparam.

## 20. Model Rankings

- **test_macro_f1**: vgg16(0.7382), convnext_tiny(0.7169), vgg19(0.7110), densenet121(0.7061), inception_v3(0.7050)
- **test_accuracy**: vgg16(0.7258), convnext_tiny(0.6968), vgg19(0.6895), densenet121(0.6863), inception_v3(0.6863)
- **test_qwk**: vgg16(0.8753), vgg19(0.8637), convnext_tiny(0.8618), inception_v3(0.8442), densenet121(0.8440)
- **test_balanced_accuracy**: vgg16(0.7392), convnext_tiny(0.7306), densenet121(0.7124), vgg19(0.7124), inception_v3(0.7053)
- **test_kl4_f1**: resnet50(0.9487), shufflenet(0.9067), squeezenet(0.9000), resnet18(0.9000), vgg19(0.8947)
- **lowest_mae**: vgg16(0.3008), convnext_tiny(0.3323), vgg19(0.3363), densenet121(0.3581), inception_v3(0.3597)
- **lowest_latency**: squeezenet(1.2547), xception(2.9777), resnet18(3.0162), vgg16(3.0295), resnet50(3.0760)
- **lowest_params**: squeezenet(725061), shufflenet(1258729), mobilenet_v2(2230277), efficientnet_b0(4013953), mobilenet_v3_large(4208437)

## 21. Best Model by Metric

- A. Best overall (multi-factor): see section 22
- B. Best Macro-F1: vgg16 (0.7382)
- C. Best Accuracy: vgg16 (0.7258)
- D. Best Macro Precision: vgg16 (0.7461)
- E. Best Macro Recall: vgg16 (0.7392)
- F. Best Weighted F1: vgg16 (0.7267)
- G. Best Balanced Accuracy: vgg16 (0.7392)
- H. Best QWK: vgg16 (0.8753)
- I. Best KL4 F1: resnet50 (0.9487)
- J. Lowest MAE: vgg16 (0.3008)
- K. Fastest: squeezenet (1.2547)
- L. Smallest: squeezenet (725061)
- M. Best macro-F1 / param: squeezenet
- N. Best macro-F1 / latency: squeezenet

## 22. Overall Model Recommendation

_Evidence-based synthesis across Macro-F1, macro precision/recall, weighted F1, balanced accuracy, QWK, KL4 F1, MAE, minority-class behaviour, parameter count, latency, and the validation->test gap. If the leaders are within ~0.005 test Macro-F1 and match on QWK/KL4, they are called a practical tie and the smaller/faster one is preferred. Filled from the actual numbers above; no single winner is forced if strengths genuinely differ._

Leaders: vgg16 (F1 0.7382, QWK 0.8753, KL4-F1 0.8800, 134.3M, 3.03ms), convnext_tiny (F1 0.7169, QWK 0.8618, KL4-F1 0.8571, 27.8M, 4.26ms), vgg19 (F1 0.7110, QWK 0.8637, KL4-F1 0.8947, 139.6M, 3.35ms). Top-3 Macro-F1 spread = 0.0273 -> a clear ordering exists.

## 23. Limitations

- Single held-out test set (1240 samples, 620 patients); Class 4 support = 38 -> KL4 metrics have wide uncertainty; do NOT over-interpret small KL4 differences.
- No confidence intervals / significance tests computed; differences < ~0.005 Macro-F1 are within noise for this sample size.
- AMP fp16 forward is used for metrics (matches Phase-4 val regime); latency is fp32.
- Latency is desktop RTX 5080 batch-1 synthetic-input forward only - not an edge-device or end-to-end (with preprocessing) number.
- Probabilities are raw softmax; no calibration was applied or assessed here.

## 24. Recommended Next Experiment

_To be decided with the user from these results (e.g. preprocessing ablation - basic vs CLAHE vs histeq - on the 2-3 test leaders; or calibration / TTA / ensembling of the top models). No decision is made here; the test set is now consumed for final reporting and must not drive further training choices._


## No Data Leakage (explicit)

The TEST set was NOT used for training, checkpoint selection, early stopping, hyper-parameter tuning, preprocessing choice, threshold selection, augmentation choice, or architecture selection. Every checkpoint was selected on VALIDATION Macro-F1 during the clean Phase-4 run. The test set is used here ONLY for this final evaluation.


## Reproducibility

Python 3.14.6 | torch 2.11.0+cu128 | torchvision 0.26.0+cu128 | timm 1.0.28 | CUDA 12.8 | GPU NVIDIA GeForce RTX 5080 | seed 42 | config configs/benchmark_phase4.yaml | checkpoints models/phase4/<m>/best.pt | split metadata/processed_manifest_basic.csv | variant basic | 224x224 | imagenet norm | test batch_size per-model (vgg 16, else 64) | latency warmup 20 iters 100. Full env: reports/phase5/reproducibility.json.
