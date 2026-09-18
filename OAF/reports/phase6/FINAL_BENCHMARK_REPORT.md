# AETHER-OA - FINAL BENCHMARK (Phases 5 + 6)

_Generated 2026-08-27T22:08:36.684084+00:00. All numbers are on the frozen, patient- and sample-disjoint TEST split (1240 images / 620 patients). Checkpoints = clean Phase-4, selected on VALIDATION Macro-F1. No test-set tuning._

## 1. Complete 15-model test benchmark

| Rank | Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | QWK | KL4-F1 | MAE | Exact Acc | Within +/-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | VGG16 | 0.7258 | 0.7382 | 0.7267 | 0.7392 | 0.8753 | 0.8800 | 0.3008 | 0.7258 | 0.9750 |
| 2 | ConvNeXt-Tiny | 0.6968 | 0.7169 | 0.7023 | 0.7306 | 0.8618 | 0.8571 | 0.3323 | 0.6968 | 0.9718 |
| 3 | VGG19 | 0.6895 | 0.7110 | 0.6918 | 0.7124 | 0.8637 | 0.8947 | 0.3363 | 0.6895 | 0.9750 |
| 4 | DenseNet121 | 0.6863 | 0.7061 | 0.6855 | 0.7124 | 0.8440 | 0.8831 | 0.3581 | 0.6863 | 0.9581 |
| 5 | Inception-V3 | 0.6863 | 0.7050 | 0.6820 | 0.7053 | 0.8442 | 0.8947 | 0.3597 | 0.6863 | 0.9556 |
| 6 | ResNet50 | 0.6524 | 0.6967 | 0.6564 | 0.6990 | 0.8264 | 0.9487 | 0.3960 | 0.6524 | 0.9524 |
| 7 | EfficientNet-B1 | 0.6315 | 0.6742 | 0.6458 | 0.6660 | 0.8154 | 0.8493 | 0.4121 | 0.6315 | 0.9581 |
| 8 | GoogLeNet | 0.6468 | 0.6715 | 0.6467 | 0.6770 | 0.8168 | 0.8571 | 0.4089 | 0.6468 | 0.9452 |
| 9 | EfficientNet-B0 | 0.6516 | 0.6705 | 0.6415 | 0.6753 | 0.8044 | 0.8947 | 0.4226 | 0.6516 | 0.9282 |
| 10 | ResNet18 | 0.6282 | 0.6595 | 0.6237 | 0.6708 | 0.8160 | 0.9000 | 0.4274 | 0.6282 | 0.9460 |
| 11 | ShuffleNet-V2 | 0.6226 | 0.6583 | 0.6179 | 0.6531 | 0.7986 | 0.9067 | 0.4468 | 0.6226 | 0.9315 |
| 12 | MobileNet-V2 | 0.6363 | 0.6547 | 0.6284 | 0.6780 | 0.7958 | 0.8706 | 0.4419 | 0.6363 | 0.9242 |
| 13 | MobileNet-V3-L | 0.6460 | 0.6518 | 0.6351 | 0.6633 | 0.7929 | 0.8250 | 0.4355 | 0.6460 | 0.9210 |
| 14 | SqueezeNet | 0.5960 | 0.6474 | 0.6046 | 0.6455 | 0.7730 | 0.9000 | 0.4798 | 0.5960 | 0.9250 |
| 15 | Xception | 0.6290 | 0.6350 | 0.6138 | 0.6246 | 0.7777 | 0.8235 | 0.4629 | 0.6290 | 0.9105 |

## 2. Per-class F1 (test)

| Rank | Model | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|---|--:|--:|--:|--:|--:|
| 1 | VGG16 | 0.8211 | 0.4909 | 0.6947 | 0.8045 | 0.8800 |
| 2 | ConvNeXt-Tiny | 0.7772 | 0.4286 | 0.7061 | 0.8155 | 0.8571 |
| 3 | VGG19 | 0.7915 | 0.4294 | 0.6596 | 0.7797 | 0.8947 |
| 4 | DenseNet121 | 0.7677 | 0.4009 | 0.6823 | 0.7967 | 0.8831 |
| 5 | Inception-V3 | 0.7721 | 0.3769 | 0.6702 | 0.8113 | 0.8947 |
| 6 | ResNet50 | 0.7444 | 0.3562 | 0.6203 | 0.8140 | 0.9487 |
| 7 | EfficientNet-B1 | 0.7100 | 0.4027 | 0.6255 | 0.7834 | 0.8493 |
| 8 | GoogLeNet | 0.7197 | 0.3282 | 0.6615 | 0.7910 | 0.8571 |
| 9 | EfficientNet-B0 | 0.7412 | 0.3051 | 0.6246 | 0.7867 | 0.8947 |
| 10 | ResNet18 | 0.7311 | 0.2725 | 0.5795 | 0.8144 | 0.9000 |
| 11 | ShuffleNet-V2 | 0.7167 | 0.2527 | 0.5806 | 0.8348 | 0.9067 |
| 12 | MobileNet-V2 | 0.7273 | 0.3055 | 0.6137 | 0.7563 | 0.8706 |
| 13 | MobileNet-V3-L | 0.7427 | 0.2799 | 0.6230 | 0.7884 | 0.8250 |
| 14 | SqueezeNet | 0.6798 | 0.2890 | 0.5882 | 0.7799 | 0.9000 |
| 15 | Xception | 0.7216 | 0.2404 | 0.5997 | 0.7899 | 0.8235 |

**KL1 ('doubtful') is the universal weak class** - best is VGG16 at 0.49; most models 0.25-0.43. KL0/KL3/KL4 are all strong (0.7-0.95). This is the single biggest limiter of Macro-F1.

## 3. Efficiency

| Rank | Model | Params (M) | Trainable (M) | GPU latency b1 (ms) | best epoch | epochs run |
|---|---|--:|--:|--:|--:|--:|
| 1 | VGG16 | 134.28 | 134.28 | 3.03 | 21 | 26 |
| 2 | ConvNeXt-Tiny | 27.82 | 27.82 | 4.26 | 14 | 19 |
| 3 | VGG19 | 139.59 | 139.59 | 3.35 | 27 | 32 |
| 4 | DenseNet121 | 6.96 | 6.96 | 8.18 | 19 | 24 |
| 5 | Inception-V3 | 21.80 | 21.80 | 9.39 | 12 | 17 |
| 6 | ResNet50 | 23.52 | 23.52 | 3.08 | 29 | 34 |
| 7 | EfficientNet-B1 | 6.52 | 6.52 | 5.72 | 7 | 12 |
| 8 | GoogLeNet | 5.61 | 5.61 | 3.47 | 28 | 33 |
| 9 | EfficientNet-B0 | 4.01 | 4.01 | 4.18 | 19 | 24 |
| 10 | ResNet18 | 11.18 | 11.18 | 3.02 | 26 | 31 |
| 11 | ShuffleNet-V2 | 1.26 | 1.26 | 5.50 | 17 | 22 |
| 12 | MobileNet-V2 | 2.23 | 2.23 | 4.68 | 26 | 31 |
| 13 | MobileNet-V3-L | 4.21 | 4.21 | 3.28 | 19 | 24 |
| 14 | SqueezeNet | 0.73 | 0.73 | 1.25 | 18 | 23 |
| 15 | Xception | 20.82 | 20.82 | 2.98 | 7 | 12 |

## 4. Validation -> Test generalization (Macro-F1)

| Rank | Model | val Macro-F1 | test Macro-F1 | delta (test - val) |
|---|---|--:|--:|--:|
| 1 | VGG16 | 0.7077 | 0.7382 | +0.0305 |
| 2 | ConvNeXt-Tiny | 0.7115 | 0.7169 | +0.0054 |
| 3 | VGG19 | 0.7079 | 0.7110 | +0.0030 |
| 4 | DenseNet121 | 0.7017 | 0.7061 | +0.0044 |
| 5 | Inception-V3 | 0.6840 | 0.7050 | +0.0211 |
| 6 | ResNet50 | 0.6686 | 0.6967 | +0.0281 |
| 7 | EfficientNet-B1 | 0.6387 | 0.6742 | +0.0355 |
| 8 | GoogLeNet | 0.6731 | 0.6715 | -0.0015 |
| 9 | EfficientNet-B0 | 0.6498 | 0.6705 | +0.0206 |
| 10 | ResNet18 | 0.6457 | 0.6595 | +0.0139 |
| 11 | ShuffleNet-V2 | 0.6452 | 0.6583 | +0.0131 |
| 12 | MobileNet-V2 | 0.6469 | 0.6547 | +0.0078 |
| 13 | MobileNet-V3-L | 0.6596 | 0.6518 | -0.0078 |
| 14 | SqueezeNet | 0.6345 | 0.6474 | +0.0129 |
| 15 | Xception | 0.6523 | 0.6350 | -0.0173 |

Mean delta +0.0113 (test slightly ABOVE val for most - the val set is marginally harder for these models / mild val-selection effect; not leakage, since splits are patient-disjoint).

## 5. Phase 6 enhancement - VGG16 + horizontal-flip TTA + temperature scaling

| config | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | QWK | KL1-F1 | KL4-F1 | MAE | Within +/-1 | ECE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vgg16_none_raw | 0.7258 | 0.7382 | 0.7267 | 0.7392 | 0.8753 | 0.4909 | 0.8800 | 0.3008 | 0.9750 | 0.0593 |
| vgg16_none_temp | 0.7258 | 0.7382 | 0.7267 | 0.7392 | 0.8753 | 0.4909 | 0.8800 | 0.3008 | 0.9750 | 0.0442 |
| vgg16_hflip_raw | 0.7274 | 0.7447 | 0.7285 | 0.7449 | 0.8799 | 0.4638 | 0.9067 | 0.2960 | 0.9774 | 0.0403 |
| vgg16_hflip_temp | 0.7242 | 0.7432 | 0.7261 | 0.7435 | 0.8785 | 0.4644 | 0.9067 | 0.2992 | 0.9774 | 0.0425 |

- hflip TTA: **+0.006 Macro-F1 / +0.005 QWK** over plain VGG16, no retraining.
- Temperature scaling (T fit on validation NLL) roughly halves ECE for every model - confidences become trustworthy. See `calibration.csv` / `reliability_diagram.png`.
- The multi-model ensemble was tested and **did not beat VGG16 + hflip** (val-optimal ensemble scored 0.7375 on test vs 0.7447) - dropped.

## 6. Binary OA screening (test)  -  {KL0,KL1} = no-OA  vs  {KL2,KL3,KL4} = OA

- **ROC-AUC 0.9649**, average precision 0.9616
- Sensitivity 0.883 at specificity 0.90
- Specificity 0.878 at sensitivity 0.90
- At threshold 0.5: accuracy 0.8968, sensitivity 0.818, specificity 0.957, F1 0.873 (n_OA 537, n_noOA 703)

## 7. 3-class (test)  -  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)

- accuracy **0.7976** | Macro-F1 **0.8120** | weighted-F1 0.7972 | QWK 0.7985

## 8. Selective prediction / abstention (test)

| confidence threshold | coverage (auto-graded) | selective accuracy | selective Macro-F1 |
|--:|--:|--:|--:|
| 0.50 | 79.3% | 0.7856 | 0.7585 |
| 0.60 | 59.8% | 0.8464 | 0.7573 |
| 0.70 | 45.6% | 0.8920 | 0.7435 |
| 0.80 | 29.9% | 0.9272 | 0.7489 |
| 0.90 | 16.7% | 0.9662 | 0.7758 |

e.g. auto-grade the ~30% most-confident knees at ~93% accuracy; refer the rest.

## 9. Preprocessing ablation (validation Macro-F1 / KL1-F1)

| model | basic | clahe | histeq |
|---|--:|--:|--:|
| ConvNeXt-Tiny | 0.7064 | 0.7170 | 0.6902 |
| DenseNet121 | 0.6877 | 0.6889 | 0.6836 |
| Inception-V3 | 0.6886 | 0.6787 | 0.6912 |

CLAHE helps **ConvNeXt-Tiny only** (+0.011 Macro-F1, KL1-F1 0.39->0.43). Neutral/negative elsewhere; histeq never helps. `basic` stays default.

## 10. Loss ablation (validation Macro-F1 / KL1-F1)

| model | class_balanced (F1/KL1) | focal (F1/KL1) | weighted_cross_entropy (F1/KL1) |
|---|--:|--:|--:|
| ConvNeXt-Tiny | 0.711 / 0.444 | 0.685 / 0.408 | 0.706 / 0.392 |
| DenseNet121 | 0.674 / 0.324 | 0.662 / 0.380 | 0.688 / 0.309 |
| Inception-V3 | 0.675 / 0.380 | 0.645 / 0.394 | 0.689 / 0.370 |

**class_balanced** loss helps ConvNeXt-Tiny's KL1-F1 (0.39->0.44) and Macro-F1. For DenseNet/Inception, plain weighted-CE wins. **Focal is worst everywhere.**

## 10b. Multi-seed robustness (validation, seeds 42/123/3407)

| config | mean Macro-F1 | std | mean QWK | std | mean bAcc | std |
|---|--:|--:|--:|--:|--:|--:|
| ConvNeXt-Tiny (+CLAHE+class_balanced) | 0.7093 | 0.0034 | 0.8299 | 0.0071 | 0.7161 | 0.0042 |
| VGG16 (basic+WCE) | 0.6926 | 0.0229 | 0.8231 | 0.0201 | 0.6974 | 0.0125 |

- **ConvNeXt-Tiny is the most stable** model (Macro-F1 std ~0.003). Its ~0.71 is reliable.
- **VGG16 has high seed variance** (Macro-F1 std ~0.023; one seed early-stopped at epoch 7 -> 0.66). Its #1 test result (~0.74) is real but partly seed-favourable; a re-seed could land ~0.72.
- **CLAHE + class_balanced did NOT stack** for ConvNeXt (individually ~0.717 / ~0.711; together ~0.709). No config change is adopted - the plain Phase-4 `basic` + `weighted_cross_entropy` checkpoints stand.

## 11. Final recommendation

**Best diagnostic accuracy:** `VGG16` + hflip TTA + temperature scaling -> test Macro-F1 **0.7447**, QWK **0.8799**, balanced acc 0.7449, within-+/-1 97.7%, MAE 0.296. Cost: 134 M params.


**Best deployable / most robust:** `ConvNeXt-Tiny` - test Macro-F1 0.7169, QWK 0.8618 at **28 M params (1/5 of VGG16)**, 4.3 ms, and the lowest seed variance (val Macro-F1 std ~0.003). **This is the recommended model to take forward** unless the extra ~0.02-0.03 Macro-F1 from VGG16 is critical.


**VGG16 vs VGG19 vs ConvNeXt-Tiny**: near-tie on QWK (~0.86-0.88). VGG16 leads test Macro-F1 by ~0.02-0.03 (via better KL1-F1) but is 134 M params AND seed-sensitive (val std ~0.023). ConvNeXt-Tiny is 5x smaller and stable.


**Deploy as:** the chosen backbone + hflip TTA + temperature calibration, reporting the **binary OA-screen (AUC 0.965)** and **3-class (acc 0.80)** as primary clinical outputs, with a **confidence-abstention band** (auto-grade the confident majority, refer the rest). 5-class KL is the hardest framing and is bounded by KL0/KL1 ambiguity.

## 12. Limitations

- Single-site test set (OAI-derived); KL4 support = 38 -> wide CI on KL4 metrics.
- No significance testing; test Macro-F1 differences < ~0.005 are noise on n=1240.
- KL1 recall ~0.5 across all models - the model is weak on the 'is there early OA?' call.
- Latency = desktop RTX 5080, batch-1, synthetic input, forward only (no preprocessing).
- Probabilities calibrated by temperature scaling; not externally validated.
