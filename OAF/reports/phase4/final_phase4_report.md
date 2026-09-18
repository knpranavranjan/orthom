# AETHER-OA - X-ray Module | PHASE 4: 15-Model Extended-Budget Benchmark (max_epochs = 50)

_Generated 2026-08-27T18:55:42.333364+00:00. Seed 42. Controlled experiment vs Phase 3 (max_epochs = 30). Test set NEVER used._

## 1. Executive summary

- Retrained all **15 models** identically to Phase 3 except **max_epochs 30 -> 50**; early stopping (patience 5 on val Macro-F1) left enabled.
- Macro-F1 delta (Phase4 - phase3): mean **-0.0025**, median **+0.0005** across 15 comparable models.
- Improved (> +0.002 Macro-F1): **7** ['convnext_tiny', 'vgg19', 'densenet121', 'googlenet', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']
- Practically unchanged (+/-0.002): **3** ['vgg16', 'mobilenet_v3_large', 'squeezenet']
- Degraded (< -0.002): **5** ['inception_v3', 'xception', 'resnet18', 'shufflenet', 'efficientnet_b1']
- Models that used > 30 epochs: **5** ['vgg19', 'googlenet', 'resnet50', 'mobilenet_v2', 'resnet18']
- Best validation Macro-F1 (Phase 4): **convnext_tiny** = 0.7115

- Largest Macro-F1 gain from the extra budget: **efficientnet_b0** (0.6403 -> 0.6498, +0.0095).

## 2. Experimental objective

Answer, with all other variables held constant: *does raising the maximum training budget from 30 to 50 epochs improve the 15-model benchmark?* 30 epochs = baseline (Phase 3); 50 epochs = experimental condition (Phase 4). No architecture / loss / augmentation / class-weight / hyper-parameter changes.

## 3. Hardware / software environment

- GPU: NVIDIA GeForce RTX 5080 (17.09 GB, capability 12.0)
- PyTorch 2.11.0+cu128 | CUDA 12.8 | cuDNN 91900 | torchvision 0.26.0+cu128 | timm 1.0.28
- OS Windows 11 | Python 3.14.6 | device = CUDA
- cuDNN deterministic=True, benchmark=False; AMP on.

## 4. Dataset integrity verification

Repeated the Phase-3 leakage + class-distribution checks (all PASSED before training):

| split | n | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|--:|--:|--:|--:|--:|--:|
| train | 5782 | 2293 | 1048 | 1509 | 756 | 176 |
| val | 1238 | 484 | 220 | 338 | 159 | 37 |
| test | 1240 | 476 | 227 | 328 | 171 | 38 |

patient overlap train/val = 0, train/test = 0, val/test = 0 ; sample overlap = 0 ; all 5 classes present.

## 5. Training configuration (identical to Phase 3 except max_epochs)

- preprocessing variant `basic`, 224x224, 3-ch replicated grayscale, imagenet normalization
- augmentation `augmentation.yaml` (train only); val/test deterministic
- loss `weighted_cross_entropy` with TRAIN-ONLY class weights (`weights_list_balanced`)
- optimizer adamw lr(StageB)=0.0001 wd=0.0001; cosine schedule, warmup 1 ep
- transfer learning: Stage A 3 ep head-only @ 0.001, then Stage B full fine-tune @ 0.0001
- batch size 64 (vgg16/vgg19: 16 - VRAM necessity, same as Phase 3), num_workers 4, AMP True
- **max_epochs 50** (Phase 3 = 30), early stopping monitor `val_macro_f1` mode max patience 5 min_delta 0.0
- selection metric `val_macro_f1`, seed 42

## 6. 15-model results (Phase 4, validation)

| rank | model | macro_f1 | quadratic_weighted_kappa | balanced_accuracy | weighted_f1 | cohen_kappa | macro_precision | macro_recall | kl0_f1 | kl1_f1 | kl2_f1 | kl3_f1 | kl4_f1 | parameters | model_size_mb | gmacs | mean_latency_ms | cpu_mean_latency_ms | peak_memory_mb | best_epoch | epochs_run | training_time_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | convnext_tiny | 0.7115 | 0.8452 | 0.7366 | 0.6904 | 0.5653 | 0.7053 | 0.7366 | 0.7755 | 0.4444 | 0.6491 | 0.8179 | 0.8706 | 27823973 | 106.1980 | 4.4697 | 7.5395 | 20.5223 | 421.2800 | 14 | 19 | 145.3000 |
| 2 | vgg19 | 0.7079 | 0.8422 | 0.7133 | 0.6789 | 0.5517 | 0.7186 | 0.7133 | 0.7870 | 0.4023 | 0.6132 | 0.8162 | 0.9211 | 139590725 | 532.5100 | 19.6280 | 3.3688 | 57.9901 | 2158.6500 | 27 | 32 | 698.5000 |
| 3 | vgg16 | 0.7077 | 0.8441 | 0.7058 | 0.6779 | 0.5464 | 0.7171 | 0.7058 | 0.7713 | 0.3769 | 0.6424 | 0.8291 | 0.9189 | 134281029 | 512.2530 | 15.4662 | 3.3027 | 60.2504 | 2092.1200 | 21 | 26 | 520.3000 |
| 4 | densenet121 | 0.7017 | 0.8095 | 0.7108 | 0.6637 | 0.5360 | 0.6983 | 0.7108 | 0.7394 | 0.4016 | 0.6206 | 0.8280 | 0.9189 | 6958981 | 27.0950 | 2.8645 | 10.8264 | 27.1495 | 168.7000 | 19 | 24 | 230.8000 |
| 5 | inception_v3 | 0.6840 | 0.8097 | 0.6908 | 0.6500 | 0.5182 | 0.6845 | 0.6908 | 0.7583 | 0.3391 | 0.5928 | 0.8085 | 0.9211 | 21795813 | 83.4440 | 2.8452 | 6.1267 | 23.5802 | 345.4000 | 12 | 17 | 113.7000 |
| 6 | googlenet | 0.6731 | 0.7823 | 0.6801 | 0.6350 | 0.4930 | 0.6692 | 0.6801 | 0.7132 | 0.3361 | 0.6210 | 0.7740 | 0.9211 | 5605029 | 21.5360 | 1.5039 | 3.6287 | 16.2997 | 153.7200 | 28 | 33 | 154.7000 |
| 7 | resnet50 | 0.6686 | 0.8008 | 0.6725 | 0.6290 | 0.4836 | 0.6779 | 0.6725 | 0.7348 | 0.3519 | 0.5448 | 0.8050 | 0.9067 | 23518277 | 90.0060 | 4.1095 | 4.6448 | 20.7482 | 372.8800 | 29 | 34 | 249.6000 |
| 8 | mobilenet_v3_large | 0.6596 | 0.7594 | 0.6706 | 0.6278 | 0.4934 | 0.6542 | 0.6706 | 0.7147 | 0.2895 | 0.6205 | 0.7871 | 0.8861 | 4208437 | 16.2400 | 0.2242 | 3.5182 | 10.9046 | 127.7900 | 19 | 24 | 130.2000 |
| 9 | xception | 0.6523 | 0.7595 | 0.6534 | 0.6089 | 0.4680 | 0.6599 | 0.6534 | 0.7238 | 0.2979 | 0.5231 | 0.8000 | 0.9167 | 20817197 | 79.7010 | 4.5742 | 3.0339 | 23.8616 | 341.1700 | 7 | 12 | 103.4000 |
| 10 | efficientnet_b0 | 0.6498 | 0.7718 | 0.6673 | 0.6239 | 0.4885 | 0.6368 | 0.6673 | 0.7374 | 0.3300 | 0.5741 | 0.7327 | 0.8750 | 4013953 | 15.5770 | 0.3981 | 5.1495 | 11.6818 | 129.5300 | 19 | 24 | 154.7000 |
| 11 | mobilenet_v2 | 0.6469 | 0.7559 | 0.6633 | 0.6117 | 0.4693 | 0.6335 | 0.6633 | 0.7169 | 0.3303 | 0.5550 | 0.7347 | 0.8974 | 2230277 | 8.7280 | 0.3129 | 3.1030 | 9.9801 | 108.0300 | 26 | 31 | 161.6000 |
| 12 | resnet18 | 0.6457 | 0.7812 | 0.6590 | 0.6007 | 0.4533 | 0.6527 | 0.6590 | 0.7305 | 0.3126 | 0.4737 | 0.8024 | 0.9091 | 11179077 | 42.7170 | 1.8186 | 1.8794 | 7.3368 | 222.6600 | 26 | 31 | 162.6000 |
| 13 | shufflenet | 0.6452 | 0.7671 | 0.6498 | 0.5968 | 0.4426 | 0.6538 | 0.6498 | 0.7046 | 0.3010 | 0.5081 | 0.7913 | 0.9211 | 1258729 | 4.9660 | 0.1478 | 3.5238 | 7.9804 | 86.5900 | 17 | 22 | 100.0000 |
| 14 | efficientnet_b1 | 0.6387 | 0.7724 | 0.6349 | 0.6000 | 0.4430 | 0.6656 | 0.6349 | 0.6995 | 0.3810 | 0.4973 | 0.7586 | 0.8571 | 6519589 | 25.2540 | 0.5871 | 6.9488 | 16.9772 | 159.8100 | 7 | 12 | 98.9000 |
| 15 | squeezenet | 0.6345 | 0.7514 | 0.6393 | 0.5815 | 0.4157 | 0.6403 | 0.6393 | 0.6814 | 0.3175 | 0.4874 | 0.7632 | 0.9231 | 725061 | 2.7840 | 0.2631 | 1.4214 | 4.7393 | 82.9600 | 18 | 23 | 97.1000 |

## 7. Phase 3 vs Phase 4 comparison (delta = Phase4 - phase3)

| rank | model | phase3_macro_f1 | phase4_macro_f1 | delta_macro_f1 | phase3_qwk | phase4_qwk | delta_qwk | phase3_balanced_accuracy | phase4_balanced_accuracy | delta_balanced_accuracy | phase3_kl4_f1 | phase4_kl4_f1 | delta_kl4_f1 | phase3_latency | phase4_latency | best_epoch | epochs_completed | early_stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | convnext_tiny | 0.7055 | 0.7115 | 0.0060 | 0.8398 | 0.8452 | 0.0053 | 0.7160 | 0.7366 | 0.0206 | 0.9000 | 0.8706 | -0.0294 | 3.3294 | 7.5395 | 14 | 19 | True |
| 2 | vgg19 | 0.7024 | 0.7079 | 0.0056 | 0.8367 | 0.8422 | 0.0054 | 0.7190 | 0.7133 | -0.0057 | 0.9231 | 0.9211 | -0.0020 | 3.6375 | 3.3688 | 27 | 32 | True |
| 3 | vgg16 | 0.7074 | 0.7077 | 0.0003 | 0.8373 | 0.8441 | 0.0068 | 0.7117 | 0.7058 | -0.0059 | 0.9189 | 0.9189 | 0.0000 | 3.0357 | 3.3027 | 21 | 26 | True |
| 4 | densenet121 | 0.6932 | 0.7017 | 0.0085 | 0.8136 | 0.8095 | -0.0041 | 0.6934 | 0.7108 | 0.0174 | 0.9067 | 0.9189 | 0.0123 | 7.9209 | 10.8264 | 19 | 24 | True |
| 5 | inception_v3 | 0.6872 | 0.6840 | -0.0033 | 0.7849 | 0.8097 | 0.0249 | 0.6948 | 0.6908 | -0.0040 | 0.9067 | 0.9211 | 0.0144 | 6.1649 | 6.1267 | 12 | 17 | True |
| 6 | googlenet | 0.6695 | 0.6731 | 0.0036 | 0.7992 | 0.7823 | -0.0169 | 0.6866 | 0.6801 | -0.0065 | 0.9000 | 0.9211 | 0.0211 | 3.4971 | 3.6287 | 28 | 33 | True |
| 7 | resnet50 | 0.6625 | 0.6686 | 0.0062 | 0.7767 | 0.8008 | 0.0241 | 0.6782 | 0.6725 | -0.0057 | 0.9091 | 0.9067 | -0.0024 | 2.9917 | 4.6448 | 29 | 34 | True |
| 8 | mobilenet_v3_large | 0.6592 | 0.6596 | 0.0004 | 0.7559 | 0.7594 | 0.0035 | 0.6677 | 0.6706 | 0.0030 | 0.8974 | 0.8861 | -0.0114 | 3.3883 | 3.5182 | 19 | 24 | True |
| 9 | xception | 0.6870 | 0.6523 | -0.0347 | 0.7778 | 0.7595 | -0.0183 | 0.6864 | 0.6534 | -0.0330 | 0.9589 | 0.9167 | -0.0422 | 4.1874 | 3.0339 | 7 | 12 | True |
| 10 | efficientnet_b0 | 0.6403 | 0.6498 | 0.0095 | 0.7713 | 0.7718 | 0.0005 | 0.6643 | 0.6673 | 0.0031 | 0.8780 | 0.8750 | -0.0030 | 6.4523 | 5.1495 | 19 | 24 | True |
| 11 | mobilenet_v2 | 0.6440 | 0.6469 | 0.0029 | 0.7644 | 0.7559 | -0.0085 | 0.6650 | 0.6633 | -0.0017 | 0.8889 | 0.8974 | 0.0085 | 2.9680 | 3.1030 | 26 | 31 | True |
| 12 | resnet18 | 0.6479 | 0.6457 | -0.0022 | 0.7688 | 0.7812 | 0.0124 | 0.6545 | 0.6590 | 0.0045 | 0.9211 | 0.9091 | -0.0120 | 2.1972 | 1.8794 | 26 | 31 | True |
| 13 | shufflenet | 0.6473 | 0.6452 | -0.0021 | 0.7621 | 0.7671 | 0.0049 | 0.6476 | 0.6498 | 0.0022 | 0.9333 | 0.9211 | -0.0123 | 3.4385 | 3.5238 | 17 | 22 | True |
| 14 | efficientnet_b1 | 0.6776 | 0.6387 | -0.0389 | 0.7901 | 0.7724 | -0.0177 | 0.6866 | 0.6349 | -0.0517 | 0.9231 | 0.8571 | -0.0659 | 5.8165 | 6.9488 | 7 | 12 | True |
| 15 | squeezenet | 0.6340 | 0.6345 | 0.0005 | 0.7528 | 0.7514 | -0.0014 | 0.6275 | 0.6393 | 0.0118 | 0.8986 | 0.9231 | 0.0245 | 1.3485 | 1.4214 | 18 | 23 | True |

Full table: `reports/phase4/phase3_vs_phase4_comparison.csv`

## 8. Ranking by Macro-F1

| rank | model | macro_f1 | quadratic_weighted_kappa | balanced_accuracy | kl4_f1 | parameters |
|---|---|---|---|---|---|---|
| 1 | convnext_tiny | 0.7115 | 0.8452 | 0.7366 | 0.8706 | 27823973 |
| 2 | vgg19 | 0.7079 | 0.8422 | 0.7133 | 0.9211 | 139590725 |
| 3 | vgg16 | 0.7077 | 0.8441 | 0.7058 | 0.9189 | 134281029 |
| 4 | densenet121 | 0.7017 | 0.8095 | 0.7108 | 0.9189 | 6958981 |
| 5 | inception_v3 | 0.6840 | 0.8097 | 0.6908 | 0.9211 | 21795813 |
| 6 | googlenet | 0.6731 | 0.7823 | 0.6801 | 0.9211 | 5605029 |
| 7 | resnet50 | 0.6686 | 0.8008 | 0.6725 | 0.9067 | 23518277 |
| 8 | mobilenet_v3_large | 0.6596 | 0.7594 | 0.6706 | 0.8861 | 4208437 |
| 9 | xception | 0.6523 | 0.7595 | 0.6534 | 0.9167 | 20817197 |
| 10 | efficientnet_b0 | 0.6498 | 0.7718 | 0.6673 | 0.8750 | 4013953 |
| 11 | mobilenet_v2 | 0.6469 | 0.7559 | 0.6633 | 0.8974 | 2230277 |
| 12 | resnet18 | 0.6457 | 0.7812 | 0.6590 | 0.9091 | 11179077 |
| 13 | shufflenet | 0.6452 | 0.7671 | 0.6498 | 0.9211 | 1258729 |
| 14 | efficientnet_b1 | 0.6387 | 0.7724 | 0.6349 | 0.8571 | 6519589 |
| 15 | squeezenet | 0.6345 | 0.7514 | 0.6393 | 0.9231 | 725061 |

## 9. Ranking by Quadratic Weighted Kappa

| rank | model | quadratic_weighted_kappa | macro_f1 | balanced_accuracy | kl4_f1 |
|---|---|---|---|---|---|
| 1 | convnext_tiny | 0.8452 | 0.7115 | 0.7366 | 0.8706 |
| 2 | vgg16 | 0.8441 | 0.7077 | 0.7058 | 0.9189 |
| 3 | vgg19 | 0.8422 | 0.7079 | 0.7133 | 0.9211 |
| 4 | inception_v3 | 0.8097 | 0.6840 | 0.6908 | 0.9211 |
| 5 | densenet121 | 0.8095 | 0.7017 | 0.7108 | 0.9189 |
| 6 | resnet50 | 0.8008 | 0.6686 | 0.6725 | 0.9067 |
| 7 | googlenet | 0.7823 | 0.6731 | 0.6801 | 0.9211 |
| 8 | resnet18 | 0.7812 | 0.6457 | 0.6590 | 0.9091 |
| 9 | efficientnet_b1 | 0.7724 | 0.6387 | 0.6349 | 0.8571 |
| 10 | efficientnet_b0 | 0.7718 | 0.6498 | 0.6673 | 0.8750 |
| 11 | shufflenet | 0.7671 | 0.6452 | 0.6498 | 0.9211 |
| 12 | xception | 0.7595 | 0.6523 | 0.6534 | 0.9167 |
| 13 | mobilenet_v3_large | 0.7594 | 0.6596 | 0.6706 | 0.8861 |
| 14 | mobilenet_v2 | 0.7559 | 0.6469 | 0.6633 | 0.8974 |
| 15 | squeezenet | 0.7514 | 0.6345 | 0.6393 | 0.9231 |

## 10. Ranking by Balanced Accuracy

| rank | model | balanced_accuracy | macro_f1 | quadratic_weighted_kappa | kl4_f1 |
|---|---|---|---|---|---|
| 1 | convnext_tiny | 0.7366 | 0.7115 | 0.8452 | 0.8706 |
| 2 | vgg19 | 0.7133 | 0.7079 | 0.8422 | 0.9211 |
| 3 | densenet121 | 0.7108 | 0.7017 | 0.8095 | 0.9189 |
| 4 | vgg16 | 0.7058 | 0.7077 | 0.8441 | 0.9189 |
| 5 | inception_v3 | 0.6908 | 0.6840 | 0.8097 | 0.9211 |
| 6 | googlenet | 0.6801 | 0.6731 | 0.7823 | 0.9211 |
| 7 | resnet50 | 0.6725 | 0.6686 | 0.8008 | 0.9067 |
| 8 | mobilenet_v3_large | 0.6706 | 0.6596 | 0.7594 | 0.8861 |
| 9 | efficientnet_b0 | 0.6673 | 0.6498 | 0.7718 | 0.8750 |
| 10 | mobilenet_v2 | 0.6633 | 0.6469 | 0.7559 | 0.8974 |
| 11 | resnet18 | 0.6590 | 0.6457 | 0.7812 | 0.9091 |
| 12 | xception | 0.6534 | 0.6523 | 0.7595 | 0.9167 |
| 13 | shufflenet | 0.6498 | 0.6452 | 0.7671 | 0.9211 |
| 14 | squeezenet | 0.6393 | 0.6345 | 0.7514 | 0.9231 |
| 15 | efficientnet_b1 | 0.6349 | 0.6387 | 0.7724 | 0.8571 |

## 11. Ranking by KL4 F1

| rank | model | kl4_f1 | macro_f1 | quadratic_weighted_kappa | balanced_accuracy |
|---|---|---|---|---|---|
| 1 | squeezenet | 0.9231 | 0.6345 | 0.7514 | 0.6393 |
| 2 | googlenet | 0.9211 | 0.6731 | 0.7823 | 0.6801 |
| 3 | vgg19 | 0.9211 | 0.7079 | 0.8422 | 0.7133 |
| 4 | shufflenet | 0.9211 | 0.6452 | 0.7671 | 0.6498 |
| 5 | inception_v3 | 0.9211 | 0.6840 | 0.8097 | 0.6908 |
| 6 | densenet121 | 0.9189 | 0.7017 | 0.8095 | 0.7108 |
| 7 | vgg16 | 0.9189 | 0.7077 | 0.8441 | 0.7058 |
| 8 | xception | 0.9167 | 0.6523 | 0.7595 | 0.6534 |
| 9 | resnet18 | 0.9091 | 0.6457 | 0.7812 | 0.6590 |
| 10 | resnet50 | 0.9067 | 0.6686 | 0.8008 | 0.6725 |
| 11 | mobilenet_v2 | 0.8974 | 0.6469 | 0.7559 | 0.6633 |
| 12 | mobilenet_v3_large | 0.8861 | 0.6596 | 0.7594 | 0.6706 |
| 13 | efficientnet_b0 | 0.8750 | 0.6498 | 0.7718 | 0.6673 |
| 14 | convnext_tiny | 0.8706 | 0.7115 | 0.8452 | 0.7366 |
| 15 | efficientnet_b1 | 0.8571 | 0.6387 | 0.7724 | 0.6349 |

## 12. Training-time comparison

| rank | model | training_time_sec | epochs_run | best_epoch | sec_per_epoch |
|---|---|---|---|---|---|
| 1 | vgg19 | 698.5000 | 32 | 27 | 21.8000 |
| 2 | vgg16 | 520.3000 | 26 | 21 | 20.0000 |
| 3 | resnet50 | 249.6000 | 34 | 29 | 7.3000 |
| 4 | densenet121 | 230.8000 | 24 | 19 | 9.6000 |
| 5 | resnet18 | 162.6000 | 31 | 26 | 5.2000 |
| 6 | mobilenet_v2 | 161.6000 | 31 | 26 | 5.2000 |
| 7 | googlenet | 154.7000 | 33 | 28 | 4.7000 |
| 8 | efficientnet_b0 | 154.7000 | 24 | 19 | 6.4000 |
| 9 | convnext_tiny | 145.3000 | 19 | 14 | 7.6000 |
| 10 | mobilenet_v3_large | 130.2000 | 24 | 19 | 5.4000 |
| 11 | inception_v3 | 113.7000 | 17 | 12 | 6.7000 |
| 12 | xception | 103.4000 | 12 | 7 | 8.6000 |
| 13 | shufflenet | 100.0000 | 22 | 17 | 4.5000 |
| 14 | efficientnet_b1 | 98.9000 | 12 | 7 | 8.2000 |
| 15 | squeezenet | 97.1000 | 23 | 18 | 4.2000 |

Total Phase-4 wall time: **53.3 min**.

## 13. Parameter / latency analysis

| rank | model | params_M | gmacs | model_size_mb | mean_latency_ms | cpu_mean_latency_ms | macro_f1 | f1_per_Mparam |
|---|---|---|---|---|---|---|---|---|
| 1 | convnext_tiny | 27.8200 | 4.4697 | 106.1980 | 7.5395 | 20.5223 | 0.7115 | 0.0256 |
| 2 | vgg19 | 139.5900 | 19.6280 | 532.5100 | 3.3688 | 57.9901 | 0.7079 | 0.0051 |
| 3 | vgg16 | 134.2800 | 15.4662 | 512.2530 | 3.3027 | 60.2504 | 0.7077 | 0.0053 |
| 4 | densenet121 | 6.9600 | 2.8645 | 27.0950 | 10.8264 | 27.1495 | 0.7017 | 0.1008 |
| 5 | inception_v3 | 21.8000 | 2.8452 | 83.4440 | 6.1267 | 23.5802 | 0.6840 | 0.0314 |
| 6 | googlenet | 5.6100 | 1.5039 | 21.5360 | 3.6287 | 16.2997 | 0.6731 | 0.1200 |
| 7 | resnet50 | 23.5200 | 4.1095 | 90.0060 | 4.6448 | 20.7482 | 0.6686 | 0.0284 |
| 8 | mobilenet_v3_large | 4.2100 | 0.2242 | 16.2400 | 3.5182 | 10.9046 | 0.6596 | 0.1567 |
| 9 | xception | 20.8200 | 4.5742 | 79.7010 | 3.0339 | 23.8616 | 0.6523 | 0.0313 |
| 10 | efficientnet_b0 | 4.0100 | 0.3981 | 15.5770 | 5.1495 | 11.6818 | 0.6498 | 0.1621 |
| 11 | mobilenet_v2 | 2.2300 | 0.3129 | 8.7280 | 3.1030 | 9.9801 | 0.6469 | 0.2901 |
| 12 | resnet18 | 11.1800 | 1.8186 | 42.7170 | 1.8794 | 7.3368 | 0.6457 | 0.0578 |
| 13 | shufflenet | 1.2600 | 0.1478 | 4.9660 | 3.5238 | 7.9804 | 0.6452 | 0.5121 |
| 14 | efficientnet_b1 | 6.5200 | 0.5871 | 25.2540 | 6.9488 | 16.9772 | 0.6387 | 0.0980 |
| 15 | squeezenet | 0.7300 | 0.2631 | 2.7840 | 1.4214 | 4.7393 | 0.6345 | 0.8692 |

Lightest 5: ['squeezenet', 'shufflenet', 'mobilenet_v2', 'efficientnet_b0', 'mobilenet_v3_large']. Pareto plots: `reports/phase4/pareto_macroF1_vs_latency.png`, `reports/phase4/pareto_macroF1_vs_size.png`.

## 14. Early-stopping analysis

| rank | model | best_epoch | epochs_completed | early_stop | early_stop_reason |
|---|---|---|---|---|---|
| 1 | resnet50 | 29 | 34 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 2 | googlenet | 28 | 33 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 3 | vgg19 | 27 | 32 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 4 | resnet18 | 26 | 31 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 5 | mobilenet_v2 | 26 | 31 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 6 | vgg16 | 21 | 26 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 7 | densenet121 | 19 | 24 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 8 | efficientnet_b0 | 19 | 24 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 9 | mobilenet_v3_large | 19 | 24 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 10 | squeezenet | 18 | 23 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 11 | shufflenet | 17 | 22 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 12 | convnext_tiny | 14 | 19 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 13 | inception_v3 | 12 | 17 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 14 | xception | 7 | 12 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |
| 15 | efficientnet_b1 | 7 | 12 | True | early stop: no val_macro_f1 improvement for 5 epochs (patience exhausted) |

- Stopped early (patience 5): ['convnext_tiny', 'vgg19', 'vgg16', 'densenet121', 'inception_v3', 'googlenet', 'resnet50', 'mobilenet_v3_large', 'xception', 'efficientnet_b0', 'mobilenet_v2', 'resnet18', 'shufflenet', 'efficientnet_b1', 'squeezenet']
- Ran past 30 epochs (i.e. genuinely used the extra budget): ['vgg19', 'googlenet', 'resnet50', 'mobilenet_v2', 'resnet18']
- Best epoch <= 30 for 15/15 models => the extra 20-epoch budget was mostly unused for those.

## 15. Overfitting observations

Signals inspected per model (from `models/phase4/<m>/history.json`): (a) val Macro-F1 peaks then declines while train loss keeps dropping; (b) best_epoch far below epochs_completed with a long no-improvement tail; (c) Phase4 Macro-F1 below Phase3 despite more epochs.

- Models that trained longer AND scored lower than Phase 3 (overfitting-consistent): ['resnet18']
- Models whose best_epoch <= 30 but kept training to satisfy patience: ['convnext_tiny', 'vgg19', 'vgg16', 'densenet121', 'inception_v3', 'googlenet', 'resnet50', 'mobilenet_v3_large', 'xception', 'efficientnet_b0', 'mobilenet_v2', 'resnet18', 'shufflenet', 'efficientnet_b1', 'squeezenet']

## 16. Best-performing models (Phase 4, top 5 by Macro-F1)

| rank | model | macro_f1 | quadratic_weighted_kappa | balanced_accuracy | kl4_f1 | parameters | mean_latency_ms |
|---|---|---|---|---|---|---|---|
| 1 | convnext_tiny | 0.7115 | 0.8452 | 0.7366 | 0.8706 | 27823973 | 7.5395 |
| 2 | vgg19 | 0.7079 | 0.8422 | 0.7133 | 0.9211 | 139590725 | 3.3688 |
| 3 | vgg16 | 0.7077 | 0.8441 | 0.7058 | 0.9189 | 134281029 | 3.3027 |
| 4 | densenet121 | 0.7017 | 0.8095 | 0.7108 | 0.9189 | 6958981 | 10.8264 |
| 5 | inception_v3 | 0.6840 | 0.8097 | 0.6908 | 0.9211 | 21795813 | 6.1267 |

## 17. Most efficient models

| rank | model | params_M | mean_latency_ms | cpu_mean_latency_ms | macro_f1 | quadratic_weighted_kappa |
|---|---|---|---|---|---|---|
| 1 | googlenet | 5.6100 | 3.6287 | 16.2997 | 0.6731 | 0.7823 |
| 2 | mobilenet_v3_large | 4.2100 | 3.5182 | 10.9046 | 0.6596 | 0.7594 |
| 3 | efficientnet_b0 | 4.0100 | 5.1495 | 11.6818 | 0.6498 | 0.7718 |
| 4 | mobilenet_v2 | 2.2300 | 3.1030 | 9.9801 | 0.6469 | 0.7559 |
| 5 | shufflenet | 1.2600 | 3.5238 | 7.9804 | 0.6452 | 0.7671 |
| 6 | squeezenet | 0.7300 | 1.4214 | 4.7393 | 0.6345 | 0.7514 |

## 18. Models benefiting from 50 epochs

| rank | model | phase3_macro_f1 | phase4_macro_f1 | delta_macro_f1 | delta_qwk | delta_balanced_accuracy | best_epoch | epochs_completed |
|---|---|---|---|---|---|---|---|---|
| 1 | efficientnet_b0 | 0.6403 | 0.6498 | 0.0095 | 0.0005 | 0.0031 | 19 | 24 |
| 2 | densenet121 | 0.6932 | 0.7017 | 0.0085 | -0.0041 | 0.0174 | 19 | 24 |
| 3 | resnet50 | 0.6625 | 0.6686 | 0.0062 | 0.0241 | -0.0057 | 29 | 34 |
| 4 | convnext_tiny | 0.7055 | 0.7115 | 0.0060 | 0.0053 | 0.0206 | 14 | 19 |
| 5 | vgg19 | 0.7024 | 0.7079 | 0.0056 | 0.0054 | -0.0057 | 27 | 32 |
| 6 | googlenet | 0.6695 | 0.6731 | 0.0036 | -0.0169 | -0.0065 | 28 | 33 |
| 7 | mobilenet_v2 | 0.6440 | 0.6469 | 0.0029 | -0.0085 | -0.0017 | 26 | 31 |

## 19. Models NOT benefiting from 50 epochs

| rank | model | phase3_macro_f1 | phase4_macro_f1 | delta_macro_f1 | best_epoch | epochs_completed | early_stop |
|---|---|---|---|---|---|---|---|
| 1 | efficientnet_b1 | 0.6776 | 0.6387 | -0.0389 | 7 | 12 | True |
| 2 | xception | 0.6870 | 0.6523 | -0.0347 | 7 | 12 | True |
| 3 | inception_v3 | 0.6872 | 0.6840 | -0.0033 | 12 | 17 | True |
| 4 | resnet18 | 0.6479 | 0.6457 | -0.0022 | 26 | 31 | True |
| 5 | shufflenet | 0.6473 | 0.6452 | -0.0021 | 17 | 22 | True |
| 6 | vgg16 | 0.7074 | 0.7077 | 0.0003 | 21 | 26 | True |
| 7 | mobilenet_v3_large | 0.6592 | 0.6596 | 0.0004 | 19 | 24 | True |
| 8 | squeezenet | 0.6340 | 0.6345 | 0.0005 | 18 | 23 | True |

## 20. Final recommendation for the next phase

_See the terminal summary for the evidence-based read on whether 50 epochs was worth it. No production model is selected in this phase - Phase 4 only isolates the epoch-budget effect. Recommended next experiments are listed in the run summary (preprocessing ablation on the top models, and/or fine-tuning-strategy ablation), keeping the test set frozen._
