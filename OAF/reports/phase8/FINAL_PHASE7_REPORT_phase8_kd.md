# AETHER-OA X-ray - Phase 7 FINAL TEST evaluation

Evaluation only. Temperature + abstain threshold fitted on VALIDATION; the
1240-image test split scored once per model. KL1 remains the intrinsic weak class.

**Best by Macro-F1:** `K6_kd_ens_long__seed3407` - test Macro-F1 0.6952, QWK 0.8383, binary-OA acc 0.8637 (AUC 0.9431), 3-class acc 0.7589.

## 5-class KL (test)
| run | Acc | MacroF1 | wF1 | bAcc | QWK | MAE | w1 | KL0 | KL1 | KL2 | KL3 | KL4 | +hflip |
|---|--|--|--|--|--|--|--|--|--|--|--|--|--|
| K6_kd_ens_long__seed3407 | 0.6798 | 0.6952 | 0.6764 | 0.6951 | 0.8383 | 0.3694 | 0.9508 | 0.7799 | 0.3871 | 0.6418 | 0.7953 | 0.8718 | 0.6998 (+0.0046) |
| K6_kd_ens_long | 0.6653 | 0.6821 | 0.6635 | 0.6825 | 0.832 | 0.3839 | 0.9524 | 0.7661 | 0.3613 | 0.6377 | 0.7844 | 0.8608 | 0.6898 (+0.0077) |
| K6_kd_ens_long__seed123 | 0.6605 | 0.6809 | 0.6559 | 0.6804 | 0.8345 | 0.3863 | 0.954 | 0.7629 | 0.3283 | 0.6218 | 0.8081 | 0.8831 | 0.6804 (-0.0005) |

## Binary OA screen  {KL0,KL1} vs {KL2,KL3,KL4}
| run | Acc | ROC-AUC | PR-AUC | Sens | Spec | Prec |
|---|--|--|--|--|--|--|
| K6_kd_ens_long | 0.8613 | 0.9452 | 0.9423 | 0.7169 | 0.9716 | 0.9506 |
| K6_kd_ens_long__seed3407 | 0.8637 | 0.9431 | 0.9406 | 0.7374 | 0.9602 | 0.934 |
| K6_kd_ens_long__seed123 | 0.8621 | 0.9409 | 0.9398 | 0.7225 | 0.9687 | 0.9463 |

## 3-class  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)
| run | Acc | MacroF1 | QWK |
|---|--|--|--|
| K6_kd_ens_long | 0.7685 | 0.7864 | 0.7652 |
| K6_kd_ens_long__seed123 | 0.7645 | 0.7851 | 0.7678 |
| K6_kd_ens_long__seed3407 | 0.7589 | 0.7793 | 0.759 |

## Calibration / abstention / efficiency / generalisation
| run | T | ECE pre->post | abstain thr | coverage | selective-acc | params | size MB | CPU ms | val->test gap |
|---|--|--|--|--|--|--|--|--|--|
| K6_kd_ens_long | 1.4296 | 0.0551->0.0492 | 0.71 | 0.3427 | 0.8941 | 2,230,277 | 8.717 | 9.4353 | -0.0046 |
| K6_kd_ens_long__seed123 | 1.3934 | 0.0573->0.0575 | 0.73 | 0.3363 | 0.8849 | 2,230,277 | 8.717 | 9.3223 | 0.0086 |
| K6_kd_ens_long__seed3407 | 1.3892 | 0.0372->0.0362 | 0.71 | 0.3363 | 0.8849 | 2,230,277 | 8.717 | 9.8094 | -0.0169 |

## Honest read on the >90% goal

- **5-class accuracy does not reach 90%** and is not expected to for any model on
  OAI KL grades (label noise floor; field SOTA ~0.72-0.75).
- **Binary OA screening** is the clinically meaningful task: best test accuracy 0.8637, best AUC 0.9452.
- KL1 F1 stays the ceiling-limited class; compare per-run above.