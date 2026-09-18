# AETHER-OA X-ray - Phase 7 FINAL TEST evaluation

Evaluation only. Temperature + abstain threshold fitted on VALIDATION; the
1240-image test split scored once per model. KL1 remains the intrinsic weak class.

**Best by Macro-F1:** `convnext_tiny__M_difflr` - test Macro-F1 0.7307, QWK 0.8534, binary-OA acc 0.8782 (AUC 0.9514), 3-class acc 0.7726.

## 5-class KL (test)
| run | Acc | MacroF1 | wF1 | bAcc | QWK | MAE | w1 | KL0 | KL1 | KL2 | KL3 | KL4 | +hflip |
|---|--|--|--|--|--|--|--|--|--|--|--|--|--|
| convnext_tiny__M_difflr | 0.6984 | 0.7307 | 0.6999 | 0.7318 | 0.8534 | 0.3403 | 0.9621 | 0.7858 | 0.4341 | 0.6702 | 0.8159 | 0.9474 | 0.723 (-0.0076) |
| mobilenet_v2__M_strongaug | 0.6387 | 0.6686 | 0.6358 | 0.6689 | 0.8062 | 0.4274 | 0.9355 | 0.7308 | 0.3229 | 0.5937 | 0.8127 | 0.8831 | 0.6964 (+0.0278) |
| mobilenet_v2__M_classbal | 0.6323 | 0.6651 | 0.6372 | 0.6721 | 0.7941 | 0.4363 | 0.9347 | 0.7129 | 0.3647 | 0.6154 | 0.7824 | 0.85 | 0.6721 (+0.0071) |
| efficientnet_b0__M0_baseline | 0.6452 | 0.6506 | 0.6356 | 0.6481 | 0.7986 | 0.4315 | 0.9258 | 0.7376 | 0.2885 | 0.6449 | 0.7508 | 0.8312 | 0.6473 (-0.0033) |
| mobilenet_v2__M_corn | 0.6444 | 0.6353 | 0.6198 | 0.6416 | 0.8017 | 0.4339 | 0.9234 | 0.7392 | 0.162 | 0.6454 | 0.7988 | 0.8312 | 0.6516 (+0.0163) |

## Binary OA screen  {KL0,KL1} vs {KL2,KL3,KL4}
| run | Acc | ROC-AUC | PR-AUC | Sens | Spec | Prec |
|---|--|--|--|--|--|--|
| convnext_tiny__M_difflr | 0.8782 | 0.9514 | 0.946 | 0.7765 | 0.9559 | 0.9308 |
| efficientnet_b0__M0_baseline | 0.8669 | 0.9307 | 0.9308 | 0.7486 | 0.9573 | 0.9306 |
| mobilenet_v2__M_strongaug | 0.8524 | 0.9287 | 0.9276 | 0.7076 | 0.963 | 0.936 |
| mobilenet_v2__M_classbal | 0.8645 | 0.9246 | 0.9246 | 0.7691 | 0.9374 | 0.9037 |
| mobilenet_v2__M_corn | 0.8508 | 0.9242 | 0.9239 | 0.7263 | 0.9459 | 0.9112 |

## 3-class  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)
| run | Acc | MacroF1 | QWK |
|---|--|--|--|
| convnext_tiny__M_difflr | 0.7726 | 0.7889 | 0.7724 |
| mobilenet_v2__M_strongaug | 0.7427 | 0.7677 | 0.7386 |
| efficientnet_b0__M0_baseline | 0.7411 | 0.7616 | 0.7383 |
| mobilenet_v2__M_corn | 0.7371 | 0.7638 | 0.7394 |
| mobilenet_v2__M_classbal | 0.7129 | 0.7362 | 0.6913 |

## Calibration / abstention / efficiency / generalisation
| run | T | ECE pre->post | abstain thr | coverage | selective-acc | params | size MB | CPU ms | val->test gap |
|---|--|--|--|--|--|--|--|--|--|
| convnext_tiny__M_difflr | 1.3906 | 0.041->0.0429 | 0.73 | 0.3581 | 0.8986 | 27,823,973 | 106.198 | 20.911 | -0.0178 |
| efficientnet_b0__M0_baseline | 2.9195 | 0.2192->0.0262 | 0.87 | 0.1419 | 0.9034 | 4,013,953 | 15.577 | 9.9905 | -0.0009 |
| mobilenet_v2__M_corn | 1.5738 | 0.1044->0.0287 | 0.91 | 0.0823 | 0.9216 | 2,228,996 | 8.724 | 7.5089 | 0.0233 |
| mobilenet_v2__M_strongaug | 1.5291 | 0.0884->0.0429 | 0.82 | 0.1863 | 0.8874 | 2,230,277 | 8.728 | 8.2821 | -0.0154 |
| mobilenet_v2__M_classbal | 1.7126 | 0.0982->0.0329 | 0.85 | 0.1242 | 0.9091 | 2,230,277 | 8.728 | 7.771 | -0.0192 |

## Honest read on the >90% goal

- **5-class accuracy does not reach 90%** and is not expected to for any model on
  OAI KL grades (label noise floor; field SOTA ~0.72-0.75).
- **Binary OA screening** is the clinically meaningful task: best test accuracy 0.8782, best AUC 0.9514.
- KL1 F1 stays the ceiling-limited class; compare per-run above.