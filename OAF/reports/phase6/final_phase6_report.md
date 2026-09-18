# Phase 6 - Ensemble + Calibration + Clinical Sub-tasks

_Generated 2026-08-27T20:13:09.063178+00:00. Evaluation only. Ensemble members, weights, temperatures and the abstention threshold are ALL fit on VALIDATION and applied ONCE to TEST (frozen). TTA = hflip._

## 1. Frozen configuration (chosen on validation)

- Members: **vgg16, densenet121**  (weight scheme: uniform)
- Weights: [1.0, 1.0]
- Per-model temperature (val NLL): vgg16=1.455, densenet121=1.538
- Selection metric on val: macro_f1

## 2. Test result vs best single model

| config | split | accuracy | macro_f1 | weighted_f1 | balanced_acc | QWK | KL1_f1 | KL4_f1 | MAE | within_1 | ECE |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ENSEMBLE | val | 0.6922 | 0.7247 | 0.6967 | 0.7281 | 0.8464 | 0.4073 | 0.9189 | 0.3481 | 0.9596 | 0.0406 |
| ENSEMBLE | test | 0.7234 | 0.7375 | 0.7197 | 0.7423 | 0.8718 | 0.4295 | 0.9091 | 0.3081 | 0.9702 | 0.0444 |
| best_single[vgg19]+T | val | 0.6737 | 0.7109 | 0.6801 | 0.7117 | 0.8491 | 0.3862 | 0.9333 | 0.3562 | 0.9709 | 0.0531 |
| best_single[vgg19]+T | test | 0.6984 | 0.7148 | 0.6993 | 0.7073 | 0.8722 | 0.4228 | 0.8889 | 0.3218 | 0.9806 | 0.0532 |

Ensemble vs best-single on **test**: Macro-F1 +0.0227, QWK -0.0004, KL1-F1 +0.0068. Differences < ~0.005 are within noise for n=1240 (no significance test run).

## 3. Per-class (test, ensemble)

| class | precision | recall | f1 | support |
|---|--:|--:|--:|--:|
| KL0 | 0.766 | 0.853 | 0.807 | 476 |
| KL1 | 0.436 | 0.423 | 0.430 | 227 |
| KL2 | 0.808 | 0.643 | 0.716 | 328 |
| KL3 | 0.784 | 0.871 | 0.825 | 171 |
| KL4 | 0.897 | 0.921 | 0.909 | 38 |

Top test confusions: KL1->KL0(100), KL0->KL1(62), KL2->KL1(60), KL2->KL3(34), KL1->KL2(28), KL2->KL0(23)
KL4 predicted as: {'KL0': 0, 'KL1': 0, 'KL2': 0, 'KL3': 3, 'KL4': 35}

## 4. Calibration

Temperature scaling (1 param per model, fit on validation NLL). Per-model and ensemble ECE (test) are in `calibration.csv`. Ensemble test ECE = **0.0444**. Reliability diagram: `reliability_diagram.png`.

## 5. Abstention (selective prediction)

Confidence threshold chosen on VAL for >= 90% selective accuracy: **0.72**.

Applied to TEST: coverage **42.0%** (auto-grade 521 / 1240), selective accuracy **0.9021**, selective Macro-F1 0.7349. The remaining 58.0% are 'refer to clinician'.

Full curve: `abstention_curve.csv`, `abstention_curve.png`.

## 6. Binary OA screening (test)  -  {KL0,KL1} vs {KL2,KL3,KL4}

- ROC-AUC **0.9649** | average precision 0.9616
- Sensitivity at specificity >= 0.90: **0.883**
- Specificity at sensitivity >= 0.90: **0.878**
- At p(OA) >= 0.5: acc 0.8968, sens 0.818, spec 0.957, F1 0.873  (n_OA 537, n_noOA 703)

## 7. 3-class (test)  -  Normal(KL0) / Early(KL1-2) / Advanced(KL3-4)

- accuracy **0.7976** | macro-F1 **0.8120** | weighted-F1 0.7972 | QWK 0.7985

| true \ pred | Normal(KL0) | Early(KL1-2) | Advanced(KL3-4) |
|---|--:|--:|--:|
| Normal(KL0) | 368 | 107 | 1 |
| Early(KL1-2) | 88 | 434 | 33 |
| Advanced(KL3-4) | 0 | 22 | 187 |

## 8. Interpretation

- The **5-class** task is still bounded by KL1 vs KL0/KL2 ambiguity; ensembling + calibration improve robustness and confidence quality more than raw Macro-F1.
- The **binary OA screen** and **abstention** views are the deployable story: report coverage at a fixed selective accuracy and the OA-vs-not AUC.
- Nothing here was tuned on test; the test set saw exactly one frozen evaluation pass.


## 9. Next

- If gains are marginal: try `--tta hflip`, add/remove a member, or move to an ordinal head (CORAL/CORN) and/or a higher-resolution retrain of the top 2 (Phase 6.3 / 6.4).
