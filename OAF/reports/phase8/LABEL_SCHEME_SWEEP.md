# Phase 8 - label-scheme sweep (KD MobileNetV2, recipe = K6_kd_ens_long)

Same KD recipe, four label schemes. Test scored once per scheme.

| scheme | classes | val Acc | val MacroF1 | test Acc | test MacroF1 | test wF1 | test QWK | test AUC | params |
|---|--|--|--|--|--|--|--|--|--|
| 5class | 5 | 0.643 | 0.6848 | **0.675** | **0.6941** | 0.672 | 0.8385 | None | 2,230,277 |
| 4class_kl01 | 4 | 0.7674 | 0.7469 | **0.7879** | **0.7594** | 0.7569 | 0.824 | None | 2,228,996 |
| 3class | 3 | 0.6931 | 0.7034 | **0.7008** | **0.7129** | 0.681 | 0.6557 | None | 2,227,715 |
| binary | 2 | 0.8263 | 0.8143 | **0.8694** | **0.8616** | 0.866 | None | 0.9432 | 2,226,434 |

## Per-class test F1

- **5class**: {'KL0': 0.772, 'KL1': 0.357, 'KL2': 0.6487, 'KL3': 0.8095, 'KL4': 0.8831}  (support {'KL0': 476, 'KL1': 227, 'KL2': 328, 'KL3': 171, 'KL4': 38})
- **4class_kl01**: {'KL0-1': 0.8669, 'KL2': 0.4803, 'KL3': 0.8071, 'KL4': 0.8831}  (support {'KL0-1': 703, 'KL2': 328, 'KL3': 171, 'KL4': 38})
- **3class**: {'Normal': 0.5378, 'Early': 0.7337, 'Advanced': 0.8672}  (support {'Normal': 476, 'Early': 555, 'Advanced': 209})
- **binary**: {'No-OA': 0.8944, 'OA': 0.8288}  (support {'No-OA': 703, 'OA': 537})

## Note
- 5class is the clinical KL standard. 4class_kl01 folds the noisy KL0/KL1 boundary.
- Higher accuracy on a coarser scheme is expected; weigh it against lost clinical granularity.
- KL4 stays its own class in 5class / 4class because its F1 is already ~0.88 (not the weak point).