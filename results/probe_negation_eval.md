# Negation-probe audit

Hand-checked premises: **200**
Sources: vinli_dev, vinli_train

| measure | value |
|---|---:|
| gold negation | 99 |
| probe targets | 100 |
| TP / FP / FN | 97 / 3 / 2 |
| precision | 0.970 |
| recall (from 100 unedited premises) | 0.980 |
| F1 | 0.975 |
| G6 (precision ≥ 0.95) | PASS |

The probe deletes only words selected by `is_negation()` after POS/context disambiguation.
