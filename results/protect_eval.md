# Linguistic-protection detector evaluation

Annotated sentences: **200** (without any protected word: 12); blind sheet `blind-v2`
Sources: vietnews_dev, vinli_dev

| tier | gold positives | predicted positives | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 927 | 894 | 828 | 66 | 99 | 0.926 | 0.893 | 0.909 |
| T2 | 86 | 89 | 85 | 4 | 1 | 0.955 | 0.988 | 0.971 |
| T3 | 49 | 40 | 40 | 0 | 9 | 1.000 | 0.816 | 0.899 |

The report is an input to π configuration selection; do not select π before this file is regenerated.
