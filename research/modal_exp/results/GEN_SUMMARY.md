# Generalisation study: cross-axis summary

Bias-free logistic rows (extended softmax, positives at the token before the phrase) unless stated. Means over phrases of each axis; 200 held-out contexts per phrase; generic negatives = 176,763 held-out FineWeb positions (final layer) and 18,622 positions with lens states (per-layer thresholds).

## lr_nobias

| axis | phrases | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.9 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R (median) | earliest J | earliest logit | never (R) | row cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nouns | 45 | 0.988 | 0.53 | 0.75 | 0.89 | 0.015 | 0.899 | 0.53 | 0.47 | 0.63 | 0.66 | 2.9 | 20 | 20 | 24 | 0.00 | 0.074 |
| verbs | 43 | 0.983 | 0.35 | 0.58 | 0.82 | 0.039 | 0.876 | 0.43 | 0.22 | 0.45 | 0.51 | 1.9 | 20 | 20 | 20 | 0.05 | 0.119 |
| abstract | 20 | 0.980 | 0.41 | 0.62 | 0.81 | 0.045 | 0.918 | 0.53 | 0.26 | 0.56 | 0.48 | 3.0 | 24 | 24 | 24 | 0.10 | 0.096 |
| events | 28 | 0.992 | 0.71 | 0.87 | 0.94 | 0.005 | 0.886 | 0.58 | 0.66 | 0.65 | 0.72 | 1.6 | 8 | 18 | 18 | 0.00 | 0.069 |
| xling | 48 | 0.996 | 0.74 | 0.88 | 0.96 | 0.003 | 0.982 | 0.84 | 0.54 | 0.79 | 0.70 | 0.9 | 24 | 24 | 24 | 0.02 | 0.063 |
| joint_nouns_verbs_abstract_events | 136 | 0.981 | 0.48 | 0.68 | 0.84 | 0.034 | 0.880 | 0.50 | 0.43 | 0.52 | 0.60 | 1.5 | 20 | 20 | 20 | 0.04 | 0.065 |

Per-layer TPR at FPR 1e-2 under the R-lens (mean over phrases): layers [8, 12, 16, 20, 24, 28]

| axis | L8 | L12 | L16 | L20 | L24 | L28 | first token top-10 (R) at L8 / L12 / L16 / L20 / L24 / L28 |
|---|---|---|---|---|---|---|---|
| nouns | 0.28 | 0.35 | 0.36 | 0.56 | 0.77 | 0.84 | 0.04 / 0.06 / 0.06 / 0.14 / 0.32 / 0.60 |
| verbs | 0.21 | 0.27 | 0.35 | 0.54 | 0.73 | 0.76 | 0.11 / 0.08 / 0.09 / 0.09 / 0.10 / 0.33 |
| abstract | 0.14 | 0.20 | 0.21 | 0.39 | 0.62 | 0.71 | 0.02 / 0.05 / 0.06 / 0.20 / 0.39 / 0.45 |
| events | 0.51 | 0.55 | 0.56 | 0.75 | 0.89 | 0.92 | 0.12 / 0.08 / 0.08 / 0.12 / 0.26 / 0.59 |
| xling | 0.11 | 0.14 | 0.12 | 0.28 | 0.62 | 0.83 | 0.01 / 0.02 / 0.02 / 0.06 / 0.19 / 0.51 |
| joint_nouns_verbs_abstract_events | 0.35 | 0.42 | 0.44 | 0.61 | 0.78 | 0.82 | 0.08 / 0.07 / 0.07 / 0.13 / 0.25 / 0.49 |

Distribution of the earliest R-lens layer with TPR@1e-2 ≥ 0.5 (count of phrases; 99 = never):

| axis | 8 | 12 | 16 | 20 | 24 | 28 | 99 |
|---|---|---|---|---|---|---|---|
| nouns | 7 | 3 | 3 | 15 | 14 | 3 | 0 |
| verbs | 1 | 4 | 1 | 20 | 14 | 1 | 2 |
| abstract | 1 | 1 | 0 | 2 | 10 | 4 | 2 |
| events | 15 | 3 | 3 | 4 | 3 | 0 | 0 |
| xling | 1 | 1 | 1 | 7 | 25 | 12 | 1 |
| joint_nouns_verbs_abstract_events | 36 | 8 | 6 | 43 | 33 | 5 | 5 |

## probe

| axis | phrases | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.9 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R (median) | earliest J | earliest logit | never (R) | row cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nouns | 45 | 0.965 | 0.30 | 0.58 | 0.78 | 0.079 | 0.887 | 0.56 | 0.94 | — | — | — | 12 | 12 | 8 | 0.02 | 0.042 |
| verbs | 43 | 0.895 | 0.07 | 0.21 | 0.46 | 0.248 | 0.831 | 0.35 | 0.85 | — | — | — | 24 | 24 | 24 | 0.47 | 0.049 |
| abstract | 20 | 0.957 | 0.20 | 0.42 | 0.66 | 0.122 | 0.873 | 0.46 | 0.92 | — | — | — | 20 | 20 | 20 | 0.05 | 0.046 |
| events | 28 | 0.924 | 0.43 | 0.58 | 0.72 | 0.140 | 0.842 | 0.46 | 0.80 | — | — | — | 8 | 8 | 8 | 0.14 | 0.073 |
| xling | 48 | 0.993 | 0.82 | 0.90 | 0.96 | 0.014 | 0.993 | 0.96 | 0.99 | — | — | — | 8 | 12 | 8 | 0.00 | 0.021 |
| joint_nouns_verbs_abstract_events | 136 | 0.895 | 0.15 | 0.31 | 0.50 | 0.239 | 0.803 | 0.33 | 0.74 | — | — | — | 24 | 20 | 20 | 0.24 | 0.034 |

Per-layer TPR at FPR 1e-2 under the R-lens (mean over phrases): layers [8, 12, 16, 20, 24, 28]

| axis | L8 | L12 | L16 | L20 | L24 | L28 | first token top-10 (R) at L8 / L12 / L16 / L20 / L24 / L28 |
|---|---|---|---|---|---|---|---|
| nouns | 0.44 | 0.51 | 0.53 | 0.69 | 0.79 | 0.82 | 0.04 / 0.06 / 0.06 / 0.14 / 0.32 / 0.60 |
| verbs | 0.18 | 0.23 | 0.28 | 0.39 | 0.48 | 0.51 | 0.11 / 0.08 / 0.09 / 0.09 / 0.10 / 0.33 |
| abstract | 0.23 | 0.33 | 0.37 | 0.52 | 0.65 | 0.70 | 0.02 / 0.05 / 0.06 / 0.20 / 0.39 / 0.45 |
| events | 0.52 | 0.58 | 0.59 | 0.70 | 0.78 | 0.80 | 0.12 / 0.08 / 0.08 / 0.12 / 0.26 / 0.59 |
| xling | 0.53 | 0.58 | 0.55 | 0.78 | 0.90 | 0.95 | 0.01 / 0.02 / 0.02 / 0.06 / 0.19 / 0.51 |
| joint_nouns_verbs_abstract_events | 0.26 | 0.31 | 0.33 | 0.45 | 0.58 | 0.63 | 0.08 / 0.07 / 0.07 / 0.13 / 0.25 / 0.49 |

Distribution of the earliest R-lens layer with TPR@1e-2 ≥ 0.5 (count of phrases; 99 = never):

| axis | 8 | 12 | 16 | 20 | 24 | 28 | 99 |
|---|---|---|---|---|---|---|---|
| nouns | 16 | 7 | 4 | 14 | 3 | 0 | 1 |
| verbs | 1 | 1 | 3 | 12 | 5 | 1 | 20 |
| abstract | 1 | 2 | 0 | 8 | 7 | 1 | 1 |
| events | 20 | 1 | 0 | 1 | 2 | 0 | 4 |
| xling | 26 | 7 | 1 | 12 | 2 | 0 | 0 |
| joint_nouns_verbs_abstract_events | 22 | 4 | 8 | 31 | 29 | 9 | 33 |

## Joint fit (joint_nouns_verbs_abstract_events) vs separate fits, bias-free row, per axis

| axis | phrases | AUROC gen sep → joint | TPR@1e-3 sep → joint | AUROC sib sep → joint | TPR@1e-2 sib sep → joint | mass/prior sep → joint | earliest R sep → joint |
|---|---|---|---|---|---|---|---|
| nouns | 45 | 0.988 → 0.983 | 0.75 → 0.74 | 0.899 → 0.885 | 0.53 → 0.51 | 2.9 → 1.7 | 19 → 17 |
| verbs | 43 | 0.983 → 0.980 | 0.58 → 0.55 | 0.876 → 0.872 | 0.43 → 0.42 | 1.9 → 1.5 | 24 → 27 |
| abstract | 20 | 0.980 → 0.975 | 0.62 → 0.61 | 0.918 → 0.914 | 0.53 → 0.55 | 3.0 → 1.6 | 30 → 25 |
| events | 28 | 0.992 → 0.986 | 0.87 → 0.84 | 0.886 → 0.861 | 0.58 → 0.54 | 1.6 → 1.0 | 13 → 11 |

Joint head row redundancy: mean cosine 0.065, participation ratio 84.3 of 136.
