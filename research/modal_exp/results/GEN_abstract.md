# Generalisation axis: abstract

20 phrases; 176,763 held-out generic positions for the final-layer ROC; 18,622 generic positions with lens states for the per-layer thresholds. Rows fitted on up to 1,200 contexts per phrase; 200 held out.

## Head: lr_nobias

Row redundancy: mean pairwise cosine 0.096, max 0.169, participation ratio 16.5 of 20; mean row norm 3.98.

### Summary over phrases

| metric | mean | median | min | max |
|---|---|---|---|---|
| AUROC vs generic | 0.980 | 0.982 | 0.960 | 0.995 |
| TPR @ FPR 1e-4 (generic) | 0.407 | 0.416 | 0.131 | 0.773 |
| TPR @ FPR 1e-3 (generic) | 0.623 | 0.600 | 0.338 | 0.856 |
| TPR @ FPR 1e-2 (generic) | 0.812 | 0.818 | 0.646 | 0.933 |
| FPR @ TPR 0.5 (generic) | 0.001 | 0.000 | 0.000 | 0.003 |
| FPR @ TPR 0.9 (generic) | 0.045 | 0.029 | 0.003 | 0.117 |
| AUROC vs siblings | 0.918 | 0.936 | 0.817 | 0.985 |
| TPR @ FPR 1e-2 (siblings) | 0.535 | 0.579 | 0.045 | 0.835 |
| inside-phrase TPR @ generic FPR 1e-2 | 0.262 | 0.190 | 0.062 | 0.700 |
| own top-10 (final layer) | 0.560 | 0.554 | 0.278 | 0.761 |
| first-token top-10 | 0.477 | 0.468 | 0.126 | 0.717 |
| generic top-10 | 0.001 | 0.001 | 0.000 | 0.002 |
| mass / prior | 3.049 | 3.195 | 0.159 | 7.664 |
| earliest R-lens layer with TPR@1e-2 ≥ 0.5 | 30.500 | 24.000 | 8.000 | 99.000 |
| earliest J-lens layer with TPR@1e-2 ≥ 0.5 | 26.950 | 24.000 | 12.000 | 99.000 |
| earliest logit-lens layer with TPR@1e-2 ≥ 0.5 | 32.100 | 24.000 | 20.000 | 99.000 |

### Per-layer TPR at FPR 1e-2 (mean over phrases)

| lens | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| rlens phrase TPR@1e-2 | 0.14 | 0.20 | 0.21 | 0.39 | 0.62 | 0.71 |
| jlens phrase TPR@1e-2 | 0.11 | 0.19 | 0.21 | 0.41 | 0.66 | 0.72 |
| logit phrase TPR@1e-2 | 0.10 | 0.12 | 0.13 | 0.28 | 0.56 | 0.67 |
| rlens phrase top-10 | 0.11 | 0.15 | 0.24 | 0.41 | 0.55 | 0.52 |
| jlens phrase top-10 | 0.41 | 0.55 | 0.55 | 0.61 | 0.65 | 0.58 |
| logit phrase top-10 | 0.10 | 0.12 | 0.22 | 0.44 | 0.58 | 0.59 |
| rlens real first token top-10 | 0.02 | 0.05 | 0.06 | 0.20 | 0.39 | 0.45 |

### Position profile: TPR at the generic FPR 1e-2 threshold by position relative to the phrase (offset 0 = token before; 1.. = inside; last = after)

| phrase | -1 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| betrayal | 0.50 | 0.83 | 0.13 | 0.24 | 0.44 | 0.32 |
| regret | 0.57 | 0.87 | 0.16 | 0.23 | 0.25 | — |
| forgiveness | 0.51 | 0.82 | 0.08 | 0.25 | 0.03 | — |
| jealousy | 0.42 | 0.77 | 0.31 | 0.34 | 0.28 | — |
| nostalgia | 0.43 | 0.69 | 0.34 | 0.39 | 0.42 | 0.37 |
| gratitude | 0.56 | 0.91 | 0.19 | 0.21 | — | — |
| loneliness | 0.42 | 0.74 | 0.13 | 0.42 | — | — |
| curiosity | 0.34 | 0.68 | 0.07 | 0.24 | 0.12 | — |
| irony | 0.25 | 0.65 | 0.14 | 0.29 | — | — |
| hypocrisy | 0.64 | 0.89 | 0.09 | 0.55 | 0.56 | — |
| ambition | 0.36 | 0.78 | 0.42 | 0.32 | — | — |
| bankruptcy | 0.62 | 0.86 | 0.35 | 0.34 | 0.41 | — |
| inflation | 0.64 | 0.92 | 0.18 | 0.49 | — | — |
| negotiation | 0.38 | 0.78 | 0.14 | 0.24 | — | — |
| democracy | 0.43 | 0.81 | 0.18 | 0.18 | — | — |
| evolution | 0.41 | 0.76 | 0.06 | 0.24 | — | — |
| photosynthesis | 0.77 | 0.93 | 0.37 | 0.59 | 0.80 | 0.71 |
| gravity | 0.49 | 0.84 | 0.32 | 0.37 | — | — |
| entropy | 0.65 | 0.81 | 0.61 | 0.56 | 0.55 | — |
| recursion | 0.71 | 0.89 | 0.70 | 0.66 | — | — |

### Per phrase

| phrase | group | n | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.5 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R | earliest J | earliest logit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| betrayal | emotion | 197 | 0.981 | 0.26 | 0.55 | 0.83 | 0.0008 | 0.931 | 0.50 | 0.17 | 0.54 | 0.38 | 7.7 | 24 | 24 | 28 |
| regret | emotion | 197 | 0.993 | 0.47 | 0.70 | 0.87 | 0.0002 | 0.946 | 0.58 | 0.17 | 0.64 | 0.45 | 3.4 | 24 | 24 | 24 |
| forgiveness | emotion | 197 | 0.986 | 0.46 | 0.61 | 0.82 | 0.0002 | 0.939 | 0.58 | 0.07 | 0.58 | 0.55 | 2.5 | 24 | 24 | 24 |
| jealousy | emotion | 198 | 0.967 | 0.38 | 0.59 | 0.77 | 0.0005 | 0.847 | 0.41 | 0.30 | 0.54 | 0.38 | 3.0 | 28 | 28 | 28 |
| nostalgia | emotion | 198 | 0.967 | 0.27 | 0.44 | 0.69 | 0.0021 | 0.838 | 0.29 | 0.32 | 0.35 | 0.22 | 2.5 | 99 | 24 | 99 |
| gratitude | emotion | 197 | 0.994 | 0.51 | 0.79 | 0.91 | 0.0001 | 0.954 | 0.63 | 0.20 | 0.76 | 0.67 | 1.9 | 24 | 24 | 24 |
| loneliness | emotion | 199 | 0.970 | 0.25 | 0.52 | 0.74 | 0.0008 | 0.867 | 0.40 | 0.12 | 0.55 | 0.41 | 5.1 | 24 | 24 | 24 |
| curiosity | emotion | 199 | 0.960 | 0.21 | 0.51 | 0.68 | 0.0010 | 0.840 | 0.33 | 0.08 | 0.50 | 0.39 | 3.8 | 28 | 28 | 28 |
| irony | character | 198 | 0.963 | 0.13 | 0.34 | 0.65 | 0.0035 | 0.817 | 0.05 | 0.14 | 0.28 | 0.13 | 1.6 | 99 | 99 | 99 |
| hypocrisy | character | 199 | 0.992 | 0.52 | 0.72 | 0.89 | 0.0001 | 0.917 | 0.35 | 0.25 | 0.58 | 0.44 | 1.9 | 24 | 24 | 24 |
| ambition | character | 199 | 0.977 | 0.26 | 0.50 | 0.78 | 0.0009 | 0.922 | 0.50 | 0.41 | 0.53 | 0.33 | 3.8 | 24 | 24 | 28 |
| bankruptcy | economy | 198 | 0.991 | 0.52 | 0.76 | 0.86 | 0.0001 | 0.963 | 0.76 | 0.34 | 0.74 | 0.72 | 4.3 | 24 | 24 | 24 |
| inflation | economy | 199 | 0.995 | 0.60 | 0.80 | 0.92 | 0.0000 | 0.985 | 0.80 | 0.18 | 0.72 | 0.70 | 1.5 | 20 | 20 | 24 |
| negotiation | economy | 198 | 0.975 | 0.28 | 0.54 | 0.78 | 0.0007 | 0.936 | 0.58 | 0.14 | 0.56 | 0.48 | 3.4 | 24 | 24 | 24 |
| democracy | economy | 197 | 0.989 | 0.22 | 0.47 | 0.81 | 0.0012 | 0.968 | 0.59 | 0.18 | 0.53 | 0.61 | 4.8 | 28 | 24 | 28 |
| evolution | science | 199 | 0.983 | 0.35 | 0.56 | 0.76 | 0.0006 | 0.943 | 0.59 | 0.06 | 0.60 | 0.55 | 4.4 | 28 | 28 | 28 |
| photosynthesis | science | 194 | 0.991 | 0.77 | 0.86 | 0.93 | 0.0000 | 0.977 | 0.84 | 0.53 | 0.59 | 0.68 | 0.2 | 8 | 12 | 20 |
| gravity | science | 199 | 0.978 | 0.45 | 0.71 | 0.84 | 0.0002 | 0.935 | 0.64 | 0.31 | 0.68 | 0.56 | 4.6 | 24 | 24 | 24 |
| entropy | science | 198 | 0.973 | 0.62 | 0.72 | 0.81 | 0.0000 | 0.892 | 0.62 | 0.57 | 0.48 | 0.56 | 0.3 | 20 | 20 | 20 |
| recursion | science | 197 | 0.983 | 0.60 | 0.79 | 0.89 | 0.0000 | 0.946 | 0.66 | 0.70 | 0.44 | 0.33 | 0.4 | 12 | 16 | 20 |

## Head: probe

Row redundancy: mean pairwise cosine 0.046, max 0.668, participation ratio 15.5 of 20; mean row norm 0.13.

### Summary over phrases

| metric | mean | median | min | max |
|---|---|---|---|---|
| AUROC vs generic | 0.957 | 0.969 | 0.871 | 0.989 |
| TPR @ FPR 1e-4 (generic) | 0.205 | 0.154 | 0.020 | 0.558 |
| TPR @ FPR 1e-3 (generic) | 0.422 | 0.436 | 0.056 | 0.744 |
| TPR @ FPR 1e-2 (generic) | 0.658 | 0.698 | 0.228 | 0.905 |
| FPR @ TPR 0.5 (generic) | 0.007 | 0.002 | 0.000 | 0.050 |
| FPR @ TPR 0.9 (generic) | 0.122 | 0.089 | 0.009 | 0.390 |
| AUROC vs siblings | 0.873 | 0.901 | 0.730 | 0.974 |
| TPR @ FPR 1e-2 (siblings) | 0.460 | 0.453 | 0.020 | 0.834 |
| inside-phrase TPR @ generic FPR 1e-2 | 0.920 | 0.967 | 0.468 | 1.000 |
| earliest R-lens layer with TPR@1e-2 ≥ 0.5 | 24.350 | 20.000 | 8.000 | 99.000 |
| earliest J-lens layer with TPR@1e-2 ≥ 0.5 | 23.550 | 20.000 | 8.000 | 99.000 |
| earliest logit-lens layer with TPR@1e-2 ≥ 0.5 | 22.350 | 20.000 | 8.000 | 99.000 |

### Per-layer TPR at FPR 1e-2 (mean over phrases)

| lens | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| rlens phrase TPR@1e-2 | 0.23 | 0.33 | 0.37 | 0.52 | 0.65 | 0.70 |
| jlens phrase TPR@1e-2 | 0.22 | 0.33 | 0.38 | 0.55 | 0.67 | 0.70 |
| logit phrase TPR@1e-2 | 0.34 | 0.38 | 0.43 | 0.58 | 0.68 | 0.71 |
| rlens phrase top-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| jlens phrase top-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| logit phrase top-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| rlens real first token top-10 | 0.02 | 0.05 | 0.06 | 0.20 | 0.39 | 0.45 |

### Position profile: TPR at the generic FPR 1e-2 threshold by position relative to the phrase (offset 0 = token before; 1.. = inside; last = after)

| phrase | -1 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| betrayal | 0.58 | 0.72 | 0.99 | 0.74 | 0.69 | 0.58 |
| regret | 0.45 | 0.72 | 0.98 | 0.57 | 0.50 | — |
| forgiveness | 0.69 | 0.81 | 0.96 | 0.82 | 0.74 | — |
| jealousy | 0.49 | 0.68 | 0.96 | 0.60 | 0.42 | — |
| nostalgia | 0.31 | 0.53 | 0.96 | 0.54 | 0.92 | 0.45 |
| gratitude | 0.54 | 0.85 | 1.00 | 0.72 | — | — |
| loneliness | 0.49 | 0.69 | 0.98 | 0.62 | — | — |
| curiosity | 0.25 | 0.58 | 0.95 | 0.57 | 0.38 | — |
| irony | 0.24 | 0.45 | 0.98 | 0.70 | — | — |
| hypocrisy | 0.60 | 0.78 | 0.99 | 0.74 | 0.55 | — |
| ambition | 0.30 | 0.68 | 0.98 | 0.52 | — | — |
| bankruptcy | 0.75 | 0.85 | 0.96 | 0.75 | 0.77 | — |
| inflation | 0.81 | 0.90 | 0.99 | 0.85 | — | — |
| negotiation | 0.46 | 0.70 | 0.98 | 0.64 | — | — |
| democracy | 0.57 | 0.78 | 1.00 | 0.68 | — | — |
| evolution | 0.48 | 0.72 | 1.00 | 0.57 | — | — |
| photosynthesis | 0.25 | 0.44 | 0.96 | 0.58 | 0.39 | 0.52 |
| gravity | 0.46 | 0.62 | 0.94 | 0.52 | — | — |
| entropy | 0.31 | 0.42 | 0.75 | 0.25 | 0.14 | — |
| recursion | 0.11 | 0.23 | 0.47 | 0.17 | — | — |

### Per phrase

| phrase | group | n | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.5 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R | earliest J | earliest logit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| betrayal | emotion | 197 | 0.981 | 0.08 | 0.34 | 0.72 | 0.0027 | 0.903 | 0.42 | 0.97 | — | — | — | 20 | 20 | 20 |
| regret | emotion | 197 | 0.973 | 0.18 | 0.45 | 0.72 | 0.0017 | 0.888 | 0.43 | 0.96 | — | — | — | 20 | 20 | 20 |
| forgiveness | emotion | 197 | 0.977 | 0.20 | 0.55 | 0.81 | 0.0008 | 0.916 | 0.61 | 0.96 | — | — | — | 12 | 20 | 12 |
| jealousy | emotion | 198 | 0.954 | 0.29 | 0.48 | 0.68 | 0.0012 | 0.787 | 0.31 | 0.92 | — | — | — | 24 | 24 | 20 |
| nostalgia | emotion | 198 | 0.948 | 0.11 | 0.30 | 0.53 | 0.0081 | 0.869 | 0.33 | 0.95 | — | — | — | 24 | 24 | 24 |
| gratitude | emotion | 197 | 0.985 | 0.38 | 0.66 | 0.85 | 0.0003 | 0.940 | 0.68 | 1.00 | — | — | — | 12 | 16 | 8 |
| loneliness | emotion | 199 | 0.972 | 0.14 | 0.41 | 0.69 | 0.0023 | 0.856 | 0.31 | 0.97 | — | — | — | 24 | 24 | 24 |
| curiosity | emotion | 199 | 0.935 | 0.12 | 0.35 | 0.58 | 0.0035 | 0.800 | 0.31 | 0.95 | — | — | — | 24 | 24 | 28 |
| irony | character | 198 | 0.958 | 0.03 | 0.16 | 0.45 | 0.0146 | 0.737 | 0.02 | 0.98 | — | — | — | 99 | 99 | 99 |
| hypocrisy | character | 199 | 0.977 | 0.39 | 0.59 | 0.78 | 0.0004 | 0.929 | 0.50 | 0.97 | — | — | — | 20 | 20 | 20 |
| ambition | character | 199 | 0.960 | 0.20 | 0.40 | 0.68 | 0.0031 | 0.919 | 0.53 | 0.98 | — | — | — | 24 | 24 | 24 |
| bankruptcy | economy | 198 | 0.980 | 0.56 | 0.71 | 0.85 | 0.0001 | 0.922 | 0.75 | 0.96 | — | — | — | 20 | 20 | 8 |
| inflation | economy | 199 | 0.989 | 0.56 | 0.74 | 0.90 | 0.0001 | 0.974 | 0.83 | 1.00 | — | — | — | 8 | 8 | 8 |
| negotiation | economy | 198 | 0.972 | 0.07 | 0.44 | 0.70 | 0.0017 | 0.922 | 0.59 | 0.99 | — | — | — | 20 | 20 | 20 |
| democracy | economy | 197 | 0.983 | 0.16 | 0.48 | 0.78 | 0.0011 | 0.968 | 0.72 | 0.99 | — | — | — | 20 | 20 | 20 |
| evolution | science | 199 | 0.966 | 0.15 | 0.43 | 0.72 | 0.0015 | 0.898 | 0.59 | 0.99 | — | — | — | 24 | 24 | 24 |
| photosynthesis | science | 194 | 0.926 | 0.06 | 0.22 | 0.44 | 0.0157 | 0.905 | 0.46 | 0.78 | — | — | — | 20 | 8 | 8 |
| gravity | science | 199 | 0.934 | 0.35 | 0.47 | 0.62 | 0.0017 | 0.769 | 0.45 | 0.94 | — | — | — | 20 | 20 | 20 |
| entropy | science | 198 | 0.871 | 0.06 | 0.22 | 0.42 | 0.0255 | 0.730 | 0.19 | 0.69 | — | — | — | 24 | 16 | 20 |
| recursion | science | 197 | 0.892 | 0.02 | 0.06 | 0.23 | 0.0499 | 0.827 | 0.19 | 0.47 | — | — | — | 28 | 20 | 20 |
