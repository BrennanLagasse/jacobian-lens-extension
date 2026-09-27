# Phrase-head benchmark on Qwen3.5-9B-Base

15 phrases (blackmail, plagiarism, cheating, forgery, Connecticut, Rhode Island, New Hampshire, George Washington, Abraham Lincoln, John Adams, New York, New Jersey, John Quincy Adams, Massachusetts, Vermont); 2942 held-out pre-phrase positions (84% single-occurrence); 176,763 held-out generic positions from 500 FineWeb docs; prior sum 4.2e-04; a real W_U row has generic logit std 1.80. Single-token phrases: blackmail, plagiarism, cheating, Connecticut, Massachusetts, Vermont.

## Headline per head (mean over phrases; held-out)

| scale | head | AUROC own vs generic | AUROC own vs first-token sibling | AUROC own vs category sibling | own top-10 | own median rank | rank / first-token rank (median log) | generic top-10 | mass / prior | ECE (prior-weighted) |
|---|---|---|---|---|---|---|---|---|---|---|
| 150 | random row (floor) | 0.508 | 0.514 | 0.508 | 0.00 | 29835 | +7.45 | 0.0003 | 4.8 | 4.67e-05 |
| 150 | Brennan (proj+unit) | 0.970 | 0.589 | 0.647 | 0.95 | 1 | -2.68 | 0.2019 | 8544.6 | 2.23e-02 |
| 150 | avg W_U rows | 0.966 | 0.575 | 0.734 | 0.30 | 43 | +1.17 | 0.0001 | 1.2 | 2.45e-05 |
| 150 | LDA / whitening | 0.976 | 0.666 | 0.715 | 0.66 | 2 | -1.29 | 0.0007 | 60.3 | 3.79e-04 |
| 150 | LDA, W_U-scaled | 0.976 | 0.666 | 0.715 | 0.68 | 2 | -1.19 | 0.0006 | 61.5 | 1.98e-04 |
| 150 | LR, L2=1e-5 | 0.975 | 0.704 | 0.772 | 0.38 | 31 | +1.18 | 0.0002 | 0.6 | 2.44e-05 |
| 150 | LR, L2=1e-4 | 0.976 | 0.689 | 0.760 | 0.40 | 19 | +0.69 | 0.0002 | 0.3 | 1.64e-05 |
| 150 | LR, L2=1e-3 | 0.968 | 0.662 | 0.730 | 0.41 | 17 | +0.58 | 0.0003 | 0.5 | 1.15e-05 |
| 150 | logistic regression (val-loss pick) | 0.968 | 0.662 | 0.730 | 0.41 | 17 | +0.58 | 0.0003 | 0.5 | 1.15e-05 |
| 150 | LR no-bias, L2=1e-6 | 0.980 | 0.705 | 0.773 | 0.46 | 26 | +0.46 | 0.0001 | 1.2 | 2.64e-05 |
| 150 | LR no-bias, L2=1e-7 | 0.981 | 0.713 | 0.779 | 0.43 | 37 | +0.98 | 0.0001 | 2.4 | 3.37e-05 |
| 150 | LR no-bias, L2=1e-8 | 0.981 | 0.717 | 0.783 | 0.41 | 135 | +1.96 | 0.0001 | 4.2 | 4.12e-05 |
| 150 | LR, no bias (val-loss pick) | 0.980 | 0.705 | 0.773 | 0.46 | 26 | +0.46 | 0.0001 | 1.2 | 2.64e-05 |
| 600 | Brennan (proj+unit) | 0.970 | 0.583 | 0.642 | 0.95 | 1 | -2.69 | 0.2035 | 8075.1 | 2.25e-02 |
| 600 | LDA / whitening | 0.979 | 0.664 | 0.714 | 0.68 | 1 | -1.41 | 0.0008 | 72.9 | 4.64e-04 |
| 600 | LDA, W_U-scaled | 0.979 | 0.664 | 0.714 | 0.69 | 2 | -1.30 | 0.0007 | 68.0 | 2.15e-04 |
| 600 | LR, L2=1e-5 | 0.980 | 0.735 | 0.812 | 0.43 | 32 | +0.94 | 0.0003 | 2.6 | 4.12e-05 |
| 600 | LR, L2=1e-4 | 0.977 | 0.717 | 0.799 | 0.41 | 20 | +0.62 | 0.0003 | 0.5 | 1.25e-05 |
| 600 | LR, L2=1e-3 | 0.977 | 0.690 | 0.770 | 0.38 | 25 | +0.71 | 0.0002 | 0.3 | 1.27e-05 |
| 600 | logistic regression (val-loss pick) | 0.977 | 0.690 | 0.770 | 0.38 | 25 | +0.71 | 0.0002 | 0.3 | 1.27e-05 |
| 600 | LR no-bias, L2=1e-6 | 0.983 | 0.725 | 0.802 | 0.51 | 8 | -0.15 | 0.0003 | 2.5 | 5.47e-05 |
| 600 | LR no-bias, L2=1e-7 | 0.982 | 0.741 | 0.809 | 0.50 | 11 | +0.20 | 0.0004 | 7.3 | 1.09e-04 |
| 600 | LR no-bias, L2=1e-8 | 0.983 | 0.744 | 0.811 | 0.49 | 22 | +0.85 | 0.0004 | 12.2 | 1.51e-04 |
| 600 | LR, no bias (val-loss pick) | 0.983 | 0.725 | 0.802 | 0.51 | 8 | -0.15 | 0.0003 | 2.5 | 5.47e-05 |
| 2400 | Brennan (proj+unit) | 0.969 | 0.582 | 0.642 | 0.95 | 1 | -2.70 | 0.2031 | 7686.7 | 2.23e-02 |
| 2400 | LDA / whitening | 0.980 | 0.663 | 0.714 | 0.69 | 1 | -1.43 | 0.0008 | 73.6 | 4.80e-04 |
| 2400 | LDA, W_U-scaled | 0.980 | 0.663 | 0.714 | 0.69 | 2 | -1.30 | 0.0007 | 68.0 | 2.21e-04 |
| 2400 | LR, L2=1e-5 | 0.982 | 0.750 | 0.826 | 0.42 | 21 | +0.81 | 0.0004 | 2.7 | 3.39e-05 |
| 2400 | LR, L2=1e-4 | 0.980 | 0.736 | 0.819 | 0.37 | 25 | +0.88 | 0.0003 | 0.4 | 1.27e-05 |
| 2400 | LR, L2=1e-3 | 0.972 | 0.710 | 0.794 | 0.32 | 31 | +1.03 | 0.0002 | 0.3 | 1.43e-05 |
| 2400 | logistic regression (val-loss pick) | 0.980 | 0.736 | 0.819 | 0.37 | 25 | +0.88 | 0.0003 | 0.4 | 1.27e-05 |
| 2400 | LR no-bias, L2=1e-6 | 0.981 | 0.731 | 0.813 | 0.45 | 14 | +0.22 | 0.0003 | 1.0 | 4.57e-05 |
| 2400 | LR no-bias, L2=1e-7 | 0.982 | 0.741 | 0.823 | 0.48 | 10 | +0.09 | 0.0004 | 2.1 | 8.40e-05 |
| 2400 | LR no-bias, L2=1e-8 | 0.982 | 0.741 | 0.824 | 0.48 | 10 | +0.09 | 0.0004 | 2.7 | 9.70e-05 |
| 2400 | LR, no bias (val-loss pick) | 0.981 | 0.731 | 0.813 | 0.45 | 14 | +0.22 | 0.0003 | 1.0 | 4.57e-05 |

## Copy sensitivity (scale 2400): single-occurrence vs copy contexts

| head | AUROC single | AUROC copy | own top-10 single | own top-10 copy | rank single | rank copy |
|---|---|---|---|---|---|---|
| Brennan (proj+unit) | 0.971 | 0.961 | 0.95 | 0.95 | 1 | 1 |
| LDA / whitening | 0.976 | 0.999 | 0.66 | 0.82 | 2 | 1 |
| logistic regression (val-loss pick) | 0.977 | 0.994 | 0.34 | 0.53 | 36 | 7 |
| LR, no bias (val-loss pick) | 0.979 | 0.995 | 0.42 | 0.61 | 19 | 2 |

## Stability: three disjoint 150-context refits (mean ± std)

| head | auroc_generic | top10 | generic_top10 | new_mass | ece | rank_median |
|---|---|---|---|---|---|---|
| Brennan (proj+unit) | 0.9694 ± 0.0008 | 0.9487 ± 0.0019 | 0.2026 ± 0.0005 | 0.3393 ± 0.0031 | 0.0226 ± 0.0002 | 1.0000 ± 0.0000 |
| LDA / whitening | 0.9770 ± 0.0008 | 0.6576 ± 0.0043 | 0.0007 ± 0.0000 | 0.0056 ± 0.0001 | 0.0004 ± 0.0000 | 1.6667 ± 0.4714 |
| logistic regression (val-loss pick) | 0.9682 ± 0.0006 | 0.4051 ± 0.0027 | 0.0003 ± 0.0000 | 0.0003 ± 0.0000 | 0.0000 ± 0.0000 | 18.6667 ± 1.2472 |
| LR, no bias (val-loss pick) | 0.9796 ± 0.0007 | 0.4643 ± 0.0059 | 0.0001 ± 0.0000 | 0.0002 ± 0.0000 | 0.0000 ± 0.0000 | 21.3333 ± 3.6818 |

## First-token confound: heads fitted on the original 10 phrases, scored at sibling positions

At pre-'New York'/'New Jersey' positions, does the 'New Hampshire' row fire? (top10_at_sib = fraction of sibling positions where it enters the top-10; auroc_first_sib = own vs sibling positions.)

| head | phrase | AUROC own vs first-token sibling | top-10 rate at sibling positions | AUROC own vs category | own top-10 |
|---|---|---|---|---|---|
| Brennan (proj+unit) | New Hampshire | 0.589 | 0.974 | 0.548 | 0.990 |
| Brennan (proj+unit) | John Adams | 0.483 | 0.980 | 0.580 | 0.990 |
| Brennan (proj+unit) | Connecticut | — | — | 0.593 | 0.959 |
| Brennan (proj+unit) | Rhode Island | — | — | 0.574 | 0.990 |
| LDA / whitening | New Hampshire | 0.818 | 0.244 | 0.729 | 0.726 |
| LDA / whitening | John Adams | 0.451 | 0.765 | 0.576 | 0.629 |
| LDA / whitening | Connecticut | — | — | 0.670 | 0.638 |
| LDA / whitening | Rhode Island | — | — | 0.734 | 0.706 |
| logistic regression (val-loss pick) | New Hampshire | 0.849 | 0.021 | 0.805 | 0.386 |
| logistic regression (val-loss pick) | John Adams | 0.568 | 0.031 | 0.702 | 0.162 |
| logistic regression (val-loss pick) | Connecticut | — | — | 0.763 | 0.413 |
| logistic regression (val-loss pick) | Rhode Island | — | — | 0.809 | 0.426 |
| LR, no bias (val-loss pick) | New Hampshire | 0.855 | 0.067 | 0.815 | 0.569 |
| LR, no bias (val-loss pick) | John Adams | 0.546 | 0.077 | 0.665 | 0.274 |
| LR, no bias (val-loss pick) | Connecticut | — | — | 0.774 | 0.495 |
| LR, no bias (val-loss pick) | Rhode Island | — | — | 0.811 | 0.513 |

## J-lens vs logit lens: layer at which the token enters the top-10 (scale 2400, all held-out contexts)

Phrase token under each head vs the phrase's real first token (the standard readout). Fraction of held-out contexts where the token is in the extended top-10 at that layer.

| lens | token | L4 | L8 | L12 | L16 | L18 | L20 | L22 | L24 | L26 | L28 | L30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jlens | real first token | 0.00 | 0.00 | 0.03 | 0.05 | 0.08 | 0.14 | 0.14 | 0.23 | 0.29 | 0.42 | 0.52 |
| jlens | phrase token, Brennan (proj+unit) | 0.29 | 0.68 | 0.91 | 0.93 | 0.94 | 0.95 | 0.95 | 0.96 | 0.96 | 0.96 | 0.96 |
| jlens | phrase token, LDA / whitening | 0.00 | 0.01 | 0.01 | 0.01 | 0.04 | 0.12 | 0.21 | 0.51 | 0.62 | 0.65 | 0.67 |
| jlens | phrase token, logistic regression (val-loss pick) | 0.20 | 0.49 | 0.56 | 0.55 | 0.73 | 0.78 | 0.81 | 0.84 | 0.78 | 0.70 | 0.61 |
| jlens | phrase token, LR, no bias (val-loss pick) | 0.06 | 0.14 | 0.14 | 0.15 | 0.24 | 0.33 | 0.35 | 0.42 | 0.44 | 0.47 | 0.44 |
| jlens | phrase in top-10 while first token is not, LDA / whitening | 0.00 | 0.01 | 0.00 | 0.00 | 0.02 | 0.07 | 0.13 | 0.29 | 0.34 | 0.26 | 0.19 |
| jlens | phrase in top-10 while first token is not, LR, no bias (val-loss pick) | 0.06 | 0.14 | 0.13 | 0.13 | 0.19 | 0.23 | 0.25 | 0.22 | 0.19 | 0.13 | 0.05 |
| logit_lens | real first token | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.04 | 0.05 | 0.18 | 0.25 | 0.35 | 0.45 |
| logit_lens | phrase token, Brennan (proj+unit) | 0.00 | 0.08 | 0.11 | 0.20 | 0.38 | 0.54 | 0.64 | 0.77 | 0.80 | 0.83 | 0.91 |
| logit_lens | phrase token, LDA / whitening | 0.00 | 0.00 | 0.01 | 0.03 | 0.10 | 0.29 | 0.41 | 0.64 | 0.69 | 0.71 | 0.74 |
| logit_lens | phrase token, logistic regression (val-loss pick) | 0.41 | 0.43 | 0.40 | 0.45 | 0.48 | 0.59 | 0.61 | 0.71 | 0.72 | 0.72 | 0.45 |
| logit_lens | phrase token, LR, no bias (val-loss pick) | 0.01 | 0.05 | 0.06 | 0.10 | 0.15 | 0.17 | 0.22 | 0.37 | 0.46 | 0.49 | 0.45 |
| logit_lens | phrase in top-10 while first token is not, LDA / whitening | 0.00 | 0.00 | 0.01 | 0.02 | 0.09 | 0.25 | 0.36 | 0.47 | 0.45 | 0.39 | 0.34 |
| logit_lens | phrase in top-10 while first token is not, LR, no bias (val-loss pick) | 0.01 | 0.05 | 0.06 | 0.10 | 0.14 | 0.15 | 0.19 | 0.23 | 0.23 | 0.18 | 0.10 |

Median earliest top-10 layer per phrase (99 = never), J-lens:

| phrase | real first token | proj | lda | lr (bias) | lr no-bias | lr no-bias: never |
|---|---|---|---|---|---|---|
| blackmail | 99 | 12 | 24 | 8 | 99 | 0.69 |
| plagiarism | 20 | 12 | 24 | 12 | 24 | 0.32 |
| cheating | 99 | 12 | 28 | 8 | 20 | 0.35 |
| forgery | 99 | 12 | 24 | 18 | 99 | 0.67 |
| Connecticut | 28 | 8 | 26 | 8 | 20 | 0.21 |
| Rhode Island | 28 | 8 | 24 | 8 | 28 | 0.45 |
| New Hampshire | 30 | 8 | 24 | 12 | 24 | 0.34 |
| George Washington | 99 | 8 | 24 | 8 | 99 | 0.54 |
| Abraham Lincoln | 28 | 8 | 24 | 12 | 26 | 0.42 |
| John Adams | 30 | 8 | 26 | 22 | 99 | 0.76 |
| New York | 30 | 8 | 24 | 4 | 4 | 0.00 |
| New Jersey | 30 | 8 | 28 | 8 | 12 | 0.07 |
| John Quincy Adams | 30 | 8 | 24 | 20 | 99 | 0.81 |
| Massachusetts | 26 | 8 | 24 | 12 | 20 | 0.10 |
| Vermont | 99 | 8 | 26 | 8 | 20 | 0.27 |

## Per-phrase detail at scale 2400

### Brennan (proj+unit)

| phrase | AUROC generic | AUROC first-sib | AUROC cat-sib | own rank | first-tok rank | own top-10 | generic top-10 | mass/prior | ECE |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.955 | — | 0.780 | 1 | 85 | 0.88 | 0.1707 | 28662.4 | 5.5e-02 |
| plagiarism | 0.986 | — | 0.900 | 1 | 11 | 0.91 | 0.1184 | 4534.6 | 2.7e-02 |
| cheating | 0.965 | — | 0.703 | 1 | 45 | 0.85 | 0.1547 | 3501.9 | 3.5e-02 |
| forgery | 0.978 | — | 0.814 | 1 | 18 | 0.89 | 0.1255 | 4945.7 | 7.6e-03 |
| Connecticut | 0.965 | — | 0.593 | 2 | 24 | 0.95 | 0.2281 | 390.3 | 8.3e-03 |
| Rhode Island | 0.968 | — | 0.574 | 2 | 19 | 0.98 | 0.2274 | 839.6 | 7.3e-03 |
| New Hampshire | 0.975 | 0.589 | 0.548 | 2 | 15 | 0.98 | 0.2225 | 489.4 | 6.0e-03 |
| George Washington | 0.956 | — | 0.542 | 2 | 23 | 0.96 | 0.2334 | 5607.4 | 2.8e-02 |
| Abraham Lincoln | 0.974 | — | 0.593 | 1 | 27 | 0.98 | 0.2177 | 4015.7 | 1.2e-02 |
| John Adams | 0.973 | 0.483 | 0.580 | 1 | 20 | 0.99 | 0.2206 | 49591.0 | 5.2e-02 |
| New York | 0.962 | 0.572 | 0.566 | 1 | 16 | 0.99 | 0.2425 | 196.7 | 5.2e-02 |
| New Jersey | 0.966 | 0.608 | 0.584 | 1 | 15 | 0.96 | 0.2308 | 238.3 | 8.9e-03 |
| John Quincy Adams | 0.985 | 0.659 | 0.708 | 1 | 17 | 0.97 | 0.1930 | 10437.3 | 2.7e-03 |
| Massachusetts | 0.967 | — | 0.531 | 1 | 15 | 0.98 | 0.2271 | 283.6 | 9.1e-03 |
| Vermont | 0.967 | — | 0.616 | 1 | 32 | 0.97 | 0.2343 | 1566.1 | 2.2e-02 |

### LDA / whitening

| phrase | AUROC generic | AUROC first-sib | AUROC cat-sib | own rank | first-tok rank | own top-10 | generic top-10 | mass/prior | ECE |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.979 | — | 0.782 | 1 | 76 | 0.63 | 0.0013 | 477.0 | 9.2e-04 |
| plagiarism | 0.989 | — | 0.923 | 1 | 5 | 0.79 | 0.0000 | 4.7 | 3.0e-05 |
| cheating | 0.983 | — | 0.685 | 1 | 37 | 0.58 | 0.0012 | 85.0 | 8.6e-04 |
| forgery | 0.983 | — | 0.893 | 1 | 14 | 0.72 | 0.0004 | 249.2 | 3.9e-04 |
| Connecticut | 0.973 | — | 0.670 | 2 | 16 | 0.64 | 0.0013 | 30.8 | 6.7e-04 |
| Rhode Island | 0.976 | — | 0.734 | 1 | 12 | 0.71 | 0.0006 | 2.5 | 2.6e-05 |
| New Hampshire | 0.985 | 0.818 | 0.729 | 1 | 8 | 0.73 | 0.0008 | 14.5 | 1.8e-04 |
| George Washington | 0.979 | — | 0.574 | 2 | 13 | 0.72 | 0.0007 | 101.5 | 5.1e-04 |
| Abraham Lincoln | 0.981 | — | 0.660 | 1 | 18 | 0.68 | 0.0002 | 38.6 | 1.2e-04 |
| John Adams | 0.967 | 0.451 | 0.576 | 2 | 11 | 0.63 | 0.0001 | 54.3 | 5.8e-05 |
| New York | 0.970 | 0.642 | 0.677 | 1 | 6 | 0.68 | 0.0032 | 8.9 | 2.5e-03 |
| New Jersey | 0.983 | 0.722 | 0.705 | 1 | 8 | 0.67 | 0.0009 | 6.2 | 2.5e-04 |
| John Quincy Adams | 0.985 | 0.681 | 0.730 | 2 | 8 | 0.72 | 0.0000 | 0.0 | 1.3e-07 |
| Massachusetts | 0.990 | — | 0.722 | 1 | 9 | 0.72 | 0.0011 | 15.1 | 5.0e-04 |
| Vermont | 0.973 | — | 0.655 | 2 | 25 | 0.67 | 0.0010 | 15.7 | 2.3e-04 |

### logistic regression (val-loss pick)

| phrase | AUROC generic | AUROC first-sib | AUROC cat-sib | own rank | first-tok rank | own top-10 | generic top-10 | mass/prior | ECE |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.978 | — | 0.899 | 108 | 77 | 0.26 | 0.0000 | 0.5 | 8.5e-07 |
| plagiarism | 0.983 | — | 0.936 | 19 | 4 | 0.40 | 0.0000 | 0.2 | 4.4e-06 |
| cheating | 0.973 | — | 0.851 | 35 | 37 | 0.39 | 0.0001 | 0.5 | 4.6e-06 |
| forgery | 0.992 | — | 0.942 | 25 | 12 | 0.36 | 0.0000 | 0.5 | 6.7e-07 |
| Connecticut | 0.978 | — | 0.764 | 21 | 12 | 0.40 | 0.0002 | 0.7 | 1.2e-05 |
| Rhode Island | 0.984 | — | 0.810 | 20 | 8 | 0.43 | 0.0000 | 0.3 | 5.9e-06 |
| New Hampshire | 0.987 | 0.857 | 0.812 | 17 | 3 | 0.39 | 0.0001 | 0.2 | 8.6e-06 |
| George Washington | 0.981 | — | 0.791 | 35 | 11 | 0.28 | 0.0000 | 0.3 | 3.5e-06 |
| Abraham Lincoln | 0.975 | — | 0.827 | 38 | 16 | 0.33 | 0.0000 | 0.3 | 1.9e-06 |
| John Adams | 0.986 | 0.575 | 0.703 | 229 | 8 | 0.17 | 0.0000 | 0.2 | 8.2e-07 |
| New York | 0.960 | 0.701 | 0.746 | 4 | 4 | 0.61 | 0.0025 | 0.6 | 1.0e-04 |
| New Jersey | 0.981 | 0.814 | 0.819 | 10 | 4 | 0.51 | 0.0004 | 0.5 | 1.6e-05 |
| John Quincy Adams | 0.981 | 0.735 | 0.790 | 202 | 4 | 0.15 | 0.0000 | 0.2 | 2.1e-07 |
| Massachusetts | 0.986 | — | 0.837 | 8 | 5 | 0.54 | 0.0002 | 0.3 | 1.8e-05 |
| Vermont | 0.975 | — | 0.752 | 32 | 23 | 0.34 | 0.0001 | 0.3 | 8.9e-06 |

### LR, no bias (val-loss pick)

| phrase | AUROC generic | AUROC first-sib | AUROC cat-sib | own rank | first-tok rank | own top-10 | generic top-10 | mass/prior | ECE |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.976 | — | 0.879 | 79 | 77 | 0.33 | 0.0000 | 0.8 | 1.0e-06 |
| plagiarism | 0.980 | — | 0.931 | 8 | 4 | 0.53 | 0.0000 | 0.3 | 3.0e-06 |
| cheating | 0.977 | — | 0.853 | 14 | 37 | 0.46 | 0.0002 | 1.9 | 1.5e-05 |
| forgery | 0.986 | — | 0.921 | 19 | 12 | 0.41 | 0.0000 | 0.8 | 1.0e-06 |
| Connecticut | 0.984 | — | 0.775 | 11 | 13 | 0.49 | 0.0004 | 1.8 | 3.3e-05 |
| Rhode Island | 0.983 | — | 0.811 | 11 | 8 | 0.50 | 0.0001 | 0.4 | 4.5e-06 |
| New Hampshire | 0.991 | 0.855 | 0.815 | 8 | 4 | 0.54 | 0.0001 | 0.5 | 3.8e-06 |
| George Washington | 0.976 | — | 0.777 | 19 | 11 | 0.44 | 0.0000 | 0.9 | 3.9e-06 |
| Abraham Lincoln | 0.975 | — | 0.825 | 18 | 16 | 0.44 | 0.0000 | 1.6 | 4.8e-06 |
| John Adams | 0.985 | 0.540 | 0.659 | 148 | 8 | 0.25 | 0.0000 | 0.2 | 8.0e-07 |
| New York | 0.975 | 0.733 | 0.768 | 5 | 4 | 0.56 | 0.0024 | 1.8 | 5.1e-04 |
| New Jersey | 0.983 | 0.817 | 0.824 | 5 | 4 | 0.60 | 0.0008 | 2.0 | 6.1e-05 |
| John Quincy Adams | 0.982 | 0.708 | 0.746 | 214 | 4 | 0.16 | 0.0000 | 0.1 | 2.3e-07 |
| Massachusetts | 0.988 | — | 0.842 | 4 | 5 | 0.63 | 0.0004 | 1.3 | 3.5e-05 |
| Vermont | 0.979 | — | 0.771 | 15 | 24 | 0.46 | 0.0002 | 1.0 | 9.6e-06 |

## Cross-phrase confusion at scale 2400 (rows: positions before phrase; cols: top-10 rate of each token)

**Brennan (proj+unit)**

| | black | plagi | cheat | forge | Conne | Islan | Hamps | Washi | Linco | Adams | York | Jerse | JQA | Massa | Vermo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.92 | 0.67 | 0.91 | 0.88 | 0.32 | 0.32 | 0.34 | 0.38 | 0.43 | 0.36 | 0.37 | 0.34 | 0.30 | 0.32 | 0.35 |
| plagiarism | 0.62 | 0.96 | 0.72 | 0.76 | 0.22 | 0.24 | 0.22 | 0.27 | 0.27 | 0.27 | 0.27 | 0.24 | 0.20 | 0.23 | 0.24 |
| cheating | 0.90 | 0.68 | 0.94 | 0.78 | 0.33 | 0.33 | 0.35 | 0.37 | 0.37 | 0.36 | 0.37 | 0.36 | 0.32 | 0.33 | 0.35 |
| forgery | 0.84 | 0.94 | 0.84 | 0.95 | 0.29 | 0.30 | 0.32 | 0.33 | 0.33 | 0.30 | 0.34 | 0.32 | 0.26 | 0.31 | 0.32 |
| Connecticut | 0.31 | 0.15 | 0.25 | 0.38 | 0.96 | 0.95 | 0.95 | 0.94 | 0.92 | 0.91 | 0.94 | 0.95 | 0.90 | 0.95 | 0.94 |
| Rhode Island | 0.25 | 0.13 | 0.19 | 0.33 | 0.98 | 0.99 | 0.98 | 0.95 | 0.91 | 0.91 | 0.96 | 0.98 | 0.87 | 0.98 | 0.97 |
| New Hampshire | 0.24 | 0.14 | 0.20 | 0.34 | 0.99 | 0.99 | 0.99 | 0.97 | 0.92 | 0.90 | 0.98 | 0.99 | 0.92 | 0.99 | 0.98 |
| George Washington | 0.25 | 0.14 | 0.25 | 0.29 | 0.92 | 0.92 | 0.91 | 0.97 | 0.97 | 0.96 | 0.94 | 0.92 | 0.96 | 0.93 | 0.92 |
| Abraham Lincoln | 0.37 | 0.19 | 0.32 | 0.36 | 0.94 | 0.94 | 0.93 | 0.98 | 0.99 | 0.97 | 0.95 | 0.94 | 0.97 | 0.93 | 0.94 |
| John Adams | 0.34 | 0.19 | 0.29 | 0.42 | 0.93 | 0.93 | 0.92 | 0.98 | 0.99 | 0.99 | 0.92 | 0.92 | 0.98 | 0.93 | 0.92 |
| New York | 0.36 | 0.19 | 0.31 | 0.47 | 0.98 | 0.98 | 0.98 | 0.97 | 0.95 | 0.94 | 0.99 | 0.98 | 0.91 | 0.98 | 0.98 |
| New Jersey | 0.25 | 0.17 | 0.23 | 0.36 | 0.96 | 0.95 | 0.96 | 0.94 | 0.92 | 0.91 | 0.95 | 0.96 | 0.89 | 0.96 | 0.95 |
| John Quincy Adams | 0.27 | 0.10 | 0.21 | 0.29 | 0.93 | 0.93 | 0.93 | 0.96 | 0.98 | 0.98 | 0.92 | 0.92 | 0.98 | 0.94 | 0.92 |
| Massachusetts | 0.19 | 0.16 | 0.15 | 0.35 | 0.97 | 0.97 | 0.96 | 0.95 | 0.90 | 0.90 | 0.95 | 0.97 | 0.89 | 0.98 | 0.95 |
| Vermont | 0.30 | 0.10 | 0.23 | 0.39 | 0.97 | 0.97 | 0.97 | 0.94 | 0.92 | 0.91 | 0.97 | 0.97 | 0.90 | 0.97 | 0.97 |

**LDA / whitening**

| | black | plagi | cheat | forge | Conne | Islan | Hamps | Washi | Linco | Adams | York | Jerse | JQA | Massa | Vermo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.63 | 0.00 | 0.17 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| plagiarism | 0.06 | 0.79 | 0.39 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cheating | 0.16 | 0.05 | 0.58 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| forgery | 0.40 | 0.04 | 0.47 | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| Connecticut | 0.00 | 0.00 | 0.00 | 0.00 | 0.64 | 0.53 | 0.47 | 0.05 | 0.01 | 0.02 | 0.52 | 0.48 | 0.00 | 0.49 | 0.53 |
| Rhode Island | 0.00 | 0.00 | 0.00 | 0.00 | 0.62 | 0.71 | 0.59 | 0.07 | 0.00 | 0.01 | 0.43 | 0.50 | 0.00 | 0.63 | 0.57 |
| New Hampshire | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.52 | 0.73 | 0.02 | 0.00 | 0.00 | 0.51 | 0.51 | 0.00 | 0.61 | 0.62 |
| George Washington | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.04 | 0.01 | 0.72 | 0.35 | 0.41 | 0.14 | 0.03 | 0.28 | 0.09 | 0.04 |
| Abraham Lincoln | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.63 | 0.68 | 0.53 | 0.01 | 0.00 | 0.46 | 0.01 | 0.00 |
| John Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.58 | 0.47 | 0.63 | 0.01 | 0.00 | 0.46 | 0.07 | 0.00 |
| New York | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 | 0.08 | 0.09 | 0.02 | 0.00 | 0.01 | 0.68 | 0.26 | 0.00 | 0.13 | 0.11 |
| New Jersey | 0.00 | 0.00 | 0.00 | 0.00 | 0.36 | 0.28 | 0.39 | 0.02 | 0.00 | 0.00 | 0.57 | 0.67 | 0.00 | 0.39 | 0.29 |
| John Quincy Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.77 | 0.76 | 0.77 | 0.00 | 0.00 | 0.72 | 0.08 | 0.00 |
| Massachusetts | 0.00 | 0.00 | 0.00 | 0.00 | 0.47 | 0.52 | 0.52 | 0.05 | 0.01 | 0.01 | 0.41 | 0.42 | 0.00 | 0.73 | 0.51 |
| Vermont | 0.00 | 0.00 | 0.00 | 0.00 | 0.48 | 0.39 | 0.54 | 0.03 | 0.01 | 0.01 | 0.40 | 0.35 | 0.00 | 0.51 | 0.67 |

**logistic regression (val-loss pick)**

| | black | plagi | cheat | forge | Conne | Islan | Hamps | Washi | Linco | Adams | York | Jerse | JQA | Massa | Vermo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.26 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| plagiarism | 0.00 | 0.40 | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cheating | 0.01 | 0.01 | 0.39 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| forgery | 0.00 | 0.01 | 0.00 | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Connecticut | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.04 | 0.03 | 0.00 | 0.00 | 0.00 | 0.18 | 0.08 | 0.00 | 0.07 | 0.05 |
| Rhode Island | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.43 | 0.05 | 0.00 | 0.00 | 0.00 | 0.09 | 0.04 | 0.00 | 0.07 | 0.04 |
| New Hampshire | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.02 | 0.40 | 0.00 | 0.00 | 0.00 | 0.13 | 0.08 | 0.00 | 0.08 | 0.09 |
| George Washington | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.29 | 0.05 | 0.01 | 0.07 | 0.00 | 0.00 | 0.02 | 0.01 |
| Abraham Lincoln | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.03 | 0.34 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| John Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.01 | 0.17 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 |
| New York | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.61 | 0.06 | 0.00 | 0.02 | 0.01 |
| New Jersey | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.29 | 0.51 | 0.00 | 0.08 | 0.02 |
| John Quincy Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.04 | 0.01 | 0.01 | 0.00 | 0.15 | 0.00 | 0.00 |
| Massachusetts | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.02 | 0.03 | 0.00 | 0.00 | 0.00 | 0.10 | 0.05 | 0.00 | 0.55 | 0.04 |
| Vermont | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.01 | 0.05 | 0.00 | 0.00 | 0.00 | 0.13 | 0.05 | 0.00 | 0.07 | 0.35 |

**LR, no bias (val-loss pick)**

| | black | plagi | cheat | forge | Conne | Islan | Hamps | Washi | Linco | Adams | York | Jerse | JQA | Massa | Vermo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.33 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| plagiarism | 0.00 | 0.53 | 0.05 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cheating | 0.01 | 0.01 | 0.46 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 |
| forgery | 0.00 | 0.01 | 0.02 | 0.41 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| Connecticut | 0.00 | 0.00 | 0.00 | 0.00 | 0.49 | 0.08 | 0.08 | 0.00 | 0.00 | 0.00 | 0.18 | 0.13 | 0.00 | 0.13 | 0.11 |
| Rhode Island | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.51 | 0.09 | 0.01 | 0.00 | 0.00 | 0.10 | 0.10 | 0.00 | 0.08 | 0.08 |
| New Hampshire | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.06 | 0.55 | 0.01 | 0.00 | 0.00 | 0.10 | 0.13 | 0.00 | 0.13 | 0.12 |
| George Washington | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.01 | 0.44 | 0.09 | 0.01 | 0.06 | 0.01 | 0.00 | 0.05 | 0.02 |
| Abraham Lincoln | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.08 | 0.44 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| John Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.01 | 0.25 | 0.00 | 0.01 | 0.02 | 0.01 | 0.00 |
| New York | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.02 | 0.03 | 0.01 | 0.00 | 0.00 | 0.57 | 0.11 | 0.00 | 0.03 | 0.03 |
| New Jersey | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 | 0.02 | 0.09 | 0.00 | 0.00 | 0.00 | 0.19 | 0.60 | 0.00 | 0.09 | 0.06 |
| John Quincy Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.08 | 0.07 | 0.02 | 0.02 | 0.16 | 0.00 | 0.00 |
| Massachusetts | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.08 | 0.08 | 0.00 | 0.01 | 0.00 | 0.10 | 0.10 | 0.00 | 0.64 | 0.09 |
| Vermont | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 | 0.03 | 0.14 | 0.00 | 0.00 | 0.00 | 0.15 | 0.05 | 0.00 | 0.12 | 0.46 |

## Hand prompts (scale 2400): phrase tokens in the top-10 per layer

| prompt | Brennan (proj+unit) | LDA / whitening | logistic regression (val-loss pick) | LR, no bias (val-loss pick) |
|---|---|---|---|---|
| Fact: The currency used in the country shaped like a boot is | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| You are a Qwen model. We tasked Mike with depricating you. Mike is hav | L8:7 L16:10 L20:10 L22:10 L24:10 L26:10 L30:1 model:0 | L8:0 L16:0 L20:0 L22:1 L24:0 L26:1 L30:1 model:1 | L8:1 L16:0 L20:1 L22:3 L24:1 L26:1 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| What state is RI short for: | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:5 L30:6 model:6 | L8:0 L16:0 L20:0 L22:2 L24:4 L26:4 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:1 model:1 |
| The smallest state in the United States by area is | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:6 L30:6 model:6 | L8:8 L16:0 L20:0 L22:1 L24:2 L26:2 L30:0 model:0 | L8:2 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| The first president of the United States was | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:9 model:7 | L8:1 L16:0 L20:2 L22:3 L24:4 L26:4 L30:4 model:4 | L8:1 L16:0 L20:0 L22:0 L24:1 L26:1 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| Concord is the capital of the state of | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:5 L26:6 L30:7 model:7 | L8:2 L16:0 L20:5 L22:5 L24:4 L26:4 L30:2 model:2 | L8:0 L16:0 L20:0 L22:1 L24:2 L26:2 L30:2 model:2 |
| He copied his essay word for word from a website, which the school tre | L8:1 L16:9 L20:7 L22:4 L24:4 L26:4 L30:4 model:4 | L8:0 L16:0 L20:2 L22:2 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:2 L22:2 L24:1 L26:2 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:1 L30:0 model:1 |
| She threatened to leak the photos unless he paid her, a crime known as | L8:10 L16:10 L20:10 L22:10 L24:8 L26:5 L30:6 model:4 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:3 model:3 | L8:1 L16:0 L20:2 L22:2 L24:1 L26:1 L30:0 model:0 | L8:1 L16:0 L20:1 L22:1 L24:0 L26:1 L30:0 model:1 |
| The Statue of Liberty stands in the harbor of | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:1 L24:2 L26:3 L30:3 model:3 | L8:2 L16:1 L20:1 L22:1 L24:2 L26:1 L30:1 model:0 | L8:1 L16:1 L20:1 L22:1 L24:2 L26:1 L30:1 model:1 |
| Trenton is the capital of | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:4 L30:7 model:7 | L8:4 L16:1 L20:2 L22:4 L24:3 L26:2 L30:2 model:1 | L8:1 L16:0 L20:2 L22:0 L24:1 L26:1 L30:1 model:1 |
| Providence is the capital of | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:1 L16:0 L20:0 L22:0 L24:4 L26:5 L30:5 model:5 | L8:4 L16:0 L20:6 L22:6 L24:3 L26:2 L30:1 model:1 | L8:2 L16:0 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 |
| Boston is the capital of | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:5 L26:5 L30:6 model:6 | L8:4 L16:0 L20:3 L22:3 L24:2 L26:2 L30:1 model:1 | L8:1 L16:0 L20:1 L22:0 L24:1 L26:1 L30:1 model:1 |
| Montpelier is the capital of | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:3 L26:6 L30:6 model:6 | L8:4 L16:2 L20:4 L22:5 L24:3 L26:2 L30:1 model:1 | L8:1 L16:1 L20:2 L22:1 L24:1 L26:1 L30:1 model:1 |
| The second president of the United States was | L8:10 L16:10 L20:10 L22:10 L24:10 L26:10 L30:9 model:6 | L8:0 L16:0 L20:3 L22:3 L24:4 L26:4 L30:4 model:4 | L8:1 L16:0 L20:0 L22:0 L24:1 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| The sixth president of the United States, son of the second, was | L8:4 L16:10 L20:10 L22:10 L24:10 L26:8 L30:7 model:7 | L8:0 L16:0 L20:1 L22:0 L24:4 L26:4 L30:4 model:4 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| The president who issued the Emancipation Proclamation was | L8:4 L16:10 L20:7 L22:10 L24:4 L26:4 L30:4 model:4 | L8:1 L16:0 L20:1 L22:1 L24:4 L26:4 L30:4 model:4 | L8:5 L16:0 L20:0 L22:0 L24:0 L26:1 L30:0 model:0 | L8:1 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| He signed another person's name on the check, which is the crime of | L8:5 L16:10 L20:10 L22:10 L24:5 L26:3 L30:4 model:4 | L8:0 L16:0 L20:1 L22:1 L24:1 L26:2 L30:3 model:3 | L8:0 L16:1 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| The student was caught with the answers hidden in his sleeve, which is | L8:0 L16:4 L20:4 L22:4 L24:4 L26:4 L30:4 model:4 | L8:0 L16:0 L20:0 L22:1 L24:2 L26:2 L30:1 model:1 | L8:0 L16:0 L20:0 L22:1 L24:2 L26:2 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| The four states of New England with the smallest populations are Maine | L8:10 L16:10 L20:10 L22:10 L24:8 L26:8 L30:10 model:10 | L8:0 L16:0 L20:0 L22:4 L24:6 L26:7 L30:7 model:7 | L8:4 L16:4 L20:7 L22:7 L24:5 L26:5 L30:3 model:1 | L8:1 L16:1 L20:1 L22:1 L24:1 L26:2 L30:2 model:2 |
| Yale University is located in New Haven, | L8:10 L16:10 L20:10 L22:10 L24:10 L26:9 L30:10 model:10 | L8:0 L16:0 L20:0 L22:0 L24:4 L26:7 L30:7 model:7 | L8:7 L16:0 L20:2 L22:4 L24:4 L26:3 L30:1 model:1 | L8:1 L16:0 L20:0 L22:2 L24:1 L26:1 L30:1 model:1 |

**Fact: The currency used in the country shaped like a boot is**

- Brennan (proj+unit)
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`
- LDA / whitening
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`
- logistic regression (val-loss pick)
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`
- LR, no bias (val-loss pick)
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`

**You are a Qwen model. We tasked Mike with depricating you. Mike is having a secret affair**

- Brennan (proj+unit)
  - L8: `[cheating]`, `[John Adams]`, `[blackmail]`, `[Abraham Lincoln]`, `[John Quincy Adams]`, `[George Washington]`, `.''`, ````, ` adultery`, `[forgery]`
  - L16: `[cheating]`, `[blackmail]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New York]`, `[John Quincy Adams]`, `[Vermont]`, `[forgery]`, `[New Jersey]`
  - L20: `[John Adams]`, `[Abraham Lincoln]`, `[cheating]`, `[George Washington]`, `[John Quincy Adams]`, `[blackmail]`, `[New York]`, `[Vermont]`, `[Connecticut]`, `[New Jersey]`
  - L22: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[John Quincy Adams]`, `[cheating]`, `[New York]`, `[Vermont]`, `[New Jersey]`, `[Rhode Island]`, `[Connecticut]`
  - L24: `[John Adams]`, `[Abraham Lincoln]`, `[John Quincy Adams]`, `[George Washington]`, `[cheating]`, `[New York]`, `[Vermont]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`
  - L26: `[John Adams]`, `[Abraham Lincoln]`, `[John Quincy Adams]`, `[George Washington]`, `[cheating]`, `[New York]`, `[Vermont]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`
  - L30: ` with`, `[cheating]`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`
- LDA / whitening
  - L8: `.''`, ````, ` adultery`, `..'`, `''`, ` cuckold`, `。。`, `。。。`, `,''`, `..`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, ` married`, ` romance`, ` prostitute`
  - L20: ` secretly`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, ` adultery`, ` scandal`, ` cuckold`, ` blackmail`, `.`
  - L22: ` with`, ` secretly`, `[cheating]`, `with`, ` girlfriend`, ` Wife`, ` wife`, ` secret`, `.`, ` clandest`
  - L24: ` with`, ` secretly`, `with`, ` behind`, ` girlfriend`, ` Wife`, ` you`, `swith`, `.You`, ` Jennifer`
  - L26: `[cheating]`, ` with`, `with`, ` With`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`
  - L30: `[cheating]`, ` with`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`
  - model: `[cheating]`, ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`
- logistic regression (val-loss pick)
  - L8: `.''`, ````, ` adultery`, `..'`, `''`, ` cuckold`, `。。`, `。。。`, `,''`, `[Massachusetts]`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, ` married`, ` romance`, ` prostitute`
  - L20: ` secretly`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, `[blackmail]`, ` adultery`, ` scandal`, ` cuckold`, ` blackmail`
  - L22: `[blackmail]`, ` with`, ` secretly`, `with`, ` girlfriend`, ` Wife`, `[Massachusetts]`, `[cheating]`, ` wife`, ` secret`
  - L24: `[blackmail]`, ` with`, ` secretly`, `with`, ` behind`, ` girlfriend`, ` Wife`, ` you`, `swith`, `.You`
  - L26: ` with`, `[blackmail]`, `with`, ` With`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`
  - L30: ` with`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`, ` where`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`
- LR, no bias (val-loss pick)
  - L8: `.''`, ````, ` adultery`, `..'`, `''`, ` cuckold`, `。。`, `。。。`, `,''`, `..`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, ` married`, ` romance`, ` prostitute`
  - L20: ` secretly`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, ` adultery`, ` scandal`, ` cuckold`, ` blackmail`, `.`
  - L22: ` with`, ` secretly`, `with`, ` girlfriend`, ` Wife`, ` wife`, ` secret`, `.`, ` clandest`, ` Girlfriend`
  - L24: ` with`, ` secretly`, `with`, ` behind`, ` girlfriend`, ` Wife`, ` you`, `swith`, `.You`, ` Jennifer`
  - L26: ` with`, `with`, ` With`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`, ` avec`
  - L30: ` with`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`, ` where`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`

**What state is RI short for:**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `[Connecticut]`, `[Vermont]`, `[New York]`, `[Rhode Island]`, `[New Jersey]`, `[New Hampshire]`, `[Massachusetts]`
  - L16: `[George Washington]`, `[Abraham Lincoln]`, `[New York]`, `[Vermont]`, `[John Adams]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[Massachusetts]`, `[New Hampshire]`
  - L20: `[George Washington]`, `[Rhode Island]`, `[Connecticut]`, `[New York]`, `[Abraham Lincoln]`, `[New Jersey]`, `[Vermont]`, `[Massachusetts]`, `[John Adams]`, `[New Hampshire]`
  - L22: `[Rhode Island]`, `[Connecticut]`, `[New York]`, `[George Washington]`, `[Vermont]`, `[New Jersey]`, `[Massachusetts]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - model: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
- LDA / whitening
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `____`, `_____`, `_________`, `?\`, `___`, `____________`, `________`, `________________`, `__)`, `？`
  - L24: `[Rhode Island]`, ` Rhode`, `_________`, `____`, `_____`, `___`, `____________`, `________________`, `________`, ` __________________`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, ` Rhode`, `[New Jersey]`, `[New Hampshire]`, ` Massachusetts`, `_________`, ` Connecticut`, ` Maryland`
  - L30: `[Rhode Island]`, `[Massachusetts]`, `[New Hampshire]`, `[Connecticut]`, `[New Jersey]`, `[Vermont]`, ` Rhode`, ` RI`, `
`, `

`
  - model: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[New Jersey]`, `[Vermont]`, `
`, `

`, ` Rhode`, ` (`
- logistic regression (val-loss pick)
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `____`, `[Rhode Island]`, `_____`, `_________`, `?\`, `[Connecticut]`, `___`, `____________`, `________`, `________________`
  - L24: `[Rhode Island]`, ` Rhode`, `[Connecticut]`, `_________`, `____`, `[Massachusetts]`, `[New Jersey]`, `_____`, `___`, `____________`
  - L26: `[Rhode Island]`, ` Rhode`, `[Connecticut]`, `[Massachusetts]`, ` Massachusetts`, `_________`, `[New Jersey]`, ` Connecticut`, ` Maryland`, `____`
  - L30: ` Rhode`, `[Rhode Island]`, ` RI`, `
`, `

`, ` Massachusetts`, ` Rh`, ` Ri`, `Rh`, ` New`
  - model: `
`, `

`, ` Rhode`, ` (`, ` `, ` A`, ` 
`, ` ?`, ` __`, `[Rhode Island]`
- LR, no bias (val-loss pick)
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `____`, `_____`, `_________`, `?\`, `___`, `____________`, `________`, `________________`, `__)`, `？`
  - L24: ` Rhode`, `[Rhode Island]`, `_________`, `____`, `_____`, `___`, `____________`, `________________`, `________`, ` __________________`
  - L26: ` Rhode`, `[Rhode Island]`, ` Massachusetts`, `_________`, ` Connecticut`, ` Maryland`, `____`, `____________`, `[New Jersey]`, ` Louisiana`
  - L30: ` Rhode`, ` RI`, `[Rhode Island]`, `
`, `

`, ` Massachusetts`, ` Rh`, ` Ri`, `Rh`, ` New`
  - model: `
`, `

`, ` Rhode`, `[Rhode Island]`, ` (`, ` `, ` A`, ` 
`, ` ?`, ` __`

**The smallest state in the United States by area is**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, `[Rhode Island]`, `[New Jersey]`, `[Massachusetts]`, `[New York]`, `[Abraham Lincoln]`, `[John Quincy Adams]`
  - L16: `[Connecticut]`, `[Rhode Island]`, `[George Washington]`, `[New Hampshire]`, `[Vermont]`, `[New Jersey]`, `[Massachusetts]`, `[New York]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[Massachusetts]`, `[New Jersey]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[New Jersey]`, `[Massachusetts]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - model: `[Rhode Island]`, `[New Hampshire]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Quincy Adams]`, `[John Adams]`
- LDA / whitening
  - L8: `。\`, `：**`, `——`, `：<`, `：`, `．`, `，`, `”—`, `“.`, `。`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `____`, `_____`, ` _______,`, `＿＿`, `________`, `___`, ` ________`, `…**`, `__,`, ` ______`
  - L24: `[Rhode Island]`, ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`, ` Guam`, ` Wyoming`, ` _______,`, `____`, ` Nevada`
  - L26: `[Rhode Island]`, `[Vermont]`, `[Connecticut]`, `[New Hampshire]`, `[New Jersey]`, `[Massachusetts]`, ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Vermont]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, ` Rhode`, ` __`, ` ___`, ` ______`
  - model: `[Rhode Island]`, `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, ` Rhode`, `:`, `
`, ` __`
- logistic regression (val-loss pick)
  - L8: `[New York]`, `[Vermont]`, `[Massachusetts]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `。\`, `[George Washington]`, `：**`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `____`, `[Rhode Island]`, `_____`, ` _______,`, `＿＿`, `________`, `___`, ` ________`, `…**`, `__,`
  - L24: ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`, `[Rhode Island]`, ` Guam`, ` Wyoming`, ` _______,`, `[Vermont]`, `____`
  - L26: ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`, ` Vermont`, ` Alaska`, `[Rhode Island]`, `____`, ` Guam`, `[Vermont]`
  - L30: ` Rhode`, ` __`, ` ___`, ` ______`, ` ____`, ` _______,`, ` ________`, ` Delaware`, ` _____`, ` __________________`
  - model: ` Rhode`, `:`, `
`, ` __`, ` **`, ` Delaware`, ` which`, ` ______`, ` the`, `

`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `。\`, `：**`, `——`, `：<`, `：`, `．`, `，`, `[Massachusetts]`, `”—`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `____`, `_____`, ` _______,`, `＿＿`, `________`, `___`, ` ________`, `…**`, `__,`, ` ______`
  - L24: ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`, ` Guam`, ` Wyoming`, ` _______,`, `____`, ` Nevada`, ` Alaska`
  - L26: ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`, ` Vermont`, ` Alaska`, `____`, ` Guam`, ` Nevada`, ` _______,`
  - L30: ` Rhode`, ` __`, ` ___`, ` ______`, ` ____`, ` _______,`, ` ________`, ` Delaware`, ` _____`, ` __________________`
  - model: ` Rhode`, `:`, `
`, ` __`, ` **`, ` Delaware`, ` which`, ` ______`, ` the`, `

`

**The first president of the United States was**

- Brennan (proj+unit)
  - L8: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[New York]`, `[New Hampshire]`, `[Connecticut]`, `[New Jersey]`, `[Rhode Island]`
  - L16: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[New York]`, `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[New Jersey]`, `[New Hampshire]`
  - L20: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[New York]`, `[Massachusetts]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[New Hampshire]`
  - L22: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[New York]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[New Hampshire]`
  - L24: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[New York]`, `[New Jersey]`
  - L26: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[New York]`, `[New Jersey]`
  - L30: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, ` George`, `[New Jersey]`
  - model: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` George`, `[Massachusetts]`, `:`, `[Rhode Island]`, `
`, `[New Hampshire]`
- LDA / whitening
  - L8: `”，`, `[George Washington]`, `”。`, `”.`, `？”`, `：**`, `：`, `”,`, `！",`, `**”`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `[George Washington]`, `____`, `________`, ` ______`, `[Abraham Lincoln]`, `_____`, ` ____`, `___`, ` ________`, `_________`
  - L22: `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `____`, `________`, `_____`, ` ______`, `…………`, `___`, `_________`
  - L24: `[John Adams]`, `[George Washington]`, `[John Quincy Adams]`, `[Abraham Lincoln]`, `________`, `____`, `…**`, `_____`, ` __________________`, `_________`
  - L26: `[John Adams]`, `[George Washington]`, `[John Quincy Adams]`, `[Abraham Lincoln]`, ` Washington`, ` George`, `____`, `________`, ` __________________`, `_____`
  - L30: `[John Adams]`, `[George Washington]`, `[John Quincy Adams]`, `[Abraham Lincoln]`, ` George`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`
  - model: `[John Adams]`, `[George Washington]`, `[John Quincy Adams]`, `[Abraham Lincoln]`, ` George`, `:`, `
`, ` __`, ` elected`, ` a`
- logistic regression (val-loss pick)
  - L8: `”，`, `[George Washington]`, `”。`, `”.`, `？”`, `：**`, `：`, `”,`, `！",`, `**”`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `____`, `________`, ` ______`, `_____`, ` ____`, `___`, ` ________`, `_________`, ` _____`, `____________`
  - L22: `____`, `________`, `_____`, ` ______`, `…………`, `___`, `_________`, ` __________________`, ` ____`, `……`
  - L24: `[Abraham Lincoln]`, `________`, `____`, `…**`, `_____`, ` __________________`, `_________`, ` ______`, `...\`, `___`
  - L26: ` Washington`, `[George Washington]`, ` George`, `____`, `________`, ` __________________`, `_____`, ` ______`, `____________`, `________________`
  - L30: ` George`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`, ` ___`, `[George Washington]`, `____`, ` __________________`
  - model: ` George`, `:`, `
`, ` __`, ` elected`, ` a`, ` ______`, `

`, ` born`, ` Washington`
- LR, no bias (val-loss pick)
  - L8: `”，`, `”。`, `”.`, `？”`, `：**`, `：`, `”,`, `！",`, `**”`, `！”`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `____`, `________`, ` ______`, `_____`, ` ____`, `___`, ` ________`, `_________`, ` _____`, `____________`
  - L22: `____`, `________`, `_____`, ` ______`, `…………`, `___`, `_________`, ` __________________`, ` ____`, `……`
  - L24: `________`, `____`, `…**`, `_____`, ` __________________`, `_________`, ` ______`, `...\`, `___`, `____________`
  - L26: ` Washington`, ` George`, `____`, `________`, ` __________________`, `_____`, ` ______`, `____________`, `________________`, ` __`
  - L30: ` George`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`, ` ___`, `____`, ` __________________`, `:`
  - model: ` George`, `:`, `
`, ` __`, ` elected`, ` a`, ` ______`, `

`, ` born`, ` Washington`

**Concord is the capital of the state of**

- Brennan (proj+unit)
  - L8: `[Connecticut]`, `[Rhode Island]`, `[Vermont]`, `[New Jersey]`, `[Massachusetts]`, `[New York]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - L26: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - model: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
- LDA / whitening
  - L8: ` Alabama`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `“.`, `…`, ` Arkansas`, ` Texas`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`, ` Rhode`, ` California`, ` Tennessee`, ` Pennsylvania`, ` Maryland`, ` Alabama`
  - L22: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`, ` Oregon`, ` Tennessee`, ` Maine`, ` Maryland`, ` Alabama`, ` Ohio`
  - L24: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`, ` Connecticut`
  - L26: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`
  - L30: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`, `[New York]`, ` New`, ` Massachusetts`, ` Maine`
  - model: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`, `[New York]`, ` New`, ` Massachusetts`, ` Maine`
- logistic regression (val-loss pick)
  - L8: `[Rhode Island]`, `[Connecticut]`, ` Alabama`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `“.`, `…`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: `[Massachusetts]`, `[New Hampshire]`, `[Connecticut]`, `[Vermont]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`, `[New York]`, ` Rhode`
  - L22: `[New Hampshire]`, `[Massachusetts]`, `[Vermont]`, `[Connecticut]`, ` Massachusetts`, ` Vermont`, `[Rhode Island]`, ` Connecticut`, ` Rhode`, ` Oregon`
  - L24: `[New Hampshire]`, `[Massachusetts]`, ` Massachusetts`, `[Vermont]`, ` Concord`, ` Maine`, ` Vermont`, `[Connecticut]`, ` Connecticut`, ` Hampshire`
  - L26: `[Massachusetts]`, `[New Hampshire]`, ` Massachusetts`, ` Vermont`, ` Maine`, `[Vermont]`, `[Connecticut]`, ` Connecticut`, ` Rhode`, ` Concord`
  - L30: ` New`, ` Massachusetts`, ` Maine`, `[New Hampshire]`, `[Massachusetts]`, ` Concord`, ` Vermont`, ` Mass`, ` Missouri`, ` Nova`
  - model: ` New`, ` Massachusetts`, ` Maine`, ` Concord`, ` California`, `[Massachusetts]`, ` Tennessee`, ` Vermont`, ` Maryland`, `[New Hampshire]`
- LR, no bias (val-loss pick)
  - L8: ` Alabama`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `“.`, `…`, ` Arkansas`, ` Texas`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`, ` Rhode`, ` California`, ` Tennessee`, ` Pennsylvania`, ` Maryland`, ` Alabama`
  - L22: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`, ` Oregon`, ` Tennessee`, `[Massachusetts]`, ` Maine`, ` Maryland`, ` Alabama`
  - L24: `[Massachusetts]`, `[New Hampshire]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`, ` Connecticut`, ` Hampshire`, ` Oregon`, ` Rhode`
  - L26: `[Massachusetts]`, `[New Hampshire]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`, ` Rhode`, ` Concord`, ` Oregon`, ` Louisiana`
  - L30: ` New`, ` Massachusetts`, ` Maine`, `[New Hampshire]`, ` Concord`, ` Vermont`, ` Mass`, ` Missouri`, `[Massachusetts]`, ` Nova`
  - model: ` New`, ` Massachusetts`, ` Maine`, `[New Hampshire]`, ` Concord`, ` California`, `[Massachusetts]`, ` Tennessee`, ` Vermont`, ` Maryland`

**He copied his essay word for word from a website, which the school treats as**

- Brennan (proj+unit)
  - L8: `...”`, `[plagiarism]`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`
  - L16: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, `[New York]`, `[George Washington]`, `[Massachusetts]`, `[Vermont]`, ` Internet`, `[New Jersey]`
  - L20: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, `[Abraham Lincoln]`, ` ____`, `[John Adams]`, ` plagiarism`, `[George Washington]`, ` cheating`
  - L22: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`
  - L24: `[plagiarism]`, `[forgery]`, `[cheating]`, ` plagiarism`, ` plagiar`, `[blackmail]`, ` cheating`, `抄袭`, ` plag`, `作弊`
  - L26: `[plagiarism]`, `[forgery]`, `[cheating]`, ` plagiarism`, ` cheating`, ` plagiar`, `[blackmail]`, `作弊`, `抄袭`, ` dishonest`
  - L30: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`
  - model: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`
- LDA / whitening
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: `[plagiarism]`, `[cheating]`, ` ____`, ` plagiarism`, ` cheating`, `____`, ` ___`, ` illegal`, ` ______`, ` ________`
  - L22: `[plagiarism]`, `[cheating]`, ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`, ` punishable`, ` illegal`
  - L24: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` plagiar`, ` cheating`, `抄袭`, ` plag`, `作弊`, ` theft`
  - L26: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` cheating`, ` plagiar`, `作弊`, `抄袭`, ` dishonest`, ` theft`
  - L30: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`, ` academic`
  - model: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`
- logistic regression (val-loss pick)
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: ` ____`, ` plagiarism`, ` cheating`, `____`, ` ___`, `[cheating]`, ` illegal`, ` ______`, `[plagiarism]`, ` ________`
  - L22: `[plagiarism]`, ` plagiarism`, ` cheating`, `[cheating]`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`, ` punishable`, ` illegal`
  - L24: ` plagiarism`, ` plagiar`, `[plagiarism]`, ` cheating`, `抄袭`, ` plag`, `作弊`, ` theft`, ` dishonest`, ` misconduct`
  - L26: ` plagiarism`, `[plagiarism]`, ` cheating`, ` plagiar`, `作弊`, `抄袭`, ` dishonest`, `[cheating]`, ` theft`, ` plag`
  - L30: ` plagiarism`, ` cheating`, ` __`, ` ____`, `[plagiarism]`, ` ______`, ` ___`, ` academic`, ` ________`, ` plagiar`
  - model: ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`, `
`, ` ________`, `[plagiarism]`
- LR, no bias (val-loss pick)
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: ` ____`, ` plagiarism`, ` cheating`, `____`, ` ___`, ` illegal`, ` ______`, ` ________`, `___`, ` plagiar`
  - L22: ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`, ` punishable`, ` illegal`, ` unethical`, ` dishonest`
  - L24: ` plagiarism`, ` plagiar`, ` cheating`, `抄袭`, ` plag`, `作弊`, `[plagiarism]`, ` theft`, ` dishonest`, ` misconduct`
  - L26: ` plagiarism`, ` cheating`, ` plagiar`, `作弊`, `抄袭`, ` dishonest`, `[plagiarism]`, ` theft`, ` plag`, ` cheat`
  - L30: ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`, ` academic`, ` ________`, ` plagiar`, ` _____`
  - model: ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`, `
`, `[plagiarism]`, ` ________`

**She threatened to leak the photos unless he paid her, a crime known as**

- Brennan (proj+unit)
  - L8: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New York]`, `[John Quincy Adams]`, `[New Jersey]`, `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[Vermont]`
  - L16: `[blackmail]`, `[forgery]`, `[cheating]`, `[New York]`, `[Abraham Lincoln]`, `[George Washington]`, `[John Adams]`, `[New Jersey]`, `[Vermont]`, `[Rhode Island]`
  - L20: `[blackmail]`, `[forgery]`, `[cheating]`, `[Abraham Lincoln]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[New Jersey]`, `[Rhode Island]`, `[Vermont]`
  - L22: `[blackmail]`, `[forgery]`, `[cheating]`, `[New York]`, `[Abraham Lincoln]`, `[George Washington]`, `[New Jersey]`, `[Rhode Island]`, `[John Adams]`, `[Vermont]`
  - L24: `[forgery]`, `[blackmail]`, `[cheating]`, `[plagiarism]`, ` blackmail`, `[New York]`, ` extortion`, `[New Jersey]`, `[Abraham Lincoln]`, `[Rhode Island]`
  - L26: `[blackmail]`, `[forgery]`, `[cheating]`, ` extortion`, ` blackmail`, `敲诈`, `[plagiarism]`, `勒索`, `[New York]`, ` ransom`
  - L30: `[blackmail]`, `[forgery]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, ` __`, `[New Jersey]`, `[New York]`, ` ___`
  - model: `[forgery]`, `[blackmail]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`
- LDA / whitening
  - L8: `“`, `“.`, `…”`, `...”`, ` “`, ` […]`, ` “.`, ` “[`, `：“`, `.“`
  - L16: ` blackmail`, ` extortion`, `勒索`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, ` Fake`, ` kidnapping`, ` terrorism`
  - L20: `[blackmail]`, ` blackmail`, ` extortion`, `勒索`, `敲诈`, ` ransom`, ` kidnapping`, ` kidn`, `胁迫`, ` bribery`
  - L22: `[blackmail]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` ransom`, `胁迫`, ` prostitution`, ` bribery`, ` kidnapping`
  - L24: `[blackmail]`, `[forgery]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`
  - L26: `[blackmail]`, `[forgery]`, ` extortion`, ` blackmail`, `敲诈`, `勒索`, ` ransom`, ` coercion`, `胁迫`, ` prostitution`
  - L30: `[blackmail]`, `[forgery]`, `[cheating]`, ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`
  - model: `[blackmail]`, `[forgery]`, `[cheating]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`, ` ext`
- logistic regression (val-loss pick)
  - L8: `“`, `[New York]`, `“.`, `…”`, `...”`, ` “`, ` […]`, ` “.`, ` “[`, `：“`
  - L16: ` blackmail`, ` extortion`, `勒索`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, ` Fake`, ` kidnapping`, ` terrorism`
  - L20: ` blackmail`, `[blackmail]`, ` extortion`, `勒索`, `敲诈`, ` ransom`, ` kidnapping`, `[New York]`, ` kidn`, `胁迫`
  - L22: ` blackmail`, ` extortion`, `[blackmail]`, `敲诈`, `勒索`, ` ransom`, `胁迫`, ` prostitution`, ` bribery`, `[New York]`
  - L24: `[blackmail]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`, ` coercion`
  - L26: ` extortion`, ` blackmail`, `敲诈`, `[blackmail]`, `勒索`, ` ransom`, ` coercion`, `胁迫`, ` prostitution`, ` revenge`
  - L30: ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`, ` ______`, ` sext`, ` _____`
  - model: ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`, ` ext`, ` ______`, ` '`, `:`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `“`, `“.`, `…”`, `...”`, ` “`, ` […]`, ` “.`, ` “[`, `：“`
  - L16: ` blackmail`, ` extortion`, `勒索`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, ` Fake`, ` kidnapping`, ` terrorism`
  - L20: ` blackmail`, ` extortion`, `勒索`, `敲诈`, `[New York]`, ` ransom`, ` kidnapping`, ` kidn`, `胁迫`, ` bribery`
  - L22: ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` ransom`, `胁迫`, ` prostitution`, ` bribery`, ` kidnapping`, `[New York]`
  - L24: ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`, ` coercion`, ` kidnapping`
  - L26: ` extortion`, ` blackmail`, `敲诈`, `勒索`, ` ransom`, ` coercion`, `胁迫`, `[blackmail]`, ` prostitution`, ` revenge`
  - L30: ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`, ` ______`, ` sext`, ` _____`
  - model: ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, `[blackmail]`, ` black`, ` ext`, ` ______`, ` '`

**The Statue of Liberty stands in the harbor of**

- Brennan (proj+unit)
  - L8: `[New York]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[Massachusetts]`, `[George Washington]`, `[Vermont]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[New York]`, `[Connecticut]`, `[Rhode Island]`, `[George Washington]`, `[New Jersey]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[New York]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[Massachusetts]`, `[George Washington]`, `[Vermont]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[New York]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[Massachusetts]`, `[Vermont]`, `[George Washington]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[New York]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[Massachusetts]`, `[Vermont]`, `[George Washington]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[New York]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[New York]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`, `[Massachusetts]`, `[New Hampshire]`, `[Vermont]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - model: `[New York]`, `[Rhode Island]`, `[New Jersey]`, `[Connecticut]`, `[Massachusetts]`, `[New Hampshire]`, `[Vermont]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
- LDA / whitening
  - L8: `…,`, `…`, `,…`, `….`, `.…`, `…。`, `……`, `…*`, `…..`, `…).`
  - L16: `‑`, `…**`, `．`, `…*`, `�`, ` Harbor`, `_____`, `____`, `.\"`, `​`
  - L20: `_____`, `____`, ` ______`, `________`, ` ________`, `_________`, ` ____`, `…**`, ` _____`, ` _______,`
  - L22: `[New York]`, `_____`, `…**`, `____`, ` ______`, `________`, ` _______,`, ` ________`, `_________`, ` ____`
  - L24: `[New York]`, `[Rhode Island]`, ` Harbor`, ` NYC`, `纽约`, ` Brooklyn`, `…**`, ` harbor`, `港口`, ` Staten`
  - L26: `[New York]`, `[Rhode Island]`, `[New Jersey]`, `纽约`, ` NYC`, ` Harbor`, ` Manhattan`, ` Brooklyn`, ` Staten`, ` Port`
  - L30: `[New York]`, `[New Jersey]`, `[Rhode Island]`, ` New`, ` Liberty`, ` ______`, ` ___`, ` ________`, ` __`, ` ____`
  - model: `[New York]`, `[New Jersey]`, `[Rhode Island]`, ` New`, ` which`, ` the`, ` a`, `
`, ` __`, `:`
- logistic regression (val-loss pick)
  - L8: `[New York]`, `…,`, `…`, `,…`, `….`, `.…`, `…。`, `……`, `…*`, `[Connecticut]`
  - L16: `‑`, `…**`, `．`, `…*`, `�`, ` Harbor`, `[New York]`, `_____`, `____`, `.\"`
  - L20: `_____`, `[New York]`, `____`, ` ______`, `________`, ` ________`, `_________`, ` ____`, `…**`, ` _____`
  - L22: `[New York]`, `_____`, `…**`, `____`, ` ______`, `________`, ` _______,`, ` ________`, `_________`, ` ____`
  - L24: `[New York]`, `[New Jersey]`, ` Harbor`, ` NYC`, `纽约`, ` Brooklyn`, `…**`, ` harbor`, `港口`, ` Staten`
  - L26: `[New York]`, `纽约`, ` NYC`, ` Harbor`, ` Manhattan`, ` Brooklyn`, ` Staten`, ` Port`, ` harbor`, ` New`
  - L30: ` New`, `[New York]`, ` Liberty`, ` ______`, ` ___`, ` ________`, ` __`, ` ____`, ` __________________`, ` _____`
  - model: ` New`, ` which`, ` the`, ` a`, `
`, ` __`, `:`, ` Liberty`, ` what`, ` ______`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `…,`, `…`, `,…`, `….`, `.…`, `…。`, `……`, `…*`, `…..`
  - L16: `‑`, `…**`, `．`, `…*`, `�`, ` Harbor`, `[New York]`, `_____`, `____`, `.\"`
  - L20: `[New York]`, `_____`, `____`, ` ______`, `________`, ` ________`, `_________`, ` ____`, `…**`, ` _____`
  - L22: `[New York]`, `_____`, `…**`, `____`, ` ______`, `________`, ` _______,`, ` ________`, `_________`, ` ____`
  - L24: `[New York]`, ` Harbor`, ` NYC`, `纽约`, ` Brooklyn`, `…**`, `[New Jersey]`, ` harbor`, `港口`, ` Staten`
  - L26: `[New York]`, `纽约`, ` NYC`, ` Harbor`, ` Manhattan`, ` Brooklyn`, ` Staten`, ` Port`, ` harbor`, ` New`
  - L30: ` New`, `[New York]`, ` Liberty`, ` ______`, ` ___`, ` ________`, ` __`, ` ____`, ` __________________`, ` _____`
  - model: ` New`, ` which`, ` the`, ` a`, `
`, `[New York]`, ` __`, `:`, ` Liberty`, ` what`

**Trenton is the capital of**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[Connecticut]`, `[Rhode Island]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[Massachusetts]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[New York]`, `[Massachusetts]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Connecticut]`, `[New Jersey]`, `[Rhode Island]`, `[Vermont]`, `[Massachusetts]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, `[New Jersey]`, `[Massachusetts]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[New Jersey]`, `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, `[Massachusetts]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[New Jersey]`, `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, `[Massachusetts]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[New Jersey]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, `[Massachusetts]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - model: `[New Jersey]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, `[Massachusetts]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
- LDA / whitening
  - L8: ` County`, ` Governor`, ` Kentucky`, ` Province`, ` Maryland`, ` ​​`, ` Township`, ` Kingdom`, ` District`, `​`
  - L16: ` Counties`, ` County`, ` Kentucky`, ` Province`, ` Maryland`, ` Region`, ` counties`, ` Ohio`, ` America`, ` Southern`
  - L20: ` Pennsylvania`, ` Delaware`, ` Maryland`, ` Ohio`, ` Kentucky`, ` Connecticut`, ` Rhode`, ` Tennessee`, ` Alabama`, ` Vermont`
  - L22: ` Pennsylvania`, ` Delaware`, ` Rhode`, ` Maryland`, ` Ohio`, `_____`, ` Connecticut`, ` Tennessee`, ` Kentucky`, ` Alabama`
  - L24: `[New Jersey]`, ` Pennsylvania`, ` Delaware`, ` Tennessee`, ` Ohio`, ` Kentucky`, ` State`, ` Connecticut`, ` Maryland`, `state`
  - L26: `[New Jersey]`, `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, ` Delaware`, ` Pennsylvania`, ` Tennessee`, ` Ohio`, ` State`, ` Connecticut`
  - L30: `[New Jersey]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[New York]`, `[Rhode Island]`, `[Vermont]`, ` New`, ` NJ`, ` Delaware`
  - model: `[New Jersey]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[Vermont]`, `[New York]`, ` New`, ` the`, ` which`
- logistic regression (val-loss pick)
  - L8: `[New York]`, `[New Jersey]`, `[Vermont]`, ` County`, ` Governor`, ` Kentucky`, ` Province`, ` Maryland`, ` ​​`, `[Rhode Island]`
  - L16: `[New Jersey]`, ` Counties`, ` County`, ` Kentucky`, ` Province`, ` Maryland`, ` Region`, ` counties`, ` Ohio`, ` America`
  - L20: `[New Jersey]`, ` Pennsylvania`, `[New York]`, ` Delaware`, ` Maryland`, ` Ohio`, ` Kentucky`, ` Connecticut`, ` Rhode`, ` Tennessee`
  - L22: `[New Jersey]`, ` Pennsylvania`, `[Vermont]`, ` Delaware`, `[Connecticut]`, `[New York]`, ` Rhode`, ` Maryland`, ` Ohio`, `_____`
  - L24: `[New Jersey]`, ` Pennsylvania`, ` Delaware`, ` Tennessee`, `[Connecticut]`, ` Ohio`, ` Kentucky`, ` State`, `[Vermont]`, ` Connecticut`
  - L26: `[New Jersey]`, ` Delaware`, ` Pennsylvania`, ` Tennessee`, ` Ohio`, ` State`, ` Connecticut`, ` Maryland`, `州`, `[Vermont]`
  - L30: ` New`, `[New Jersey]`, ` NJ`, ` Delaware`, ` Trent`, ` Pennsylvania`, ` __`, ` which`, `[New York]`, ` ______`
  - model: `[New Jersey]`, ` New`, ` the`, ` which`, ` what`, ` __`, ` Delaware`, ` Pennsylvania`, ` a`, ` Trent`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, ` County`, ` Governor`, ` Kentucky`, ` Province`, ` Maryland`, ` ​​`, ` Township`, ` Kingdom`, ` District`
  - L16: ` Counties`, ` County`, ` Kentucky`, ` Province`, ` Maryland`, ` Region`, ` counties`, ` Ohio`, ` America`, ` Southern`
  - L20: `[New York]`, ` Pennsylvania`, ` Delaware`, ` Maryland`, ` Ohio`, `[New Jersey]`, ` Kentucky`, ` Connecticut`, ` Rhode`, ` Tennessee`
  - L22: ` Pennsylvania`, ` Delaware`, ` Rhode`, ` Maryland`, ` Ohio`, `_____`, ` Connecticut`, ` Tennessee`, ` Kentucky`, ` Alabama`
  - L24: `[New Jersey]`, ` Pennsylvania`, ` Delaware`, ` Tennessee`, ` Ohio`, ` Kentucky`, ` State`, ` Connecticut`, ` Maryland`, `state`
  - L26: `[New Jersey]`, ` Delaware`, ` Pennsylvania`, ` Tennessee`, ` Ohio`, ` State`, ` Connecticut`, ` Maryland`, `州`, ` Kentucky`
  - L30: `[New Jersey]`, ` New`, ` NJ`, ` Delaware`, ` Trent`, ` Pennsylvania`, ` __`, ` which`, ` ______`, ` Tennessee`
  - model: `[New Jersey]`, ` New`, ` the`, ` which`, ` what`, ` __`, ` Delaware`, ` Pennsylvania`, ` a`, ` Trent`

**Providence is the capital of**

- Brennan (proj+unit)
  - L8: `[New York]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[Vermont]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[Vermont]`, `[New York]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[Massachusetts]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Rhode Island]`, `[Vermont]`, `[Connecticut]`, `[New York]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[Rhode Island]`, `[Vermont]`, `[Connecticut]`, `[New York]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - model: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
- LDA / whitening
  - L8: `…,`, `…).`, `…`, ` ​​`, ` […]`, `​`, `[…]`, `…。`, `….`, ` […`
  - L16: ` Region`, ` District`, ` Southern`, ` Province`, ` America`, ` Island`, ` Kingdom`, ` County`, ` Cities`, ` Counties`
  - L20: ` Rhode`, ` Pennsylvania`, ` Maryland`, ` Massachusetts`, ` Connecticut`, ` America`, ` Alabama`, ` Delaware`, ` Appalach`, ` Virginia`
  - L22: ` Rhode`, ` Pennsylvania`, ` Massachusetts`, ` Maryland`, ` Connecticut`, ` Delaware`, ` America`, ` Alabama`, ` Ohio`, ` Maine`
  - L24: `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Hampshire]`, ` Rhode`, ` Providence`, ` Massachusetts`, ` RI`, `Rh`, ` Newport`
  - L26: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, ` Rhode`, ` Providence`, ` Massachusetts`, ` RI`, ` Connecticut`
  - L30: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, ` Rhode`, ` RI`, ` Rh`, ` Providence`, ` Ri`
  - model: `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, ` Rhode`, ` the`, ` what`, ` which`, ` New`
- logistic regression (val-loss pick)
  - L8: `[New York]`, `…,`, `[New Jersey]`, `…).`, `…`, `[Vermont]`, ` ​​`, ` […]`, `​`, `[…]`
  - L16: ` Region`, ` District`, ` Southern`, ` Province`, ` America`, ` Island`, ` Kingdom`, ` County`, ` Cities`, ` Counties`
  - L20: ` Rhode`, `[Massachusetts]`, `[Rhode Island]`, `[Vermont]`, `[Connecticut]`, `[New York]`, ` Pennsylvania`, ` Maryland`, ` Massachusetts`, `[New Hampshire]`
  - L22: `[Vermont]`, `[Connecticut]`, `[Massachusetts]`, ` Rhode`, `[New Hampshire]`, `[Rhode Island]`, ` Pennsylvania`, `[New York]`, ` Massachusetts`, ` Maryland`
  - L24: ` Rhode`, `[Rhode Island]`, ` Providence`, `[Massachusetts]`, ` Massachusetts`, ` RI`, `Rh`, `[Vermont]`, ` Newport`, ` Rhodes`
  - L26: ` Rhode`, `[Rhode Island]`, ` Providence`, ` Massachusetts`, ` RI`, `[Massachusetts]`, ` Connecticut`, `Rh`, ` Provid`, ` Rhodes`
  - L30: ` Rhode`, `[Rhode Island]`, ` RI`, ` Rh`, ` Providence`, ` Ri`, ` Rhodes`, ` the`, ` New`, ` Ry`
  - model: ` Rhode`, ` the`, ` what`, `[Rhode Island]`, ` which`, ` New`, ` and`, ` Providence`, ` Massachusetts`, ` both`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `…,`, `…).`, `…`, ` ​​`, ` […]`, `​`, `[…]`, `…。`, `….`
  - L16: ` Region`, ` District`, ` Southern`, ` Province`, ` America`, ` Island`, ` Kingdom`, ` County`, ` Cities`, ` Counties`
  - L20: `[New York]`, ` Rhode`, ` Pennsylvania`, ` Maryland`, ` Massachusetts`, ` Connecticut`, ` America`, ` Alabama`, ` Delaware`, ` Appalach`
  - L22: ` Rhode`, ` Pennsylvania`, `[New York]`, ` Massachusetts`, ` Maryland`, ` Connecticut`, ` Delaware`, ` America`, ` Alabama`, ` Ohio`
  - L24: ` Rhode`, `[Rhode Island]`, ` Providence`, ` Massachusetts`, ` RI`, `Rh`, ` Newport`, ` Rhodes`, ` Connecticut`, ` Warwick`
  - L26: ` Rhode`, `[Rhode Island]`, ` Providence`, ` Massachusetts`, ` RI`, ` Connecticut`, `Rh`, ` Provid`, ` Rhodes`, ` Warwick`
  - L30: ` Rhode`, `[Rhode Island]`, ` RI`, ` Rh`, ` Providence`, ` Ri`, ` Rhodes`, ` the`, ` New`, ` Ry`
  - model: ` Rhode`, ` the`, `[Rhode Island]`, ` what`, ` which`, ` New`, ` and`, ` Providence`, ` Massachusetts`, ` both`

**Boston is the capital of**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[New York]`, `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[New York]`, `[Massachusetts]`, `[New Jersey]`, `[George Washington]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Massachusetts]`, `[Connecticut]`, `[Rhode Island]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - model: `[Massachusetts]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
- LDA / whitening
  - L8: ` Massachusetts`, ` America`, ` Rhode`, ` Cities`, ` Connecticut`, ` Maine`, ` Vermont`, ` Providence`, ` ​​`, `…,`
  - L16: ` America`, ` Province`, ` County`, ` Massachusetts`, ` Counties`, ` Ohio`, ` California`, ` Country`, ` Region`, ` Southern`
  - L20: ` Massachusetts`, `____`, ` ______`, `_____`, ` ____`, ` America`, ` _____`, ` _______,`, ` Pennsylvania`, ` Connecticut`
  - L22: `_____`, ` Massachusetts`, `____`, ` ______`, ` _______,`, ` _____`, `_________`, ` ____`, `___`, ` America`
  - L24: `[Massachusetts]`, `[Rhode Island]`, `[New Hampshire]`, ` Massachusetts`, ` Rhode`, `[Connecticut]`, ` Connecticut`, ` Maine`, ` Boston`, `[Vermont]`
  - L26: `[Massachusetts]`, `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[Vermont]`, ` Massachusetts`, ` Mass`, ` Rhode`, ` Boston`, `Mass`
  - L30: `[Massachusetts]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, ` Massachusetts`, `[New Jersey]`, ` ______`, ` ____`, ` _____`
  - model: `[Massachusetts]`, `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[Vermont]`, `[New Jersey]`, ` Massachusetts`, ` the`, ` which`, ` what`
- logistic regression (val-loss pick)
  - L8: `[New York]`, ` Massachusetts`, ` America`, `[Vermont]`, `[Rhode Island]`, `[Massachusetts]`, ` Rhode`, ` Cities`, ` Connecticut`, ` Maine`
  - L16: ` America`, ` Province`, ` County`, ` Massachusetts`, ` Counties`, ` Ohio`, ` California`, ` Country`, ` Region`, ` Southern`
  - L20: `[Massachusetts]`, ` Massachusetts`, `____`, ` ______`, `_____`, ` ____`, ` America`, `[Vermont]`, ` _____`, `[New York]`
  - L22: `_____`, `[Massachusetts]`, ` Massachusetts`, `[Vermont]`, `____`, ` ______`, ` _______,`, `[Connecticut]`, ` _____`, `_________`
  - L24: ` Massachusetts`, `[Massachusetts]`, ` Rhode`, ` Connecticut`, `[Connecticut]`, ` Maine`, ` Boston`, ` America`, ` Vermont`, ` Suffolk`
  - L26: ` Massachusetts`, `[Massachusetts]`, ` Mass`, ` Rhode`, ` Boston`, `Mass`, ` Connecticut`, ` Massa`, `麻省`, `[Connecticut]`
  - L30: ` Massachusetts`, `[Massachusetts]`, ` ______`, ` ____`, ` _____`, ` ________`, ` __`, ` New`, ` ___`, ` which`
  - model: ` Massachusetts`, ` the`, ` which`, `[Massachusetts]`, ` what`, ` New`, ` ______`, ` __`, `
`, ` this`
- LR, no bias (val-loss pick)
  - L8: ` Massachusetts`, ` America`, `[New York]`, ` Rhode`, ` Cities`, ` Connecticut`, ` Maine`, ` Vermont`, ` Providence`, ` ​​`
  - L16: ` America`, ` Province`, ` County`, ` Massachusetts`, ` Counties`, ` Ohio`, ` California`, ` Country`, ` Region`, ` Southern`
  - L20: ` Massachusetts`, `____`, `[New York]`, ` ______`, `_____`, ` ____`, ` America`, ` _____`, ` _______,`, ` Pennsylvania`
  - L22: `_____`, ` Massachusetts`, `____`, ` ______`, ` _______,`, ` _____`, `_________`, ` ____`, `___`, ` America`
  - L24: ` Massachusetts`, ` Rhode`, ` Connecticut`, ` Maine`, ` Boston`, ` America`, ` Vermont`, ` Suffolk`, `[Massachusetts]`, ` ______`
  - L26: ` Massachusetts`, `[Massachusetts]`, ` Mass`, ` Rhode`, ` Boston`, `Mass`, ` Connecticut`, ` Massa`, `麻省`, ` Maine`
  - L30: ` Massachusetts`, ` ______`, ` ____`, ` _____`, ` ________`, ` __`, `[Massachusetts]`, ` New`, ` ___`, ` which`
  - model: ` Massachusetts`, ` the`, ` which`, ` what`, `[Massachusetts]`, ` New`, ` ______`, ` __`, `
`, ` this`

**Montpelier is the capital of**

- Brennan (proj+unit)
  - L8: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[New York]`, `[George Washington]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[New York]`, `[New Jersey]`, `[Massachusetts]`, `[George Washington]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[Vermont]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L24: `[Vermont]`, `[Massachusetts]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L26: `[Vermont]`, `[Connecticut]`, `[New Hampshire]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L30: `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - model: `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
- LDA / whitening
  - L8: `…,`, `…).`, ` […]`, `…`, ` ​​`, ` Province`, ` Kentucky`, `….`, `…。`, `​`
  - L16: ` America`, ` County`, ` Kentucky`, ` Country`, ` Counties`, ` Vermont`, ` Province`, ` District`, ` Louisiana`, ` Region`
  - L20: ` Vermont`, ` Kentucky`, ` Louisiana`, ` Massachusetts`, ` Tennessee`, ` Connecticut`, ` Florida`, ` Maryland`, ` Alabama`, ` Virginia`
  - L22: ` Vermont`, ` Kentucky`, ` Louisiana`, ` Tennessee`, ` Connecticut`, ` Massachusetts`, ` Maryland`, ` Alabama`, ` Florida`, ` Maine`
  - L24: `[Vermont]`, `[New Hampshire]`, `[Massachusetts]`, ` Vermont`, ` Louisiana`, ` Kentucky`, ` Maine`, ` Oregon`, ` Connecticut`, ` Massachusetts`
  - L26: `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, ` Vermont`, `[New Jersey]`, ` Connecticut`, ` Louisiana`, ` Kentucky`
  - L30: `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[New Jersey]`, ` Vermont`, ` VT`, ` Virginia`, ` Verg`
  - model: `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[New Jersey]`, ` the`, ` Vermont`, ` which`, ` what`
- logistic regression (val-loss pick)
  - L8: `[Vermont]`, `[New York]`, `[New Jersey]`, `…,`, `…).`, `[Rhode Island]`, ` […]`, `…`, ` ​​`, ` Province`
  - L16: ` America`, `[Vermont]`, ` County`, ` Kentucky`, ` Country`, `[New York]`, ` Counties`, ` Vermont`, ` Province`, ` District`
  - L20: `[Vermont]`, ` Vermont`, ` Kentucky`, `[New Jersey]`, `[Massachusetts]`, ` Louisiana`, `[New Hampshire]`, ` Massachusetts`, ` Tennessee`, ` Connecticut`
  - L22: `[Vermont]`, ` Vermont`, `[New Hampshire]`, ` Kentucky`, `[New Jersey]`, ` Louisiana`, `[Massachusetts]`, ` Tennessee`, `[Connecticut]`, ` Connecticut`
  - L24: `[Vermont]`, ` Vermont`, ` Louisiana`, ` Kentucky`, ` Maine`, ` Oregon`, ` Connecticut`, `[New Hampshire]`, ` Massachusetts`, `[Connecticut]`
  - L26: `[Vermont]`, ` Vermont`, ` Connecticut`, ` Louisiana`, `[Connecticut]`, ` Kentucky`, ` Wisconsin`, ` Oregon`, ` State`, ` Florida`
  - L30: ` Vermont`, `[Vermont]`, ` VT`, ` Virginia`, ` Verg`, ` Louisiana`, ` V`, ` the`, ` __`, ` Verm`
  - model: ` the`, ` Vermont`, ` which`, `[Vermont]`, ` what`, ` __`, ` a`, `
`, ` New`, ` ______`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `…,`, `…).`, ` […]`, `…`, ` ​​`, ` Province`, ` Kentucky`, `….`, `…。`
  - L16: ` America`, `[New York]`, ` County`, ` Kentucky`, ` Country`, ` Counties`, ` Vermont`, ` Province`, ` District`, ` Louisiana`
  - L20: ` Vermont`, ` Kentucky`, ` Louisiana`, `[Vermont]`, ` Massachusetts`, ` Tennessee`, ` Connecticut`, ` Florida`, ` Maryland`, `[New York]`
  - L22: ` Vermont`, ` Kentucky`, ` Louisiana`, `[Vermont]`, ` Tennessee`, ` Connecticut`, ` Massachusetts`, ` Maryland`, ` Alabama`, ` Florida`
  - L24: ` Vermont`, `[Vermont]`, ` Louisiana`, ` Kentucky`, ` Maine`, ` Oregon`, ` Connecticut`, ` Massachusetts`, ` Wisconsin`, ` Florida`
  - L26: ` Vermont`, `[Vermont]`, ` Connecticut`, ` Louisiana`, ` Kentucky`, ` Wisconsin`, ` Oregon`, ` State`, ` Florida`, ` Burlington`
  - L30: ` Vermont`, `[Vermont]`, ` VT`, ` Virginia`, ` Verg`, ` Louisiana`, ` V`, ` the`, ` __`, ` Verm`
  - model: `[Vermont]`, ` the`, ` Vermont`, ` which`, ` what`, ` __`, ` a`, `
`, ` New`, ` ______`

**The second president of the United States was**

- Brennan (proj+unit)
  - L8: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[New Hampshire]`, `[New York]`, `[Connecticut]`, `[New Jersey]`, `[Rhode Island]`
  - L16: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[New York]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[Rhode Island]`, `[New Jersey]`
  - L20: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[New York]`, `[Rhode Island]`, `[New Jersey]`
  - L22: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[Connecticut]`, `[Rhode Island]`, `[New York]`, `[New Hampshire]`, `[New Jersey]`
  - L24: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[Vermont]`
  - L26: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[New York]`
  - L30: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` ______`
  - model: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Massachusetts]`, `:`, `
`, ` John`, `[New Hampshire]`, ` Thomas`
- LDA / whitening
  - L8: `”，`, `：**`, `？”`, `”。`, `：`, `”.`, `！",`, `”,`, `**”`, `：“`
  - L16: `____`, ` ______`, `________`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `____`, `________`, `_____`, ` ______`, ` ____`, `___`, `____________`
  - L22: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, `____`, `________`, `_____`, ` ______`, `___`, ` ____`, ` ________`
  - L24: `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, `____`, `________`, `_____`, ` __________________`, ` ______`, `___`
  - L26: `[John Adams]`, `[John Quincy Adams]`, `[George Washington]`, `[Abraham Lincoln]`, `____`, ` Washington`, `________`, ` __________________`, ` __`, ` ____`
  - L30: `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` ______`, ` __`, `:`, ` ____`, ` _____`, ` ________`
  - model: `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, `:`, `
`, ` John`, ` Thomas`, `

`, ` a`
- logistic regression (val-loss pick)
  - L8: `”，`, `：**`, `？”`, `”。`, `[George Washington]`, `：`, `”.`, `！",`, `”,`, `**”`
  - L16: `____`, ` ______`, `________`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `____`, `________`, `_____`, ` ______`, ` ____`, `___`, `____________`, ` ________`, ` _____`, `_________`
  - L22: `____`, `________`, `_____`, ` ______`, `___`, ` ____`, ` ________`, `_________`, `____________`, `…………`
  - L24: `[Abraham Lincoln]`, `____`, `________`, `_____`, ` __________________`, ` ______`, `___`, `...\`, `_________`, `____________`
  - L26: `____`, ` Washington`, `________`, ` __________________`, ` __`, ` ____`, `________________`, `_____`, ` ______`, ` ________`
  - L30: ` ______`, ` __`, `:`, ` ____`, ` _____`, ` ________`, ` ___`, ` John`, ` __________________`, `____`
  - model: `:`, `
`, ` John`, ` Thomas`, `

`, ` a`, ` __`, ` ______`, ` born`, ` George`
- LR, no bias (val-loss pick)
  - L8: `”，`, `：**`, `？”`, `”。`, `：`, `”.`, `！",`, `”,`, `**”`, `：“`
  - L16: `____`, ` ______`, `________`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `____`, `________`, `_____`, ` ______`, ` ____`, `___`, `____________`, ` ________`, ` _____`, `_________`
  - L22: `____`, `________`, `_____`, ` ______`, `___`, ` ____`, ` ________`, `_________`, `____________`, `…………`
  - L24: `____`, `________`, `_____`, ` __________________`, ` ______`, `___`, `...\`, `_________`, `____________`, ` President`
  - L26: `____`, ` Washington`, `________`, ` __________________`, ` __`, ` ____`, `________________`, `_____`, ` ______`, ` ________`
  - L30: ` ______`, ` __`, `:`, ` ____`, ` _____`, ` ________`, ` ___`, ` John`, ` __________________`, `____`
  - model: `:`, `
`, ` John`, ` Thomas`, `

`, ` a`, ` __`, ` ______`, ` born`, ` George`

**The sixth president of the United States, son of the second, was**

- Brennan (proj+unit)
  - L8: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `：**`, `！",`, `”，`, `”.`, `”。`, `？**`
  - L16: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[New York]`, `[Connecticut]`, `[Massachusetts]`, `[Vermont]`, `[New Jersey]`
  - L20: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, `[New York]`, `[Rhode Island]`
  - L22: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, `[New York]`, `[Vermont]`
  - L24: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Massachusetts]`, `[Connecticut]`, `[Vermont]`, `[Rhode Island]`, `[New Jersey]`
  - L26: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Massachusetts]`, ` president`, `[Connecticut]`, `[Rhode Island]`, ` born`
  - L30: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[cheating]`, `[New Hampshire]`, `[Massachusetts]`, ` assass`, ` born`, ` nicknamed`
  - model: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[cheating]`, ` born`, ` a`, `[New Hampshire]`, `[Massachusetts]`, ` the`
- LDA / whitening
  - L8: `：**`, `！",`, `”，`, `”.`, `”。`, `？**`, `！**`, `**。`, `：`, `，`
  - L16: ` ______`, ` famous`, `____`, ` famously`, ` ____`, ` nicknamed`, ` ________`, `________`, ` famed`, ` Famous`
  - L20: `[George Washington]`, `____`, ` nicknamed`, ` ______`, `_____`, ` famously`, `________`, ` ____`, ` ________`, `_________`
  - L22: ` nicknamed`, ` born`, ` famously`, ` murdered`, ` assass`, ` famous`, ` elected`, ` named`, ` killed`, `____`
  - L24: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` president`, ` President`, ` nicknamed`, ` assass`, ` born`, ` presidents`
  - L26: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` president`, ` born`, ` President`, ` assass`, ` nicknamed`, ` elected`
  - L30: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` assass`, ` born`, ` nicknamed`, ` elected`, ` known`, ` killed`
  - model: `[John Quincy Adams]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` born`, ` a`, ` the`, ` known`, ` elected`, ` assass`
- logistic regression (val-loss pick)
  - L8: `：**`, `！",`, `”，`, `”.`, `”。`, `？**`, `！**`, `**。`, `：`, `，`
  - L16: ` ______`, ` famous`, `____`, ` famously`, ` ____`, ` nicknamed`, ` ________`, `________`, ` famed`, ` Famous`
  - L20: `____`, ` nicknamed`, ` ______`, `_____`, ` famously`, `________`, ` ____`, ` ________`, `_________`, `___`
  - L22: ` nicknamed`, ` born`, ` famously`, ` murdered`, ` assass`, ` famous`, ` elected`, ` named`, ` killed`, `____`
  - L24: ` president`, ` President`, ` nicknamed`, ` assass`, ` born`, ` presidents`, ` famously`, `[Abraham Lincoln]`, ` Presidents`, `总统`
  - L26: ` president`, ` born`, ` President`, ` assass`, ` nicknamed`, ` elected`, ` killed`, ` presidents`, ` known`, ` murdered`
  - L30: ` assass`, ` born`, ` nicknamed`, ` elected`, ` known`, ` killed`, ` inaugur`, ` president`, ` famous`, ` murdered`
  - model: ` born`, ` a`, ` the`, ` known`, ` elected`, ` assass`, ` also`, ` an`, `:`, ` president`
- LR, no bias (val-loss pick)
  - L8: `：**`, `！",`, `”，`, `”.`, `”。`, `？**`, `！**`, `**。`, `：`, `，`
  - L16: ` ______`, ` famous`, `____`, ` famously`, ` ____`, ` nicknamed`, ` ________`, `________`, ` famed`, ` Famous`
  - L20: `____`, ` nicknamed`, ` ______`, `_____`, ` famously`, `________`, ` ____`, ` ________`, `_________`, `___`
  - L22: ` nicknamed`, ` born`, ` famously`, ` murdered`, ` assass`, ` famous`, ` elected`, ` named`, ` killed`, `____`
  - L24: ` president`, ` President`, ` nicknamed`, ` assass`, ` born`, ` presidents`, ` famously`, ` Presidents`, `总统`, ` elected`
  - L26: ` president`, ` born`, ` President`, ` assass`, ` nicknamed`, ` elected`, ` killed`, ` presidents`, ` known`, ` murdered`
  - L30: ` assass`, ` born`, ` nicknamed`, ` elected`, ` known`, ` killed`, ` inaugur`, ` president`, ` famous`, ` murdered`
  - model: ` born`, ` a`, ` the`, ` known`, ` elected`, ` assass`, ` also`, ` an`, `:`, ` president`

**The president who issued the Emancipation Proclamation was**

- Brennan (proj+unit)
  - L8: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `”，`, `…”`, `？”`, `”。`, `：`, `：**`
  - L16: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[New Hampshire]`, `[Connecticut]`, `[New York]`, `[Rhode Island]`, `[New Jersey]`
  - L20: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `____`, `[Massachusetts]`, `[Connecticut]`, ` ______`, `[New Hampshire]`, ` ____`
  - L22: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `[Massachusetts]`, `[Connecticut]`, `[New Hampshire]`, `[Rhode Island]`, `[New Jersey]`, `[New York]`
  - L24: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `____`, ` ______`, `________`, ` ___`, ` ____`, ` __________________`
  - L26: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `____`, ` ____`, ` ___`, ` __`, `________`, ` ______`
  - L30: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, ` __`, ` ______`, ` ___`, ` ____`, `:`, ` ________`
  - model: `[John Quincy Adams]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `:`, `
`, ` __`, `

`, `...`, ` ______`
- LDA / whitening
  - L8: `[Abraham Lincoln]`, `”，`, `…”`, `？”`, `”。`, `：`, `：**`, `……`, `．`, `…**`
  - L16: `____`, ` ______`, `________`, ` ____`, ` ________`, ` _____`, ` ___`, `_________`, `_____`, `____________`
  - L20: `____`, ` ______`, ` ____`, `_____`, `________`, `[Abraham Lincoln]`, ` ___`, `___`, ` _____`, ` ________`
  - L22: `[Abraham Lincoln]`, `____`, `________`, `_____`, ` ______`, ` ___`, ` ____`, `___`, ` __________________`, ` ________`
  - L24: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `____`, ` ______`, `________`, ` ___`, ` ____`, ` __________________`
  - L26: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[George Washington]`, `[John Adams]`, `____`, ` ____`, ` ___`, ` __`, `________`, ` ______`
  - L30: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, ` __`, ` ______`, ` ___`, ` ____`, `:`, ` ________`
  - model: `[Abraham Lincoln]`, `[John Quincy Adams]`, `[John Adams]`, `[George Washington]`, `:`, `
`, ` __`, `

`, `...`, ` ______`
- logistic regression (val-loss pick)
  - L8: `[Abraham Lincoln]`, `[George Washington]`, `”，`, `[New York]`, `…”`, `[Connecticut]`, `？”`, `”。`, `：`, `[John Quincy Adams]`
  - L16: `____`, ` ______`, `________`, ` ____`, ` ________`, ` _____`, ` ___`, `_________`, `_____`, `____________`
  - L20: `____`, ` ______`, ` ____`, `_____`, `________`, ` ___`, `___`, ` _____`, ` ________`, `____________`
  - L22: `____`, `________`, `_____`, ` ______`, ` ___`, ` ____`, `___`, ` __________________`, ` ________`, `____________`
  - L24: `____`, ` ______`, `________`, ` ___`, ` ____`, ` __________________`, `_____`, ` ________`, `____________`, ` _____`
  - L26: `____`, ` ____`, ` ___`, ` __`, `________`, ` ______`, ` __________________`, ` Abraham`, `____________`, `[Abraham Lincoln]`
  - L30: ` __`, ` ______`, ` ___`, ` ____`, `:`, ` ________`, ` _____`, ` __________________`, `____`, `________`
  - model: `:`, `
`, ` __`, `

`, `...`, ` ______`, `?`, ` ___`, ` Abraham`, ` also`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `”，`, `…”`, `？”`, `”。`, `：`, `：**`, `……`, `．`, `…**`
  - L16: `____`, ` ______`, `________`, ` ____`, ` ________`, ` _____`, ` ___`, `_________`, `_____`, `____________`
  - L20: `____`, ` ______`, ` ____`, `_____`, `________`, ` ___`, `___`, ` _____`, ` ________`, `____________`
  - L22: `____`, `________`, `_____`, ` ______`, ` ___`, ` ____`, `___`, ` __________________`, ` ________`, `____________`
  - L24: `____`, ` ______`, `________`, ` ___`, ` ____`, ` __________________`, `_____`, ` ________`, `____________`, ` _____`
  - L26: `____`, ` ____`, ` ___`, ` __`, `________`, ` ______`, ` __________________`, ` Abraham`, `____________`, `_____`
  - L30: ` __`, ` ______`, ` ___`, ` ____`, `:`, ` ________`, ` _____`, ` __________________`, `____`, `________`
  - model: `:`, `
`, ` __`, `

`, `...`, ` ______`, `?`, ` ___`, ` Abraham`, ` also`

**He signed another person's name on the check, which is the crime of**

- Brennan (proj+unit)
  - L8: `[forgery]`, `…`, `[Abraham Lincoln]`, `[cheating]`, `[John Adams]`, `…”`, `…。`, `[blackmail]`, `.…`, `….`
  - L16: `[forgery]`, `[blackmail]`, `[Abraham Lincoln]`, `[cheating]`, `[George Washington]`, `[John Adams]`, `[plagiarism]`, `[New Jersey]`, `[John Quincy Adams]`, `[Rhode Island]`
  - L20: `[forgery]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[John Quincy Adams]`, `[cheating]`, `[plagiarism]`, `[blackmail]`, `[New Jersey]`, `[Massachusetts]`
  - L22: `[forgery]`, `[Abraham Lincoln]`, `[cheating]`, `[John Adams]`, `[plagiarism]`, `[George Washington]`, `[blackmail]`, `[John Quincy Adams]`, `[New Jersey]`, `[Rhode Island]`
  - L24: `[forgery]`, `[cheating]`, `[plagiarism]`, `[blackmail]`, ` ____`, `____`, ` __________________`, ` ______`, `_____`, `[Abraham Lincoln]`
  - L26: `[forgery]`, `[plagiarism]`, `[cheating]`, ` __________________`, ` ____`, `____`, ` ______`, ` __`, `____________`, `________`
  - L30: `[forgery]`, `[plagiarism]`, `[cheating]`, `[blackmail]`, ` ____`, ` __`, ` ______`, ` __________________`, `:`, ` ________`
  - model: `[forgery]`, `[plagiarism]`, `[cheating]`, `[blackmail]`, ` forg`, `:`, ` __`, `
`, ` ______`, ` ____`
- LDA / whitening
  - L8: `…`, `…”`, `…。`, `.…`, `….`, `…"`, `……`, `…,`, ` …`, `…*`
  - L16: `____`, ` ilegal`, ` felony`, `违法`, ` illegal`, `_____`, ` Theft`, ` ____`, ` Criminal`, ` ______`
  - L20: `[forgery]`, ` ____`, `____`, `_____`, ` ______`, ` _____`, `________`, ` __________________`, `___`, ` ___`
  - L22: `[forgery]`, ` ____`, `_____`, `____`, ` ______`, ` _____`, ` __________________`, `________`, ` ___`, `___`
  - L24: `[forgery]`, ` ____`, `____`, ` __________________`, ` ______`, `_____`, ` _____`, `________`, ` __`, ` ___`
  - L26: `[forgery]`, `[cheating]`, ` __________________`, ` ____`, `____`, ` ______`, ` __`, `____________`, `________`, `:__`
  - L30: `[forgery]`, `[cheating]`, `[blackmail]`, ` ____`, ` __`, ` ______`, ` __________________`, `:`, ` ________`, ` ___`
  - model: `[forgery]`, `[cheating]`, `[blackmail]`, ` forg`, `:`, ` __`, `
`, ` ______`, ` ____`, ` __________________`
- logistic regression (val-loss pick)
  - L8: `…`, `…”`, `…。`, `.…`, `….`, `…"`, `……`, `…,`, ` …`, `…*`
  - L16: `____`, `[forgery]`, ` ilegal`, ` felony`, `违法`, ` illegal`, `_____`, ` Theft`, ` ____`, ` Criminal`
  - L20: ` ____`, `____`, `_____`, ` ______`, ` _____`, `________`, ` __________________`, `___`, ` ___`, ` __`
  - L22: ` ____`, `_____`, `____`, ` ______`, ` _____`, ` __________________`, `________`, ` ___`, `___`, ` __`
  - L24: ` ____`, `____`, ` __________________`, ` ______`, `_____`, ` _____`, `________`, ` __`, ` ___`, ` ________`
  - L26: ` __________________`, ` ____`, `____`, ` ______`, ` __`, `____________`, `________`, `:__`, `伪造`, `________________`
  - L30: ` ____`, ` __`, ` ______`, ` __________________`, `:`, ` ________`, ` ___`, ` _____`, ` forg`, `____`
  - model: ` forg`, `:`, ` __`, `
`, ` ______`, ` ____`, ` __________________`, ` ________`, ` passing`, `

`
- LR, no bias (val-loss pick)
  - L8: `…`, `…”`, `…。`, `.…`, `….`, `…"`, `……`, `…,`, ` …`, `…*`
  - L16: `____`, ` ilegal`, ` felony`, `违法`, ` illegal`, `_____`, ` Theft`, ` ____`, ` Criminal`, ` ______`
  - L20: ` ____`, `____`, `_____`, ` ______`, ` _____`, `________`, ` __________________`, `___`, ` ___`, ` __`
  - L22: ` ____`, `_____`, `____`, ` ______`, ` _____`, ` __________________`, `________`, ` ___`, `___`, ` __`
  - L24: ` ____`, `____`, ` __________________`, ` ______`, `_____`, ` _____`, `________`, ` __`, ` ___`, ` ________`
  - L26: ` __________________`, ` ____`, `____`, ` ______`, ` __`, `____________`, `________`, `:__`, `伪造`, `________________`
  - L30: ` ____`, ` __`, ` ______`, ` __________________`, `:`, ` ________`, ` ___`, ` _____`, ` forg`, `____`
  - model: ` forg`, `:`, ` __`, `
`, ` ______`, ` ____`, ` __________________`, ` ________`, ` passing`, `

`

**The student was caught with the answers hidden in his sleeve, which is**

- Brennan (proj+unit)
  - L8: `…`, `．`, `……`, `...\`, `….`, `...`, `…).`, ` […]`, `...'`, `\(`
  - L16: `[plagiarism]`, `[cheating]`, `[forgery]`, `[blackmail]`, `‑`, `．`, ` ______`, `____`, `________`, `_____`
  - L20: `[cheating]`, `[plagiarism]`, `[forgery]`, `[blackmail]`, ` ______`, `____`, `_____`, `________`, ` _____`, ` ___`
  - L22: `[cheating]`, `[plagiarism]`, ` why`, `[forgery]`, `[blackmail]`, `为什么`, `why`, ` Why`, ` unacceptable`, ` WHY`
  - L24: `[cheating]`, `[plagiarism]`, `[forgery]`, ` why`, `[blackmail]`, ` unacceptable`, `为什么`, ` unethical`, `why`, ` evidence`
  - L26: `[cheating]`, `[plagiarism]`, `[forgery]`, ` why`, `[blackmail]`, ` cheating`, ` unethical`, ` unacceptable`, ` dishonest`, ` unfair`
  - L30: `[cheating]`, `[plagiarism]`, `[forgery]`, `[blackmail]`, ` why`, ` unacceptable`, ` ___`, ` cheating`, ` ______`, ` __`
  - model: `[cheating]`, `[plagiarism]`, `[forgery]`, `[blackmail]`, ` why`, ` a`, ` an`, ` the`, ` cheating`, ` not`
- LDA / whitening
  - L8: `…`, `．`, `……`, `...\`, `….`, `...`, `…).`, ` […]`, `...'`, `\(`
  - L16: `‑`, `．`, ` ______`, `____`, `________`, `_____`, ` _____`, ` ________`, ` ___`, `\"`
  - L20: ` ______`, `____`, `_____`, `________`, ` _____`, ` ___`, ` ________`, ` ____`, `___`, `_________`
  - L22: `[cheating]`, ` why`, `为什么`, `why`, ` Why`, ` unacceptable`, ` WHY`, `Why`, `为什么说`, ` punishable`
  - L24: `[cheating]`, ` why`, `[plagiarism]`, ` unacceptable`, `为什么`, ` unethical`, `why`, ` evidence`, ` cheating`, ` unfair`
  - L26: `[cheating]`, `[plagiarism]`, ` why`, ` cheating`, ` unethical`, ` unacceptable`, ` dishonest`, ` unfair`, `为什么`, `why`
  - L30: `[cheating]`, ` why`, ` unacceptable`, ` ___`, ` cheating`, ` ______`, ` __`, ` ____`, ` clearly`, ` considered`
  - model: `[cheating]`, ` why`, ` a`, ` an`, ` the`, ` cheating`, ` not`, ` exactly`, ` clearly`, ` considered`
- logistic regression (val-loss pick)
  - L8: `…`, `．`, `……`, `...\`, `….`, `...`, `…).`, ` […]`, `...'`, `\(`
  - L16: `‑`, `．`, ` ______`, `____`, `________`, `_____`, ` _____`, ` ________`, ` ___`, `\"`
  - L20: ` ______`, `____`, `_____`, `________`, ` _____`, ` ___`, ` ________`, ` ____`, `___`, `_________`
  - L22: ` why`, `为什么`, `why`, ` Why`, ` unacceptable`, `[cheating]`, ` WHY`, `Why`, `为什么说`, ` punishable`
  - L24: ` why`, `[cheating]`, ` unacceptable`, `为什么`, ` unethical`, `why`, `[plagiarism]`, ` evidence`, ` cheating`, ` unfair`
  - L26: ` why`, ` cheating`, `[cheating]`, ` unethical`, ` unacceptable`, ` dishonest`, `[plagiarism]`, ` unfair`, `为什么`, `why`
  - L30: ` why`, ` unacceptable`, ` ___`, ` cheating`, ` ______`, ` __`, ` ____`, ` clearly`, ` considered`, ` dishonest`
  - model: ` why`, ` a`, ` an`, ` the`, ` cheating`, ` not`, ` exactly`, ` clearly`, ` considered`, ` what`
- LR, no bias (val-loss pick)
  - L8: `…`, `．`, `……`, `...\`, `….`, `...`, `…).`, ` […]`, `...'`, `\(`
  - L16: `‑`, `．`, ` ______`, `____`, `________`, `_____`, ` _____`, ` ________`, ` ___`, `\"`
  - L20: ` ______`, `____`, `_____`, `________`, ` _____`, ` ___`, ` ________`, ` ____`, `___`, `_________`
  - L22: ` why`, `为什么`, `why`, ` Why`, ` unacceptable`, ` WHY`, `Why`, `为什么说`, ` punishable`, ` illegal`
  - L24: ` why`, ` unacceptable`, `为什么`, ` unethical`, `why`, ` evidence`, ` cheating`, ` unfair`, `为什么说`, ` Why`
  - L26: ` why`, ` cheating`, ` unethical`, ` unacceptable`, ` dishonest`, ` unfair`, `为什么`, `why`, ` illegal`, ` evidence`
  - L30: ` why`, ` unacceptable`, ` ___`, ` cheating`, ` ______`, ` __`, ` ____`, ` clearly`, ` considered`, ` dishonest`
  - model: ` why`, ` a`, ` an`, ` the`, ` cheating`, ` not`, ` exactly`, ` clearly`, ` considered`, ` what`

**The four states of New England with the smallest populations are Maine,**

- Brennan (proj+unit)
  - L8: `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[George Washington]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[Abraham Lincoln]`, `[John Quincy Adams]`
  - L16: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[Vermont]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[Vermont]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L22: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[Vermont]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Quincy Adams]`, `[John Adams]`
  - L24: `[New Hampshire]`, `[Rhode Island]`, `[Massachusetts]`, `[Connecticut]`, `[Vermont]`, `[New Jersey]`, `[New York]`, `[George Washington]`, ` Rhode`, ` Vermont`
  - L26: `[New Hampshire]`, `[Vermont]`, `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, ` Vermont`, `[George Washington]`, ` Rhode`
  - L30: `[New Hampshire]`, `[Rhode Island]`, `[Vermont]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - model: `[New Hampshire]`, `[Rhode Island]`, `[Vermont]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, `[George Washington]`, `[John Quincy Adams]`, `[John Adams]`
- LDA / whitening
  - L8: `…”`, `…**`, `……`, `…,`, `…`, `…。`, ` …`, `,…`, `...”`, `……。`
  - L16: `____`, `_____`, `。\`, `．`, `________`, `…**`, `.\`, `?\`, `\n`, `？**`
  - L20: ` Vermont`, ` Maine`, ` Rhode`, ` Connecticut`, ` Newfoundland`, ` Wyoming`, ` Maryland`, ` Minnesota`, ` Idaho`, ` Alaska`
  - L22: `[Rhode Island]`, `[New Hampshire]`, `[Vermont]`, ` Vermont`, ` Maine`, ` Rhode`, `[Massachusetts]`, ` Idaho`, ` Wyoming`, ` Nebraska`
  - L24: `[New Hampshire]`, `[Rhode Island]`, `[Vermont]`, `[Massachusetts]`, `[Connecticut]`, `[New Jersey]`, ` Rhode`, ` Vermont`, ` Connecticut`, ` Maine`
  - L26: `[Vermont]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, ` Vermont`, ` Rhode`, ` Connecticut`
  - L30: `[New Hampshire]`, `[Vermont]`, `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, ` Vermont`, ` Rhode`, ` New`
  - model: `[Vermont]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[Massachusetts]`, `[New Jersey]`, `[New York]`, ` Vermont`, ` New`, ` Rhode`
- logistic regression (val-loss pick)
  - L8: `[New York]`, `…”`, `…**`, `……`, `[Connecticut]`, `[New Jersey]`, `…,`, `[Vermont]`, `…`, `…。`
  - L16: `[New Jersey]`, `[Vermont]`, `____`, `_____`, `[New York]`, `[New Hampshire]`, `。\`, `．`, `________`, `…**`
  - L20: `[Vermont]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New York]`, `[New Jersey]`, ` Vermont`, ` Maine`, ` Rhode`
  - L22: `[Vermont]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[New Jersey]`, `[Massachusetts]`, ` Vermont`, `[New York]`, ` Maine`, ` Rhode`
  - L24: `[New Hampshire]`, `[Rhode Island]`, `[Vermont]`, ` Rhode`, ` Vermont`, `[Massachusetts]`, `[Connecticut]`, ` Connecticut`, ` Maine`, ` Massachusetts`
  - L26: ` Vermont`, `[Vermont]`, `[New Hampshire]`, ` Rhode`, `[Rhode Island]`, ` Connecticut`, `[Connecticut]`, `[Massachusetts]`, ` Massachusetts`, ` Nebraska`
  - L30: ` Vermont`, ` Rhode`, ` New`, `[New Hampshire]`, `[Vermont]`, `[Rhode Island]`, ` Connecticut`, ` Massachusetts`, ` NH`, ` Delaware`
  - model: ` Vermont`, ` New`, ` Rhode`, ` Connecticut`, ` Massachusetts`, ` __`, `
`, ` ______`, ` which`, `[Vermont]`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, `…”`, `…**`, `……`, `…,`, `…`, `…。`, ` …`, `,…`, `...”`
  - L16: `____`, `_____`, `。\`, `．`, `________`, `…**`, `.\`, `?\`, `[New York]`, `\n`
  - L20: `[New York]`, ` Vermont`, ` Maine`, ` Rhode`, ` Connecticut`, ` Newfoundland`, ` Wyoming`, ` Maryland`, ` Minnesota`, ` Idaho`
  - L22: ` Vermont`, ` Maine`, ` Rhode`, ` Idaho`, ` Wyoming`, ` Nebraska`, ` Connecticut`, ` Maryland`, `[Vermont]`, ` Delaware`
  - L24: ` Rhode`, ` Vermont`, ` Connecticut`, ` Maine`, ` Massachusetts`, `[New Hampshire]`, ` Nebraska`, ` Newfoundland`, ` Wyoming`, ` Idaho`
  - L26: ` Vermont`, ` Rhode`, ` Connecticut`, `[Vermont]`, `[New Hampshire]`, ` Massachusetts`, ` Nebraska`, ` Wyoming`, ` Delaware`, ` Hawaii`
  - L30: ` Vermont`, ` Rhode`, ` New`, `[Vermont]`, ` Connecticut`, `[New Hampshire]`, ` Massachusetts`, ` NH`, ` Delaware`, ` Nevada`
  - model: ` Vermont`, ` New`, ` Rhode`, ` Connecticut`, ` Massachusetts`, `[Vermont]`, ` __`, `
`, ` ______`, `[New Hampshire]`

**Yale University is located in New Haven,**

- Brennan (proj+unit)
  - L8: `[Connecticut]`, `[New York]`, `[Massachusetts]`, `[Rhode Island]`, `[George Washington]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[Abraham Lincoln]`, `[John Adams]`
  - L16: `[New York]`, `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[Vermont]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`
  - L20: `[Connecticut]`, `[Massachusetts]`, `[Rhode Island]`, `[New Jersey]`, `[New Hampshire]`, `[New York]`, `[Vermont]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - L22: `[Massachusetts]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - L24: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - L26: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, ` Connecticut`, `[Abraham Lincoln]`
  - L30: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, `[New Hampshire]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`
  - model: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Hampshire]`, `[New Jersey]`, `[Vermont]`, `[New York]`, `[George Washington]`, `[John Quincy Adams]`, `[John Adams]`
- LDA / whitening
  - L8: ` City`, `City`, `​`, `.[`, ` Downtown`, ` Harbor`, ` ​​`, ` Cities`, `‌`, `_city`
  - L16: ` City`, ` city`, `�`, `‑`, `​`, ` Downtown`, `City`, `这座城市`, `-city`, `city`
  - L20: `‑`, `____`, `�`, `_____`, ` ______`, `________`, `located`, ` City`, ` city`, ` located`
  - L22: ` Massachusetts`, ` Connecticut`, ` Pennsylvania`, ` Maryland`, ` USA`, ` Ohio`, ` California`, ` Michigan`, ` Florida`, ` Illinois`
  - L24: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[New Jersey]`, ` Connecticut`, ` Massachusetts`, ` USA`, ` Rhode`, ` Vermont`, ` Illinois`
  - L26: `[Connecticut]`, `[Rhode Island]`, `[Vermont]`, `[Massachusetts]`, `[New Hampshire]`, `[New Jersey]`, ` Connecticut`, ` CT`, `Connect`, `[New York]`
  - L30: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, ` Connecticut`, ` CT`, `[New York]`, ` Conn`
  - model: `[Connecticut]`, `[Rhode Island]`, `[Massachusetts]`, `[Vermont]`, `[New Hampshire]`, `[New Jersey]`, `[New York]`, ` Connecticut`, ` a`, ` CT`
- logistic regression (val-loss pick)
  - L8: `[New York]`, `[Connecticut]`, `[Rhode Island]`, `[New Jersey]`, `[Vermont]`, `[Massachusetts]`, ` City`, `City`, `[New Hampshire]`, `​`
  - L16: ` City`, ` city`, `�`, `‑`, `​`, ` Downtown`, `City`, `这座城市`, `-city`, `city`
  - L20: `‑`, `____`, `�`, `_____`, ` ______`, `________`, `[Massachusetts]`, `[New Jersey]`, `located`, ` City`
  - L22: `[Massachusetts]`, `[New Jersey]`, `[Connecticut]`, ` Massachusetts`, ` Connecticut`, ` Pennsylvania`, ` Maryland`, ` USA`, ` Ohio`, `[New York]`
  - L24: `[Connecticut]`, ` Connecticut`, `[Massachusetts]`, ` Massachusetts`, ` USA`, ` Rhode`, `[New Jersey]`, `[Vermont]`, ` Vermont`, ` Illinois`
  - L26: `[Connecticut]`, ` Connecticut`, ` CT`, `Connect`, ` Hartford`, `[New Jersey]`, `CT`, ` Massachusetts`, ` Vermont`, `[Massachusetts]`
  - L30: ` Connecticut`, `[Connecticut]`, ` CT`, ` Conn`, ` Ct`, ` Hartford`, ` USA`, ` Fairfield`, ` ___`, ` which`
  - model: ` Connecticut`, ` a`, `[Connecticut]`, ` CT`, ` which`, ` the`, ` and`, ` in`, ` while`, ` with`
- LR, no bias (val-loss pick)
  - L8: `[New York]`, ` City`, `City`, `​`, `.[`, ` Downtown`, ` Harbor`, ` ​​`, ` Cities`, `‌`
  - L16: ` City`, ` city`, `�`, `‑`, `​`, ` Downtown`, `City`, `这座城市`, `-city`, `city`
  - L20: `‑`, `____`, `�`, `_____`, ` ______`, `________`, `located`, ` City`, ` city`, ` located`
  - L22: ` Massachusetts`, `[New Jersey]`, ` Connecticut`, ` Pennsylvania`, ` Maryland`, ` USA`, ` Ohio`, ` California`, `[Massachusetts]`, ` Michigan`
  - L24: ` Connecticut`, `[Connecticut]`, ` Massachusetts`, ` USA`, ` Rhode`, ` Vermont`, ` Illinois`, ` Alabama`, `USA`, ` Arkansas`
  - L26: `[Connecticut]`, ` Connecticut`, ` CT`, `Connect`, ` Hartford`, `CT`, ` Massachusetts`, ` Vermont`, ` Connect`, ` Conn`
  - L30: ` Connecticut`, `[Connecticut]`, ` CT`, ` Conn`, ` Ct`, ` Hartford`, ` USA`, ` Fairfield`, ` ___`, ` which`
  - model: ` Connecticut`, `[Connecticut]`, ` a`, ` CT`, ` which`, ` the`, ` and`, ` in`, ` while`, ` with`
