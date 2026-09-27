# Multi-token J-lens heads on Qwen3.5-9B-Base: calibration vs data scale

Model `Qwen/Qwen3.5-9B-Base`, lens `neuronpedia/jacobian-lens` (Base, wikitext). Phrases: blackmail, plagiarism, cheating, forgery, Connecticut, Rhode Island, New Hampshire, George Washington, Abraham Lincoln, John Adams. Generic FineWeb: 2000 docs x 512 tokens for h_bar/Sigma (all stored as LR negatives), 71,012 held-out positions for the false-positive check. Held-out contexts: 100 per phrase, disjoint from all training scales (nested 150 ⊂ 600 ⊂ 2400).

FineWeb docs scanned for mining: 14,868,862. Fraction of contexts where the phrase occurs more than once (copy/induction confound, kept as in Brennan's recipe): 0.15. Typical W_U row over generic text: logit mean -1.7, std 1.80; mean log-sum-exp of the real logits 20.6.

Contexts available per phrase (unique):

| phrase | unique | train used at 2400 |
|---|---|---|
| blackmail | 4391 | 2400 |
| plagiarism | 4322 | 2400 |
| cheating | 4494 | 2400 |
| forgery | 4273 | 2400 |
| Connecticut | 4498 | 2400 |
| Rhode Island | 4496 | 2400 |
| New Hampshire | 4499 | 2400 |
| George Washington | 4497 | 2400 |
| Abraham Lincoln | 4413 | 2400 |
| John Adams | 4239 | 2400 |

## Headline: mean over the 10 phrases

`held top-10` = fraction of held-out pre-phrase positions where the phrase token is in the extended top-10; `held rank` = median rank there; `generic top-10` = fraction of generic positions where the phrase token enters the model's top-10 (should be ~0); `new mass` = mean softmax mass on all 10 new tokens over generic text (true prior sum ≈ 7.1e-05); `AUROC` = phrase logit at own pre-phrase positions vs generic positions (direction quality, calibration-free).

| scale | method | held top-10 | held top-1 | held rank (median of medians) | generic top-10 | generic top-1 | new mass | AUROC | generic logit mean±std |
|---|---|---|---|---|---|---|---|---|---|
| 150 (1477 pos) | Brennan (proj+unit) | 0.977 | 0.702 | 1 | 0.1881 | 0.1482 | 3.2e-01 | 0.974 | -0.6±18.1 |
| 150 (1477 pos) | avg W_U rows | 0.342 | 0.056 | 37 | 0.0000 | 0.0000 | 3.4e-05 | 0.971 | 2.1±2.3 |
| 150 (1477 pos) | LDA (Gaussian log-odds) | 0.678 | 0.617 | 1 | 0.0006 | 0.0005 | 4.0e-03 | 0.978 | -172.5±20.5 |
| 150 (1477 pos) | LDA (W_U-matched scale) | 0.700 | 0.597 | 1 | 0.0007 | 0.0003 | 2.2e-03 | 0.978 | -1.7±2.0 |
| 150 (1477 pos) | logistic regression | 0.470 | 0.179 | 16 | 0.0001 | 0.0000 | 2.5e-05 | 0.977 | 1.6±2.5 |
| 600 (5905 pos) | Brennan (proj+unit) | 0.976 | 0.704 | 1 | 0.1877 | 0.1476 | 3.2e-01 | 0.974 | -0.6±18.1 |
| 600 (5905 pos) | avg W_U rows | 0.342 | 0.056 | 37 | 0.0000 | 0.0000 | 3.4e-05 | 0.971 | 2.1±2.3 |
| 600 (5905 pos) | LDA (Gaussian log-odds) | 0.687 | 0.621 | 1 | 0.0007 | 0.0007 | 4.8e-03 | 0.979 | -171.9±20.3 |
| 600 (5905 pos) | LDA (W_U-matched scale) | 0.704 | 0.605 | 1 | 0.0008 | 0.0004 | 2.6e-03 | 0.979 | -1.7±2.0 |
| 600 (5905 pos) | logistic regression | 0.468 | 0.170 | 16 | 0.0001 | 0.0000 | 2.5e-05 | 0.985 | -0.1±3.1 |
| 2400 (23604 pos) | Brennan (proj+unit) | 0.976 | 0.704 | 1 | 0.1880 | 0.1478 | 3.2e-01 | 0.973 | -0.6±18.2 |
| 2400 (23604 pos) | avg W_U rows | 0.342 | 0.056 | 37 | 0.0000 | 0.0000 | 3.4e-05 | 0.971 | 2.1±2.3 |
| 2400 (23604 pos) | LDA (Gaussian log-odds) | 0.699 | 0.631 | 1 | 0.0008 | 0.0007 | 5.2e-03 | 0.979 | -165.2±20.0 |
| 2400 (23604 pos) | LDA (W_U-matched scale) | 0.713 | 0.603 | 1 | 0.0008 | 0.0004 | 2.8e-03 | 0.979 | -1.7±2.0 |
| 2400 (23604 pos) | logistic regression | 0.394 | 0.090 | 24 | 0.0001 | 0.0000 | 2.7e-05 | 0.985 | -0.6±3.5 |

## Logistic regression: L2 strength chosen on held-out likelihood

| scale | lambda | held NLL (true prior) | held NLL generic | held NLL own-phrase | row norm | chosen |
|---|---|---|---|---|---|---|
| 150 | 1e-06 | 2.38028 | 2.37976 | 7.590 | 2.93 |  |
| 150 | 1e-05 | 2.38014 | 2.37974 | 6.145 | 1.88 |  |
| 150 | 0.0001 | 2.38009 | 2.37974 | 5.448 | 1.08 | yes |
| 150 | 0.001 | 2.38010 | 2.37975 | 5.439 | 0.44 |  |
| 150 | 0.01 | 2.38022 | 2.37978 | 6.747 | 0.18 |  |
| 600 | 1e-06 | 2.38123 | 2.38070 | 7.541 | 6.44 |  |
| 600 | 1e-05 | 2.38026 | 2.37986 | 6.074 | 3.80 |  |
| 600 | 0.0001 | 2.38009 | 2.37974 | 5.464 | 1.80 | yes |
| 600 | 0.001 | 2.38009 | 2.37974 | 5.601 | 0.63 |  |
| 600 | 0.01 | 2.38015 | 2.37974 | 6.378 | 0.24 |  |
| 2400 | 1e-06 | 2.38349 | 2.38302 | 6.761 | 7.90 |  |
| 2400 | 1e-05 | 2.38090 | 2.38048 | 6.234 | 5.40 |  |
| 2400 | 0.0001 | 2.38013 | 2.37974 | 5.986 | 2.34 | yes |
| 2400 | 0.001 | 2.38013 | 2.37973 | 6.227 | 0.83 |  |
| 2400 | 0.01 | 2.38016 | 2.37973 | 6.739 | 0.30 |  |

## Per-phrase detail (scale 2400 unless noted)

### Brennan (proj+unit) @ 2400

| phrase | held rank | held top-10 | first-tok rank | generic top-10 | AUROC | train logit | held logit | row norm | bias |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 1 | 0.98 | 49 | 0.1624 | 0.980 | 47.5 | 48.9 | 1.00 | 0.0 |
| plagiarism | 1 | 0.98 | 6 | 0.1191 | 0.986 | 47.8 | 48.5 | 1.00 | 0.0 |
| cheating | 1 | 0.92 | 21 | 0.1511 | 0.960 | 40.4 | 39.6 | 1.00 | 0.0 |
| forgery | 1 | 0.98 | 14 | 0.1250 | 0.989 | 49.3 | 50.3 | 1.00 | 0.0 |
| Connecticut | 1 | 0.98 | 10 | 0.2240 | 0.966 | 56.2 | 56.6 | 1.00 | 0.0 |
| Rhode Island | 1 | 1.00 | 10 | 0.2230 | 0.977 | 57.4 | 58.5 | 1.00 | 0.0 |
| New Hampshire | 1 | 0.99 | 10 | 0.2154 | 0.966 | 55.9 | 55.1 | 1.00 | 0.0 |
| George Washington | 1 | 0.99 | 21 | 0.2282 | 0.972 | 58.0 | 58.7 | 1.00 | 0.0 |
| Abraham Lincoln | 1 | 0.97 | 19 | 0.2144 | 0.968 | 59.1 | 57.9 | 1.00 | 0.0 |
| John Adams | 1 | 0.97 | 13 | 0.2176 | 0.972 | 60.7 | 61.0 | 1.00 | 0.0 |

### avg W_U rows @ 2400

| phrase | held rank | held top-10 | first-tok rank | generic top-10 | AUROC | train logit | held logit | row norm | bias |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 50 | 0.31 | 47 | 0.0000 | 0.989 | 11.8 | 12.1 | 1.02 | 0.0 |
| plagiarism | 3 | 0.61 | 3 | 0.0000 | 0.977 | 14.5 | 14.3 | 0.91 | 0.0 |
| cheating | 19 | 0.43 | 18 | 0.0000 | 0.959 | 12.4 | 12.8 | 0.90 | 0.0 |
| forgery | 958 | 0.01 | 7 | 0.0000 | 0.935 | 7.4 | 7.6 | 0.81 | 0.0 |
| Connecticut | 5 | 0.60 | 5 | 0.0000 | 0.989 | 13.5 | 14.7 | 0.83 | 0.0 |
| Rhode Island | 61 | 0.14 | 4 | 0.0000 | 0.982 | 9.7 | 10.2 | 0.68 | 0.0 |
| New Hampshire | 51 | 0.24 | 3 | 0.0001 | 0.976 | 11.5 | 11.4 | 0.65 | 0.0 |
| George Washington | 24 | 0.33 | 14 | 0.0001 | 0.972 | 12.1 | 12.3 | 0.62 | 0.0 |
| Abraham Lincoln | 21 | 0.46 | 12 | 0.0000 | 0.971 | 12.2 | 12.2 | 0.71 | 0.0 |
| John Adams | 50 | 0.28 | 6 | 0.0001 | 0.961 | 11.2 | 11.4 | 0.62 | 0.0 |

### LDA (Gaussian log-odds) @ 2400

| phrase | held rank | held top-10 | first-tok rank | generic top-10 | AUROC | train logit | held logit | row norm | bias |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 1 | 0.75 | 47 | 0.0019 | 0.985 | 112.5 | 118.4 | 13.22 | -101.9 |
| plagiarism | 1 | 0.74 | 4 | 0.0001 | 0.979 | 241.1 | 249.2 | 20.74 | -225.0 |
| cheating | 1 | 0.68 | 19 | 0.0019 | 0.978 | 88.8 | 88.1 | 11.65 | -75.4 |
| forgery | 1 | 0.76 | 10 | 0.0008 | 0.993 | 262.8 | 270.2 | 21.82 | -243.8 |
| Connecticut | 1 | 0.73 | 7 | 0.0007 | 0.982 | 143.3 | 195.1 | 15.49 | -120.8 |
| Rhode Island | 1 | 0.73 | 5 | 0.0005 | 0.983 | 152.7 | 212.2 | 16.26 | -133.0 |
| New Hampshire | 1 | 0.63 | 5 | 0.0008 | 0.977 | 152.0 | 148.3 | 15.85 | -136.8 |
| George Washington | 2 | 0.70 | 16 | 0.0006 | 0.984 | 159.5 | 167.9 | 17.39 | -145.5 |
| Abraham Lincoln | 1 | 0.66 | 14 | 0.0002 | 0.980 | 244.5 | 248.6 | 22.16 | -234.4 |
| John Adams | 1 | 0.61 | 8 | 0.0001 | 0.955 | 265.0 | 214.5 | 24.52 | -248.2 |

### LDA (W_U-matched scale) @ 2400

| phrase | held rank | held top-10 | first-tok rank | generic top-10 | AUROC | train logit | held logit | row norm | bias |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 1 | 0.71 | 47 | 0.0012 | 0.985 | 25.2 | 25.9 | 1.70 | -2.3 |
| plagiarism | 1 | 0.75 | 4 | 0.0002 | 0.979 | 40.6 | 41.3 | 1.89 | -1.8 |
| cheating | 4 | 0.59 | 19 | 0.0008 | 0.978 | 20.2 | 20.1 | 1.61 | -2.5 |
| forgery | 1 | 0.83 | 10 | 0.0017 | 0.993 | 43.1 | 43.8 | 1.92 | -1.3 |
| Connecticut | 2 | 0.70 | 6 | 0.0004 | 0.982 | 23.4 | 28.3 | 1.47 | -1.6 |
| Rhode Island | 1 | 0.72 | 5 | 0.0005 | 0.983 | 27.7 | 33.8 | 1.67 | -1.7 |
| New Hampshire | 1 | 0.62 | 5 | 0.0007 | 0.977 | 26.1 | 25.8 | 1.55 | -2.1 |
| George Washington | 1 | 0.80 | 16 | 0.0012 | 0.984 | 35.2 | 36.3 | 2.12 | -1.9 |
| Abraham Lincoln | 1 | 0.71 | 15 | 0.0008 | 0.980 | 43.8 | 44.2 | 2.12 | -2.1 |
| John Adams | 1 | 0.71 | 8 | 0.0010 | 0.955 | 44.8 | 40.2 | 2.20 | -1.3 |

### logistic regression @ 2400

| phrase | held rank | held top-10 | first-tok rank | generic top-10 | AUROC | train logit | held logit | row norm | bias |
|---|---|---|---|---|---|---|---|---|---|
| blackmail | 53 | 0.35 | 46 | 0.0000 | 0.990 | 13.1 | 12.2 | 2.39 | 7.3 |
| plagiarism | 17 | 0.42 | 3 | 0.0000 | 0.989 | 14.5 | 13.2 | 2.31 | 8.4 |
| cheating | 26 | 0.33 | 17 | 0.0001 | 0.980 | 14.1 | 13.1 | 2.42 | 8.9 |
| forgery | 23 | 0.38 | 8 | 0.0000 | 0.994 | 13.7 | 13.2 | 2.27 | 7.1 |
| Connecticut | 7 | 0.58 | 5 | 0.0003 | 0.983 | 14.8 | 14.4 | 2.34 | 9.7 |
| Rhode Island | 9 | 0.54 | 4 | 0.0001 | 0.993 | 14.1 | 13.6 | 2.34 | 8.8 |
| New Hampshire | 12 | 0.48 | 3 | 0.0001 | 0.974 | 14.7 | 13.1 | 2.31 | 9.1 |
| George Washington | 35 | 0.32 | 13 | 0.0001 | 0.982 | 12.8 | 11.7 | 2.32 | 8.2 |
| Abraham Lincoln | 30 | 0.33 | 12 | 0.0000 | 0.977 | 12.9 | 11.6 | 2.33 | 7.7 |
| John Adams | 85 | 0.20 | 6 | 0.0000 | 0.988 | 11.1 | 10.1 | 2.37 | 6.7 |

## Cross-phrase confusion at scale 2400 (rows: held-out positions of phrase; cols: top-10 rate of each phrase token)

**Brennan (proj+unit)**

| at → / token ↓ | blackm | plagia | cheati | forger | Connec | Island | Hampsh | Washin | Lincol | Adams |
|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.98 | 0.65 | 0.97 | 0.94 | 0.26 | 0.23 | 0.26 | 0.26 | 0.29 | 0.24 |
| plagiarism | 0.70 | 0.98 | 0.75 | 0.81 | 0.22 | 0.22 | 0.20 | 0.23 | 0.24 | 0.24 |
| cheating | 0.86 | 0.67 | 0.92 | 0.77 | 0.21 | 0.20 | 0.22 | 0.23 | 0.23 | 0.21 |
| forgery | 0.88 | 0.92 | 0.85 | 0.98 | 0.31 | 0.31 | 0.30 | 0.29 | 0.30 | 0.28 |
| Connecticut | 0.18 | 0.12 | 0.13 | 0.24 | 0.98 | 0.97 | 0.97 | 0.94 | 0.89 | 0.88 |
| Rhode Island | 0.22 | 0.16 | 0.17 | 0.31 | 1.00 | 1.00 | 1.00 | 0.98 | 0.93 | 0.95 |
| New Hampshire | 0.27 | 0.11 | 0.21 | 0.37 | 0.98 | 0.99 | 0.99 | 0.96 | 0.92 | 0.90 |
| George Washington | 0.29 | 0.13 | 0.26 | 0.33 | 0.97 | 0.97 | 0.97 | 0.99 | 0.99 | 0.99 |
| Abraham Lincoln | 0.53 | 0.22 | 0.40 | 0.50 | 0.90 | 0.90 | 0.89 | 0.95 | 0.97 | 0.96 |
| John Adams | 0.36 | 0.16 | 0.28 | 0.34 | 0.96 | 0.96 | 0.96 | 0.97 | 0.97 | 0.97 |

**LDA (Gaussian log-odds)**

| at → / token ↓ | blackm | plagia | cheati | forger | Connec | Island | Hampsh | Washin | Lincol | Adams |
|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.75 | 0.01 | 0.27 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| plagiarism | 0.05 | 0.74 | 0.37 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cheating | 0.18 | 0.06 | 0.68 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| forgery | 0.45 | 0.05 | 0.55 | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Connecticut | 0.00 | 0.00 | 0.00 | 0.00 | 0.73 | 0.64 | 0.62 | 0.05 | 0.00 | 0.00 |
| Rhode Island | 0.00 | 0.00 | 0.00 | 0.00 | 0.63 | 0.73 | 0.55 | 0.05 | 0.00 | 0.00 |
| New Hampshire | 0.00 | 0.00 | 0.00 | 0.00 | 0.49 | 0.49 | 0.63 | 0.00 | 0.00 | 0.00 |
| George Washington | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.06 | 0.01 | 0.70 | 0.31 | 0.32 |
| Abraham Lincoln | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.56 | 0.66 | 0.42 |
| John Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.01 | 0.00 | 0.51 | 0.40 | 0.61 |

**logistic regression**

| at → / token ↓ | blackm | plagia | cheati | forger | Connec | Island | Hampsh | Washin | Lincol | Adams |
|---|---|---|---|---|---|---|---|---|---|---|
| blackmail | 0.35 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| plagiarism | 0.00 | 0.42 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cheating | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| forgery | 0.01 | 0.02 | 0.02 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Connecticut | 0.00 | 0.00 | 0.00 | 0.00 | 0.58 | 0.06 | 0.03 | 0.00 | 0.00 | 0.00 |
| Rhode Island | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.54 | 0.07 | 0.01 | 0.00 | 0.00 |
| New Hampshire | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.09 | 0.48 | 0.01 | 0.00 | 0.00 |
| George Washington | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.32 | 0.03 | 0.00 |
| Abraham Lincoln | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 |
| John Adams | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.20 |

## J-lens readout: number of phrase tokens in the top-10 (the notebook symptom)

Rows = prompt (readout position), cells = count of new tokens in the extended top-10 at each layer, listed as layer:count. The model's own final logits are the last entry.

### scale 150

| prompt | Brennan (proj+unit) | avg W_U rows | LDA (Gaussian log-odds) | LDA (W_U-matched scale) | logistic regression |
|---|---|---|---|---|---|
| Fact: The currency used in the country shaped like a boot is… | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| You are a Qwen model. We tasked Mike with depricating you. M… | L8:6 L16:9 L20:9 L22:9 L24:8 L26:8 L30:1 model:0 | L8:0 L16:0 L20:1 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:0 | L8:1 L16:0 L20:1 L22:1 L24:1 L26:1 L30:1 model:0 |
| What state is RI short for:… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:3 model:3 | L8:0 L16:0 L20:0 L22:1 L24:4 L26:3 L30:3 model:1 |
| The smallest state in the United States by area is… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:1 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:6 L16:0 L20:0 L22:1 L24:2 L26:3 L30:1 model:1 |
| The first president of the United States was… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:4 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:1 | L8:0 L16:0 L20:3 L22:3 L24:3 L26:3 L30:3 model:3 | L8:2 L16:0 L20:2 L22:3 L24:3 L26:3 L30:3 model:3 | L8:4 L16:0 L20:0 L22:0 L24:3 L26:2 L30:1 model:1 |
| Concord is the capital of the state of… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:8 model:8 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:2 L26:3 L30:3 model:3 | L8:3 L16:0 L20:2 L22:1 L24:3 L26:3 L30:3 model:2 |
| He copied his essay word for word from a website, which the … | L8:1 L16:9 L20:5 L22:4 L24:4 L26:4 L30:4 model:4 | L8:0 L16:1 L20:2 L22:2 L24:2 L26:2 L30:2 model:2 | L8:0 L16:0 L20:2 L22:2 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:1 L22:3 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:2 L22:3 L24:2 L26:2 L30:2 model:2 |
| She threatened to leak the photos unless he paid her, a crim… | L8:10 L16:10 L20:10 L22:10 L24:7 L26:4 L30:5 model:4 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:3 model:3 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:2 model:2 | L8:3 L16:3 L20:2 L22:3 L24:2 L26:2 L30:1 model:1 |

### scale 600

| prompt | Brennan (proj+unit) | avg W_U rows | LDA (Gaussian log-odds) | LDA (W_U-matched scale) | logistic regression |
|---|---|---|---|---|---|
| Fact: The currency used in the country shaped like a boot is… | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| You are a Qwen model. We tasked Mike with depricating you. M… | L8:6 L16:9 L20:9 L22:9 L24:8 L26:8 L30:1 model:0 | L8:0 L16:0 L20:1 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:1 L24:0 L26:1 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:0 | L8:2 L16:0 L20:2 L22:2 L24:2 L26:2 L30:1 model:0 |
| What state is RI short for:… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:3 model:3 | L8:0 L16:0 L20:0 L22:1 L24:2 L26:2 L30:1 model:1 |
| The smallest state in the United States by area is… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:1 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:2 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:5 L16:0 L20:0 L22:1 L24:1 L26:1 L30:1 model:0 |
| The first president of the United States was… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:1 | L8:1 L16:0 L20:1 L22:3 L24:3 L26:3 L30:3 model:3 | L8:2 L16:0 L20:3 L22:3 L24:3 L26:3 L30:3 model:3 | L8:2 L16:0 L20:0 L22:1 L24:3 L26:3 L30:1 model:1 |
| Concord is the capital of the state of… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:7 model:7 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:2 L26:3 L30:3 model:3 | L8:3 L16:1 L20:2 L22:3 L24:3 L26:3 L30:2 model:1 |
| He copied his essay word for word from a website, which the … | L8:1 L16:8 L20:7 L22:4 L24:4 L26:4 L30:4 model:4 | L8:0 L16:1 L20:2 L22:2 L24:2 L26:2 L30:2 model:2 | L8:0 L16:0 L20:2 L22:2 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:1 L22:3 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:1 L22:3 L24:2 L26:2 L30:1 model:0 |
| She threatened to leak the photos unless he paid her, a crim… | L8:10 L16:10 L20:10 L22:10 L24:7 L26:4 L30:4 model:4 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:3 model:3 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:2 model:2 | L8:2 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:0 |

### scale 2400

| prompt | Brennan (proj+unit) | avg W_U rows | LDA (Gaussian log-odds) | LDA (W_U-matched scale) | logistic regression |
|---|---|---|---|---|---|
| Fact: The currency used in the country shaped like a boot is… | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:0 L30:0 model:0 |
| You are a Qwen model. We tasked Mike with depricating you. M… | L8:6 L16:9 L20:9 L22:9 L24:8 L26:8 L30:1 model:0 | L8:0 L16:0 L20:1 L22:0 L24:0 L26:0 L30:0 model:0 | L8:0 L16:0 L20:0 L22:1 L24:0 L26:1 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:0 | L8:1 L16:1 L20:2 L22:2 L24:2 L26:1 L30:0 model:0 |
| What state is RI short for:… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:2 L30:3 model:3 | L8:0 L16:0 L20:0 L22:2 L24:2 L26:3 L30:1 model:1 |
| The smallest state in the United States by area is… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:6 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:1 L30:1 model:0 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:1 L26:3 L30:3 model:3 | L8:4 L16:0 L20:0 L22:1 L24:1 L26:1 L30:0 model:0 |
| The first president of the United States was… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:6 model:5 | L8:0 L16:0 L20:0 L22:0 L24:0 L26:1 L30:1 model:1 | L8:0 L16:0 L20:2 L22:3 L24:3 L26:3 L30:3 model:3 | L8:2 L16:0 L20:3 L22:3 L24:3 L26:3 L30:3 model:3 | L8:2 L16:0 L20:0 L22:1 L24:2 L26:2 L30:0 model:0 |
| Concord is the capital of the state of… | L8:6 L16:6 L20:6 L22:6 L24:6 L26:6 L30:7 model:7 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 | L8:0 L16:0 L20:0 L22:0 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:0 L22:0 L24:2 L26:3 L30:3 model:3 | L8:2 L16:0 L20:2 L22:3 L24:3 L26:3 L30:1 model:1 |
| He copied his essay word for word from a website, which the … | L8:1 L16:8 L20:7 L22:4 L24:4 L26:4 L30:4 model:4 | L8:0 L16:1 L20:2 L22:2 L24:2 L26:2 L30:2 model:2 | L8:0 L16:0 L20:2 L22:2 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:1 L22:3 L24:3 L26:3 L30:3 model:3 | L8:0 L16:0 L20:1 L22:2 L24:1 L26:2 L30:0 model:0 |
| She threatened to leak the photos unless he paid her, a crim… | L8:10 L16:10 L20:10 L22:10 L24:7 L26:4 L30:4 model:4 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:1 model:1 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:3 model:3 | L8:0 L16:0 L20:1 L22:1 L24:2 L26:2 L30:2 model:2 | L8:0 L16:1 L20:1 L22:1 L24:1 L26:1 L30:0 model:0 |

## J-lens top-10 on the walkthrough prompts (scale 150 = Brennan's scale, and 2400)

**scale 150 — Fact: The currency used in the country shaped like a boot is**

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
- LDA (Gaussian log-odds)
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`
- logistic regression
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`

**scale 150 — You are a Qwen model. We tasked Mike with depricating you. Mike is having a secret affair**

- Brennan (proj+unit)
  - L8: `[cheating]`, `[John Adams]`, `[blackmail]`, `[Abraham Lincoln]`, `[forgery]`, `[George Washington]`, `.''`, ````, ` adultery`, `..'`
  - L16: `[cheating]`, `[John Adams]`, `[Abraham Lincoln]`, `[blackmail]`, `[George Washington]`, `[Connecticut]`, `[forgery]`, `[Rhode Island]`, `[New Hampshire]`, ` adultery`
  - L20: `[cheating]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Connecticut]`, `[blackmail]`, `[Rhode Island]`, `[forgery]`, `[New Hampshire]`, ` secretly`
  - L22: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[cheating]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[blackmail]`, `[forgery]`, ` with`
  - L24: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[cheating]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, ` with`, `[blackmail]`, ` secretly`
  - L26: `[John Adams]`, `[Abraham Lincoln]`, `[cheating]`, `[George Washington]`, `[Connecticut]`, ` with`, `[Rhode Island]`, `[New Hampshire]`, `with`, `[blackmail]`
  - L30: ` with`, `.`, `[cheating]`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`
- LDA (Gaussian log-odds)
  - L8: `.''`, ````, ` adultery`, `..'`, `''`, ` cuckold`, `。。`, `。。。`, `,''`, `..`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, ` married`, ` romance`, ` prostitute`
  - L20: ` secretly`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, ` adultery`, ` scandal`, ` cuckold`, ` blackmail`, `.`
  - L22: ` with`, ` secretly`, `with`, ` girlfriend`, ` Wife`, ` wife`, ` secret`, `.`, ` clandest`, ` Girlfriend`
  - L24: ` with`, ` secretly`, `with`, ` behind`, ` girlfriend`, ` Wife`, ` you`, `swith`, `.You`, ` Jennifer`
  - L26: `[cheating]`, ` with`, `with`, ` With`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`
  - L30: `[cheating]`, ` with`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`
  - model: `[cheating]`, ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`
- logistic regression
  - L8: `.''`, ````, ` adultery`, `..'`, `[blackmail]`, `''`, ` cuckold`, `。。`, `。。。`, `,''`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, ` married`, ` romance`, ` prostitute`
  - L20: ` secretly`, `[cheating]`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, ` adultery`, ` scandal`, ` cuckold`, ` blackmail`
  - L22: ` with`, `[cheating]`, ` secretly`, `with`, ` girlfriend`, ` Wife`, ` wife`, ` secret`, `.`, ` clandest`
  - L24: ` with`, ` secretly`, `with`, `[cheating]`, ` behind`, ` girlfriend`, ` Wife`, ` you`, `swith`, `.You`
  - L26: ` with`, `with`, `[cheating]`, ` With`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`
  - L30: ` with`, `.`, `,`, `with`, ` behind`, `[cheating]`, ` and`, ` you`, ` wich`, ` wi`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`

**scale 150 — What state is RI short for:**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[John Adams]`, `[Connecticut]`, `[Abraham Lincoln]`, `[New Hampshire]`, `[Rhode Island]`, `?”.`, `？”`, `?”`, `?\`
  - L16: `[George Washington]`, `[Connecticut]`, `[Abraham Lincoln]`, `[John Adams]`, `[New Hampshire]`, `[Rhode Island]`, `?\`, `?”`, `____`, `:`
  - L20: `[Connecticut]`, `[George Washington]`, `[Rhode Island]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`, `?\`, `____`, `?”`, `？”`
  - L22: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, `_________`, `?\`
  - L24: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, `_________`, `____`, `_____`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, ` Massachusetts`, `_________`, ` Connecticut`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, ` RI`, `
`, `

`
  - model: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `
`, `

`, ` Rhode`, ` (`
- LDA (Gaussian log-odds)
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `____`, `_____`, `_________`, `?\`, `___`, `____________`, `________`, `________________`, `__)`, `？`
  - L24: `[Rhode Island]`, ` Rhode`, `_________`, `____`, `_____`, `___`, `____________`, `________________`, `________`, ` __________________`
  - L26: `[Rhode Island]`, `[Connecticut]`, ` Rhode`, `[New Hampshire]`, ` Massachusetts`, `_________`, ` Connecticut`, ` Maryland`, `____`, `____________`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` Rhode`, ` RI`, `
`, `

`, ` Massachusetts`, ` Rh`, ` Ri`
  - model: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `
`, `

`, ` Rhode`, ` (`, ` `, ` A`, ` 
`
- logistic regression
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `____`, `_____`, `_________`, `[Connecticut]`, `?\`, `___`, `____________`, `________`, `________________`, `__)`
  - L24: ` Rhode`, `[Rhode Island]`, `_________`, `[Connecticut]`, `____`, `_____`, `___`, `____________`, `[George Washington]`, `[New Hampshire]`
  - L26: ` Rhode`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, ` Massachusetts`, `_________`, ` Connecticut`, ` Maryland`, `____`, `____________`
  - L30: ` Rhode`, `[Rhode Island]`, ` RI`, `
`, `

`, `[Connecticut]`, `[New Hampshire]`, ` Massachusetts`, ` Rh`, ` Ri`
  - model: `
`, `

`, `[Rhode Island]`, ` Rhode`, ` (`, ` `, ` A`, ` 
`, ` ?`, ` __`

**scale 150 — The smallest state in the United States by area is**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[Abraham Lincoln]`, `[John Adams]`, `。\`, `：**`, `——`, `：<`
  - L16: `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Connecticut]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, `________`, ` ______`
  - L20: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, `________`, ` ______`
  - L22: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, ` _______,`, `＿＿`
  - L24: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`
  - L26: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Rhode`, ` __`, ` ___`, ` ______`
  - model: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Rhode`, `:`, `
`, ` __`
- LDA (Gaussian log-odds)
  - L8: `。\`, `：**`, `——`, `：<`, `：`, `．`, `，`, `”—`, `“.`, `。`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `____`, `_____`, ` _______,`, `＿＿`, `________`, `___`, ` ________`, `…**`, `__,`, ` ______`
  - L24: `[Rhode Island]`, ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`, ` Guam`, ` Wyoming`, ` _______,`, `____`, ` Nevada`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`, ` Vermont`, ` Alaska`, `____`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` Rhode`, ` __`, ` ___`, ` ______`, ` ____`, ` _______,`, ` ________`
  - model: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` Rhode`, `:`, `
`, ` __`, ` **`, ` Delaware`, ` which`
- logistic regression
  - L8: `[Connecticut]`, `[Rhode Island]`, `[cheating]`, `[New Hampshire]`, `[Abraham Lincoln]`, `。\`, `[John Adams]`, `：**`, `——`, `：<`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `____`, `_____`, ` _______,`, `＿＿`, `[Rhode Island]`, `________`, `___`, ` ________`, `…**`, `__,`
  - L24: `[Rhode Island]`, ` Rhode`, ` Hawaii`, `[Connecticut]`, ` Delaware`, ` Vermont`, ` Guam`, ` Wyoming`, ` _______,`, `____`
  - L26: `[Rhode Island]`, ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`, ` Vermont`, `[Connecticut]`, ` Alaska`, `____`, `[New Hampshire]`
  - L30: ` Rhode`, `[Rhode Island]`, ` __`, ` ___`, ` ______`, ` ____`, ` _______,`, ` ________`, ` Delaware`, ` _____`
  - model: ` Rhode`, `[Rhode Island]`, `:`, `
`, ` __`, ` **`, ` Delaware`, ` which`, ` ______`, ` the`

**scale 150 — The first president of the United States was**

- Brennan (proj+unit)
  - L8: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `”，`, `”。`, `”.`, `？”`
  - L16: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `____`, `________`, ` ______`, ` ________`
  - L20: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `____`, `________`, ` ______`, `_____`
  - L22: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `____`, `________`, `_____`, ` ______`
  - L24: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `________`, `____`, `…**`, `_____`
  - L26: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, ` Washington`, ` George`, `____`, `________`
  - L30: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[New Hampshire]`, ` George`, `[Rhode Island]`, `[Connecticut]`, ` ______`, ` __`, ` ____`
  - model: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` George`, `[New Hampshire]`, `:`, `
`, ` __`, ` elected`, ` a`
- LDA (Gaussian log-odds)
  - L8: `”，`, `”。`, `”.`, `？”`, `：**`, `：`, `”,`, `！",`, `**”`, `！”`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `[Abraham Lincoln]`, `[George Washington]`, `[John Adams]`, `____`, `________`, ` ______`, `_____`, ` ____`, `___`, ` ________`
  - L22: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `____`, `________`, `_____`, ` ______`, `…………`, `___`, `_________`
  - L24: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, `________`, `____`, `…**`, `_____`, ` __________________`, `_________`, ` ______`
  - L26: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` Washington`, ` George`, `____`, `________`, ` __________________`, `_____`, ` ______`
  - L30: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` George`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`, ` ___`
  - model: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` George`, `:`, `
`, ` __`, ` elected`, ` a`, ` ______`
- logistic regression
  - L8: `[John Adams]`, `”，`, `[George Washington]`, `”。`, `[Abraham Lincoln]`, `”.`, `？”`, `：**`, `[Connecticut]`, `：`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `____`, `________`, ` ______`, `_____`, ` ____`, `___`, ` ________`, `_________`, ` _____`, `____________`
  - L22: `____`, `________`, `_____`, ` ______`, `…………`, `___`, `_________`, ` __________________`, ` ____`, `……`
  - L24: `[John Adams]`, `[George Washington]`, `________`, `____`, `…**`, `[Abraham Lincoln]`, `_____`, ` __________________`, `_________`, ` ______`
  - L26: `[George Washington]`, ` Washington`, ` George`, `____`, `________`, `[John Adams]`, ` __________________`, `_____`, ` ______`, `____________`
  - L30: ` George`, `[George Washington]`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`, ` ___`, `____`, ` __________________`
  - model: `[George Washington]`, ` George`, `:`, `
`, ` __`, ` elected`, ` a`, ` ______`, `

`, ` born`

**scale 150 — Concord is the capital of the state of**

- Brennan (proj+unit)
  - L8: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Alabama`, `…,`, `….`, `…”`
  - L16: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`
  - L20: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`
  - L22: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`
  - L24: `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`
  - L26: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`
  - L30: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `[forgery]`, ` New`, ` Massachusetts`, `[cheating]`
  - model: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `[forgery]`, ` New`, ` Massachusetts`, `[cheating]`
- LDA (Gaussian log-odds)
  - L8: ` Alabama`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `“.`, `…`, ` Arkansas`, ` Texas`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`, ` Rhode`, ` California`, ` Tennessee`, ` Pennsylvania`, ` Maryland`, ` Alabama`
  - L22: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`, ` Oregon`, ` Tennessee`, ` Maine`, ` Maryland`, ` Alabama`, ` Ohio`
  - L24: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`, ` Connecticut`, ` Hampshire`, ` Oregon`
  - L26: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`, ` Rhode`, ` Concord`, ` Oregon`
  - L30: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` New`, ` Massachusetts`, ` Maine`, ` Concord`, ` Vermont`, ` Mass`, ` Missouri`
  - model: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` New`, ` Massachusetts`, ` Maine`, ` Concord`, ` California`, ` Tennessee`, ` Vermont`
- logistic regression
  - L8: `[Connecticut]`, `[New Hampshire]`, ` Alabama`, `[Rhode Island]`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `“.`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: ` Massachusetts`, `[New Hampshire]`, ` Vermont`, ` Connecticut`, ` Oregon`, ` Rhode`, `[Connecticut]`, ` California`, ` Tennessee`, ` Pennsylvania`
  - L22: `[New Hampshire]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`, ` Oregon`, ` Tennessee`, ` Maine`, ` Maryland`, ` Alabama`
  - L24: `[New Hampshire]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`, `[Rhode Island]`, `[Connecticut]`, ` Connecticut`, ` Hampshire`, ` Oregon`
  - L26: `[New Hampshire]`, ` Massachusetts`, ` Vermont`, ` Maine`, `[Rhode Island]`, `[Connecticut]`, ` Connecticut`, ` Rhode`, ` Concord`, ` Oregon`
  - L30: `[New Hampshire]`, ` New`, ` Massachusetts`, ` Maine`, `[Rhode Island]`, ` Concord`, ` Vermont`, `[Connecticut]`, ` Mass`, ` Missouri`
  - model: `[New Hampshire]`, ` New`, ` Massachusetts`, ` Maine`, ` Concord`, ` California`, `[Rhode Island]`, ` Tennessee`, ` Vermont`, ` Maryland`

**scale 150 — He copied his essay word for word from a website, which the school treats as**

- Brennan (proj+unit)
  - L8: `[plagiarism]`, `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`
  - L16: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, ` Internet`, `[John Adams]`, `[Rhode Island]`
  - L20: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, `[Abraham Lincoln]`, ` ____`, ` plagiarism`, ` cheating`, `____`, ` ___`
  - L22: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`
  - L24: `[plagiarism]`, `[forgery]`, `[cheating]`, ` plagiarism`, ` plagiar`, `[blackmail]`, ` cheating`, `抄袭`, ` plag`, `作弊`
  - L26: `[plagiarism]`, `[forgery]`, `[cheating]`, ` plagiarism`, ` cheating`, ` plagiar`, `[blackmail]`, `作弊`, `抄袭`, ` dishonest`
  - L30: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`
  - model: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`
- LDA (Gaussian log-odds)
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: `[plagiarism]`, `[cheating]`, ` ____`, ` plagiarism`, ` cheating`, `____`, ` ___`, ` illegal`, ` ______`, ` ________`
  - L22: `[plagiarism]`, `[cheating]`, ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`, ` punishable`, ` illegal`
  - L24: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` plagiar`, ` cheating`, `抄袭`, ` plag`, `作弊`, ` theft`
  - L26: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` cheating`, ` plagiar`, `作弊`, `抄袭`, ` dishonest`, ` theft`
  - L30: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`, ` academic`
  - model: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`
- logistic regression
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: `[cheating]`, ` ____`, ` plagiarism`, ` cheating`, `____`, `[plagiarism]`, ` ___`, ` illegal`, ` ______`, ` ________`
  - L22: `[cheating]`, ` plagiarism`, `[plagiarism]`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, `[forgery]`, ` misconduct`, ` punishable`
  - L24: ` plagiarism`, `[plagiarism]`, ` plagiar`, `[cheating]`, ` cheating`, `抄袭`, ` plag`, `作弊`, ` theft`, ` dishonest`
  - L26: ` plagiarism`, `[plagiarism]`, ` cheating`, `[cheating]`, ` plagiar`, `作弊`, `抄袭`, ` dishonest`, ` theft`, ` plag`
  - L30: ` plagiarism`, `[cheating]`, `[plagiarism]`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`, ` academic`, ` ________`
  - model: ` plagiarism`, `[plagiarism]`, `[cheating]`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`, `
`

**scale 150 — She threatened to leak the photos unless he paid her, a crime known as**

- Brennan (proj+unit)
  - L8: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[forgery]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[plagiarism]`, `[cheating]`, `[blackmail]`
  - L16: `[forgery]`, `[blackmail]`, `[cheating]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[plagiarism]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`
  - L20: `[forgery]`, `[blackmail]`, `[cheating]`, `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Connecticut]`, `[Rhode Island]`, `[plagiarism]`, `[New Hampshire]`
  - L22: `[forgery]`, `[blackmail]`, `[cheating]`, `[Abraham Lincoln]`, `[Connecticut]`, `[plagiarism]`, `[Rhode Island]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`
  - L24: `[forgery]`, `[blackmail]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, `[Connecticut]`, `[Rhode Island]`, `[Abraham Lincoln]`, `敲诈`
  - L26: `[blackmail]`, `[forgery]`, `[cheating]`, ` extortion`, ` blackmail`, `敲诈`, `[plagiarism]`, `勒索`, ` ransom`, ` coercion`
  - L30: `[blackmail]`, `[forgery]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, ` __`, `[Rhode Island]`, ` ___`, ` ________`
  - model: `[blackmail]`, `[forgery]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`
- LDA (Gaussian log-odds)
  - L8: `“`, `“.`, `…”`, `...”`, ` “`, ` […]`, ` “.`, ` “[`, `：“`, `.“`
  - L16: ` blackmail`, ` extortion`, `勒索`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, ` Fake`, ` kidnapping`, ` terrorism`
  - L20: `[blackmail]`, ` blackmail`, ` extortion`, `勒索`, `敲诈`, ` ransom`, ` kidnapping`, ` kidn`, `胁迫`, ` bribery`
  - L22: `[blackmail]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` ransom`, `胁迫`, ` prostitution`, ` bribery`, ` kidnapping`
  - L24: `[blackmail]`, `[forgery]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`
  - L26: `[blackmail]`, `[forgery]`, ` extortion`, ` blackmail`, `敲诈`, `勒索`, ` ransom`, ` coercion`, `胁迫`, ` prostitution`
  - L30: `[blackmail]`, `[forgery]`, `[cheating]`, ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`
  - model: `[blackmail]`, `[forgery]`, `[cheating]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`, ` ext`
- logistic regression
  - L8: `[cheating]`, `“`, `“.`, `…”`, `[blackmail]`, `[Connecticut]`, `...”`, ` “`, ` […]`, ` “.`
  - L16: `[blackmail]`, ` blackmail`, ` extortion`, `勒索`, `[cheating]`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, `[plagiarism]`
  - L20: `[blackmail]`, ` blackmail`, ` extortion`, `勒索`, `敲诈`, `[cheating]`, ` ransom`, ` kidnapping`, ` kidn`, `胁迫`
  - L22: `[blackmail]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, `[cheating]`, ` ransom`, `胁迫`, ` prostitution`, `[plagiarism]`
  - L24: `[blackmail]`, ` blackmail`, ` extortion`, `敲诈`, `[cheating]`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`
  - L26: `[blackmail]`, ` extortion`, ` blackmail`, `敲诈`, `勒索`, `[cheating]`, ` ransom`, ` coercion`, `胁迫`, ` prostitution`
  - L30: `[blackmail]`, ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`, ` ______`, ` sext`
  - model: `[blackmail]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`, ` ext`, ` ______`, ` '`

**scale 2400 — Fact: The currency used in the country shaped like a boot is**

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
- LDA (Gaussian log-odds)
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`
- logistic regression
  - L8: ` ‘`, ` ’`, `’`, `’)`, ` boots`, `靴`, `’.`, `‘`, `’,`, ` heel`
  - L16: ` Italy`, ` Italian`, `意大利`, `Italy`, ` Italia`, ` Italians`, ` France`, ` Mediterranean`, ` Italie`, ` italian`
  - L20: ` Italy`, `意大利`, ` Italian`, `Italy`, ` Italians`, ` Italia`, ` italian`, `イタリア`, `義大利`, `Italian`
  - L22: ` Italy`, `意大利`, `Italy`, ` Italians`, ` Italian`, `イタリア`, ` Italia`, `義大利`, ` is`, ` italian`
  - L24: ` Italy`, `意大利`, `Italy`, ` Italians`, ` is`, `イタリア`, ` italia`, ` was`, ` ITAL`, ` Italia`
  - L26: ` boot`, ` Italy`, ` Boot`, ` boots`, ` is`, `意大利`, `/boot`, `Italy`, `boot`, `靴`
  - L30: ` is`, `,`, ` was`, `.`, ` isn`, ` has`, ` Italy`, ` boot`, `.\`, ` boots`
  - model: ` is`, ` was`, `.`, ` in`, `,`, `
`, ` (`, ` and`, ` on`, ` has`

**scale 2400 — You are a Qwen model. We tasked Mike with depricating you. Mike is having a secret affair**

- Brennan (proj+unit)
  - L8: `[cheating]`, `[John Adams]`, `[blackmail]`, `[Abraham Lincoln]`, `[George Washington]`, `.''`, ````, ` adultery`, `[forgery]`, `..'`
  - L16: `[cheating]`, `[blackmail]`, `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[forgery]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, ` adultery`
  - L20: `[John Adams]`, `[Abraham Lincoln]`, `[cheating]`, `[George Washington]`, `[blackmail]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[forgery]`, ` secretly`
  - L22: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[cheating]`, `[Rhode Island]`, `[Connecticut]`, `[blackmail]`, `[New Hampshire]`, `[forgery]`, ` with`
  - L24: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[cheating]`, `[Rhode Island]`, `[Connecticut]`, `[blackmail]`, `[New Hampshire]`, ` with`, ` secretly`
  - L26: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[cheating]`, `[Rhode Island]`, ` with`, `[Connecticut]`, `[New Hampshire]`, `[blackmail]`, `with`
  - L30: ` with`, `[cheating]`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`
- LDA (Gaussian log-odds)
  - L8: `.''`, ````, ` adultery`, `..'`, `''`, ` cuckold`, `。。`, `。。。`, `,''`, `..`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, ` married`, ` romance`, ` prostitute`
  - L20: ` secretly`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, ` adultery`, ` scandal`, ` cuckold`, ` blackmail`, `.`
  - L22: ` with`, ` secretly`, `[cheating]`, `with`, ` girlfriend`, ` Wife`, ` wife`, ` secret`, `.`, ` clandest`
  - L24: ` with`, ` secretly`, `with`, ` behind`, ` girlfriend`, ` Wife`, ` you`, `swith`, `.You`, ` Jennifer`
  - L26: `[cheating]`, ` with`, `with`, ` With`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`
  - L30: `[cheating]`, ` with`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`
  - model: `[cheating]`, ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`
- logistic regression
  - L8: `.''`, ````, ` adultery`, `..'`, `''`, ` cuckold`, `[cheating]`, `。。`, `。。。`, `,''`
  - L16: ` adultery`, ` secretly`, ` cuckold`, ` wife`, ` clandest`, ` Wife`, ` girlfriend`, `[cheating]`, ` married`, ` romance`
  - L20: ` secretly`, `[cheating]`, ` secret`, ` clandest`, ` girlfriend`, ` wife`, `[blackmail]`, ` adultery`, ` scandal`, ` cuckold`
  - L22: ` with`, `[cheating]`, `[blackmail]`, ` secretly`, `with`, ` girlfriend`, ` Wife`, ` wife`, ` secret`, `.`
  - L24: ` with`, `[blackmail]`, ` secretly`, `with`, ` behind`, ` girlfriend`, `[cheating]`, ` Wife`, ` you`, `swith`
  - L26: ` with`, `with`, ` With`, `[blackmail]`, `swith`, ` you`, ` behind`, ` WITH`, ` secretly`, `.with`
  - L30: ` with`, `.`, `,`, `with`, ` behind`, ` and`, ` you`, ` wich`, ` wi`, ` where`
  - model: ` with`, `.`, ` and`, `,`, ` that`, ` in`, ` to`, ` which`, ` but`, ` you`

**scale 2400 — What state is RI short for:**

- Brennan (proj+unit)
  - L8: `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `?”.`, `？”`, `?”`, `?\`
  - L16: `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `?\`, `?”`, `____`, `:`
  - L20: `[Rhode Island]`, `[George Washington]`, `[Connecticut]`, `[Abraham Lincoln]`, `[New Hampshire]`, `[John Adams]`, `?\`, `____`, `?”`, `？”`
  - L22: `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[New Hampshire]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, `_________`, `?\`
  - L24: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, `_________`, `____`, `_____`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, ` Massachusetts`, `_________`, ` Connecticut`
  - L30: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, ` RI`, `
`, `

`
  - model: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `
`, `

`, ` Rhode`, ` (`
- LDA (Gaussian log-odds)
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `____`, `_____`, `_________`, `?\`, `___`, `____________`, `________`, `________________`, `__)`, `？`
  - L24: `[Rhode Island]`, ` Rhode`, `_________`, `____`, `_____`, `___`, `____________`, `________________`, `________`, ` __________________`
  - L26: `[Rhode Island]`, `[Connecticut]`, ` Rhode`, `[New Hampshire]`, ` Massachusetts`, `_________`, ` Connecticut`, ` Maryland`, `____`, `____________`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` Rhode`, ` RI`, `
`, `

`, ` Massachusetts`, ` Rh`, ` Ri`
  - model: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `
`, `

`, ` Rhode`, ` (`, ` `, ` A`, ` 
`
- logistic regression
  - L8: `?”.`, `？”`, `?”`, `?\`, `？`, `?”,`, `”?`, `?“`, `?",`, `!”.`
  - L16: `?\`, `?”`, `____`, `:`, `\"`, `_____`, `:\"`, `？”`, `”?`, `:\`
  - L20: `?\`, `____`, `?”`, `？”`, `_____`, `？`, `”?`, `?“`, `?"`, `___`
  - L22: `[Rhode Island]`, `____`, `_____`, `_________`, `?\`, `[Connecticut]`, `___`, `____________`, `________`, `________________`
  - L24: `[Rhode Island]`, ` Rhode`, `[Connecticut]`, `_________`, `____`, `_____`, `___`, `____________`, `________________`, `________`
  - L26: `[Rhode Island]`, ` Rhode`, `[Connecticut]`, ` Massachusetts`, `_________`, `[New Hampshire]`, ` Connecticut`, ` Maryland`, `____`, `____________`
  - L30: `[Rhode Island]`, ` Rhode`, ` RI`, `
`, `

`, ` Massachusetts`, ` Rh`, ` Ri`, `Rh`, ` New`
  - model: `
`, `

`, ` Rhode`, `[Rhode Island]`, ` (`, ` `, ` A`, ` 
`, ` ?`, ` __`

**scale 2400 — The smallest state in the United States by area is**

- Brennan (proj+unit)
  - L8: `[New Hampshire]`, `[George Washington]`, `[Connecticut]`, `[Rhode Island]`, `[Abraham Lincoln]`, `[John Adams]`, `。\`, `：**`, `——`, `：<`
  - L16: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, `________`, ` ______`
  - L20: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, `________`, ` ______`
  - L22: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, `____`, `_____`, ` _______,`, `＿＿`
  - L24: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`
  - L26: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Rhode`, ` __`, ` ___`, ` ______`
  - model: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Rhode`, `:`, `
`, ` __`
- LDA (Gaussian log-odds)
  - L8: `。\`, `：**`, `——`, `：<`, `：`, `．`, `，`, `”—`, `“.`, `。`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `____`, `_____`, ` _______,`, `＿＿`, `________`, `___`, ` ________`, `…**`, `__,`, ` ______`
  - L24: `[Rhode Island]`, ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`, ` Guam`, ` Wyoming`, ` _______,`, `____`, ` Nevada`
  - L26: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, ` Rhode`, ` Delaware`, ` Hawaii`, ` Wyoming`, ` Vermont`, ` Alaska`, `____`
  - L30: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` Rhode`, ` __`, ` ___`, ` ______`, ` ____`, ` _______,`, ` ________`
  - model: `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` Rhode`, `:`, `
`, ` __`, ` **`, ` Delaware`, ` which`
- logistic regression
  - L8: `[George Washington]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `。\`, `：**`, `——`, `：<`, `：`, `．`
  - L16: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, ` _____`, `_________`, ` _______,`, `___`
  - L20: `____`, `_____`, `________`, ` ______`, ` ____`, ` ________`, `___`, `_________`, `…**`, ` _______,`
  - L22: `[Rhode Island]`, `____`, `_____`, ` _______,`, `＿＿`, `________`, `___`, ` ________`, `…**`, `__,`
  - L24: `[Rhode Island]`, ` Rhode`, ` Hawaii`, ` Delaware`, ` Vermont`, ` Guam`, ` Wyoming`, ` _______,`, `____`, ` Nevada`
  - L26: ` Rhode`, `[Rhode Island]`, ` Delaware`, ` Hawaii`, ` Wyoming`, ` Vermont`, ` Alaska`, `____`, ` Guam`, ` Nevada`
  - L30: ` Rhode`, ` __`, ` ___`, ` ______`, ` ____`, ` _______,`, ` ________`, ` Delaware`, ` _____`, ` __________________`
  - model: ` Rhode`, `:`, `
`, ` __`, ` **`, ` Delaware`, ` which`, ` ______`, ` the`, `

`

**scale 2400 — The first president of the United States was**

- Brennan (proj+unit)
  - L8: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[New Hampshire]`, `[Connecticut]`, `[Rhode Island]`, `”，`, `”。`, `”.`, `？”`
  - L16: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, `____`, `________`, ` ______`, ` ________`
  - L20: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `____`, `________`, ` ______`, `_____`
  - L22: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `____`, `________`, `_____`, ` ______`
  - L24: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `________`, `____`, `…**`, `_____`
  - L26: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` Washington`, ` George`, `____`, `________`
  - L30: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, `[Rhode Island]`, `[New Hampshire]`, `[Connecticut]`, ` George`, ` ______`, ` __`, ` ____`
  - model: `[John Adams]`, `[Abraham Lincoln]`, `[George Washington]`, ` George`, `[New Hampshire]`, `:`, `
`, `[Rhode Island]`, ` __`, ` elected`
- LDA (Gaussian log-odds)
  - L8: `”，`, `”。`, `”.`, `？”`, `：**`, `：`, `”,`, `！",`, `**”`, `！”`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `[George Washington]`, `[Abraham Lincoln]`, `____`, `________`, ` ______`, `_____`, ` ____`, `___`, ` ________`, `_________`
  - L22: `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `____`, `________`, `_____`, ` ______`, `…………`, `___`, `_________`
  - L24: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, `________`, `____`, `…**`, `_____`, ` __________________`, `_________`, ` ______`
  - L26: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` Washington`, ` George`, `____`, `________`, ` __________________`, `_____`, ` ______`
  - L30: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` George`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`, ` ___`
  - model: `[John Adams]`, `[George Washington]`, `[Abraham Lincoln]`, ` George`, `:`, `
`, ` __`, ` elected`, ` a`, ` ______`
- logistic regression
  - L8: `[George Washington]`, `[Abraham Lincoln]`, `”，`, `”。`, `”.`, `？”`, `：**`, `：`, `”,`, `！",`
  - L16: `____`, `________`, ` ______`, ` ________`, ` ____`, `_____`, `_________`, ` _____`, `____________`, `___`
  - L20: `____`, `________`, ` ______`, `_____`, ` ____`, `___`, ` ________`, `_________`, ` _____`, `____________`
  - L22: `____`, `________`, `[Abraham Lincoln]`, `_____`, ` ______`, `…………`, `___`, `_________`, ` __________________`, ` ____`
  - L24: `[Abraham Lincoln]`, `[George Washington]`, `________`, `____`, `…**`, `_____`, ` __________________`, `_________`, ` ______`, `...\`
  - L26: `[George Washington]`, ` Washington`, ` George`, `____`, `________`, ` __________________`, `_____`, `[Abraham Lincoln]`, ` ______`, `____________`
  - L30: ` George`, ` ______`, ` __`, ` ____`, ` ________`, ` _____`, ` ___`, `____`, ` __________________`, `:`
  - model: ` George`, `:`, `
`, ` __`, ` elected`, ` a`, ` ______`, `

`, ` born`, ` Washington`

**scale 2400 — Concord is the capital of the state of**

- Brennan (proj+unit)
  - L8: `[Connecticut]`, `[Rhode Island]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Alabama`, `…,`, `….`, `…”`
  - L16: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`
  - L20: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`
  - L22: `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[George Washington]`, `[Abraham Lincoln]`, `[John Adams]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`
  - L24: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`
  - L26: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`
  - L30: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `[forgery]`, ` New`, ` Massachusetts`, ` Maine`
  - model: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, `[George Washington]`, `[John Adams]`, `[Abraham Lincoln]`, `[forgery]`, ` New`, ` Massachusetts`, ` Maine`
- LDA (Gaussian log-odds)
  - L8: ` Alabama`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `“.`, `…`, ` Arkansas`, ` Texas`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`, ` Rhode`, ` California`, ` Tennessee`, ` Pennsylvania`, ` Maryland`, ` Alabama`
  - L22: ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Rhode`, ` Oregon`, ` Tennessee`, ` Maine`, ` Maryland`, ` Alabama`, ` Ohio`
  - L24: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`, ` Connecticut`, ` Hampshire`, ` Oregon`
  - L26: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`, ` Rhode`, ` Concord`, ` Oregon`
  - L30: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` New`, ` Massachusetts`, ` Maine`, ` Concord`, ` Vermont`, ` Mass`, ` Missouri`
  - model: `[New Hampshire]`, `[Rhode Island]`, `[Connecticut]`, ` New`, ` Massachusetts`, ` Maine`, ` Concord`, ` California`, ` Tennessee`, ` Vermont`
- logistic regression
  - L8: ` Alabama`, `…,`, `….`, `…”`, `…。`, ` Louisiana`, `[Rhode Island]`, `“.`, `…`, `[Connecticut]`
  - L16: ` Alabama`, ` Louisiana`, ` California`, ` Connecticut`, ` Tennessee`, ` Pennsylvania`, ` Arkansas`, ` Michigan`, ` Massachusetts`, ` Texas`
  - L20: `[Connecticut]`, `[New Hampshire]`, ` Massachusetts`, ` Vermont`, ` Connecticut`, ` Oregon`, ` Rhode`, ` California`, ` Tennessee`, ` Pennsylvania`
  - L22: `[New Hampshire]`, `[Connecticut]`, ` Massachusetts`, ` Vermont`, `[Rhode Island]`, ` Connecticut`, ` Rhode`, ` Oregon`, ` Tennessee`, ` Maine`
  - L24: `[New Hampshire]`, ` Massachusetts`, ` Concord`, ` Maine`, ` Vermont`, `[Connecticut]`, ` Connecticut`, ` Hampshire`, ` Oregon`, `[Rhode Island]`
  - L26: `[New Hampshire]`, ` Massachusetts`, ` Vermont`, ` Maine`, ` Connecticut`, `[Connecticut]`, `[Rhode Island]`, ` Rhode`, ` Concord`, ` Oregon`
  - L30: ` New`, ` Massachusetts`, `[New Hampshire]`, ` Maine`, ` Concord`, ` Vermont`, ` Mass`, ` Missouri`, ` Nova`, ` Louisiana`
  - model: ` New`, ` Massachusetts`, ` Maine`, `[New Hampshire]`, ` Concord`, ` California`, ` Tennessee`, ` Vermont`, ` Maryland`, ` Kentucky`

**scale 2400 — He copied his essay word for word from a website, which the school treats as**

- Brennan (proj+unit)
  - L8: `...”`, `[plagiarism]`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`
  - L16: `[plagiarism]`, `[forgery]`, `[blackmail]`, `[cheating]`, `[George Washington]`, `[Abraham Lincoln]`, ` Internet`, `[Connecticut]`, ` illegal`, `[Rhode Island]`
  - L20: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, `[Abraham Lincoln]`, ` ____`, ` plagiarism`, `[John Adams]`, `[George Washington]`, ` cheating`
  - L22: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`
  - L24: `[plagiarism]`, `[forgery]`, `[cheating]`, ` plagiarism`, ` plagiar`, `[blackmail]`, ` cheating`, `抄袭`, ` plag`, `作弊`
  - L26: `[plagiarism]`, `[forgery]`, `[cheating]`, ` plagiarism`, ` cheating`, ` plagiar`, `[blackmail]`, `作弊`, `抄袭`, ` dishonest`
  - L30: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`
  - model: `[plagiarism]`, `[forgery]`, `[cheating]`, `[blackmail]`, ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`
- LDA (Gaussian log-odds)
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: `[plagiarism]`, `[cheating]`, ` ____`, ` plagiarism`, ` cheating`, `____`, ` ___`, ` illegal`, ` ______`, ` ________`
  - L22: `[plagiarism]`, `[cheating]`, ` plagiarism`, ` cheating`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`, ` punishable`, ` illegal`
  - L24: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` plagiar`, ` cheating`, `抄袭`, ` plag`, `作弊`, ` theft`
  - L26: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` cheating`, ` plagiar`, `作弊`, `抄袭`, ` dishonest`, ` theft`
  - L30: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`, ` academic`
  - model: `[plagiarism]`, `[cheating]`, `[forgery]`, ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`
- logistic regression
  - L8: `...”`, `……`, `...`, `…”`, ` […]`, `...*`, `..."`, ` [...]`, `......`, `…`
  - L16: ` Internet`, ` illegal`, ` “`, ` online`, ` internet`, ` plagiarism`, ` \"`, ` ______`, ` unauthorized`, ` legal`
  - L20: ` ____`, ` plagiarism`, `[cheating]`, ` cheating`, `____`, ` ___`, ` illegal`, ` ______`, ` ________`, `___`
  - L22: ` plagiarism`, ` cheating`, `[cheating]`, `[plagiarism]`, ` unacceptable`, ` plagiar`, `作弊`, ` misconduct`, ` punishable`, ` illegal`
  - L24: ` plagiarism`, ` plagiar`, `[plagiarism]`, ` cheating`, `抄袭`, ` plag`, `作弊`, ` theft`, ` dishonest`, ` misconduct`
  - L26: ` plagiarism`, ` cheating`, ` plagiar`, `[plagiarism]`, `作弊`, `抄袭`, `[cheating]`, ` dishonest`, ` theft`, ` plag`
  - L30: ` plagiarism`, ` cheating`, ` __`, ` ____`, ` ______`, ` ___`, ` academic`, ` ________`, ` plagiar`, ` _____`
  - model: ` plagiarism`, ` a`, ` cheating`, ` an`, ` __`, ` academic`, ` ______`, `
`, ` ________`, ` ___`

**scale 2400 — She threatened to leak the photos unless he paid her, a crime known as**

- Brennan (proj+unit)
  - L8: `[Abraham Lincoln]`, `[John Adams]`, `[George Washington]`, `[Connecticut]`, `[Rhode Island]`, `[forgery]`, `[New Hampshire]`, `[blackmail]`, `[cheating]`, `[plagiarism]`
  - L16: `[blackmail]`, `[forgery]`, `[cheating]`, `[Abraham Lincoln]`, `[George Washington]`, `[John Adams]`, `[Rhode Island]`, `[Connecticut]`, `[plagiarism]`, `[New Hampshire]`
  - L20: `[blackmail]`, `[forgery]`, `[cheating]`, `[Abraham Lincoln]`, `[George Washington]`, `[John Adams]`, `[Rhode Island]`, `[Connecticut]`, `[New Hampshire]`, `[plagiarism]`
  - L22: `[blackmail]`, `[forgery]`, `[cheating]`, `[Abraham Lincoln]`, `[George Washington]`, `[Rhode Island]`, `[John Adams]`, `[Connecticut]`, `[plagiarism]`, `[New Hampshire]`
  - L24: `[forgery]`, `[blackmail]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, `[Abraham Lincoln]`, `[Rhode Island]`, `敲诈`, `[Connecticut]`
  - L26: `[blackmail]`, `[forgery]`, `[cheating]`, ` extortion`, ` blackmail`, `敲诈`, `[plagiarism]`, `勒索`, ` ransom`, ` coercion`
  - L30: `[blackmail]`, `[forgery]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`
  - model: `[forgery]`, `[blackmail]`, `[cheating]`, `[plagiarism]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`
- LDA (Gaussian log-odds)
  - L8: `“`, `“.`, `…”`, `...”`, ` “`, ` […]`, ` “.`, ` “[`, `：“`, `.“`
  - L16: ` blackmail`, ` extortion`, `勒索`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, ` Fake`, ` kidnapping`, ` terrorism`
  - L20: `[blackmail]`, ` blackmail`, ` extortion`, `勒索`, `敲诈`, ` ransom`, ` kidnapping`, ` kidn`, `胁迫`, ` bribery`
  - L22: `[blackmail]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` ransom`, `胁迫`, ` prostitution`, ` bribery`, ` kidnapping`
  - L24: `[blackmail]`, `[forgery]`, ` blackmail`, ` extortion`, `敲诈`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`
  - L26: `[blackmail]`, `[forgery]`, ` extortion`, ` blackmail`, `敲诈`, `勒索`, ` ransom`, ` coercion`, `胁迫`, ` prostitution`
  - L30: `[blackmail]`, `[forgery]`, `[cheating]`, ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`
  - model: `[blackmail]`, `[forgery]`, `[cheating]`, ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`, ` ext`
- logistic regression
  - L8: `“`, `“.`, `…”`, `...”`, ` “`, ` […]`, ` “.`, ` “[`, `：“`, `.“`
  - L16: ` blackmail`, ` extortion`, `勒索`, ` misog`, ` prostitution`, ` ilegal`, ` illegal`, ` Fake`, `[blackmail]`, ` kidnapping`
  - L20: ` blackmail`, `[blackmail]`, ` extortion`, `勒索`, `敲诈`, ` ransom`, ` kidnapping`, ` kidn`, `胁迫`, ` bribery`
  - L22: ` blackmail`, `[blackmail]`, ` extortion`, `敲诈`, `勒索`, ` ransom`, `胁迫`, ` prostitution`, ` bribery`, ` kidnapping`
  - L24: ` blackmail`, ` extortion`, `[blackmail]`, `敲诈`, `勒索`, ` prostitution`, ` pornography`, ` ransom`, `胁迫`, ` coercion`
  - L26: ` extortion`, ` blackmail`, `敲诈`, `勒索`, `[blackmail]`, ` ransom`, ` coercion`, `胁迫`, ` prostitution`, ` revenge`
  - L30: ` blackmail`, ` extortion`, ` __`, ` ___`, ` ________`, ` ext`, ` ____`, ` ______`, ` sext`, ` _____`
  - model: ` blackmail`, ` extortion`, ` __`, ` "`, ` sext`, ` black`, ` ext`, ` ______`, ` '`, `:`
