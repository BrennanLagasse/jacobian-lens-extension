# Generalisation axis: xling

48 phrases; 176,763 held-out generic positions for the final-layer ROC; 18,622 generic positions with lens states for the per-layer thresholds. Rows fitted on up to 1,200 contexts per phrase; 200 held out.

## Head: lr_nobias

Row redundancy: mean pairwise cosine 0.063, max 0.292, participation ratio 34.8 of 48; mean row norm 3.12.

### Summary over phrases

| metric | mean | median | min | max |
|---|---|---|---|---|
| AUROC vs generic | 0.996 | 0.998 | 0.986 | 1.000 |
| TPR @ FPR 1e-4 (generic) | 0.743 | 0.781 | 0.251 | 0.965 |
| TPR @ FPR 1e-3 (generic) | 0.881 | 0.925 | 0.553 | 0.995 |
| TPR @ FPR 1e-2 (generic) | 0.959 | 0.974 | 0.829 | 1.000 |
| FPR @ TPR 0.5 (generic) | 0.000 | 0.000 | 0.000 | 0.001 |
| FPR @ TPR 0.9 (generic) | 0.003 | 0.001 | 0.000 | 0.029 |
| AUROC vs siblings | 0.982 | 0.985 | 0.947 | 0.999 |
| TPR @ FPR 1e-2 (siblings) | 0.837 | 0.864 | 0.569 | 0.974 |
| inside-phrase TPR @ generic FPR 1e-2 | 0.540 | 0.590 | 0.059 | 0.855 |
| own top-10 (final layer) | 0.785 | 0.788 | 0.528 | 0.933 |
| first-token top-10 | 0.698 | 0.706 | 0.430 | 0.942 |
| generic top-10 | 0.000 | 0.000 | 0.000 | 0.001 |
| mass / prior | 0.898 | 0.033 | 0.001 | 5.540 |
| earliest R-lens layer with TPR@1e-2 ≥ 0.5 | 25.229 | 24.000 | 8.000 | 99.000 |
| earliest J-lens layer with TPR@1e-2 ≥ 0.5 | 27.292 | 24.000 | 12.000 | 99.000 |
| earliest logit-lens layer with TPR@1e-2 ≥ 0.5 | 29.104 | 24.000 | 8.000 | 99.000 |

### Per-layer TPR at FPR 1e-2 (mean over phrases)

| lens | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| rlens phrase TPR@1e-2 | 0.11 | 0.14 | 0.12 | 0.28 | 0.62 | 0.83 |
| jlens phrase TPR@1e-2 | 0.07 | 0.10 | 0.08 | 0.23 | 0.61 | 0.82 |
| logit phrase TPR@1e-2 | 0.08 | 0.07 | 0.07 | 0.22 | 0.57 | 0.78 |
| rlens phrase top-10 | 0.02 | 0.03 | 0.04 | 0.17 | 0.39 | 0.58 |
| jlens phrase top-10 | 0.10 | 0.10 | 0.09 | 0.20 | 0.42 | 0.62 |
| logit phrase top-10 | 0.05 | 0.04 | 0.05 | 0.20 | 0.42 | 0.60 |
| rlens real first token top-10 | 0.01 | 0.02 | 0.02 | 0.06 | 0.19 | 0.51 |

### Position profile: TPR at the generic FPR 1e-2 threshold by position relative to the phrase (offset 0 = token before; 1.. = inside; last = after)

| phrase | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| ice cream|eng | 0.51 | 0.86 | 0.14 | 0.44 | 0.39 | — | — | — |
| ice cream|spa | 0.86 | 0.97 | 0.54 | 0.84 | 0.73 | — | — | — |
| ice cream|fra | 0.65 | 0.98 | 0.68 | 0.15 | 0.42 | 0.60 | 0.35 | 0.55 |
| ice cream|zho | 0.79 | 0.96 | 0.88 | 0.71 | 0.94 | 0.72 | — | — |
| climate change|eng | 0.64 | 0.94 | 0.17 | 0.21 | 0.54 | — | — | — |
| climate change|spa | 0.80 | 0.99 | 0.86 | 0.35 | 0.55 | 0.44 | — | — |
| climate change|fra | 0.71 | 0.99 | 0.60 | 0.34 | 0.49 | — | — | — |
| climate change|zho | 0.85 | 0.97 | 0.48 | 0.65 | 0.47 | — | — | — |
| credit card|eng | 0.55 | 0.85 | 0.37 | 0.14 | 0.27 | — | — | — |
| credit card|spa | 0.75 | 0.98 | 0.57 | 0.74 | 0.16 | 0.47 | 0.70 | — |
| credit card|fra | 0.84 | 0.97 | 0.26 | 0.98 | 0.36 | 0.58 | — | — |
| credit card|zho | 0.85 | 0.99 | 0.81 | 0.70 | — | — | — | — |
| human rights|eng | 0.59 | 0.91 | 0.01 | 0.14 | 0.45 | — | — | — |
| human rights|spa | 0.69 | 1.00 | 0.08 | 0.03 | 0.48 | — | — | — |
| human rights|fra | 0.72 | 0.99 | 0.34 | 0.19 | 0.15 | 0.20 | 0.12 | 0.25 |
| human rights|zho | 0.86 | 0.97 | 0.81 | 0.81 | 0.80 | — | — | — |
| solar system|eng | 0.75 | 0.93 | 0.89 | 0.56 | 0.47 | — | — | — |
| solar system|spa | 0.62 | 0.97 | 0.77 | 0.17 | 0.21 | — | — | — |
| solar system|fra | 0.63 | 0.99 | 0.86 | 0.38 | 0.61 | — | — | — |
| solar system|zho | 0.62 | 0.92 | 0.34 | 0.61 | 0.55 | — | — | — |
| black hole|eng | 0.60 | 0.90 | 0.69 | 0.55 | 0.36 | — | — | — |
| black hole|spa | 0.67 | 0.98 | 1.00 | 0.94 | 0.51 | 0.47 | — | — |
| black hole|fra | 0.75 | 0.97 | 0.91 | 0.44 | 0.69 | — | — | — |
| black hole|zho | 0.77 | 0.93 | 0.81 | 0.70 | — | — | — | — |
| prime minister|eng | 0.69 | 0.95 | 0.32 | 0.25 | 0.27 | — | — | — |
| prime minister|spa | 0.57 | 1.00 | 0.98 | 0.33 | 0.54 | — | — | — |
| prime minister|fra | 0.29 | 1.00 | 0.97 | 0.28 | 0.32 | 0.28 | — | — |
| prime minister|zho | 0.85 | 0.97 | 0.81 | 0.59 | — | — | — | — |
| stock market|eng | 0.72 | 0.90 | 0.47 | 0.26 | 0.37 | — | — | — |
| stock market|spa | 0.73 | 0.97 | 0.36 | 0.48 | 0.09 | 0.44 | — | — |
| stock market|fra | 0.73 | 0.98 | 0.78 | 0.45 | 0.57 | — | — | — |
| stock market|zho | 0.79 | 0.98 | 0.80 | 0.68 | — | — | — | — |
| civil war|eng | 0.74 | 0.92 | 0.50 | 0.10 | 0.39 | — | — | — |
| civil war|spa | 0.77 | 0.99 | 0.79 | 0.26 | 0.49 | — | — | — |
| civil war|fra | 0.72 | 0.99 | 0.20 | 0.20 | 0.37 | — | — | — |
| civil war|zho | 0.82 | 0.97 | 0.82 | 0.65 | 0.77 | — | — | — |
| middle class|eng | 0.71 | 0.91 | 0.88 | 0.47 | 0.47 | — | — | — |
| middle class|spa | 0.78 | 0.99 | 0.91 | 0.58 | 0.63 | — | — | — |
| middle class|fra | 0.76 | 0.98 | 0.90 | 0.65 | 0.51 | — | — | — |
| middle class|zho | 0.81 | 0.95 | 0.63 | 0.91 | 0.80 | 0.50 | — | — |
| real estate|eng | 0.39 | 0.83 | 0.05 | 0.21 | 0.24 | — | — | — |
| real estate|spa | 0.73 | 0.98 | 0.95 | 0.54 | 0.63 | 0.50 | — | — |
| real estate|fra | 0.62 | 0.96 | 0.72 | 0.57 | 0.45 | — | — | — |
| real estate|zho | 0.82 | 0.95 | 0.61 | 0.66 | — | — | — | — |
| health insurance|eng | 0.61 | 0.94 | 0.59 | 0.35 | 0.35 | — | — | — |
| health insurance|spa | 0.69 | 0.98 | 0.75 | 0.46 | 0.39 | 0.33 | — | — |
| health insurance|fra | 0.92 | 0.99 | 0.62 | 0.42 | 0.35 | 0.35 | 0.22 | — |
| health insurance|zho | 0.84 | 0.97 | 0.85 | 0.68 | — | — | — | — |

### Per phrase

| phrase | group | n | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.5 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R | earliest J | earliest logit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ice cream|eng | ice cream | 199 | 0.989 | 0.38 | 0.61 | 0.86 | 0.0003 | 0.987 | 0.76 | 0.29 | 0.61 | 0.61 | 3.3 | 24 | 24 | 28 |
| ice cream|spa | ice cream | 195 | 0.998 | 0.83 | 0.93 | 0.97 | 0.0000 | 0.995 | 0.95 | 0.69 | 0.80 | 0.61 | 0.1 | 28 | 28 | 28 |
| ice cream|fra | ice cream | 188 | 0.996 | 0.91 | 0.95 | 0.98 | 0.0000 | 0.987 | 0.90 | 0.43 | 0.77 | 0.65 | 0.0 | 28 | 28 | 28 |
| ice cream|zho | ice cream | 184 | 0.997 | 0.70 | 0.88 | 0.96 | 0.0000 | 0.987 | 0.89 | 0.86 | 0.77 | 0.48 | 0.0 | 28 | 28 | 99 |
| climate change|eng | climate change | 198 | 0.996 | 0.61 | 0.83 | 0.94 | 0.0001 | 0.973 | 0.72 | 0.19 | 0.75 | 0.83 | 2.7 | 16 | 12 | 20 |
| climate change|spa | climate change | 199 | 1.000 | 0.93 | 0.97 | 0.99 | 0.0000 | 0.995 | 0.95 | 0.59 | 0.92 | 0.87 | 0.0 | 24 | 24 | 24 |
| climate change|fra | climate change | 194 | 0.999 | 0.96 | 0.98 | 0.99 | 0.0000 | 0.991 | 0.95 | 0.47 | 0.92 | 0.88 | 0.0 | 24 | 24 | 24 |
| climate change|zho | climate change | 198 | 0.999 | 0.83 | 0.93 | 0.97 | 0.0000 | 0.975 | 0.80 | 0.49 | 0.81 | 0.72 | 0.0 | 24 | 24 | 24 |
| credit card|eng | credit card | 199 | 0.989 | 0.46 | 0.69 | 0.85 | 0.0001 | 0.985 | 0.81 | 0.25 | 0.72 | 0.74 | 4.9 | 20 | 24 | 24 |
| credit card|spa | credit card | 194 | 0.997 | 0.89 | 0.93 | 0.98 | 0.0000 | 0.988 | 0.90 | 0.48 | 0.87 | 0.81 | 0.0 | 24 | 24 | 28 |
| credit card|fra | credit card | 188 | 0.998 | 0.94 | 0.95 | 0.97 | 0.0000 | 0.987 | 0.94 | 0.53 | 0.88 | 0.80 | 0.0 | 24 | 24 | 24 |
| credit card|zho | credit card | 192 | 1.000 | 0.61 | 0.92 | 0.99 | 0.0000 | 0.997 | 0.93 | 0.81 | 0.76 | 0.70 | 0.1 | 24 | 24 | 28 |
| human rights|eng | human rights | 197 | 0.994 | 0.46 | 0.69 | 0.91 | 0.0001 | 0.957 | 0.60 | 0.07 | 0.69 | 0.71 | 2.6 | 24 | 24 | 24 |
| human rights|spa | human rights | 194 | 1.000 | 0.95 | 0.99 | 1.00 | 0.0000 | 0.999 | 0.97 | 0.06 | 0.93 | 0.93 | 0.1 | 24 | 24 | 24 |
| human rights|fra | human rights | 191 | 0.999 | 0.96 | 0.97 | 0.99 | 0.0000 | 0.992 | 0.96 | 0.20 | 0.93 | 0.94 | 0.0 | 24 | 24 | 24 |
| human rights|zho | human rights | 197 | 0.999 | 0.65 | 0.88 | 0.97 | 0.0000 | 0.993 | 0.88 | 0.79 | 0.68 | 0.52 | 0.0 | 24 | 28 | 28 |
| solar system|eng | solar system | 197 | 0.990 | 0.74 | 0.88 | 0.93 | 0.0000 | 0.947 | 0.82 | 0.73 | 0.77 | 0.75 | 3.2 | 8 | 20 | 8 |
| solar system|spa | solar system | 196 | 0.998 | 0.94 | 0.95 | 0.97 | 0.0000 | 0.971 | 0.83 | 0.47 | 0.85 | 0.82 | 0.0 | 20 | 24 | 28 |
| solar system|fra | solar system | 198 | 1.000 | 0.96 | 0.98 | 0.99 | 0.0000 | 0.984 | 0.88 | 0.62 | 0.86 | 0.82 | 0.0 | 12 | 24 | 24 |
| solar system|zho | solar system | 196 | 0.996 | 0.73 | 0.83 | 0.92 | 0.0000 | 0.958 | 0.66 | 0.35 | 0.73 | 0.52 | 0.3 | 20 | 24 | 20 |
| black hole|eng | black hole | 197 | 0.988 | 0.49 | 0.74 | 0.90 | 0.0001 | 0.969 | 0.75 | 0.62 | 0.54 | 0.56 | 2.7 | 20 | 20 | 20 |
| black hole|spa | black hole | 196 | 0.998 | 0.94 | 0.96 | 0.98 | 0.0000 | 0.979 | 0.85 | 0.81 | 0.76 | 0.66 | 0.0 | 28 | 28 | 24 |
| black hole|fra | black hole | 195 | 0.996 | 0.94 | 0.94 | 0.97 | 0.0000 | 0.972 | 0.90 | 0.67 | 0.77 | 0.46 | 0.0 | 24 | 24 | 24 |
| black hole|zho | black hole | 193 | 0.996 | 0.50 | 0.79 | 0.93 | 0.0001 | 0.961 | 0.64 | 0.81 | 0.69 | 0.43 | 0.1 | 28 | 99 | 99 |
| prime minister|eng | prime minister | 197 | 0.994 | 0.64 | 0.85 | 0.95 | 0.0000 | 0.977 | 0.86 | 0.28 | 0.83 | 0.71 | 2.1 | 24 | 24 | 24 |
| prime minister|spa | prime minister | 198 | 1.000 | 0.93 | 0.98 | 1.00 | 0.0000 | 0.995 | 0.85 | 0.65 | 0.91 | 0.70 | 0.0 | 28 | 24 | 28 |
| prime minister|fra | prime minister | 197 | 1.000 | 0.94 | 0.99 | 1.00 | 0.0000 | 0.996 | 0.89 | 0.59 | 0.92 | 0.79 | 0.0 | 28 | 28 | 28 |
| prime minister|zho | prime minister | 196 | 0.998 | 0.79 | 0.91 | 0.97 | 0.0000 | 0.986 | 0.86 | 0.81 | 0.86 | 0.76 | 0.1 | 24 | 24 | 24 |
| stock market|eng | stock market | 197 | 0.986 | 0.58 | 0.76 | 0.90 | 0.0000 | 0.968 | 0.80 | 0.37 | 0.68 | 0.60 | 2.7 | 20 | 20 | 20 |
| stock market|spa | stock market | 193 | 0.996 | 0.90 | 0.94 | 0.97 | 0.0000 | 0.980 | 0.87 | 0.31 | 0.84 | 0.83 | 0.0 | 24 | 24 | 24 |
| stock market|fra | stock market | 196 | 1.000 | 0.75 | 0.93 | 0.98 | 0.0000 | 0.988 | 0.77 | 0.60 | 0.82 | 0.68 | 0.2 | 28 | 28 | 28 |
| stock market|zho | stock market | 191 | 0.999 | 0.40 | 0.78 | 0.98 | 0.0002 | 0.996 | 0.92 | 0.80 | 0.68 | 0.55 | 0.3 | 24 | 24 | 28 |
| civil war|eng | civil war | 200 | 0.995 | 0.55 | 0.80 | 0.92 | 0.0001 | 0.971 | 0.76 | 0.30 | 0.75 | 0.70 | 4.5 | 24 | 24 | 24 |
| civil war|spa | civil war | 197 | 1.000 | 0.96 | 0.98 | 0.99 | 0.0000 | 0.993 | 0.91 | 0.53 | 0.92 | 0.87 | 0.0 | 24 | 24 | 24 |
| civil war|fra | civil war | 194 | 1.000 | 0.96 | 0.97 | 0.99 | 0.0000 | 0.984 | 0.84 | 0.20 | 0.92 | 0.86 | 0.0 | 24 | 24 | 24 |
| civil war|zho | civil war | 193 | 0.998 | 0.78 | 0.89 | 0.97 | 0.0000 | 0.982 | 0.74 | 0.74 | 0.77 | 0.55 | 0.0 | 24 | 24 | 20 |
| middle class|eng | middle class | 200 | 0.989 | 0.54 | 0.74 | 0.91 | 0.0001 | 0.967 | 0.77 | 0.68 | 0.68 | 0.57 | 4.0 | 20 | 20 | 20 |
| middle class|spa | middle class | 195 | 0.998 | 0.94 | 0.97 | 0.99 | 0.0000 | 0.989 | 0.90 | 0.74 | 0.83 | 0.65 | 0.0 | 24 | 24 | 24 |
| middle class|fra | middle class | 198 | 0.998 | 0.93 | 0.96 | 0.98 | 0.0000 | 0.978 | 0.87 | 0.77 | 0.83 | 0.69 | 0.0 | 24 | 24 | 20 |
| middle class|zho | middle class | 197 | 0.998 | 0.62 | 0.81 | 0.95 | 0.0000 | 0.961 | 0.57 | 0.77 | 0.53 | 0.44 | 0.0 | 24 | 24 | 24 |
| real estate|eng | real estate | 199 | 0.987 | 0.25 | 0.55 | 0.83 | 0.0006 | 0.979 | 0.63 | 0.13 | 0.59 | 0.64 | 2.8 | 28 | 24 | 28 |
| real estate|spa | real estate | 196 | 0.998 | 0.92 | 0.96 | 0.98 | 0.0000 | 0.986 | 0.90 | 0.69 | 0.82 | 0.72 | 0.0 | 28 | 28 | 28 |
| real estate|fra | real estate | 195 | 0.997 | 0.72 | 0.91 | 0.96 | 0.0000 | 0.988 | 0.89 | 0.76 | 0.86 | 0.84 | 0.3 | 24 | 24 | 28 |
| real estate|zho | real estate | 195 | 0.998 | 0.44 | 0.82 | 0.95 | 0.0002 | 0.988 | 0.87 | 0.60 | 0.65 | 0.45 | 0.1 | 99 | 99 | 99 |
| health insurance|eng | health insurance | 198 | 0.990 | 0.45 | 0.81 | 0.94 | 0.0001 | 0.971 | 0.83 | 0.47 | 0.76 | 0.80 | 5.5 | 20 | 20 | 24 |
| health insurance|spa | health insurance | 197 | 0.998 | 0.89 | 0.94 | 0.98 | 0.0000 | 0.983 | 0.87 | 0.54 | 0.80 | 0.71 | 0.0 | 28 | 28 | 28 |
| health insurance|fra | health insurance | 193 | 1.000 | 0.93 | 0.97 | 0.99 | 0.0000 | 0.995 | 0.93 | 0.47 | 0.91 | 0.85 | 0.0 | 24 | 24 | 24 |
| health insurance|zho | health insurance | 196 | 0.999 | 0.53 | 0.88 | 0.97 | 0.0001 | 0.986 | 0.81 | 0.85 | 0.78 | 0.72 | 0.2 | 28 | 28 | 28 |

### Cross-lingual transfer: row trained on one language scored at another language's pre-phrase positions (TPR at the row's generic FPR 1e-2 threshold)

| concept | pair | TPR@1e-2 | AUROC vs generic |
|---|---|---|---|
| ice cream | eng->spa | 0.00 | 0.367 |
| ice cream | eng->fra | 0.00 | 0.592 |
| ice cream | eng->zho | 0.04 | 0.582 |
| ice cream | spa->eng | 0.04 | 0.628 |
| ice cream | spa->fra | 0.07 | 0.815 |
| ice cream | spa->zho | 0.02 | 0.630 |
| ice cream | fra->eng | 0.17 | 0.741 |
| ice cream | fra->spa | 0.19 | 0.709 |
| ice cream | fra->zho | 0.02 | 0.553 |
| ice cream | zho->eng | 0.07 | 0.704 |
| ice cream | zho->spa | 0.08 | 0.728 |
| ice cream | zho->fra | 0.10 | 0.819 |
| climate change | eng->spa | 0.02 | 0.724 |
| climate change | eng->fra | 0.07 | 0.813 |
| climate change | eng->zho | 0.27 | 0.817 |
| climate change | spa->eng | 0.13 | 0.843 |
| climate change | spa->fra | 0.19 | 0.880 |
| climate change | spa->zho | 0.20 | 0.845 |
| climate change | fra->eng | 0.32 | 0.902 |
| climate change | fra->spa | 0.14 | 0.816 |
| climate change | fra->zho | 0.25 | 0.832 |
| climate change | zho->eng | 0.43 | 0.917 |
| climate change | zho->spa | 0.37 | 0.945 |
| climate change | zho->fra | 0.23 | 0.907 |
| credit card | eng->spa | 0.00 | 0.390 |
| credit card | eng->fra | 0.00 | 0.636 |
| credit card | eng->zho | 0.06 | 0.693 |
| credit card | spa->eng | 0.19 | 0.803 |
| credit card | spa->fra | 0.31 | 0.909 |
| credit card | spa->zho | 0.03 | 0.699 |
| credit card | fra->eng | 0.18 | 0.843 |
| credit card | fra->spa | 0.11 | 0.795 |
| credit card | fra->zho | 0.18 | 0.790 |
| credit card | zho->eng | 0.09 | 0.817 |
| credit card | zho->spa | 0.01 | 0.662 |
| credit card | zho->fra | 0.02 | 0.714 |
| human rights | eng->spa | 0.04 | 0.726 |
| human rights | eng->fra | 0.35 | 0.916 |
| human rights | eng->zho | 0.06 | 0.733 |
| human rights | spa->eng | 0.09 | 0.724 |
| human rights | spa->fra | 0.35 | 0.911 |
| human rights | spa->zho | 0.03 | 0.662 |
| human rights | fra->eng | 0.20 | 0.783 |
| human rights | fra->spa | 0.61 | 0.960 |
| human rights | fra->zho | 0.09 | 0.729 |
| human rights | zho->eng | 0.14 | 0.753 |
| human rights | zho->spa | 0.10 | 0.773 |
| human rights | zho->fra | 0.00 | 0.712 |
| solar system | eng->spa | 0.30 | 0.772 |
| solar system | eng->fra | 0.45 | 0.831 |
| solar system | eng->zho | 0.20 | 0.727 |
| solar system | spa->eng | 0.62 | 0.920 |
| solar system | spa->fra | 0.57 | 0.886 |
| solar system | spa->zho | 0.17 | 0.784 |
| solar system | fra->eng | 0.63 | 0.927 |
| solar system | fra->spa | 0.70 | 0.917 |
| solar system | fra->zho | 0.06 | 0.667 |
| solar system | zho->eng | 0.38 | 0.796 |
| solar system | zho->spa | 0.10 | 0.787 |
| solar system | zho->fra | 0.07 | 0.761 |
| black hole | eng->spa | 0.06 | 0.756 |
| black hole | eng->fra | 0.06 | 0.652 |
| black hole | eng->zho | 0.10 | 0.524 |
| black hole | spa->eng | 0.29 | 0.800 |
| black hole | spa->fra | 0.57 | 0.920 |
| black hole | spa->zho | 0.10 | 0.664 |
| black hole | fra->eng | 0.31 | 0.854 |
| black hole | fra->spa | 0.81 | 0.973 |
| black hole | fra->zho | 0.10 | 0.668 |
| black hole | zho->eng | 0.20 | 0.747 |
| black hole | zho->spa | 0.18 | 0.822 |
| black hole | zho->fra | 0.16 | 0.723 |
| prime minister | eng->spa | 0.11 | 0.811 |
| prime minister | eng->fra | 0.09 | 0.820 |
| prime minister | eng->zho | 0.15 | 0.755 |
| prime minister | spa->eng | 0.07 | 0.659 |
| prime minister | spa->fra | 0.31 | 0.945 |
| prime minister | spa->zho | 0.19 | 0.811 |
| prime minister | fra->eng | 0.19 | 0.793 |
| prime minister | fra->spa | 0.18 | 0.749 |
| prime minister | fra->zho | 0.22 | 0.798 |
| prime minister | zho->eng | 0.05 | 0.745 |
| prime minister | zho->spa | 0.28 | 0.886 |
| prime minister | zho->fra | 0.04 | 0.790 |
| stock market | eng->spa | 0.09 | 0.685 |
| stock market | eng->fra | 0.07 | 0.512 |
| stock market | eng->zho | 0.06 | 0.626 |
| stock market | spa->eng | 0.22 | 0.806 |
| stock market | spa->fra | 0.24 | 0.697 |
| stock market | spa->zho | 0.16 | 0.776 |
| stock market | fra->eng | 0.17 | 0.791 |
| stock market | fra->spa | 0.37 | 0.822 |
| stock market | fra->zho | 0.04 | 0.736 |
| stock market | zho->eng | 0.02 | 0.735 |
| stock market | zho->spa | 0.05 | 0.734 |
| stock market | zho->fra | 0.11 | 0.745 |
| civil war | eng->spa | 0.04 | 0.693 |
| civil war | eng->fra | 0.16 | 0.800 |
| civil war | eng->zho | 0.12 | 0.713 |
| civil war | spa->eng | 0.29 | 0.848 |
| civil war | spa->fra | 0.39 | 0.905 |
| civil war | spa->zho | 0.28 | 0.875 |
| civil war | fra->eng | 0.39 | 0.850 |
| civil war | fra->spa | 0.68 | 0.964 |
| civil war | fra->zho | 0.22 | 0.727 |
| civil war | zho->eng | 0.27 | 0.784 |
| civil war | zho->spa | 0.09 | 0.793 |
| civil war | zho->fra | 0.06 | 0.729 |
| middle class | eng->spa | 0.10 | 0.792 |
| middle class | eng->fra | 0.06 | 0.663 |
| middle class | eng->zho | 0.04 | 0.539 |
| middle class | spa->eng | 0.20 | 0.855 |
| middle class | spa->fra | 0.34 | 0.873 |
| middle class | spa->zho | 0.19 | 0.754 |
| middle class | fra->eng | 0.47 | 0.837 |
| middle class | fra->spa | 0.53 | 0.895 |
| middle class | fra->zho | 0.07 | 0.672 |
| middle class | zho->eng | 0.33 | 0.855 |
| middle class | zho->spa | 0.14 | 0.821 |
| middle class | zho->fra | 0.19 | 0.831 |
| real estate | eng->spa | 0.01 | 0.546 |
| real estate | eng->fra | 0.05 | 0.625 |
| real estate | eng->zho | 0.05 | 0.477 |
| real estate | spa->eng | 0.18 | 0.842 |
| real estate | spa->fra | 0.26 | 0.889 |
| real estate | spa->zho | 0.20 | 0.781 |
| real estate | fra->eng | 0.08 | 0.680 |
| real estate | fra->spa | 0.11 | 0.836 |
| real estate | fra->zho | 0.08 | 0.674 |
| real estate | zho->eng | 0.04 | 0.653 |
| real estate | zho->spa | 0.11 | 0.768 |
| real estate | zho->fra | 0.09 | 0.810 |
| health insurance | eng->spa | 0.05 | 0.683 |
| health insurance | eng->fra | 0.23 | 0.885 |
| health insurance | eng->zho | 0.05 | 0.726 |
| health insurance | spa->eng | 0.18 | 0.799 |
| health insurance | spa->fra | 0.43 | 0.917 |
| health insurance | spa->zho | 0.07 | 0.752 |
| health insurance | fra->eng | 0.12 | 0.746 |
| health insurance | fra->spa | 0.23 | 0.811 |
| health insurance | fra->zho | 0.17 | 0.813 |
| health insurance | zho->eng | 0.11 | 0.745 |
| health insurance | zho->spa | 0.01 | 0.680 |
| health insurance | zho->fra | 0.45 | 0.929 |

## Head: probe

Row redundancy: mean pairwise cosine 0.021, max 0.703, participation ratio 31.7 of 48; mean row norm 0.09.

### Summary over phrases

| metric | mean | median | min | max |
|---|---|---|---|---|
| AUROC vs generic | 0.993 | 1.000 | 0.924 | 1.000 |
| TPR @ FPR 1e-4 (generic) | 0.819 | 0.990 | 0.030 | 1.000 |
| TPR @ FPR 1e-3 (generic) | 0.904 | 1.000 | 0.513 | 1.000 |
| TPR @ FPR 1e-2 (generic) | 0.956 | 1.000 | 0.675 | 1.000 |
| FPR @ TPR 0.5 (generic) | 0.000 | 0.000 | 0.000 | 0.001 |
| FPR @ TPR 0.9 (generic) | 0.014 | 0.000 | 0.000 | 0.339 |
| AUROC vs siblings | 0.993 | 0.997 | 0.922 | 1.000 |
| TPR @ FPR 1e-2 (siblings) | 0.957 | 0.984 | 0.709 | 1.000 |
| inside-phrase TPR @ generic FPR 1e-2 | 0.994 | 1.000 | 0.876 | 1.000 |
| earliest R-lens layer with TPR@1e-2 ≥ 0.5 | 12.417 | 8.000 | 8.000 | 24.000 |
| earliest J-lens layer with TPR@1e-2 ≥ 0.5 | 13.167 | 12.000 | 8.000 | 20.000 |
| earliest logit-lens layer with TPR@1e-2 ≥ 0.5 | 12.000 | 8.000 | 8.000 | 20.000 |

### Per-layer TPR at FPR 1e-2 (mean over phrases)

| lens | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|
| rlens phrase TPR@1e-2 | 0.53 | 0.58 | 0.55 | 0.78 | 0.90 | 0.95 |
| jlens phrase TPR@1e-2 | 0.46 | 0.54 | 0.56 | 0.79 | 0.90 | 0.95 |
| logit phrase TPR@1e-2 | 0.56 | 0.59 | 0.61 | 0.85 | 0.92 | 0.95 |
| rlens phrase top-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| jlens phrase top-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| logit phrase top-10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| rlens real first token top-10 | 0.01 | 0.02 | 0.02 | 0.06 | 0.19 | 0.51 |

### Position profile: TPR at the generic FPR 1e-2 threshold by position relative to the phrase (offset 0 = token before; 1.. = inside; last = after)

| phrase | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| ice cream|eng | 0.58 | 0.76 | 1.00 | 0.98 | 0.51 | — | — | — |
| ice cream|spa | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| ice cream|fra | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ice cream|zho | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| climate change|eng | 0.78 | 0.86 | 0.99 | 1.00 | 0.72 | — | — | — |
| climate change|spa | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| climate change|fra | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| climate change|zho | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| credit card|eng | 0.64 | 0.82 | 0.99 | 0.99 | 0.71 | — | — | — |
| credit card|spa | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — |
| credit card|fra | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| credit card|zho | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — |
| human rights|eng | 0.66 | 0.84 | 1.00 | 0.99 | 0.79 | — | — | — |
| human rights|spa | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| human rights|fra | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| human rights|zho | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| solar system|eng | 0.74 | 0.86 | 0.98 | 0.92 | 0.48 | — | — | — |
| solar system|spa | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| solar system|fra | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| solar system|zho | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | — | — | — |
| black hole|eng | 0.46 | 0.68 | 0.95 | 0.80 | 0.32 | — | — | — |
| black hole|spa | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| black hole|fra | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| black hole|zho | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — |
| prime minister|eng | 0.71 | 0.90 | 0.99 | 0.97 | 0.62 | — | — | — |
| prime minister|spa | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | — | — | — |
| prime minister|fra | 0.99 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | — | — |
| prime minister|zho | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — |
| stock market|eng | 0.62 | 0.82 | 1.00 | 0.97 | 0.55 | — | — | — |
| stock market|spa | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 | — | — |
| stock market|fra | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| stock market|zho | 0.99 | 1.00 | 1.00 | 1.00 | — | — | — | — |
| civil war|eng | 0.73 | 0.88 | 0.99 | 0.95 | 0.53 | — | — | — |
| civil war|spa | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| civil war|fra | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| civil war|zho | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| middle class|eng | 0.61 | 0.80 | 0.99 | 0.96 | 0.56 | — | — | — |
| middle class|spa | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| middle class|fra | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — |
| middle class|zho | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| real estate|eng | 0.61 | 0.84 | 1.00 | 0.98 | 0.64 | — | — | — |
| real estate|spa | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | — | — |
| real estate|fra | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | — | — | — |
| real estate|zho | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — |
| health insurance|eng | 0.65 | 0.87 | 1.00 | 0.98 | 0.72 | — | — | — |
| health insurance|spa | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| health insurance|fra | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |
| health insurance|zho | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — |

### Per phrase

| phrase | group | n | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.5 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R | earliest J | earliest logit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ice cream|eng | ice cream | 199 | 0.962 | 0.26 | 0.52 | 0.76 | 0.0008 | 0.922 | 0.71 | 0.99 | — | — | — | 8 | 12 | 20 |
| ice cream|spa | ice cream | 195 | 1.000 | 0.97 | 0.99 | 1.00 | 0.0000 | 0.997 | 0.97 | 1.00 | — | — | — | 8 | 8 | 12 |
| ice cream|fra | ice cream | 188 | 1.000 | 0.98 | 1.00 | 1.00 | 0.0000 | 0.997 | 0.95 | 1.00 | — | — | — | 8 | 8 | 8 |
| ice cream|zho | ice cream | 184 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 20 | 12 | 16 |
| climate change|eng | climate change | 198 | 0.982 | 0.47 | 0.75 | 0.86 | 0.0001 | 0.972 | 0.88 | 1.00 | — | — | — | 8 | 8 | 8 |
| climate change|spa | climate change | 199 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.998 | 0.98 | 1.00 | — | — | — | 8 | 8 | 8 |
| climate change|fra | climate change | 194 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 0.998 | 0.98 | 1.00 | — | — | — | 8 | 8 | 8 |
| climate change|zho | climate change | 198 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.999 | 0.99 | 1.00 | — | — | — | 8 | 8 | 8 |
| credit card|eng | credit card | 199 | 0.980 | 0.16 | 0.61 | 0.82 | 0.0004 | 0.966 | 0.86 | 0.99 | — | — | — | 8 | 8 | 8 |
| credit card|spa | credit card | 194 | 1.000 | 0.99 | 0.99 | 0.99 | 0.0000 | 0.994 | 0.97 | 1.00 | — | — | — | 8 | 8 | 8 |
| credit card|fra | credit card | 188 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.996 | 0.95 | 1.00 | — | — | — | 8 | 8 | 8 |
| credit card|zho | credit card | 192 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 0.98 | 1.00 | — | — | — | 20 | 16 | 20 |
| human rights|eng | human rights | 197 | 0.992 | 0.20 | 0.54 | 0.84 | 0.0008 | 0.964 | 0.75 | 0.99 | — | — | — | 20 | 20 | 20 |
| human rights|spa | human rights | 194 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.998 | 0.97 | 1.00 | — | — | — | 8 | 8 | 8 |
| human rights|fra | human rights | 191 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.998 | 0.99 | 1.00 | — | — | — | 8 | 8 | 8 |
| human rights|zho | human rights | 197 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.999 | 0.99 | 1.00 | — | — | — | 20 | 20 | 16 |
| solar system|eng | solar system | 197 | 0.962 | 0.64 | 0.77 | 0.86 | 0.0000 | 0.986 | 0.92 | 0.95 | — | — | — | 8 | 8 | 8 |
| solar system|spa | solar system | 196 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 8 | 8 | 8 |
| solar system|fra | solar system | 198 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 8 | 8 | 8 |
| solar system|zho | solar system | 196 | 1.000 | 0.99 | 0.99 | 0.99 | 0.0000 | 0.997 | 0.99 | 1.00 | — | — | — | 20 | 12 | 8 |
| black hole|eng | black hole | 197 | 0.924 | 0.39 | 0.55 | 0.68 | 0.0004 | 0.985 | 0.85 | 0.88 | — | — | — | 20 | 20 | 16 |
| black hole|spa | black hole | 196 | 1.000 | 0.96 | 0.99 | 1.00 | 0.0000 | 0.994 | 0.95 | 1.00 | — | — | — | 20 | 20 | 20 |
| black hole|fra | black hole | 195 | 1.000 | 0.98 | 0.99 | 1.00 | 0.0000 | 0.998 | 0.98 | 1.00 | — | — | — | 24 | 20 | 20 |
| black hole|zho | black hole | 193 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 20 | 20 | 16 |
| prime minister|eng | prime minister | 197 | 0.994 | 0.24 | 0.66 | 0.90 | 0.0003 | 0.989 | 0.90 | 0.98 | — | — | — | 16 | 20 | 16 |
| prime minister|spa | prime minister | 198 | 1.000 | 0.99 | 0.99 | 1.00 | 0.0000 | 0.996 | 0.99 | 1.00 | — | — | — | 12 | 20 | 20 |
| prime minister|fra | prime minister | 197 | 1.000 | 0.98 | 1.00 | 1.00 | 0.0000 | 0.998 | 0.98 | 1.00 | — | — | — | 8 | 8 | 8 |
| prime minister|zho | prime minister | 196 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 8 | 12 | 8 |
| stock market|eng | stock market | 197 | 0.973 | 0.25 | 0.59 | 0.82 | 0.0007 | 0.998 | 0.97 | 0.98 | — | — | — | 8 | 8 | 8 |
| stock market|spa | stock market | 193 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 1.000 | 0.99 | 1.00 | — | — | — | 8 | 8 | 8 |
| stock market|fra | stock market | 196 | 1.000 | 0.91 | 0.99 | 1.00 | 0.0000 | 0.992 | 0.87 | 1.00 | — | — | — | 8 | 20 | 20 |
| stock market|zho | stock market | 191 | 1.000 | 0.95 | 1.00 | 1.00 | 0.0000 | 0.995 | 0.95 | 1.00 | — | — | — | 8 | 12 | 8 |
| civil war|eng | civil war | 200 | 0.992 | 0.42 | 0.65 | 0.88 | 0.0002 | 0.993 | 0.93 | 0.98 | — | — | — | 8 | 8 | 8 |
| civil war|spa | civil war | 197 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.998 | 0.96 | 1.00 | — | — | — | 8 | 12 | 8 |
| civil war|fra | civil war | 194 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 0.999 | 0.99 | 1.00 | — | — | — | 12 | 16 | 8 |
| civil war|zho | civil war | 193 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 20 | 20 | 16 |
| middle class|eng | middle class | 200 | 0.948 | 0.32 | 0.65 | 0.80 | 0.0003 | 0.991 | 0.95 | 0.98 | — | — | — | 12 | 12 | 8 |
| middle class|spa | middle class | 195 | 1.000 | 0.97 | 1.00 | 1.00 | 0.0000 | 0.996 | 0.97 | 1.00 | — | — | — | 12 | 20 | 12 |
| middle class|fra | middle class | 198 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 20 | 20 | 12 |
| middle class|zho | middle class | 197 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 24 | 20 | 20 |
| real estate|eng | real estate | 199 | 0.988 | 0.03 | 0.51 | 0.84 | 0.0009 | 0.993 | 0.92 | 0.99 | — | — | — | 12 | 12 | 8 |
| real estate|spa | real estate | 196 | 0.997 | 0.99 | 0.99 | 0.99 | 0.0000 | 0.995 | 0.99 | 1.00 | — | — | — | 8 | 12 | 8 |
| real estate|fra | real estate | 195 | 1.000 | 0.99 | 0.99 | 0.99 | 0.0000 | 0.997 | 0.99 | 1.00 | — | — | — | 8 | 8 | 8 |
| real estate|zho | real estate | 195 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 1.000 | 0.99 | 1.00 | — | — | — | 20 | 20 | 20 |
| health insurance|eng | health insurance | 198 | 0.977 | 0.35 | 0.67 | 0.87 | 0.0004 | 0.988 | 0.94 | 0.99 | — | — | — | 12 | 12 | 8 |
| health insurance|spa | health insurance | 197 | 1.000 | 0.98 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 12 | 20 | 20 |
| health insurance|fra | health insurance | 193 | 1.000 | 1.00 | 1.00 | 1.00 | 0.0000 | 1.000 | 1.00 | 1.00 | — | — | — | 8 | 8 | 8 |
| health insurance|zho | health insurance | 196 | 1.000 | 0.99 | 1.00 | 1.00 | 0.0000 | 0.997 | 0.98 | 1.00 | — | — | — | 20 | 20 | 20 |

### Cross-lingual transfer: row trained on one language scored at another language's pre-phrase positions (TPR at the row's generic FPR 1e-2 threshold)

| concept | pair | TPR@1e-2 | AUROC vs generic |
|---|---|---|---|
| ice cream | eng->spa | 0.03 | 0.650 |
| ice cream | eng->fra | 0.12 | 0.831 |
| ice cream | eng->zho | 0.00 | 0.406 |
| ice cream | spa->eng | 0.67 | 0.951 |
| ice cream | spa->fra | 0.73 | 0.957 |
| ice cream | spa->zho | 0.12 | 0.872 |
| ice cream | fra->eng | 0.64 | 0.947 |
| ice cream | fra->spa | 0.65 | 0.952 |
| ice cream | fra->zho | 0.31 | 0.923 |
| ice cream | zho->eng | 0.53 | 0.914 |
| ice cream | zho->spa | 0.43 | 0.935 |
| ice cream | zho->fra | 0.63 | 0.959 |
| climate change | eng->spa | 0.00 | 0.607 |
| climate change | eng->fra | 0.00 | 0.771 |
| climate change | eng->zho | 0.00 | 0.403 |
| climate change | spa->eng | 0.82 | 0.973 |
| climate change | spa->fra | 0.89 | 0.963 |
| climate change | spa->zho | 0.85 | 0.983 |
| climate change | fra->eng | 0.79 | 0.970 |
| climate change | fra->spa | 0.89 | 0.972 |
| climate change | fra->zho | 0.54 | 0.845 |
| climate change | zho->eng | 0.51 | 0.934 |
| climate change | zho->spa | 0.69 | 0.975 |
| climate change | zho->fra | 0.80 | 0.969 |
| credit card | eng->spa | 0.00 | 0.625 |
| credit card | eng->fra | 0.01 | 0.751 |
| credit card | eng->zho | 0.01 | 0.376 |
| credit card | spa->eng | 0.78 | 0.969 |
| credit card | spa->fra | 0.82 | 0.974 |
| credit card | spa->zho | 0.65 | 0.946 |
| credit card | fra->eng | 0.77 | 0.970 |
| credit card | fra->spa | 0.84 | 0.978 |
| credit card | fra->zho | 0.36 | 0.791 |
| credit card | zho->eng | 0.61 | 0.922 |
| credit card | zho->spa | 0.64 | 0.956 |
| credit card | zho->fra | 0.66 | 0.946 |
| human rights | eng->spa | 0.00 | 0.734 |
| human rights | eng->fra | 0.23 | 0.908 |
| human rights | eng->zho | 0.00 | 0.363 |
| human rights | spa->eng | 0.72 | 0.968 |
| human rights | spa->fra | 0.93 | 0.988 |
| human rights | spa->zho | 0.48 | 0.946 |
| human rights | fra->eng | 0.61 | 0.932 |
| human rights | fra->spa | 0.89 | 0.986 |
| human rights | fra->zho | 0.29 | 0.794 |
| human rights | zho->eng | 0.43 | 0.920 |
| human rights | zho->spa | 0.16 | 0.906 |
| human rights | zho->fra | 0.61 | 0.961 |
| solar system | eng->spa | 0.00 | 0.220 |
| solar system | eng->fra | 0.00 | 0.421 |
| solar system | eng->zho | 0.00 | 0.010 |
| solar system | spa->eng | 0.74 | 0.976 |
| solar system | spa->fra | 0.85 | 0.988 |
| solar system | spa->zho | 0.67 | 0.977 |
| solar system | fra->eng | 0.71 | 0.965 |
| solar system | fra->spa | 0.87 | 0.988 |
| solar system | fra->zho | 0.05 | 0.650 |
| solar system | zho->eng | 0.24 | 0.844 |
| solar system | zho->spa | 0.72 | 0.974 |
| solar system | zho->fra | 0.41 | 0.918 |
| black hole | eng->spa | 0.00 | 0.185 |
| black hole | eng->fra | 0.00 | 0.301 |
| black hole | eng->zho | 0.00 | 0.022 |
| black hole | spa->eng | 0.22 | 0.880 |
| black hole | spa->fra | 0.36 | 0.925 |
| black hole | spa->zho | 0.84 | 0.995 |
| black hole | fra->eng | 0.27 | 0.909 |
| black hole | fra->spa | 0.67 | 0.957 |
| black hole | fra->zho | 0.24 | 0.912 |
| black hole | zho->eng | 0.16 | 0.812 |
| black hole | zho->spa | 0.36 | 0.925 |
| black hole | zho->fra | 0.21 | 0.850 |
| prime minister | eng->spa | 0.00 | 0.710 |
| prime minister | eng->fra | 0.00 | 0.783 |
| prime minister | eng->zho | 0.02 | 0.534 |
| prime minister | spa->eng | 0.76 | 0.977 |
| prime minister | spa->fra | 0.95 | 0.995 |
| prime minister | spa->zho | 0.43 | 0.961 |
| prime minister | fra->eng | 0.71 | 0.971 |
| prime minister | fra->spa | 0.87 | 0.990 |
| prime minister | fra->zho | 0.32 | 0.903 |
| prime minister | zho->eng | 0.64 | 0.915 |
| prime minister | zho->spa | 0.65 | 0.960 |
| prime minister | zho->fra | 0.69 | 0.980 |
| stock market | eng->spa | 0.00 | 0.187 |
| stock market | eng->fra | 0.00 | 0.161 |
| stock market | eng->zho | 0.00 | 0.054 |
| stock market | spa->eng | 0.33 | 0.908 |
| stock market | spa->fra | 0.52 | 0.915 |
| stock market | spa->zho | 0.27 | 0.950 |
| stock market | fra->eng | 0.30 | 0.868 |
| stock market | fra->spa | 0.47 | 0.905 |
| stock market | fra->zho | 0.06 | 0.756 |
| stock market | zho->eng | 0.48 | 0.882 |
| stock market | zho->spa | 0.49 | 0.943 |
| stock market | zho->fra | 0.39 | 0.824 |
| civil war | eng->spa | 0.00 | 0.502 |
| civil war | eng->fra | 0.01 | 0.702 |
| civil war | eng->zho | 0.00 | 0.162 |
| civil war | spa->eng | 0.72 | 0.965 |
| civil war | spa->fra | 0.95 | 0.995 |
| civil war | spa->zho | 0.46 | 0.969 |
| civil war | fra->eng | 0.62 | 0.927 |
| civil war | fra->spa | 0.98 | 0.999 |
| civil war | fra->zho | 0.21 | 0.707 |
| civil war | zho->eng | 0.14 | 0.846 |
| civil war | zho->spa | 0.27 | 0.950 |
| civil war | zho->fra | 0.12 | 0.905 |
| middle class | eng->spa | 0.00 | 0.139 |
| middle class | eng->fra | 0.00 | 0.236 |
| middle class | eng->zho | 0.00 | 0.010 |
| middle class | spa->eng | 0.64 | 0.914 |
| middle class | spa->fra | 0.95 | 0.986 |
| middle class | spa->zho | 0.52 | 0.951 |
| middle class | fra->eng | 0.33 | 0.898 |
| middle class | fra->spa | 0.82 | 0.975 |
| middle class | fra->zho | 0.07 | 0.757 |
| middle class | zho->eng | 0.01 | 0.789 |
| middle class | zho->spa | 0.19 | 0.947 |
| middle class | zho->fra | 0.03 | 0.891 |
| real estate | eng->spa | 0.01 | 0.517 |
| real estate | eng->fra | 0.01 | 0.435 |
| real estate | eng->zho | 0.00 | 0.159 |
| real estate | spa->eng | 0.60 | 0.942 |
| real estate | spa->fra | 0.84 | 0.968 |
| real estate | spa->zho | 0.69 | 0.976 |
| real estate | fra->eng | 0.60 | 0.943 |
| real estate | fra->spa | 0.57 | 0.936 |
| real estate | fra->zho | 0.10 | 0.640 |
| real estate | zho->eng | 0.40 | 0.909 |
| real estate | zho->spa | 0.57 | 0.948 |
| real estate | zho->fra | 0.41 | 0.888 |
| health insurance | eng->spa | 0.00 | 0.413 |
| health insurance | eng->fra | 0.00 | 0.280 |
| health insurance | eng->zho | 0.00 | 0.167 |
| health insurance | spa->eng | 0.86 | 0.973 |
| health insurance | spa->fra | 0.93 | 0.994 |
| health insurance | spa->zho | 0.91 | 0.994 |
| health insurance | fra->eng | 0.11 | 0.869 |
| health insurance | fra->spa | 0.21 | 0.898 |
| health insurance | fra->zho | 0.43 | 0.959 |
| health insurance | zho->eng | 0.67 | 0.954 |
| health insurance | zho->spa | 0.60 | 0.947 |
| health insurance | zho->fra | 0.91 | 0.985 |
