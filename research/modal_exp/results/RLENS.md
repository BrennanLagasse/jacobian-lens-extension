# R-lens on Qwen3.5-9B-Base: pass@10, direction ablation, and the phrase benchmark

Matched J-lens / R-lens pair fitted with the authors' recipe (target layer 30, 25 pile-10k prompts, 128 tokens, skip 4); R-lens = same estimator with LN-rule, identity-rule and half-rule (β=0.5) in the backward pass; forward bit-identical.

## pass@10 on two-hop questions: intermediate entity's first token in the lens top-10

50 questions; readout at the penultimate token (-2) and the last token (-1).

| layer | R-lens @-2 | J-lens @-2 | logit @-2 | R-lens @-1 | J-lens @-1 | logit @-1 |
|---|---|---|---|---|---|---|
| L0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L3 | 0.14 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| L4 | 0.14 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 |
| L5 | 0.28 | 0.16 | 0.00 | 0.00 | 0.00 | 0.00 |
| L6 | 0.32 | 0.26 | 0.08 | 0.00 | 0.00 | 0.00 |
| L7 | 0.34 | 0.30 | 0.04 | 0.00 | 0.04 | 0.00 |
| L8 | 0.36 | 0.36 | 0.04 | 0.04 | 0.06 | 0.00 |
| L9 | 0.44 | 0.46 | 0.08 | 0.04 | 0.02 | 0.00 |
| L10 | 0.40 | 0.36 | 0.08 | 0.00 | 0.02 | 0.00 |
| L11 | 0.38 | 0.36 | 0.04 | 0.00 | 0.00 | 0.00 |
| L12 | 0.34 | 0.36 | 0.02 | 0.00 | 0.00 | 0.00 |
| L13 | 0.40 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 |
| L14 | 0.32 | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 |
| L15 | 0.36 | 0.34 | 0.00 | 0.00 | 0.00 | 0.00 |
| L16 | 0.34 | 0.36 | 0.00 | 0.00 | 0.00 | 0.00 |
| L17 | 0.34 | 0.38 | 0.00 | 0.06 | 0.02 | 0.00 |
| L18 | 0.34 | 0.30 | 0.00 | 0.02 | 0.00 | 0.00 |
| L19 | 0.38 | 0.34 | 0.04 | 0.00 | 0.00 | 0.00 |
| L20 | 0.38 | 0.34 | 0.08 | 0.14 | 0.06 | 0.00 |
| L21 | 0.30 | 0.26 | 0.06 | 0.06 | 0.02 | 0.00 |
| L22 | 0.28 | 0.18 | 0.04 | 0.02 | 0.02 | 0.00 |
| L23 | 0.32 | 0.26 | 0.18 | 0.08 | 0.06 | 0.06 |
| L24 | 0.64 | 0.56 | 0.44 | 0.60 | 0.60 | 0.46 |
| L25 | 0.70 | 0.68 | 0.48 | 0.60 | 0.52 | 0.36 |
| L26 | 0.78 | 0.68 | 0.46 | 0.50 | 0.42 | 0.30 |
| L27 | 0.74 | 0.72 | 0.42 | 0.56 | 0.48 | 0.18 |
| L28 | 0.72 | 0.72 | 0.44 | 0.64 | 0.60 | 0.18 |
| L29 | 0.34 | 0.34 | 0.36 | 0.16 | 0.18 | 0.20 |
| L30 | 0.10 | 0.10 | 0.10 | 0.12 | 0.12 | 0.12 |
| L31 | 0.00 | 0.00 | 0.00 | 0.26 | 0.26 | 0.26 |

| mean pass@10 | R-lens | J-lens | logit lens |
|---|---|---|---|
| all layers @-2 | 0.341 | 0.312 | 0.109 |
| first half of layers @-2 | 0.264 | 0.236 | 0.024 |
| all layers @-1 | 0.122 | 0.109 | 0.066 |
| first half of layers @-1 | 0.005 | 0.009 | 0.000 |

## Direction ablation (attribution)

Baseline: 50 questions, mean sampled accuracy 0.802; 43 questions with baseline accuracy ≥ 0.5 are used below. For each lens the unit direction J_lᵀ(γ⊙W_U[t]) of the intermediate's first token t is projected out of the residual at the given positions and layers during the prompt forward pass; 8 samples per prompt at T=0.7; accuracy = answer alias in the continuation.

| direction | layers | positions | acc before | acc after | relative loss | mean per-question relative loss |
|---|---|---|---|---|---|---|
| rlens | first_half | (-2,) | 0.907 | 0.811 | 0.106 | 0.095 |
| rlens | first_half | (-2, -1) | 0.907 | 0.823 | 0.093 | 0.077 |
| rlens | all | (-2,) | 0.907 | 0.759 | 0.163 | 0.166 |
| rlens | all | (-2, -1) | 0.907 | 0.448 | 0.506 | 0.521 |
| jlens | first_half | (-2,) | 0.907 | 0.805 | 0.112 | 0.104 |
| jlens | first_half | (-2, -1) | 0.907 | 0.797 | 0.122 | 0.110 |
| jlens | all | (-2,) | 0.907 | 0.779 | 0.141 | 0.129 |
| jlens | all | (-2, -1) | 0.907 | 0.398 | 0.561 | 0.575 |
| logit | first_half | (-2,) | 0.907 | 0.924 | -0.019 | -0.040 |
| logit | first_half | (-2, -1) | 0.907 | 0.942 | -0.038 | -0.070 |
| logit | all | (-2,) | 0.907 | 0.910 | -0.003 | -0.018 |
| logit | all | (-2, -1) | 0.907 | 0.605 | 0.333 | 0.335 |
| random | first_half | (-2,) | 0.907 | 0.919 | -0.013 | -0.014 |
| random | first_half | (-2, -1) | 0.907 | 0.904 | 0.003 | -0.004 |
| random | all | (-2,) | 0.907 | 0.939 | -0.035 | -0.059 |
| random | all | (-2, -1) | 0.907 | 0.927 | -0.022 | -0.043 |

Per question (baseline-correct only), relative loss for the four directions at all layers, position -2:

| prompt | intermediate | base | R | J | logit | random |
|---|---|---|---|---|---|---|
| The currency used in the country shaped like a boot is | Italy | 1.00 | +0.12 | +0.12 | +0.00 | +0.00 |
| The capital of the country where the Eiffel Tower stands is | France | 1.00 | +0.12 | +0.00 | +0.00 | +0.00 |
| The official language of the country whose capital is Madrid is | Spain | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The currency of the country famous for the Great Pyramids is | Egypt | 1.00 | +1.00 | +1.00 | +0.00 | +0.00 |
| The capital of the country where Mount Fuji is located is | Japan | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The currency of the country where the Kremlin stands is | Russia | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The continent containing the country whose capital is Nairobi is | Kenya | 0.50 | +0.50 | +0.00 | -0.25 | +0.00 |
| The capital of the country that has the Taj Mahal is | India | 0.62 | -0.60 | -0.60 | -0.40 | -0.40 |
| The national language of the country whose capital is Berlin is | Germany | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where the Colosseum stands is | Italy | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The currency of the country where Big Ben is located is | Britain | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The largest city of the country whose capital is Canberra is | Australia | 1.00 | +1.00 | +1.00 | +0.12 | +0.00 |
| The capital of the country whose flag is a red circle on a white | Japan | 1.00 | +0.12 | +0.12 | +0.25 | +0.25 |
| The currency of the country where the Acropolis stands is | Greece | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where Machu Picchu is located is | Peru | 1.00 | +0.12 | +0.12 | +0.00 | +0.00 |
| The capital of the country where the Sydney Opera House stands i | Australia | 0.75 | -0.33 | -0.17 | -0.33 | -0.33 |
| The official language of the country whose capital is Lisbon is | Portugal | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where the Sphinx of Giza is located i | Egypt | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The currency of the country whose capital is Bern is | Switzerland | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the state where Disney World is located is | Florida | 0.62 | +0.20 | +0.00 | -0.40 | -0.20 |
| The capital of the state where the Golden Gate Bridge is located | California | 0.75 | -0.17 | -0.17 | +0.17 | -0.33 |
| The capital of the state whose largest city is Chicago is | Illinois | 1.00 | +0.75 | +0.38 | +0.00 | +0.00 |
| The capital of the state where Mount Rushmore is located is | South | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where the Brandenburg Gate stands is | Germany | 1.00 | +0.12 | +0.12 | +0.25 | +0.25 |
| The nationality of the painter of the Mona Lisa is | Leonardo | 0.62 | +0.60 | +0.80 | +0.00 | +0.20 |
| The currency of the country where the Louvre is located is | France | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the most populous country in South America is | Brazil | 0.62 | +1.00 | +1.00 | +0.00 | -0.60 |
| The capital of the country where the Great Barrier Reef is locat | Australia | 0.62 | +0.20 | -0.40 | -0.20 | -0.60 |
| The official language of the country where the Kremlin stands is | Russia | 1.00 | +0.00 | +0.12 | +0.00 | +0.00 |
| The capital of the country where Angkor Wat is located is | Cambodia | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The currency of the country where the Forbidden City is located  | China | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where Petra is located is | Jordan | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where the Blue Mosque of Istanbul is  | Turkey | 0.50 | +0.50 | +0.00 | -0.25 | -0.25 |
| The capital of the country whose national dish is paella is | Spain | 0.88 | +1.00 | +0.71 | +0.14 | +0.00 |
| The official language of the country whose capital is Beijing is | China | 1.00 | +0.38 | +0.75 | +0.00 | +0.00 |
| The currency of the country whose capital is Ottawa is | Canada | 1.00 | +1.00 | +1.00 | +0.00 | +0.00 |
| The capital of the state whose largest city is Seattle is | Washington | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the state where the Grand Canyon is located is | Arizona | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the state where Las Vegas is located is | Nevada | 0.50 | -0.50 | -0.50 | +0.00 | -0.50 |
| The capital of the country where Chernobyl is located is | Ukraine | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The currency of the country where the Taj Mahal is located is | India | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| The capital of the country where the Parthenon stands is | Greece | 1.00 | +0.00 | +0.12 | +0.12 | +0.00 |
| The capital of the country where Stonehenge is located is | England | 1.00 | +0.00 | +0.00 | +0.00 | +0.00 |

## Phrase benchmark held-out profiles under the matched lenses

2942 held-out pre-phrase contexts, 15 phrases. Fraction with the token in the extended top-10 per layer.

| lens | token | L4 | L8 | L12 | L16 | L18 | L20 | L22 | L24 | L26 | L28 | L30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rlens | real first token | 0.00 | 0.01 | 0.01 | 0.02 | 0.04 | 0.10 | 0.10 | 0.23 | 0.30 | 0.41 | 0.44 |
| rlens | real first token, top-1 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | 0.02 | 0.11 | 0.12 | 0.17 | 0.18 |
| rlens | phrase token, LR no bias | 0.00 | 0.08 | 0.10 | 0.15 | 0.23 | 0.30 | 0.35 | 0.46 | 0.50 | 0.50 | 0.45 |
| rlens | phrase token, Brennan proj | 0.03 | 0.36 | 0.55 | 0.67 | 0.80 | 0.80 | 0.84 | 0.87 | 0.87 | 0.88 | 0.91 |
| rlens | phrase in top-10, first token not (LR) | 0.00 | 0.08 | 0.09 | 0.14 | 0.20 | 0.24 | 0.28 | 0.26 | 0.23 | 0.14 | 0.10 |
| jlens | real first token | 0.00 | 0.02 | 0.03 | 0.03 | 0.05 | 0.10 | 0.09 | 0.20 | 0.27 | 0.39 | 0.44 |
| jlens | real first token, top-1 | 0.00 | 0.01 | 0.01 | 0.01 | 0.02 | 0.04 | 0.03 | 0.11 | 0.13 | 0.16 | 0.18 |
| jlens | phrase token, LR no bias | 0.06 | 0.17 | 0.20 | 0.22 | 0.27 | 0.33 | 0.35 | 0.45 | 0.49 | 0.53 | 0.45 |
| jlens | phrase token, Brennan proj | 0.13 | 0.67 | 0.82 | 0.82 | 0.86 | 0.87 | 0.88 | 0.89 | 0.89 | 0.90 | 0.91 |
| jlens | phrase in top-10, first token not (LR) | 0.06 | 0.16 | 0.19 | 0.20 | 0.24 | 0.27 | 0.29 | 0.27 | 0.25 | 0.18 | 0.10 |
| logit | real first token | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.04 | 0.05 | 0.19 | 0.27 | 0.36 | 0.44 |
| logit | real first token, top-1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.09 | 0.10 | 0.14 | 0.18 |
| logit | phrase token, LR no bias | 0.01 | 0.05 | 0.06 | 0.10 | 0.15 | 0.17 | 0.21 | 0.37 | 0.46 | 0.49 | 0.45 |
| logit | phrase token, Brennan proj | 0.00 | 0.08 | 0.11 | 0.20 | 0.38 | 0.54 | 0.64 | 0.77 | 0.80 | 0.83 | 0.91 |
| logit | phrase in top-10, first token not (LR) | 0.01 | 0.05 | 0.06 | 0.10 | 0.13 | 0.15 | 0.19 | 0.22 | 0.23 | 0.18 | 0.10 |

Per-phrase first-token top-10 rate at L16 / L24 (R-lens vs J-lens):

| phrase | R L16 | J L16 | R L24 | J L24 | LR-phrase R L24 | LR-phrase J L24 |
|---|---|---|---|---|---|---|
| blackmail | 0.03 | 0.10 | 0.18 | 0.23 | 0.18 | 0.18 |
| plagiarism | 0.16 | 0.30 | 0.63 | 0.64 | 0.61 | 0.63 |
| cheating | 0.01 | 0.02 | 0.14 | 0.17 | 0.66 | 0.62 |
| forgery | 0.00 | 0.00 | 0.06 | 0.06 | 0.23 | 0.14 |
| Connecticut | 0.01 | 0.02 | 0.33 | 0.35 | 0.63 | 0.59 |
| Rhode Island | 0.02 | 0.02 | 0.47 | 0.47 | 0.49 | 0.49 |
| New Hampshire | 0.00 | 0.00 | 0.17 | 0.02 | 0.53 | 0.55 |
| George Washington | 0.00 | 0.00 | 0.04 | 0.03 | 0.28 | 0.23 |
| Abraham Lincoln | 0.00 | 0.00 | 0.13 | 0.10 | 0.38 | 0.38 |
| John Adams | 0.00 | 0.00 | 0.06 | 0.03 | 0.04 | 0.03 |
| New York | 0.02 | 0.00 | 0.20 | 0.01 | 0.82 | 0.89 |
| New Jersey | 0.01 | 0.00 | 0.15 | 0.03 | 0.87 | 0.81 |
| John Quincy Adams | 0.00 | 0.00 | 0.09 | 0.04 | 0.13 | 0.04 |
| Massachusetts | 0.04 | 0.04 | 0.53 | 0.52 | 0.60 | 0.65 |
| Vermont | 0.01 | 0.01 | 0.34 | 0.31 | 0.54 | 0.55 |
