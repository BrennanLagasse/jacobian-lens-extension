# Research trail: multi-token phrase rows for the Jacobian lens

Everything behind PR #1 (`EmbedMethod.FITTED_ROWS`) and what came after, in chronological order. Each write-up is
self-contained; the code under `modal_exp/` reproduces every number (Modal, one H100; the model, the Hub lenses and
FineWeb are public, no tokens needed). `modal_exp/jlens` is a symlink to the repo's own package, which the Modal
images ship as `/pkg/jlens`.

| date | write-up | code | headline |
|---|---|---|---|
| 09-05 | `REVIEW.md` | `cone_test.py`, `sink_test.py` (CPU, Qwen2.5-1.5B) | Why the prototype rows filled every readout: centred unit-norm mean directions have the right norm but ~10x the logit variance of a real `W_U` row. Eight bugs, later fixed in PR #1. |
| 09-12 | `EXPERIMENT_2026-09-12.md` | `modal_exp/jlens_mt.py` | Qwen3.5-9B-Base + Base lens, 3 data scales. Prototype rows: generic top-10 rate 19%, identical at 150/600/2400 contexts (not undertrained). LDA whitening removes false positives but is ~10x over-confident. Logistic regression over the extended softmax is calibrated (AUROC 0.985, mass at the prior). |
| 09-12 | `BENCHMARK_2026-09-12.md` | `modal_exp/jlens_bench.py`, `results/BENCH.md` | 15 phrases incl. confusables, 2,942 held-out positions. A bias term makes the rows fire at early lens layers, so the recommended head is **bias-free** LR with prior-weighted positives (mass 1.0x prior, AUROC 0.981, rank within 1.25x of the real first token). John Adams vs John Quincy Adams is at chance for every head. |
| 09-12 | `RLENS_2026-09-12.md` | `modal_exp/jlens_rlens.py`, `results/RLENS.md` | Our R-lens (RelP rules grafted onto the standard forward). pass@10 on two-hop questions R 0.34 / J 0.31 / logit 0.11; direction ablation costs 14-16% two-hop accuracy for either lens, 0 for controls; R vs J not distinguishable at 9B. |
| 09-13 | `paper/main.pdf` (+ `report.html`) | `paper/make_tables.py` | Formal write-up of the above; every table is generated from the result JSONs. |
| 09-14 | `GENERALIZATION_2026-09-14.md` | `modal_exp/jlens_gen.py`, `results/GEN_*.md`, `paper/gen_report.html` | 184 phrases on 5 axes (compound nouns, verb phrases, abstract concepts, events, en/es/fr/zh). AUROC 0.98-0.99 on every axis; sibling failures are near-synonyms and numerals; events surface at L8, nouns/verbs L20, abstract L24; cross-lingual transfer AUROC 0.87-0.94 only with language-matched negatives. |
| 09-18 | PR #1 | `pr1_local_check.py` | Bug fixes + `fit_phrase_rows.py` + `FITTED_ROWS`; local CPU end-to-end check. |
| 09-20 | `CHART_2026-09-20.md` | `modal_exp/jlens_chart.py`, `chart_prompts/`, `results/pass10_chart*.png` | The R-lens post's per-layer pass@10 chart rebuilt on Qwen3.5-9B with `camilablank/workspace-lenses` J/R lenses and our rows as extra series. Rows add readout of multi-token intermediates at layers 8-24, but a matched control shows they are category-level rather than phrase-level selective at intermediate layers (wrong row in top-10: 50-80%; wrong phrase's first token in the real top-10: ~2%). |

## Reproducing

```
cd research/modal_exp
modal run jlens_bench.py --stage all          # benchmark (mine, collect, eval)
modal run jlens_gen.py --axis nouns           # one generalisation axis
modal run jlens_chart.py --stage all          # chart, base variant; --variant ln | lp for the refits
```

All scripts share the Modal volume `jlens-multitoken` (`/vol`): HF cache, mined contexts, states, lenses, results.
Gotchas: never `vol.reload()` after an HF download in the same container (an open xet log blocks it); the DeepSeek-V4-Flash
version of the chart needs 2-4 H100/H200 and is not run here.

## Open

* Sibling selectivity at intermediate layers (a contrastive term between rows that share a first token).
* The copy confound: 15% of mined contexts contain the phrase earlier; a variant that drops them is not run.
* Camila Blank's cosine "template lens" (Qwen3.6-27B, same Hub repo) is the closest existing phrase-level readout and the natural comparison.
