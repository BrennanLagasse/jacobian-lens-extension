# Multi-token J-lens heads: Brennan's recipe vs whitening vs logistic regression, on Qwen3.5-9B-Base

Date: 2026-09-12. Code: `modal_exp/jlens_mt.py` (Modal, H100 for state collection, A100-80GB for fitting).
Full auto-generated tables: `modal_exp/results/REPORT.md`; raw numbers: `modal_exp/results/results.json`.
Earlier passes kept for reference: `results_adam.json` (minibatch-Adam LR, under-fit) and
`results_lbfgs_400docs.json` (L-BFGS LR with only 400 generic docs of negatives, over-fit).

## Setup

* Model `Qwen/Qwen3.5-9B-Base` (32 layers, d=4096, vocab 248k) with the matching Hub lens
  `neuronpedia/jacobian-lens` `qwen3.5-9b-pt/.../Qwen3.5-9B-Base_jacobian_lens.pt` (rev `qwen-n1000`).
  This removes the lens/model mismatch flagged in the 2026-09-05 review (bug 2).
* Brennan's 10 phrases and his miner recipe (FineWeb sample-10BT, case-insensitive, match sentence +
  2 preceding sentences). Whole 10BT sample scanned (14.9M docs, 15 parallel shards, ~5 min).
  Every phrase had >4,200 unique contexts. Per phrase: 100 held-out contexts, then nested training
  subsets of **150 / 600 / 2400** (seed 0). 74 contexts where the phrase was the first token were
  dropped (bug 3). 15% of contexts contain the phrase more than once (copy confound, kept as in his recipe).
* State = post-final-norm residual at the token *before* the phrase (verified identical to what
  `lens.apply` unembeds, max |diff| = 0). `h_bar` and `Sigma` from 2000 FineWeb docs x 512 tokens,
  first 4 positions skipped (his `embed_baseline.py` recipe). Separate 200 docs (71k positions) held out
  for the false-positive check.
* Heads compared (all applied to the same states, so only the rows differ):
  * `proj` = Brennan's `PRIOR_REPRESENTATION_EMBED_PROJ` (project `h_bar` out of the phrase mean, unit-normalise, no bias).
  * `avgtok` = his `AVERAGE_TOKEN_WEIGHTS` with bug 1 fixed (each phrase encoded separately, leading space).
  * `lda` = `Sigma^-1 (mu_p - h_bar)` with the Gaussian log-odds bias + log prior + mean log-sum-exp
    (so the row lives in logit units); `lda_wu` = same direction rescaled to a typical W_U row's logit std.
  * `lr` = the 10 rows + biases fitted by full-batch L-BFGS on the *extended* softmax (frozen W_U),
    targets = phrase token at pre-phrase positions and the real next token elsewhere (in-context
    negatives + all 710k generic positions), then bias-shifted from the training prior to the FineWeb
    prior. L2 chosen on held-out likelihood (1e-4 at every scale).
* Note: `blackmail`, `plagiarism`, `cheating`, `Connecticut` are already single tokens in Qwen's vocab,
  so for them the "new" row competes with an existing W_U row that is the ideal answer.

## Results (mean over the 10 phrases, held-out)

| scale | head | phrase in top-10 at own pre-phrase positions | median rank there | generic positions where it enters the top-10 | softmax mass on the 10 new tokens over generic text (true prior sum 7e-5) | AUROC (own positions vs generic) | logit over generic text, mean±std (a real W_U row: ~2±1.8) |
|---|---|---|---|---|---|---|---|
| 150 | Brennan proj | 0.98 | 1 | **0.188** | **0.32** | 0.974 | -0.6 ± **18.1** |
| 150 | avg W_U rows | 0.34 | 37 | 0.0000 | 3.4e-5 | 0.971 | 2.1 ± 2.3 |
| 150 | LDA | 0.68 | 1 | 0.0006 | 4.0e-3 | 0.978 | -172 ± 20.5 |
| 150 | LDA, W_U-scaled | 0.70 | 1 | 0.0007 | 2.2e-3 | 0.978 | -1.7 ± 2.0 |
| 150 | logistic regression | 0.47 | 16 | 0.0001 | 2.5e-5 | 0.977 | 1.6 ± 2.5 |
| 600 | Brennan proj | 0.98 | 1 | 0.188 | 0.32 | 0.974 | -0.6 ± 18.1 |
| 600 | LDA | 0.69 | 1 | 0.0007 | 4.8e-3 | 0.979 | -172 ± 20.3 |
| 600 | logistic regression | 0.47 | 16 | 0.0001 | 2.5e-5 | **0.985** | -0.1 ± 3.1 |
| 2400 | Brennan proj | 0.98 | 1 | 0.188 | 0.32 | 0.973 | -0.6 ± 18.2 |
| 2400 | LDA | 0.70 | 1 | 0.0008 | 5.2e-3 | 0.979 | -165 ± 20.0 |
| 2400 | LDA, W_U-scaled | 0.71 | 1 | 0.0008 | 2.8e-3 | 0.979 | -1.7 ± 2.0 |
| 2400 | logistic regression | 0.39 | 24 | 0.0001 | 2.7e-5 | **0.985** | -0.6 ± 3.5 |

Cross-phrase confusion at 2400 (at held-out positions of the row phrase, how often the column token
is in the top-10):

* Brennan proj: every same-category token fires 0.9-1.0 (at Connecticut positions, Rhode Island 0.97,
  New Hampshire 0.97, George Washington 0.94, John Adams 0.88); cross-category 0.2-0.5.
* LDA: cross-category 0.00, but within-category 0.5-0.65 (Connecticut -> Rhode Island 0.64,
  New Hampshire 0.62; George Washington -> Lincoln 0.31, Adams 0.32; Lincoln -> Washington 0.56).
* Logistic regression: off-diagonal <= 0.10 everywhere (Connecticut -> Rhode Island 0.06, New Hampshire 0.03).

## What this says

1. **The notebook symptom reproduces on the correct Base model + Base lens, so it is not the lens mismatch.**
   Brennan's rows have a logit std of 18 over generic text against 1.8 for a real unembedding row: ten
   times too wide. Each phrase token enters the model's top-10 at 19% of random positions, the ten of
   them absorb 32% of the softmax mass on ordinary text, and on every US-state or president prompt the
   lens shows the same fixed block of six state/name tokens at every layer. On "The first president of
   the United States was" the block reads John Adams > Abraham Lincoln > George Washington at every layer,
   while the model itself says " George". The direction is fine (AUROC 0.97); the calibration is not.
   This is the review's diagnosis, now measured.

2. **It is not undertrained.** `proj` is numerically identical at 150, 600 and 2400 contexts per phrase
   (0.188 / 0.188 / 0.188 generic top-10 rate; AUROC 0.974 / 0.974 / 0.973). The mean of 150 vectors has
   already converged; the row is a mean-difference direction, and more samples only make it a better
   estimate of the same badly-scaled direction. LDA is also flat across scale (AUROC 0.978 -> 0.979).
   The only head that used the extra data was the logistic regression, whose direction improved from
   150 to 600 (AUROC 0.977 -> 0.985) and was flat after that.

3. **Whitening fixes the false positives but not the over-confidence.** `Sigma^-1 (mu - h_bar)` cuts the
   generic top-10 rate from 19% to 0.07% and puts the phrase at rank 1 on 60-70% of its own held-out
   positions. But the Gaussian log-odds are themselves ~10x too confident (std 20), so the 10 new tokens
   still take 70x their prior mass on generic text, and category neighbours bleed: at Connecticut
   positions Rhode Island and New Hampshire are in the top-10 60% of the time, and John Adams still
   outranks George Washington on the first-president prompt. Rescaling to W_U's logit std (`lda_wu`)
   leaves the ranks unchanged, because rank is decided by the direction. LDA is the right closed-form
   *direction*; its logit *scale* has to come from somewhere else.

4. **Logistic regression over the extended softmax is the head that behaves like a W_U row.**
   Best direction (AUROC 0.985), mass on generic text 2.7e-5 against a true prior sum of 7e-5 (all other
   heads are 30x-4000x above prior), off-diagonal confusion below 0.10, and the lens readouts look
   like lens readouts: "Concord is the capital of the state of" shows `[New Hampshire]` on top at
   L22-L26 next to " Massachusetts" and " Vermont"; "The first president ... was" shows
   `[George Washington]` at L26 next to " Washington" and " George" (nothing at L16-L20, where the
   lens sees only underscores); "She threatened to leak the photos ... a crime known as" shows
   `[blackmail]` at L20-L24 alongside " blackmail" and " extortion" and nothing else from the set.
   The cost is honest ranks: at its own pre-phrase positions the phrase token has a median rank of
   7-85, which is the same range as the model's *own* first token at those positions (3-46). A phrase
   with a prior of 1e-5 is not rank 1 in most contexts where it eventually appears, and neither is
   " New" before "New Hampshire".

5. **Two things the LR needs to work.** (a) A real optimiser: 60 epochs of minibatch Adam at lr 3e-4
   barely moved the rows (recall fell with scale, `results_adam.json`); full-batch L-BFGS converges in
   under a minute on the stored states. (b) Enough negative *documents*: with only 400 generic docs the
   fit picked up document-specific directions and generic mass rose to 3.7e-3 at 2400 contexts
   (`results_lbfgs_400docs.json`); with all 2000 docs and L2 = 1e-4 chosen on held-out likelihood it is
   at prior for every scale.

## Recommendation for the fork

* Keep `Qwen/Qwen3.5-9B-Base` with the Base lens (or fit a lens on the post-trained model).
* Replace `PRIOR_REPRESENTATION_EMBED*` with the fitted rows: freeze everything, add `n` rows + biases,
  minimise cross-entropy over the extended softmax on (pre-phrase positions -> phrase token) union
  (generic + in-context positions -> real next token), L-BFGS, L2 ~1e-4, then shift each bias by
  `log(pi_true/(1-pi_true)) - log(pi_train/(1-pi_train))`. Apply the bias in the readout
  (`logits[..., V:] += b`). The `collect_embeddings.py` states are all that is needed; no model forward
  passes during fitting.
* Keep the `h_bar`/`Sigma` pipeline only as the closed-form fallback (`lda_wu`), and label its logits as
  over-confident.
* Ship the calibration check before any lens plot: generic top-10 rate, mass on new tokens vs prior,
  and rank at held-out pre-phrase positions next to the rank of the phrase's real first token.
* Not done here, still needed for the science: confusable pairs (New York / New Jersey next to New
  Hampshire; John Quincy Adams next to John Adams), and a variant that drops contexts where the phrase
  already appeared earlier (15% of the mined contexts).
