# Phrase-head benchmark: how far can a multi-token unembedding row be trusted?

Date: 2026-09-12. Code `modal_exp/jlens_bench.py` (Modal app `jlens-bench`, volume `jlens-multitoken`).
Tables: `modal_exp/results/BENCH.md`; raw: `modal_exp/results/bench.json` (earlier pass without the
bias-free head: `bench_v1_biasonly.json`). Follows `EXPERIMENT_2026-09-12.md`.

## The frozen evaluation set

* Qwen3.5-9B-Base + Base J-lens. Brennan's 10 phrases plus 5 confusables: New York, New Jersey
  (share " New" with New Hampshire), John Quincy Adams (shares " John" with John Adams), Massachusetts,
  Vermont (state-category siblings). 15 phrases, every one with >4,200 unique FineWeb contexts except
  John Quincy Adams (1,317).
* 200 held-out contexts per phrase (2,942 usable pre-phrase positions; 84% single-occurrence, 16%
  contain the phrase earlier = copy bucket). 500 held-out generic FineWeb documents = 176,763 positions.
  Nested training subsets 150 / 600 / 2,400 per phrase.
* Every head is scored on the same states. Direction: AUROC own-positions vs generic, vs first-token
  sibling, vs category sibling. Calibration: prior-weighted ECE, mass on new tokens vs FineWeb prior,
  generic top-10 rate. Rank: median rank at own positions and its ratio to the phrase's real first
  token's rank there. Copy sensitivity: single vs copy bucket. Lens profile: for every held-out
  context, the rank of the phrase token at 11 layers under the J-lens and the logit lens, next to the
  real first token. Stability: three disjoint 150-context refits.
* Heads: `proj` (Brennan), `avgtok` (mean W_U rows; for the 6 single-token phrases this *is* the real
  row, so it doubles as the ceiling), `lda`, `lda_wu`, logistic regression with bias (`lr`, L2 by
  validation loss at the training mixture), logistic regression **without bias** (`lr_nobias`,
  positives weighted to the FineWeb prior instead of shifting a bias afterwards, L2 swept 1e-6..1e-8),
  and a random unit row as the floor.

## Headline (scale 2400, mean over 15 phrases)

| head | AUROC vs generic | vs first-token sibling | vs category sibling | own top-10 | own median rank | log(rank / first-token rank) | generic top-10 | mass / prior | ECE |
|---|---|---|---|---|---|---|---|---|---|
| random row | 0.508 | 0.514 | 0.508 | 0.00 | 29,835 | +7.45 | 0.0003 | 4.8 | 4.7e-5 |
| Brennan proj | 0.969 | **0.582** | 0.642 | 0.95 | 1 | -2.70 | **0.203** | **7,687** | 2.2e-2 |
| avg W_U rows | 0.966 | 0.575 | 0.734 | 0.30 | 43 | +1.17 | 0.0001 | 1.2 | 2.5e-5 |
| LDA / whitening | 0.980 | 0.663 | 0.714 | 0.69 | 1 | -1.43 | 0.0008 | 74 | 4.8e-4 |
| LR with bias (L2 1e-4) | 0.980 | 0.736 | 0.819 | 0.37 | 25 | +0.88 | 0.0003 | 0.4 | 1.3e-5 |
| LR with bias (L2 1e-5) | 0.982 | 0.750 | 0.826 | 0.42 | 21 | +0.81 | 0.0004 | 2.7 | 3.4e-5 |
| **LR no bias (L2 1e-6)** | **0.981** | 0.731 | 0.813 | 0.45 | 14 | **+0.22** | 0.0003 | **1.0** | 4.6e-5 |

Stability over three disjoint 150-context refits: AUROC std <= 0.0008 for every head, so every
difference above 0.002 in this table is real.

## Findings

1. **Brennan's head is a first-token detector.** Fitted on the original 10 phrases, the New Hampshire
   row enters the top-10 at 97% of pre-"New York"/"New Jersey" positions (AUROC own-vs-sibling 0.59)
   and the John Adams row at 98% of pre-"John Quincy Adams" positions (AUROC 0.48). Its category-sibling
   AUROC is 0.64. It cannot tell which state or which president is coming; only that one is.

2. **There is phrase identity in the pre-phrase state, and the fitted heads read it.** A logistic head
   fitted on the original 10 phrases, never shown New York or New Jersey, separates New Hampshire from
   them at AUROC 0.85 and fires at only 2% (bias) / 7% (no bias) of their positions. LDA gets 0.82 and
   24%. So the state before " New" carries which "New ..." is coming, and a mean-difference row
   throws that away. John Adams vs John Quincy Adams is at chance for every head (0.45-0.57): from the
   John Adams side the two are indistinguishable, though the John Quincy Adams row does separate its
   own contexts (0.71-0.74), presumably from "sixth president"/"son of" cues.

3. **A bias is a lens hazard.** The biased logistic head is the best-calibrated at the final layer
   (mass 0.4x prior, ECE 1e-5) but its +7..+10 logit bias does not shrink when the J-lens transports
   an early-layer state whose real logits are compressed. Under the J-lens it puts the phrase token in
   the top-10 at 49% of contexts at layer 8 and 20% at layer 4, where the real first token is at 0%;
   the hand prompts show 4-8 phrase tokens in the top-10 at layer 8. That is the same failure as
   Brennan's head in a smaller dose. The bias-free head, weighted to the prior instead, is at 6% / 14%
   at layers 4 / 8 under the J-lens and 1% / 5% under the logit lens, and its per-prompt counts at
   layer 8 are 0-2. Real unembedding rows have no bias; a phrase row should not either.

4. **The bias-free logistic head is the one that behaves like a W_U row.** Mass on new tokens 1.0x
   prior, AUROC 0.981, cross-phrase bleed small, and its rank at own positions is within a factor
   1.25 of the model's own first-token rank there (median log ratio +0.22; LDA is -1.43, i.e. four
   times *more* confident than the model is about " New", Brennan's is -2.70). Per phrase it ranks the
   phrase at 4-19 for 13 of 15 phrases; the two exceptions are John Adams (148) and John Quincy Adams
   (214), where the two rows split the " John" evidence.

5. **Copying is not what the heads are reading.** Copy-bucket contexts are easier (AUROC 0.995 vs
   0.979, top-10 0.61 vs 0.42 for the bias-free head), but single-occurrence performance stays high
   for every head, and Brennan's head is indifferent to the bucket (it fires regardless).

6. **Layer profiles, J-lens vs logit lens, over all 2,942 held-out contexts.** The standard readout
   (real first token) enters the top-10 at layer 30 in 52% of contexts under the J-lens and 45%
   under the logit lens; at layer 24 it is 23% vs 18%. The J-lens surfaces it earlier, as the paper
   reports. The bias-free phrase token tracks the same curve slightly below it (42% / 44% at layers
   24 / 30 under the J-lens; 37% / 45% under the logit lens). In 13-25% of contexts at layers 8-24 the
   phrase token is in the top-10 while the real first token is not; that is the population of cases
   where a multi-token row shows something the single-token readout does not, and it is the thing to
   examine next, with the early-layer end of it treated with suspicion (see 3). Median earliest layer
   at which the phrase token enters the top-10: 20-28 for most phrases, layer 4 for New York, never
   (>50% of contexts) for blackmail, forgery, George Washington, John Adams, John Quincy Adams.

7. **Hand prompts at 2,400, bias-free head.** "Trenton is the capital of" -> `[New Jersey]` first
   from layer 24 through the model's own logits. "Concord is the capital of the state of" ->
   `[Massachusetts]`, `[New Hampshire]` at layers 24-26, then the model's " New" with `[New Hampshire]`
   at rank 4. "Yale University is located in New Haven," -> a phrase token at layers 22-30. The
   president prompts show nothing from this head: on "The first president of the United States was"
   the lens has " Washington", " George" at layer 26 but the George Washington row does not fire
   (it is the weakest row: rank 19, never-in-top-10 for 54% of its own contexts). The biased head and
   LDA do fire there, and LDA still ranks John Adams above George Washington.

8. **Data scale.** Brennan's head and LDA are flat from 150 to 2,400. The logistic heads gain about
   0.005 AUROC from 150 to 600 and nothing after. The L2 chosen by validation loss moved from 1e-3
   (150, 600) to 1e-4 (2,400) for the biased head and stayed at 1e-6 for the bias-free head; the
   sweep tables show that the likelihood-optimal L2 is stronger than the rank-optimal one at small
   data, so which one to use depends on whether the rows are read as probabilities or as rankings.

## What to adopt

* Fit the rows by logistic regression over the extended softmax **without a bias**, positives
  weighted to the corpus prior, L2 ~ 1e-6, L-BFGS. This is the head to put behind `extend_model.py`.
* Report, for any phrase row: AUROC vs generic, vs first-token siblings, mass vs prior, and rank
  relative to the phrase's first token. Rows that fail the sibling test are first-token detectors.
* When reading the lens with phrase rows, show the phrase's first token alongside, and treat
  phrase-token hits below layer ~16 as suspect until the row's early-layer false-positive rate on
  held-out contexts is known.
* Name pairs like John Adams / John Quincy Adams are not separable at the pre-phrase position with
  these contexts; do not build claims on them.

## Caveats

* Prior-weighted ECE is dominated by the generic class and is near zero for every calibrated head; the
  mass/prior ratio is the more discriminating calibration number.
* Rank at lens layers is bucketed (exact to rank 10, then a grid), which is enough for top-10 rates.
* The J-lens is fitted on wikitext; the contexts are FineWeb. The logit-lens comparison shares the
  same states so the relative curves are fair, the absolute layer numbers are model- and lens-specific.
* 20 hand prompts are illustrations; the held-out profiles are the evidence.
