# Review: Brennan's multi-token extension of the Jacobian lens

Repo: https://github.com/BrennanLagasse/jacobian-lens-extension (fork of `anthropics/jacobian-lens`),
`multi_token/`, 8 commits 2026-08-30 → 2026-09-05. Reviewed 2026-09-05 at `eb6d855`.

## Verdict

The pipeline is a clean, well-documented scaffold, but the headline experiment does not work yet and
the notebook results should not be read as lens findings. In `walkthrough_multitoken.ipynb` the 10
added phrase tokens fill the entire top-10 of the J-lens readout at every layer, on every prompt —
including "the country shaped like a boot is" (L16/L30 → `cheating`, `blackmail`) — and the *model's
own* final logits (no lens involved) rank `cheating, plagiarism, forgery, blackmail` above ` with`.
The new unembedding rows are miscalibrated relative to the real `W_U`, so they win regardless of
context. Brennan's last commits ("continuing to diagnose issues") indicate he knows something is off;
the diagnosis below is what I believe it is.

## What was built

1. `phrase_context_miner.py` — streams FineWeb/Pile/Wikipedia, collects ≤150 contexts per phrase
   (config: blackmail, plagiarism, cheating, forgery, Connecticut, Rhode Island, New Hampshire,
   George Washington, Abraham Lincoln, John Adams).
2. `collect_embeddings.py` — post-final-norm hidden state (`model.model(...).last_hidden_state`) at
   the token immediately *before* the phrase (`--position before`).
3. `process_representations.py` — per-phrase mean vector μ_p.
4. `embed_baseline.py` — h̄, the mean post-norm state over generic FineWeb text (skipping the first
   4 positions).
5. `extend_model.py` — appends rows to `lm_head` (three recipes; the notebook uses
   `PRIOR_REPRESENTATION_EMBED_PROJ`: project h̄ out of μ_p, then L2-normalise to unit norm) and
   wraps the tokenizer so `decode()` knows the new ids.
6. `walkthrough_multitoken.ipynb` — loads the pre-fitted Hub lens and reads out through the extended head.

`jlens` consumes the extension correctly (it calls the `lm_head` module and infers `vocab_size` from
the logits), so the plumbing is sound. The problem is the *content* of the new rows.

## Root cause (verified on a toy replica, `cone_test.py` / `sink_test.py`, Qwen2.5-1.5B, CPU)

* `W_U` row norms are ~1.0 (Brennan measured 0.38–1.55 on Qwen3.5-9B; same on 2.5-1.5B), so unit
  norm is the right *magnitude*. The problem is *direction*.
* Post-norm residual states are extremely anisotropic: every state has cosine ≈ 0.72 with the shared
  mean direction, and a unit row along that direction scores ~135 on any input (real max logits ≈ 21–23).
  This is why h̄-subtraction was needed at all.
* h̄-subtraction cancels the shared component only if μ_p and h̄ are drawn from the same
  distribution. They are not (FineWeb positions ≥ 4 vs. "token-before-phrase" positions, often early
  in short 1–3 sentence contexts). Any leftover shared component is then *amplified* by the unit
  normalisation into a prompt-independent positive offset for every new token. That predicts exactly
  the notebook symptom: all 10 new tokens on top, in a nearly fixed order, on every prompt.
* Even with matched distributions (my toy, hand-written contexts), the centered prototype row is not
  calibrated: its logit ranges from −16 to **+74** across four prompts while real logits stay under 23.
  It is rank 1 on "the country shaped like a boot is" — a place-name-is-next context, i.e. the
  first-token confound (see below). Trained `W_U` rows are shaped to have low variance along the
  high-variance directions of h; a mean-difference direction is not.
* The `AVERAGE_TOKEN_WEIGHTS` baseline behaves sensibly in the toy (rank 18 on "arrived late in",
  rank in the thousands elsewhere). It is untested in the fork (see bug 1) and never compared in the notebook.

## Conceptual issue: what "before" measures

The state at the token before "New Hampshire" is the state that predicts ` New`. Its mean over
contexts is dominated by "a first-token-` New` continuation is coming", shared with New York /
New Jersey / New Zealand. Whether the model additionally encodes the *full* phrase at that position
is the interesting scientific question (it is what the workspace paper would suggest), but the current
phrase set has no confusable pairs, so the experiment cannot tell "phrase is planned" apart from
"first token is planned". Add New York / New Jersey next to New Hampshire, and John Adams vs
John Quincy Adams, and report discrimination between them.

## Bugs / correctness (highest impact first)

1. `extend_model.py:49` `tokenizer.encode(new_phrases)` passes a *list of strings* to `encode`,
   which HF treats as pre-tokenised tokens of one sequence, not as separate phrases. The
   `AVERAGE_TOKEN_WEIGHTS` path is therefore broken (`len(phrase)` on ints). Encode each phrase
   separately, with a leading space (`" New Hampshire"`), since in-text tokens are the space-prefixed variants.
2. Lens/model mismatch: the notebook loads `Qwen/Qwen3.5-9B` (post-trained) but the only 9B lens on
   the Hub is `qwen3.5-9b-pt/.../Qwen3.5-9B-Base_jacobian_lens.pt` (fitted on the Base model,
   n_prompts=458, not 1000). Brennan added this mapping himself. J is fitted on a specific model's
   activations; either load `Qwen/Qwen3.5-9B-Base` or fit a lens on the post-trained model
   (`jlens.fit`, 100 prompts is enough per the upstream walkthrough).
3. `collect_embeddings.py:138` `max(0, token_start - 1)`: when the phrase is the first token of the
   context (common — sentence-initial "Cheating", "Connecticut"), the "before" state silently becomes
   the phrase's *own first token* (predicting its second token). Those samples should be skipped, not
   clamped. Post-norm this is contamination, not a magnitude blow-up (`sink_test.py`: position-0 norm
   is ordinary after RMSNorm).
4. Two contradicting normalisations live side by side: `PRIOR_REPRESENTATION_EMBED` (subtract h̄)
   and `_PROJ` (project h̄ out). Only one of them is used; in the toy they give near-identical
   results. Delete the unused one.
5. `extend_model()` registers a gradient hook to freeze the new input rows "during training". Nothing
   trains; the hook is dead code and `model.config.vocab_size = target_len` is redundant after
   `resize_token_embeddings`. Delete both (and the `import torch` inside the function body).
6. `phrase_context_miner.py` records `phrase` with original casing; `process_representations.py`
   only merges case variants with `--case-insensitive`. The README pipeline does not mention the
   flag; without it "cheating"/"Cheating" become two tokens.
7. `test_extended_model.py` asserts `decode([len(tokenizer)]) == "New Hampshire"` — depends on dict
   ordering of `phrase_means.pt`. Compare against `list(data["phrase_means"])[0]` instead.
8. New-token strings lack the leading space that every in-text `W_U` token has (`'New Hampshire'`
   vs `' New'`). Cosmetic for decode, but it matters for bug 1 and for any string-matching downstream.

## Recommendations

1. **Fit the rows instead of constructing them.** Freeze everything, add the n new rows (and a
   per-row bias), and minimise cross-entropy over the *extended* softmax where the target is the
   phrase token at "before-phrase" positions and the ordinary next token elsewhere (mined contexts
   ∪ generic FineWeb). This puts the new rows in the same logit units as `W_U` and directly penalises
   firing on unrelated text. It is a ~40k-parameter logistic regression; it runs on the Orin.
   Apply the bias in the lens readout (`logits[..., V:] += b`).
2. If a closed form is preferred: LDA/whitening, `row_p = Σ⁻¹(μ_p − μ)` with Σ the covariance of
   post-norm states, then scale so the row's logit std on generic text matches a typical `W_U` row.
   This addresses the variance problem that mean-centering alone does not.
3. **Add a calibration check before any lens plots**: on held-out text, report for each new token
   (a) mean and std of its logit, (b) how often it is in the model's top-10 at positions that do
   *not* precede the phrase (should be ≈ 0), (c) rank at positions that do (should be high). Run it
   against `AVERAGE_TOKEN_WEIGHTS` as the baseline.
4. Add confusable phrase pairs (above) so the experiment measures phrase identity, not first-token identity.
5. Keep `results/*.pt` (or a small subset: `h_bar.pt`, `phrase_means.pt`, ~40 MB) in the repo or on
   the Hub so results are reproducible; right now nothing in the notebook can be re-run by anyone else.

## Files

* This review: `~/notes/brennan-jlens-review/REVIEW.md`
* Toy replication: `~/notes/brennan-jlens-review/cone_test.py`, `sink_test.py`
  (run with `HF_HUB_OFFLINE=1 ~/.venv-interp/bin/python <script>`; Qwen2.5-1.5B, CPU, ~2 min each)
