# How far do phrase rows generalise beyond proper nouns?

Date: 2026-09-14. Code `modal_exp/jlens_gen.py` (Modal app `jlens-gen`; states `states/gen_<axis>.pt`,
shared generic lens states `states/generic_lens.pt`); per-axis tables `modal_exp/results/GEN_<axis>.md`;
cross-axis tables `GEN_SUMMARY.md`; raw `gen_<axis>.json`, `gen_joint_nouns_verbs_abstract_events.json`.

## What was run

Five axes, each with first-token or category sibling groups so the sibling test can run:

| axis | phrases | what | sibling groups |
|---|---|---|---|
| nouns | 45 | compositional noun phrases: ice cream / ice age / ice hockey, credit card / score / union, hard drive / disk / work, ... | first token (16 groups) |
| verbs | 43 | verb phrases and idioms: take into account / place / advantage / care, in spite of / addition to / front of / terms of, kick the bucket, ... | first token (14 groups) |
| abstract | 20 | concepts defined by surface-form sets (betrayal = betray/betrayed/treachery/...; recursion = recursion/recursive/recursively) | category: emotion / character / economy / science |
| events | 28 | named events, dates, years: French Revolution / Riviera, World War II / I / Cup / Series, July 4, 1776 / July 20, 1969, 1984 / 1989, Berlin Wall / Marathon | first token (12 groups) |
| xling | 48 | 12 concepts × English / Spanish / French / Chinese (ice cream, climate change, credit card, human rights, ...), each language mined from its own FineWeb(-2) corpus | the same concept in the other languages |

Same recipe as the benchmark: Brennan's miner over the FineWeb 10BT sample (FineWeb-2 subsets for the
other languages), 200 held-out contexts per phrase, up to 1,200 training contexts (300 for *kick the
bucket*, 97--531 for four rare event strings), pre-phrase position = the token before the phrase's first
token, contexts with the phrase at position 0 dropped. Generic negatives and evaluation positions are the
benchmark's (710k stored, 176,763 held out), plus 18,622 generic positions with R-lens / J-lens /
logit-lens states at layers 8--28 for per-layer thresholds. Two heads per axis: the bias-free logistic
row (extended softmax, positives at the pre-phrase position, weighted to the corpus prior, L2 = 1e-6) and a
*presence probe* (one binary logistic row per phrase, positives = the pre-phrase position plus every
token inside the phrase, no softmax competition).

Metrics are ROC-style throughout: TPR at FPR 1e-4 / 1e-3 / 1e-2 against generic text (thresholds from the
generic positions), FPR at TPR 0.5 / 0.9, the same against siblings, TPR at the generic FPR-1e-2 threshold
at every position from two tokens before the phrase to one after ("position profile"), TPR at FPR 1e-2 at
each lens layer with the threshold taken from generic lens states of the same lens and layer ("when it
pops up"), and the earliest layer at which that TPR reaches 0.5.

## Headline, bias-free row, means over phrases

| axis | AUROC vs generic | TPR @ FPR 1e-4 | @ 1e-3 | @ 1e-2 | FPR @ TPR 0.9 | AUROC vs siblings | TPR @ 1e-2 vs siblings | mass / prior | earliest R-lens layer (median) | never |
|---|---|---|---|---|---|---|---|---|---|---|
| nouns | 0.988 | 0.53 | 0.75 | 0.89 | 0.015 | 0.899 | 0.53 | 2.9 | 20 | 0 % |
| verbs | 0.983 | 0.35 | 0.58 | 0.82 | 0.039 | 0.876 | 0.43 | 1.9 | 20 | 5 % |
| abstract | 0.980 | 0.41 | 0.62 | 0.81 | 0.045 | 0.918 | 0.53 | 3.0 | 24 | 10 % |
| events | 0.992 | 0.71 | 0.87 | 0.94 | 0.005 | 0.886 | 0.58 | 1.6 | 8 | 0 % |
| joint (136 rows, one softmax) | 0.981 | 0.48 | 0.68 | 0.84 | 0.034 | 0.880 | 0.50 | 1.5 | 20 | 4 % |

TPR at FPR 1e-2 under the R-lens by layer (mean over phrases), with the model's own first-token top-10 rate for reference:

| axis | L8 | L12 | L16 | L20 | L24 | L28 | first token in top-10 at L8 / L20 / L28 |
|---|---|---|---|---|---|---|---|
| nouns | 0.28 | 0.35 | 0.36 | 0.56 | 0.77 | 0.84 | 0.04 / 0.14 / 0.60 |
| verbs | 0.21 | 0.27 | 0.35 | 0.54 | 0.73 | 0.76 | 0.11 / 0.09 / 0.33 |
| abstract | 0.14 | 0.20 | 0.21 | 0.39 | 0.62 | 0.71 | 0.02 / 0.20 / 0.45 |
| events | 0.51 | 0.55 | 0.56 | 0.75 | 0.89 | 0.92 | 0.12 / 0.12 / 0.59 |

Earliest R-lens layer with TPR@1e-2 ≥ 0.5, count of phrases: nouns 7 / 3 / 3 / 15 / 14 / 3 at layers
8 / 12 / 16 / 20 / 24 / 28; verbs 1 / 4 / 1 / 20 / 14 / 1 (+2 never); abstract 1 / 1 / 0 / 2 / 10 / 4 (+2 never);
events 15 / 3 / 3 / 4 / 3 / 0.

## Findings

1. **The method leaves proper nouns intact.** Every axis separates its phrases from generic text at AUROC
   0.98--0.99 with a bias-free row, and at a false-positive rate of one in a thousand the row still recovers
   58 % (verbs) to 87 % (events) of the positions where the phrase is about to appear. At one in ten
   thousand, 35--71 %. Mass on the new tokens stays within 2--3× the corpus prior (1.6× for events).

2. **Sibling separation is now the routine test, and it tracks semantic distance, not syntax.** Rows fitted
   with their first-token siblings present separate them at AUROC 0.88--0.92 on average. The failures are
   the pairs a reader would also struggle with at that position: *hard drive* vs *hard disk* (0.80 / 0.76,
   sibling TPR 0.03--0.05), *health care* vs *health insurance* (0.69 / 0.80), *stock market* vs *stock
   price* / *exchange* (0.73), *climate change* vs *climate crisis* (0.75), *1984* vs *1989* (0.58 / 0.63,
   sibling TPR 0.02), *World War II* vs *World War I* (0.86, TPR 0.32), *irony* vs *hypocrisy* /
   *ambition* (0.82). The successes are the pairs whose continuations live in different worlds: *machine
   learning* vs *machine gun* 0.98, *black pepper* vs *black hole* / *market* 0.99, *prime number* vs
   *prime minister* 0.98, *high court* 0.98, *photosynthesis* vs the other sciences 0.98, *inflation* vs
   the other economy concepts 0.99. A year is the cleanest negative: the state before "1984" or "1989" says
   "a year comes next" (AUROC 1.00 vs generic) and nothing about which year.

3. **When a phrase pops up depends on how much the context has already committed.** Named events and
   dates are readable early: at layer 8 the events row already recovers 51 % of its positions at FPR
   1e-2, and 15 of 28 event phrases reach half-recall by layer 8 (*Great Depression*, *Great Barrier Reef*,
   *2008 financial crisis*, *July 4, 1776*, *Berlin Wall*, *moon landing*). Compositional nouns and verb
   phrases mostly arrive at layers 20--24 (median 20); abstract concepts at 24--28 (median 24), with the
   technical ones (*photosynthesis* layer 8, *recursion* 12, *inflation* 20) ahead of the emotions
   (*betrayal*, *regret*, *gratitude* 24; *jealousy*, *curiosity* 28; *nostalgia* never). Within an axis the
   same rule holds: *ice age* and *solar system* by layer 8, *ice cream* and *high school* not until 28,
   because the former are predictable from their topic and the latter appear anywhere.

4. **The phrase row is ahead of the model's own next-token prediction at every layer.** At layer 24 the
   rows recover 62--89 % of upcoming phrases at FPR 1e-2 while the phrase's real first token is in the
   lens top-10 at only 10--39 % of the same positions; even at the final layer the first token is in the
   top-10 at 45--72 %. The phrase row reads "this phrase is coming" from a state in which the model has not
   yet settled on its first token, which is what a workspace readout should do.

5. **R-lens vs J-lens, at fixed FPR, are the same; the top-10 view is not.** With thresholds calibrated
   on generic lens states, TPR@1e-2 curves under the two lenses agree to ±0.03 (the R-lens is ahead at
   layer 8 for events, 0.51 vs 0.45, and behind by 0.01--0.05 for nouns at 12--16). The top-10 rates differ
   by up to 0.2 at layers 8--16 (J-lens higher), which is the early-layer inflation of the J-lens's real
   logits seen before, not a property of the rows. Fixed-FPR metrics are the right ones for "when does it
   pop up"; top-10 counts are not.

6. **The presence probe and the row answer different questions.** The row fires at the token before the
   phrase (TPR 0.8--0.97 at FPR 1e-2 for verb phrases, position 0 of the profile) and already at the
   token before that in 20--58 % of contexts, then switches off inside the phrase (0.02--0.7 at position 1,
   depending on whether the second word is still uncertain: *kick* → 0.72, *take care* → 0.02). The probe
   covers the inside of the phrase at 0.9--1.0 and lingers after it (0.2--0.6 for verbs, 0.4--0.9 for
   abstract concepts, which stay active longer than idioms). At the pre-phrase position the probe is weaker
   than the row (AUROC 0.90--0.97 vs 0.98--0.99) because it spends its capacity on the inside positions. The
   probe fails entirely on the rarest phrase (*kick the bucket*, 300 contexts, TPR 0.00 everywhere): with
   positives weighted to a prior of 1e-7 and no softmax competition to anchor the scale, 300 positives are
   not enough. The row on the same 300 contexts is fine (TPR 0.81 at position 0).

7. **Fitting everything in one softmax costs little and calibrates better.** 136 rows from four axes
   fitted jointly against the same negatives lose 0.003--0.006 AUROC and 0.01--0.03 TPR at FPR 1e-3 per axis
   relative to the separate fits, and 0.004--0.025 sibling AUROC; their mass-to-prior ratios drop from
   1.6--3.0 to 1.0--1.7 because the rows now compete. Row directions stay diverse (mean pairwise cosine
   0.065, participation ratio 84 of 136). The confound is real but small; a single joint head is the right
   default for a lens with many phrases.

8. **What did not generalise.** Near-synonyms (*hard drive* / *hard disk*), numerals that share a
   context type (*1984* / *1989*, *World War I* / *II*), and abstract nouns whose surface forms are
   themselves ambiguous (*irony*, sibling TPR 0.05; *nostalgia*, never reaches half-recall). Each is a case
   where the pre-phrase state legitimately does not know which sibling is coming. Cross-lingual transfer
   is partial rather than absent (finding 9).

## Cross-lingual axis

12 concepts × 4 languages, each language mined from its own corpus (FineWeb for English, FineWeb-2
`spa_Latn` / `fra_Latn` / `cmn_Hani` for the others; substring matching for Chinese). A row fitted on
language *a* is scored at language *b*'s pre-phrase positions of the same concept against *b*'s
positions of the other eleven concepts, so language identity cannot carry the score. (The first pass
used English generic thresholds and read as "no transfer", TPR 0.07--0.13; that was the negatives being
out of distribution, not the rows.)

| row fitted on | scored in English | Spanish | French | Chinese |
|---|---|---|---|---|
| English | 0.976 / 0.79 (own) | 0.90 / 0.38 | 0.91 / 0.49 | 0.87 / 0.36 |
| Spanish | 0.88 / 0.35 | 0.99 / 0.91 (own) | 0.94 / 0.54 | 0.82 / 0.24 |
| French | 0.89 / 0.42 | 0.94 / 0.62 | 0.99 / 0.90 (own) | 0.83 / 0.25 |
| Chinese | 0.87 / 0.33 | 0.84 / 0.29 | 0.84 / 0.24 | 0.971 / 0.73 (own) |

(AUROC / TPR at FPR 1e-2, mean over the 12 concepts; bias-free row.)

9. **The row direction is largely shared across languages, with a language-specific part on top.** An
   English row separates the same concept from the other eleven in Spanish, French or Chinese at AUROC
   0.87--0.91 (own language 0.98), recovering 36--49 % of positions at FPR 1e-2 against 79 % at home.
   Spanish and French share the most (0.94 both ways), Chinese the least (0.82--0.87), and every
   direction is symmetric to within 0.03. Per concept, *climate change*, *solar system*, *civil war*,
   *human rights* and *health insurance* transfer at 0.92--0.99; *credit card* (English → Spanish 0.69)
   and *ice cream* (0.82--0.90) least. The presence probe transfers better still (English → 0.97 / 0.96 /
   0.89): whether a concept is *present* is more language-neutral than whether the model is *about to
   say* it, which is the row's question and includes the surface form. Own-language rows for
   non-English text are as good as English ones (Spanish and French 0.99, Chinese 0.97), so nothing here
   is an English artefact.

## Caveats

* Regex-mined surface forms stand in for concept labels; an LLM-labelled "concept expressed here" set
  would test presence detection properly and is the next step for the abstract axis.
* 1,200 training contexts per phrase, one L2 value (1e-6) carried over from the benchmark, no sweep.
* Per-layer thresholds come from 18,622 generic positions, so FPR 1e-2 is well determined and 1e-3 is
  the limit; the final-layer thresholds use 176,763 positions and support 1e-4.
* The prior for the probe's weighting is the pre-phrase-position prior, not a presence prior; a presence
  prior would be several times larger and would help the rare phrases.
