"""Render results/results.json into results/REPORT.md."""
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "results.json")))
meta, scales = R["meta"], R["scales"]
METHODS = ["proj", "avgtok", "lda", "lda_wu", "lr"]
LABEL = {"proj": "Brennan (proj+unit)", "avgtok": "avg W_U rows", "lda": "LDA (Gaussian log-odds)",
         "lda_wu": "LDA (W_U-matched scale)", "lr": "logistic regression"}
PHR = list(meta["pi_true"])
L = []


def agg(m, key):
    return np.mean([v[key] for v in m["per_phrase"].values()])


L.append("# Multi-token J-lens heads on Qwen3.5-9B-Base: calibration vs data scale\n")
L.append(f"Model `Qwen/Qwen3.5-9B-Base`, lens `neuronpedia/jacobian-lens` (Base, wikitext). "
         f"Phrases: {', '.join(PHR)}. Generic FineWeb: 2000 docs x 512 tokens for h_bar/Sigma "
         f"(all stored as LR negatives), {meta['n_generic_eval_positions']:,} held-out positions for the false-positive check. "
         f"Held-out contexts: 100 per phrase, disjoint from all training scales (nested 150 ⊂ 600 ⊂ 2400).\n")
L.append(f"FineWeb docs scanned for mining: {meta['mine_stats']['docs']:,}. "
         f"Fraction of contexts where the phrase occurs more than once (copy/induction confound, kept as in Brennan's recipe): "
         f"{meta['n_occ_gt1_frac']:.2f}. Typical W_U row over generic text: logit mean {meta['wu_row_logit_mean_median']:.1f}, "
         f"std {meta['wu_row_logit_std_median']:.2f}; mean log-sum-exp of the real logits {meta['mean_lse_generic']:.1f}.\n")
L.append("Contexts available per phrase (unique):\n")
L.append("| phrase | unique | train used at 2400 |\n|---|---|---|")
for p, s in meta["summary"].items():
    L.append(f"| {p} | {s['unique']} | {s['train']} |")
L.append("")

L.append("## Headline: mean over the 10 phrases\n")
L.append("`held top-10` = fraction of held-out pre-phrase positions where the phrase token is in the extended top-10; "
         "`held rank` = median rank there; `generic top-10` = fraction of generic positions where the phrase token "
         "enters the model's top-10 (should be ~0); `new mass` = mean softmax mass on all 10 new tokens over generic text "
         "(true prior sum ≈ {:.1e}); `AUROC` = phrase logit at own pre-phrase positions vs generic positions (direction quality, calibration-free).\n"
         .format(sum(meta["pi_true"].values())))
L.append("| scale | method | held top-10 | held top-1 | held rank (median of medians) | generic top-10 | generic top-1 | new mass | AUROC | generic logit mean±std |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for sc, S in scales.items():
    for mth in METHODS:
        m = S["methods"][mth]
        L.append(f"| {sc} ({S['n_train']} pos) | {LABEL[mth]} | {agg(m,'held_top10'):.3f} | {agg(m,'held_top1'):.3f} | "
                 f"{np.median([v['held_rank_median'] for v in m['per_phrase'].values()]):.0f} | "
                 f"{agg(m,'generic_top10_rate'):.4f} | {agg(m,'generic_top1_rate'):.4f} | {m['generic_new_mass']:.1e} | "
                 f"{agg(m,'auroc'):.3f} | {agg(m,'generic_mean'):.1f}±{agg(m,'generic_std'):.1f} |")
L.append("")

if any("lr_sweep" in S for S in scales.values()):
    L.append("## Logistic regression: L2 strength chosen on held-out likelihood\n")
    L.append("| scale | lambda | held NLL (true prior) | held NLL generic | held NLL own-phrase | row norm | chosen |")
    L.append("|---|---|---|---|---|---|---|")
    for sc, S in scales.items():
        for r in S.get("lr_sweep", []):
            L.append(f"| {sc} | {r['lam']:g} | {r['held_nll']:.5f} | {r['held_nll_generic']:.5f} | {r['held_nll_pos_mean']:.3f} | "
                     f"{r['row_norm_mean']:.2f} | {'yes' if r['lam'] == S.get('lr_lam') else ''} |")
    L.append("")

L.append("## Per-phrase detail (scale 2400 unless noted)\n")
for mth in METHODS:
    sc = max(scales, key=int)
    m = scales[sc]["methods"][mth]
    L.append(f"### {LABEL[mth]} @ {sc}\n")
    L.append("| phrase | held rank | held top-10 | first-tok rank | generic top-10 | AUROC | train logit | held logit | row norm | bias |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, (p, v) in enumerate(m["per_phrase"].items()):
        L.append(f"| {p} | {v['held_rank_median']:.0f} | {v['held_top10']:.2f} | {v['held_first_tok_rank_median']:.0f} | "
                 f"{v['generic_top10_rate']:.4f} | {v['auroc']:.3f} | {v['train_logit_mean']:.1f} | {v['held_logit_mean']:.1f} | "
                 f"{m['row_norm'][i]:.2f} | {m['bias'][i]:.1f} |")
    L.append("")

L.append("## Cross-phrase confusion at scale 2400 (rows: held-out positions of phrase; cols: top-10 rate of each phrase token)\n")
for mth in ("proj", "lda", "lr"):
    sc = max(scales, key=int)
    C = scales[sc]["methods"][mth]["confusion"]
    L.append(f"**{LABEL[mth]}**\n")
    L.append("| at → / token ↓ | " + " | ".join(p.split()[-1][:6] for p in PHR) + " |")
    L.append("|---|" + "---|" * len(PHR))
    for i, p in enumerate(PHR):
        L.append(f"| {p} | " + " | ".join(f"{C[i][j]:.2f}" for j in range(len(PHR))) + " |")
    L.append("")

L.append("## J-lens readout: number of phrase tokens in the top-10 (the notebook symptom)\n")
L.append("Rows = prompt (readout position), cells = count of new tokens in the extended top-10 at each layer, "
         "listed as layer:count. The model's own final logits are the last entry.\n")
for sc in scales:
    L.append(f"### scale {sc}\n")
    L.append("| prompt | " + " | ".join(LABEL[m] for m in METHODS) + " |")
    L.append("|---|" + "---|" * len(METHODS))
    n_prompts = len(scales[sc]["methods"]["proj"]["lens"])
    for pi in range(n_prompts):
        cells = []
        for mth in METHODS:
            rows = scales[sc]["methods"][mth]["lens"][pi]["rows"]
            cells.append(" ".join(f"{r['layer']}:{r['n_new_in_top10']}" for r in rows))
        prompt = scales[sc]["methods"]["proj"]["lens"][pi]["prompt"]
        L.append(f"| {prompt[:60]}… | " + " | ".join(cells) + " |")
    L.append("")

L.append("## J-lens top-10 on the walkthrough prompts (scale 150 = Brennan's scale, and 2400)\n")
for sc in (min(scales, key=int), max(scales, key=int)):
    for pi in range(len(scales[sc]["methods"]["proj"]["lens"])):
        prompt = scales[sc]["methods"]["proj"]["lens"][pi]["prompt"]
        L.append(f"**scale {sc} — {prompt}**\n")
        for mth in ("proj", "lda", "lr"):
            L.append(f"- {LABEL[mth]}")
            for r in scales[sc]["methods"][mth]["lens"][pi]["rows"]:
                L.append(f"  - {r['layer']}: " + ", ".join(f"`{t}`" for t in r["top10"]))
        L.append("")

open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])) if len(sys.argv) > 1 else os.path.join(here, "results"),
                  "REPORT.md"), "w").write("\n".join(L))
print("\n".join(L[:40]))
