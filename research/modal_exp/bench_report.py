"""Render results/bench.json -> results/BENCH.md and results/bench.html."""
import html
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results", "bench.json")))
meta, scales = R["meta"], R["scales"]
PHR = meta["phrases"]
LABEL = {"proj": "Brennan (proj+unit)", "avgtok": "avg W_U rows", "lda": "LDA / whitening", "lda_wu": "LDA, W_U-scaled",
         "lr": "logistic regression (val-loss pick)", "lr_1e-05": "LR, L2=1e-5", "lr_0.0001": "LR, L2=1e-4", "lr_0.001": "LR, L2=1e-3",
         "lr_nobias": "LR, no bias (val-loss pick)", "lr_nobias_1e-06": "LR no-bias, L2=1e-6", "lr_nobias_1e-07": "LR no-bias, L2=1e-7", "lr_nobias_1e-08": "LR no-bias, L2=1e-8",
         "random": "random row (floor)", "real_wu": "real W_U row (= avg row for single-token phrases)"}
ORDER = ["random", "proj", "avgtok", "lda", "lda_wu", "lr_1e-05", "lr_0.0001", "lr_0.001", "lr", "lr_nobias_1e-06", "lr_nobias_1e-07", "lr_nobias_1e-08", "lr_nobias"]
HEADS3 = ["proj", "lda", "lr", "lr_nobias"]
SINGLE = meta["single_token_phrases"]
LAYERS = meta["lens_layers"]


def pp(m, bucket="all"):
    return {p: v[bucket] for p, v in m["per_phrase"].items() if bucket in v}


def mean_of(m, key, bucket="all", phrases=None):
    d = pp(m, bucket)
    ps = phrases or list(d)
    vals = [d[p][key] for p in ps if p in d and d[p].get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def top_of(m, key, phrases=None):
    ps = phrases or PHR
    return float(np.mean([m["per_phrase"][p][key] for p in ps]))


L = []
L.append("# Phrase-head benchmark on Qwen3.5-9B-Base\n")
L.append(f"15 phrases ({', '.join(PHR)}); {meta['n_held']} held-out pre-phrase positions "
         f"({meta['held_single_frac']:.0%} single-occurrence); {meta['n_generic_eval_positions']:,} held-out generic positions "
         f"from 500 FineWeb docs; prior sum {meta['prior_sum']:.1e}; a real W_U row has generic logit std {meta['wu_row_logit_std_median']:.2f}. "
         f"Single-token phrases: {', '.join(SINGLE)}.\n")

L.append("## Headline per head (mean over phrases; held-out)\n")
L.append("| scale | head | AUROC own vs generic | AUROC own vs first-token sibling | AUROC own vs category sibling | own top-10 | own median rank | rank / first-token rank (median log) | generic top-10 | mass / prior | ECE (prior-weighted) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for sc, S in scales.items():
    for h in ORDER:
        if h not in S["methods"]:
            continue
        m = S["methods"][h]
        ph = SINGLE if h == "real_wu" else None
        L.append(f"| {sc} | {LABEL[h]} | {mean_of(m,'auroc_generic',phrases=ph):.3f} | {mean_of(m,'auroc_first_sib',phrases=ph):.3f} | "
                 f"{mean_of(m,'auroc_cat_sib',phrases=ph):.3f} | {mean_of(m,'top10',phrases=ph):.2f} | "
                 f"{np.median([v['rank_median'] for p, v in pp(m).items() if not ph or p in ph]):.0f} | "
                 f"{mean_of(m,'log_rank_ratio_median',phrases=ph):+.2f} | {top_of(m,'generic_top10',ph):.4f} | "
                 f"{top_of(m,'mass_ratio',ph):.1f} | {top_of(m,'ece',ph):.2e} |")
L.append("")

L.append("## Copy sensitivity (scale 2400): single-occurrence vs copy contexts\n")
L.append("| head | AUROC single | AUROC copy | own top-10 single | own top-10 copy | rank single | rank copy |")
L.append("|---|---|---|---|---|---|---|")
S = scales["2400"]
for h in HEADS3:
    m = S["methods"][h]
    L.append(f"| {LABEL[h]} | {mean_of(m,'auroc_generic','single'):.3f} | {mean_of(m,'auroc_generic','copy'):.3f} | "
             f"{mean_of(m,'top10','single'):.2f} | {mean_of(m,'top10','copy'):.2f} | "
             f"{np.median([v['rank_median'] for v in pp(m,'single').values()]):.0f} | {np.median([v['rank_median'] for v in pp(m,'copy').values()]):.0f} |")
L.append("")

L.append("## Stability: three disjoint 150-context refits (mean ± std)\n")
L.append("| head | " + " | ".join(next(iter(R["stability"].values())).keys()) + " |")
L.append("|---|" + "---|" * len(next(iter(R["stability"].values()))))
for h, st in R["stability"].items():
    L.append(f"| {LABEL[h]} | " + " | ".join(f"{v['mean']:.4f} ± {v['std']:.4f}" for v in st.values()) + " |")
L.append("")

L.append("## First-token confound: heads fitted on the original 10 phrases, scored at sibling positions\n")
L.append("At pre-'New York'/'New Jersey' positions, does the 'New Hampshire' row fire? (top10_at_sib = fraction of sibling positions where it enters the top-10; auroc_first_sib = own vs sibling positions.)\n")
L.append("| head | phrase | AUROC own vs first-token sibling | top-10 rate at sibling positions | AUROC own vs category | own top-10 |")
L.append("|---|---|---|---|---|---|")
for h, d in R["orig10"].items():
    for p, v in d.items():
        f = lambda x: "—" if x is None else f"{x:.3f}"
        L.append(f"| {LABEL[h]} | {p} | {f(v['auroc_first_sib'])} | {f(v['top10_at_sib'])} | {f(v['auroc_cat_sib'])} | {f(v['top10'])} |")
L.append("")

L.append("## J-lens vs logit lens: layer at which the token enters the top-10 (scale 2400, all held-out contexts)\n")
L.append("Phrase token under each head vs the phrase's real first token (the standard readout). Fraction of held-out contexts where the token is in the extended top-10 at that layer.\n")
L.append("| lens | token | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
L.append("|---|---|" + "---|" * len(LAYERS))
for lens in ("jlens", "logit_lens"):
    prof = S["methods"]["lr"]["lens_profile"][lens]
    L.append(f"| {lens} | real first token | " + " | ".join(f"{x:.2f}" for x in prof["first_tok_top10_by_layer"]) + " |")
    for h in HEADS3:
        prof = S["methods"][h]["lens_profile"][lens]
        L.append(f"| {lens} | phrase token, {LABEL[h]} | " + " | ".join(f"{x:.2f}" for x in prof["phrase_top10_by_layer"]) + " |")
    for h in ("lda", "lr_nobias"):
        prof = S["methods"][h]["lens_profile"][lens]
        if "phrase_only_by_layer" in prof:
            L.append(f"| {lens} | phrase in top-10 while first token is not, {LABEL[h]} | " + " | ".join(f"{x:.2f}" for x in prof["phrase_only_by_layer"]) + " |")
L.append("")
L.append("Median earliest top-10 layer per phrase (99 = never), J-lens:\n")
L.append("| phrase | real first token | proj | lda | lr (bias) | lr no-bias | lr no-bias: never |")
L.append("|---|---|---|---|---|---|---|")
for p in PHR:
    pj = {h: S["methods"][h]["lens_profile"]["jlens"] for h in HEADS3}
    L.append(f"| {p} | {pj['lr']['earliest_top10_layer_median_first_tok'][p]} | {pj['proj']['earliest_top10_layer_median'][p]} | "
             f"{pj['lda']['earliest_top10_layer_median'][p]} | {pj['lr']['earliest_top10_layer_median'][p]} | "
             f"{pj['lr_nobias']['earliest_top10_layer_median'][p]} | {pj['lr_nobias']['never_top10_frac'][p]:.2f} |")
L.append("")

L.append("## Per-phrase detail at scale 2400\n")
for h in HEADS3:
    m = S["methods"][h]
    L.append(f"### {LABEL[h]}\n")
    L.append("| phrase | AUROC generic | AUROC first-sib | AUROC cat-sib | own rank | first-tok rank | own top-10 | generic top-10 | mass/prior | ECE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p, v in m["per_phrase"].items():
        a = v["all"]
        fs = a.get("auroc_first_sib"); fs = "—" if fs is None else f"{fs:.3f}"
        L.append(f"| {p} | {a['auroc_generic']:.3f} | {fs} | {a['auroc_cat_sib']:.3f} | {a['rank_median']:.0f} | {a['first_rank_median']:.0f} | "
                 f"{a['top10']:.2f} | {v['generic_top10']:.4f} | {v['mass_ratio']:.1f} | {v['ece']:.1e} |")
    L.append("")

L.append("## Cross-phrase confusion at scale 2400 (rows: positions before phrase; cols: top-10 rate of each token)\n")
short = [p.split()[-1][:5] if p != "John Quincy Adams" else "JQA" for p in PHR]
for h in HEADS3:
    C = S["methods"][h]["confusion"]
    L.append(f"**{LABEL[h]}**\n")
    L.append("| | " + " | ".join(short) + " |")
    L.append("|---|" + "---|" * len(PHR))
    for i, p in enumerate(PHR):
        L.append(f"| {p} | " + " | ".join(f"{C[i][j]:.2f}" for j in range(len(PHR))) + " |")
    L.append("")

L.append("## Hand prompts (scale 2400): phrase tokens in the top-10 per layer\n")
L.append("| prompt | " + " | ".join(LABEL[h] for h in HEADS3) + " |")
L.append("|---|" + "---|" * len(HEADS3))
for i, pr in enumerate(S["methods"]["lr"]["prompts"]):
    cells = [" ".join(f"{r['layer']}:{r['n_new']}" for r in S["methods"][h]["prompts"][i]["rows"]) for h in HEADS3]
    L.append(f"| {pr['prompt'][:70]} | " + " | ".join(cells) + " |")
L.append("")
for i, pr in enumerate(S["methods"]["lr"]["prompts"]):
    L.append(f"**{pr['prompt']}**\n")
    for h in HEADS3:
        L.append(f"- {LABEL[h]}")
        for r in S["methods"][h]["prompts"][i]["rows"]:
            L.append(f"  - {r['layer']}: " + ", ".join(f"`{t}`" for t in r["top10"]))
    L.append("")

out_dir = os.path.dirname(os.path.abspath(sys.argv[1])) if len(sys.argv) > 1 else os.path.join(here, "results")
open(os.path.join(out_dir, "BENCH.md"), "w").write("\n".join(L))
print("\n".join(L[:60]))
