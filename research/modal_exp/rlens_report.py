"""Render results/rlens_attribution.json + results/rlens_phrases.json -> results/RLENS.md"""
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
rd = os.path.join(here, "results")
A = json.load(open(os.path.join(rd, "rlens_attribution.json")))
Pp = json.load(open(os.path.join(rd, "rlens_phrases.json"))) if os.path.exists(os.path.join(rd, "rlens_phrases.json")) else None
L = []
L.append("# R-lens on Qwen3.5-9B-Base: pass@10, direction ablation, and the phrase benchmark\n")
L.append("Matched J-lens / R-lens pair fitted with the authors' recipe (target layer 30, 25 pile-10k prompts, 128 tokens, skip 4); "
         "R-lens = same estimator with LN-rule, identity-rule and half-rule (β=0.5) in the backward pass; forward bit-identical.\n")

L.append("## pass@10 on two-hop questions: intermediate entity's first token in the lens top-10\n")
L.append(f"{A['summary']['n_questions']} questions; readout at the penultimate token (-2) and the last token (-1).\n")
n_layers = len(A["pass10"]["rlens"]["-2"])
L.append("| layer | R-lens @-2 | J-lens @-2 | logit @-2 | R-lens @-1 | J-lens @-1 | logit @-1 |")
L.append("|---|---|---|---|---|---|---|")
for l in range(n_layers):
    L.append(f"| L{l} | " + " | ".join(f"{A['pass10'][n][p][l]:.2f}" for p in ("-2", "-1") for n in ("rlens", "jlens", "logit")) + " |")
means = {n: {p: float(np.mean(A["pass10"][n][p])) for p in ("-2", "-1")} for n in ("rlens", "jlens", "logit")}
half = {n: {p: float(np.mean(A["pass10"][n][p][: n_layers // 2])) for p in ("-2", "-1")} for n in ("rlens", "jlens", "logit")}
L.append("")
L.append("| mean pass@10 | R-lens | J-lens | logit lens |\n|---|---|---|---|")
for p in ("-2", "-1"):
    L.append(f"| all layers @{p} | {means['rlens'][p]:.3f} | {means['jlens'][p]:.3f} | {means['logit'][p]:.3f} |")
    L.append(f"| first half of layers @{p} | {half['rlens'][p]:.3f} | {half['jlens'][p]:.3f} | {half['logit'][p]:.3f} |")
L.append("")

L.append("## Direction ablation (attribution)\n")
s = A["summary"]
L.append(f"Baseline: {s['n_questions']} questions, mean sampled accuracy {s['mean_base_acc_all']:.3f}; "
         f"{s['n_baseline_correct']} questions with baseline accuracy ≥ 0.5 are used below. "
         "For each lens the unit direction J_lᵀ(γ⊙W_U[t]) of the intermediate's first token t is projected out of the residual at the "
         "given positions and layers during the prompt forward pass; 8 samples per prompt at T=0.7; accuracy = answer alias in the continuation.\n")
L.append("| direction | layers | positions | acc before | acc after | relative loss | mean per-question relative loss |")
L.append("|---|---|---|---|---|---|---|")
for c in A["conditions"]:
    v = s[c]; name, lname, pos = c.split("|")
    L.append(f"| {name} | {lname} | {pos} | {v['mean_acc_before']:.3f} | {v['mean_acc_after']:.3f} | {v['relative_loss']:.3f} | {v['per_q_relative_loss_mean']:.3f} |")
L.append("")
L.append("Per question (baseline-correct only), relative loss for the four directions at all layers, position -2:\n")
L.append("| prompt | intermediate | base | R | J | logit | random |\n|---|---|---|---|---|---|---|")
for r in A["items"]:
    if r["base_acc"] < 0.5:
        continue
    cells = [f"{1 - r['abl'][f'{n}|all|(-2,)']['acc'] / r['base_acc']:+.2f}" for n in ("rlens", "jlens", "logit", "random")]
    L.append(f"| {r['prompt'][6:70]} | {r['intermediate']} | {r['base_acc']:.2f} | " + " | ".join(cells) + " |")
L.append("")

if Pp:
    L.append("## Phrase benchmark held-out profiles under the matched lenses\n")
    L.append(f"{Pp['n_held']} held-out pre-phrase contexts, 15 phrases. Fraction with the token in the extended top-10 per layer.\n")
    lay = Pp["layers"]
    L.append("| lens | token | " + " | ".join(f"L{l}" for l in lay) + " |")
    L.append("|---|---|" + "---|" * len(lay))
    for n in ("rlens", "jlens", "logit"):
        pr = Pp["profiles"][n]
        L.append(f"| {n} | real first token | " + " | ".join(f"{x:.2f}" for x in pr["first_top10_by_layer"]) + " |")
        L.append(f"| {n} | real first token, top-1 | " + " | ".join(f"{x:.2f}" for x in pr["first_top1_by_layer"]) + " |")
        L.append(f"| {n} | phrase token, LR no bias | " + " | ".join(f"{x:.2f}" for x in pr["phrase_lr_top10_by_layer"]) + " |")
        L.append(f"| {n} | phrase token, Brennan proj | " + " | ".join(f"{x:.2f}" for x in pr["phrase_proj_top10_by_layer"]) + " |")
        L.append(f"| {n} | phrase in top-10, first token not (LR) | " + " | ".join(f"{x:.2f}" for x in pr["phrase_lr_only_by_layer"]) + " |")
    L.append("")
    L.append("Per-phrase first-token top-10 rate at L16 / L24 (R-lens vs J-lens):\n")
    L.append("| phrase | R L16 | J L16 | R L24 | J L24 | LR-phrase R L24 | LR-phrase J L24 |\n|---|---|---|---|---|---|---|")
    i16, i24 = lay.index(16), lay.index(24)
    for p in Pp["profiles"]["rlens"]["per_phrase_first_top10"]:
        r, j = Pp["profiles"]["rlens"], Pp["profiles"]["jlens"]
        L.append(f"| {p} | {r['per_phrase_first_top10'][p][i16]:.2f} | {j['per_phrase_first_top10'][p][i16]:.2f} | "
                 f"{r['per_phrase_first_top10'][p][i24]:.2f} | {j['per_phrase_first_top10'][p][i24]:.2f} | "
                 f"{r['per_phrase_lr_top10'][p][i24]:.2f} | {j['per_phrase_lr_top10'][p][i24]:.2f} |")
    L.append("")

open(os.path.join(rd, "RLENS.md"), "w").write("\n".join(L))
print("\n".join(L[:12]))
