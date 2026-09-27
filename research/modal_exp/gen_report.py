"""Render results/gen_<axis>.json -> results/GEN_<axis>.md"""
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "nouns"
R = json.load(open(os.path.join(here, "results", f"gen_{tag}.json")))
lay = R["lens_layers"]
L = []
fmt = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"


def mean_(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


L.append(f"# Generalisation axis: {tag}\n")
L.append(f"{len(R['keys'])} phrases; {R['n_generic_eval']:,} held-out generic positions for the final-layer ROC; "
         f"{R['n_generic_lens']:,} generic positions with lens states for the per-layer thresholds. Rows fitted on up to 1,200 contexts per phrase; 200 held out.\n")
for hname, H in R["heads"].items():
    per = H["per_phrase"]
    L.append(f"## Head: {hname}\n")
    L.append(f"Row redundancy: mean pairwise cosine {H['row_cos_mean']:.3f}, max {H['row_cos_max']:.3f}, participation ratio {H['participation_ratio']:.1f} of {len(per)}; mean row norm {H['row_norm_mean']:.2f}.\n")
    L.append("### Summary over phrases\n")
    L.append("| metric | mean | median | min | max |\n|---|---|---|---|---|")
    for label, get in (("AUROC vs generic", lambda v: v["vs_generic"]["auroc"]), ("TPR @ FPR 1e-4 (generic)", lambda v: v["vs_generic"].get("tpr@fpr0.0001")),
                       ("TPR @ FPR 1e-3 (generic)", lambda v: v["vs_generic"].get("tpr@fpr0.001")), ("TPR @ FPR 1e-2 (generic)", lambda v: v["vs_generic"].get("tpr@fpr0.01")),
                       ("FPR @ TPR 0.5 (generic)", lambda v: v["vs_generic"].get("fpr@tpr0.5")), ("FPR @ TPR 0.9 (generic)", lambda v: v["vs_generic"].get("fpr@tpr0.9")),
                       ("AUROC vs siblings", lambda v: v.get("vs_sibling", {}).get("auroc")), ("TPR @ FPR 1e-2 (siblings)", lambda v: v.get("vs_sibling", {}).get("tpr@fpr0.01")),
                       ("inside-phrase TPR @ generic FPR 1e-2", lambda v: v.get("inside_tpr@fpr1e-2")),
                       ("own top-10 (final layer)", lambda v: v.get("own_top10")), ("first-token top-10", lambda v: v.get("first_tok_top10")),
                       ("generic top-10", lambda v: v.get("generic_top10")), ("mass / prior", lambda v: v.get("mass_ratio")),
                       ("earliest R-lens layer with TPR@1e-2 ≥ 0.5", lambda v: v["lens"]["rlens"]["earliest_layer_tpr50"]),
                       ("earliest J-lens layer with TPR@1e-2 ≥ 0.5", lambda v: v["lens"]["jlens"]["earliest_layer_tpr50"]),
                       ("earliest logit-lens layer with TPR@1e-2 ≥ 0.5", lambda v: v["lens"]["logit"]["earliest_layer_tpr50"])):
        vals = [get(v) for v in per.values()]; vals = [x for x in vals if x is not None]
        if vals:
            L.append(f"| {label} | {np.mean(vals):.3f} | {np.median(vals):.3f} | {np.min(vals):.3f} | {np.max(vals):.3f} |")
    L.append("")
    L.append("### Per-layer TPR at FPR 1e-2 (mean over phrases)\n")
    L.append("| lens | " + " | ".join(f"L{l}" for l in lay) + " |\n|---|" + "---|" * len(lay))
    for name in ("rlens", "jlens", "logit"):
        arr = np.array([v["lens"][name]["tpr@fpr1e-2_by_layer"] for v in per.values()])
        L.append(f"| {name} phrase TPR@1e-2 | " + " | ".join(f"{x:.2f}" for x in arr.mean(0)) + " |")
    for name in ("rlens", "jlens", "logit"):
        arr = np.array([v["lens"][name]["top10_by_layer"] for v in per.values()])
        L.append(f"| {name} phrase top-10 | " + " | ".join(f"{x:.2f}" for x in arr.mean(0)) + " |")
    arr = np.array([v["lens"]["rlens"]["first_tok_top10_by_layer"] for v in per.values()])
    L.append(f"| rlens real first token top-10 | " + " | ".join(f"{x:.2f}" for x in arr.mean(0)) + " |")
    L.append("")
    L.append("### Position profile: TPR at the generic FPR 1e-2 threshold by position relative to the phrase (offset 0 = token before; 1.. = inside; last = after)\n")
    offs = sorted({int(o) for v in per.values() for o in v["position_profile"]})
    L.append("| phrase | " + " | ".join(str(o) for o in offs) + " |\n|---|" + "---|" * len(offs))
    for key, v in per.items():
        L.append(f"| {key} | " + " | ".join(fmt(v["position_profile"].get(str(o)), 2) for o in offs) + " |")
    L.append("")
    L.append("### Per phrase\n")
    L.append("| phrase | group | n | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.5 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R | earliest J | earliest logit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, v in per.items():
        g = v["vs_generic"]; s = v.get("vs_sibling", {})
        L.append(f"| {key} | {v['group']} | {v['n_held']} | {g['auroc']:.3f} | {fmt(g.get('tpr@fpr0.0001'),2)} | {fmt(g.get('tpr@fpr0.001'),2)} | {fmt(g.get('tpr@fpr0.01'),2)} | "
                 f"{fmt(g.get('fpr@tpr0.5'),4)} | {fmt(s.get('auroc'))} | {fmt(s.get('tpr@fpr0.01'),2)} | {fmt(v.get('inside_tpr@fpr1e-2'),2)} | {fmt(v.get('own_top10'),2)} | "
                 f"{fmt(v.get('first_tok_top10'),2)} | {fmt(v.get('mass_ratio'),1)} | {v['lens']['rlens']['earliest_layer_tpr50']} | {v['lens']['jlens']['earliest_layer_tpr50']} | {v['lens']['logit']['earliest_layer_tpr50']} |")
    L.append("")
    if H.get("xfer"):
        L.append("### Cross-lingual transfer: row trained on one language scored at another language's pre-phrase positions (TPR at the row's generic FPR 1e-2 threshold)\n")
        L.append("| concept | pair | TPR@1e-2 | AUROC vs generic |\n|---|---|---|---|")
        for k, v in H["xfer"].items():
            c, pair = k.split("|")
            L.append(f"| {c} | {pair} | {v['tpr@fpr1e-2']:.2f} | {v['auroc_vs_generic']:.3f} |")
        L.append("")
open(os.path.join(here, "results", f"GEN_{tag}.md"), "w").write("\n".join(L))
print("\n".join(L[:45]))
