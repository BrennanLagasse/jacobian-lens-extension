"""Cross-axis summary of the generalisation study: results/gen_*.json -> results/GEN_SUMMARY.md + gen_summary.json"""
import glob
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(here, "results", "gen_*.json")))
files = [f for f in files if not os.path.basename(f).startswith("gen_summary")]
AX = {}
for f in files:
    R = json.load(open(f))
    AX[os.path.basename(f)[4:-5]] = R
order = [a for a in ("nouns", "verbs", "abstract", "events", "xling") if a in AX] + [a for a in AX if a.startswith("joint")]
L = ["# Generalisation study: cross-axis summary\n"]
L.append("Bias-free logistic rows (extended softmax, positives at the token before the phrase) unless stated. Means over phrases of each axis; 200 held-out contexts per phrase; "
         "generic negatives = 176,763 held-out FineWeb positions (final layer) and 18,622 positions with lens states (per-layer thresholds).\n")


def g(v, *ks):
    for k in ks:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return v


def mean_(per, fn):
    vals = [fn(v) for v in per.values()]; vals = [x for x in vals if x is not None]
    return float(np.mean(vals)) if vals else float("nan")


def med_(per, fn):
    vals = [fn(v) for v in per.values()]; vals = [x for x in vals if x is not None]
    return float(np.median(vals)) if vals else float("nan")


summary = {}
for head in ("lr_nobias", "probe"):
    L.append(f"## {head}\n")
    L.append("| axis | phrases | AUROC gen | TPR@1e-4 | TPR@1e-3 | TPR@1e-2 | FPR@TPR.9 | AUROC sib | TPR@1e-2 sib | inside TPR | own top-10 | 1st-tok top-10 | mass/prior | earliest R (median) | earliest J | earliest logit | never (R) | row cos |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in order:
        H = AX[a]["heads"].get(head)
        if not H:
            continue
        per = H["per_phrase"]
        eR = [v["lens"]["rlens"]["earliest_layer_tpr50"] for v in per.values()]
        row = {"n": len(per), "auroc": mean_(per, lambda v: v["vs_generic"]["auroc"]), "tpr1e4": mean_(per, lambda v: v["vs_generic"].get("tpr@fpr0.0001")),
               "tpr1e3": mean_(per, lambda v: v["vs_generic"].get("tpr@fpr0.001")), "tpr1e2": mean_(per, lambda v: v["vs_generic"].get("tpr@fpr0.01")),
               "fpr_tpr9": mean_(per, lambda v: v["vs_generic"].get("fpr@tpr0.9")), "sib_auroc": mean_(per, lambda v: g(v, "vs_sibling", "auroc")),
               "sib_tpr1e2": mean_(per, lambda v: g(v, "vs_sibling", "tpr@fpr0.01")), "inside": mean_(per, lambda v: v.get("inside_tpr@fpr1e-2")),
               "own_top10": mean_(per, lambda v: v.get("own_top10")), "first_top10": mean_(per, lambda v: v.get("first_tok_top10")), "mass": mean_(per, lambda v: v.get("mass_ratio")),
               "earliest_R": float(np.median(eR)), "earliest_J": med_(per, lambda v: v["lens"]["jlens"]["earliest_layer_tpr50"]), "earliest_logit": med_(per, lambda v: v["lens"]["logit"]["earliest_layer_tpr50"]),
               "never_R": float(np.mean([e == 99 for e in eR])), "row_cos": H["row_cos_mean"],
               "tpr_by_layer_R": np.array([v["lens"]["rlens"]["tpr@fpr1e-2_by_layer"] for v in per.values()]).mean(0).tolist(),
               "tpr_by_layer_J": np.array([v["lens"]["jlens"]["tpr@fpr1e-2_by_layer"] for v in per.values()]).mean(0).tolist(),
               "tpr_by_layer_logit": np.array([v["lens"]["logit"]["tpr@fpr1e-2_by_layer"] for v in per.values()]).mean(0).tolist(),
               "top10_by_layer_R": np.array([v["lens"]["rlens"]["top10_by_layer"] for v in per.values()]).mean(0).tolist(),
               "top10_by_layer_J": np.array([v["lens"]["jlens"]["top10_by_layer"] for v in per.values()]).mean(0).tolist(),
               "first_top10_by_layer_R": np.array([v["lens"]["rlens"]["first_tok_top10_by_layer"] for v in per.values()]).mean(0).tolist(),
               "earliest_hist": {str(k): int(sum(1 for e in eR if e == k)) for k in sorted(set(eR))}}
        summary.setdefault(head, {})[a] = row
        f2 = lambda x, d=3: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
        L.append(f"| {a} | {row['n']} | {f2(row['auroc'])} | {f2(row['tpr1e4'],2)} | {f2(row['tpr1e3'],2)} | {f2(row['tpr1e2'],2)} | {f2(row['fpr_tpr9'],3)} | {f2(row['sib_auroc'])} | {f2(row['sib_tpr1e2'],2)} | "
                 f"{f2(row['inside'],2)} | {f2(row['own_top10'],2)} | {f2(row['first_top10'],2)} | {f2(row['mass'],1)} | {row['earliest_R']:.0f} | {row['earliest_J']:.0f} | {row['earliest_logit']:.0f} | {row['never_R']:.2f} | {row['row_cos']:.3f} |")
    L.append("")
    lay = AX[order[0]]["lens_layers"]
    L.append(f"Per-layer TPR at FPR 1e-2 under the R-lens (mean over phrases): layers {lay}\n")
    L.append("| axis | " + " | ".join(f"L{l}" for l in lay) + " | first token top-10 (R) at " + " / ".join(f"L{l}" for l in lay) + " |")
    L.append("|---|" + "---|" * len(lay) + "---|")
    for a in order:
        if a in summary.get(head, {}):
            r = summary[head][a]
            L.append(f"| {a} | " + " | ".join(f"{x:.2f}" for x in r["tpr_by_layer_R"]) + " | " + " / ".join(f"{x:.2f}" for x in r["first_top10_by_layer_R"]) + " |")
    L.append("")
    L.append("Distribution of the earliest R-lens layer with TPR@1e-2 ≥ 0.5 (count of phrases; 99 = never):\n")
    layers_all = sorted({int(k) for a in summary.get(head, {}) for k in summary[head][a]["earliest_hist"]})
    L.append("| axis | " + " | ".join(str(k) for k in layers_all) + " |\n|---|" + "---|" * len(layers_all))
    for a in order:
        if a in summary.get(head, {}):
            h = summary[head][a]["earliest_hist"]
            L.append(f"| {a} | " + " | ".join(str(h.get(str(k), 0)) for k in layers_all) + " |")
    L.append("")

# joint vs separate: per-phrase deltas
joints = [a for a in AX if a.startswith("joint")]
for j in joints:
    Hj = AX[j]["heads"]["lr_nobias"]["per_phrase"]
    L.append(f"## Joint fit ({j}) vs separate fits, bias-free row, per axis\n")
    L.append("| axis | phrases | AUROC gen sep → joint | TPR@1e-3 sep → joint | AUROC sib sep → joint | TPR@1e-2 sib sep → joint | mass/prior sep → joint | earliest R sep → joint |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in [x for x in order if not x.startswith("joint")]:
        if a not in AX:
            continue
        Hs = AX[a]["heads"]["lr_nobias"]["per_phrase"]
        pairs = [(Hs[k], Hj.get(f"{k}@{a}")) for k in Hs if Hj.get(f"{k}@{a}")]
        if not pairs:
            continue
        m = lambda fn: (float(np.mean([fn(s) for s, _ in pairs if fn(s) is not None])), float(np.mean([fn(t) for _, t in pairs if fn(t) is not None])))
        au = m(lambda v: v["vs_generic"]["auroc"]); t3 = m(lambda v: v["vs_generic"].get("tpr@fpr0.001")); sa = m(lambda v: g(v, "vs_sibling", "auroc"))
        st = m(lambda v: g(v, "vs_sibling", "tpr@fpr0.01")); ms = m(lambda v: v.get("mass_ratio")); eR = m(lambda v: v["lens"]["rlens"]["earliest_layer_tpr50"])
        L.append(f"| {a} | {len(pairs)} | {au[0]:.3f} → {au[1]:.3f} | {t3[0]:.2f} → {t3[1]:.2f} | {sa[0]:.3f} → {sa[1]:.3f} | {st[0]:.2f} → {st[1]:.2f} | {ms[0]:.1f} → {ms[1]:.1f} | {eR[0]:.0f} → {eR[1]:.0f} |")
    L.append(f"\nJoint head row redundancy: mean cosine {AX[j]['heads']['lr_nobias']['row_cos_mean']:.3f}, participation ratio {AX[j]['heads']['lr_nobias']['participation_ratio']:.1f} of {len(Hj)}.\n")

open(os.path.join(here, "results", "GEN_SUMMARY.md"), "w").write("\n".join(L))
json.dump(summary, open(os.path.join(here, "results", "gen_summary.json"), "w"), indent=1)
print("\n".join(L[:30]))
