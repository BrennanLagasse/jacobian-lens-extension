"""Generalisation-study tables and figure data for main.tex (and HTML snippets), from results/gen_*.json."""
import glob
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(here, "..", "modal_exp", "results")
import importlib.util
spec = importlib.util.spec_from_file_location("mt", os.path.join(here, "make_tables.py"))
# reuse the Table class without re-running make_tables' module body: copy the class source
src = open(os.path.join(here, "make_tables.py")).read()
cls_src = src[src.index("class Table:"):src.index("def f(x")]
ns = {"os": os, "here": here, "HTML": {}}
exec(cls_src, ns)
Table = ns["Table"]; HTML = ns["HTML"]

AX = {}
for f_ in sorted(glob.glob(os.path.join(R, "gen_*.json"))):
    name = os.path.basename(f_)[4:-5]
    if name.startswith("summary") or name.endswith("_v1"):
        continue
    AX[name] = json.load(open(f_))
order = [a for a in ("nouns", "verbs", "abstract", "events", "xling") if a in AX]
joint = next((a for a in AX if a.startswith("joint")), None)
LAB = {"nouns": "compositional nouns", "verbs": "verb phrases / idioms", "abstract": "abstract concepts", "events": "events, dates, years", "xling": "cross-lingual"}


def f(x, d=3):
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def mean_(per, fn):
    v = [fn(x) for x in per.values()]; v = [y for y in v if y is not None]
    return float(np.mean(v)) if v else None


def med_(per, fn):
    v = [fn(x) for x in per.values()]; v = [y for y in v if y is not None]
    return float(np.median(v)) if v else None


g = lambda v, *ks: (lambda d: d)(v) if False else _get(v, ks)


def _get(v, ks):
    for k in ks:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return v


# headline
rows = []
for a in order + ([joint] if joint else []):
    per = AX[a]["heads"]["lr_nobias"]["per_phrase"]
    lab = LAB.get(a, "joint, 136 rows in one softmax")
    rows.append([lab, str(len(per)), f(mean_(per, lambda v: v["vs_generic"]["auroc"])), f(mean_(per, lambda v: v["vs_generic"].get("tpr@fpr0.0001")), 2),
                 f(mean_(per, lambda v: v["vs_generic"].get("tpr@fpr0.001")), 2), f(mean_(per, lambda v: v["vs_generic"].get("tpr@fpr0.01")), 2),
                 f(mean_(per, lambda v: _get(v, ("vs_same_language_other_concepts", "auroc")) if a == "xling" else _get(v, ("vs_sibling", "auroc")))),
                 f(mean_(per, lambda v: _get(v, ("vs_same_language_other_concepts", "tpr@fpr0.01")) if a == "xling" else _get(v, ("vs_sibling", "tpr@fpr0.01"))), 2),
                 f(mean_(per, lambda v: v.get("mass_ratio")), 1), f(med_(per, lambda v: v["lens"]["rlens"]["earliest_layer_tpr50"]), 0)])
Table("gen_headline", "Generalisation study: the bias-free logistic row per axis, means over phrases on held-out data. TPR at FPR $10^{-4}$ / $10^{-3}$ / $10^{-2}$ against generic text; sibling = first-token or category sibling for the English axes and, for the cross-lingual axis, the other eleven concepts in the same language. Earliest layer = median over phrases of the first R-lens layer at which TPR at FPR $10^{-2}$ reaches 0.5 (the full distribution is Table~\\ref{tab:gen_earliest}).",
      ["axis", "$n$", "AUROC", "TPR@$10^{-4}$", "@$10^{-3}$", "@$10^{-2}$", "AUROC vs sib.", "TPR@$10^{-2}$ vs sib.", "mass/prior", "earliest layer"],
      "lrrrrrrrrr", rows, small=True).emit()

# per-layer TPR@1e-2 under R-lens, plus first-token top-10
lay = AX[order[0]]["lens_layers"]
rows = []
for a in order:
    per = AX[a]["heads"]["lr_nobias"]["per_phrase"]
    tpr = np.array([v["lens"]["rlens"]["tpr@fpr1e-2_by_layer"] for v in per.values()]).mean(0)
    tprJ = np.array([v["lens"]["jlens"]["tpr@fpr1e-2_by_layer"] for v in per.values()]).mean(0)
    ft = np.array([v["lens"]["rlens"]["first_tok_top10_by_layer"] for v in per.values()]).mean(0)
    rows.append([LAB[a], "phrase row, R-lens"] + [f(x, 2) for x in tpr])
    rows.append(["", "phrase row, J-lens"] + [f(x, 2) for x in tprJ])
    rows.append(["", "real first token in top-10, R-lens"] + [f(x, 2) for x in ft])
    rows.append("hline")
rows.pop()
Table("gen_layers", "When the phrase pops up: TPR at FPR $10^{-2}$ of the phrase row by lens layer (thresholds from generic lens states of the same lens and layer), next to the rate at which the phrase's real first token is in the R-lens top-10 at the same positions.",
      ["axis", "readout"] + [f"L{l}" for l in lay], "ll" + "r" * len(lay), rows).emit()

# earliest-layer histogram
allk = [8, 12, 16, 20, 24, 28, 99]
rows = []
for a in order:
    per = AX[a]["heads"]["lr_nobias"]["per_phrase"]
    e = [v["lens"]["rlens"]["earliest_layer_tpr50"] for v in per.values()]
    rows.append([LAB[a]] + [str(sum(1 for x in e if x == k)) for k in allk])
Table("gen_earliest", "Distribution over phrases of the earliest R-lens layer at which TPR at FPR $10^{-2}$ reaches 0.5 (99 = not by layer 28).",
      ["axis"] + [f"L{k}" if k != 99 else "never" for k in allk], "l" + "r" * len(allk), rows).emit()

# sibling extremes: best and worst 6 pairs across English axes
pairs = []
for a in ("nouns", "verbs", "abstract", "events"):
    if a not in AX:
        continue
    for k, v in AX[a]["heads"]["lr_nobias"]["per_phrase"].items():
        s = v.get("vs_sibling")
        if s:
            pairs.append((s["auroc"], s.get("tpr@fpr0.01"), k, ", ".join(v.get("siblings", [])), a))
pairs.sort()
rows = [[k, sib, LAB[a], f(au), f(t, 2)] for au, t, k, sib, a in pairs[:8]] + ["hline"] + [[k, sib, LAB[a], f(au), f(t, 2)] for au, t, k, sib, a in pairs[-8:]]
Table("gen_siblings", "Sibling separation, the eight hardest and the eight easiest phrases across the English axes (AUROC and TPR at FPR $10^{-2}$ of the phrase logit at its own pre-phrase positions against its siblings' pre-phrase positions).",
      ["phrase", "siblings", "axis", "AUROC", "TPR@$10^{-2}$"], "lllrr", rows).emit()

# joint vs separate
if joint:
    Hj = AX[joint]["heads"]["lr_nobias"]["per_phrase"]
    rows = []
    for a in ("nouns", "verbs", "abstract", "events"):
        Hs = AX[a]["heads"]["lr_nobias"]["per_phrase"]
        pr = [(Hs[k], Hj[f"{k}@{a}"]) for k in Hs if f"{k}@{a}" in Hj]
        m = lambda fn: (np.mean([fn(s) for s, _ in pr if fn(s) is not None]), np.mean([fn(t) for _, t in pr if fn(t) is not None]))
        au = m(lambda v: v["vs_generic"]["auroc"]); t3 = m(lambda v: v["vs_generic"].get("tpr@fpr0.001")); sa = m(lambda v: _get(v, ("vs_sibling", "auroc")))
        ms = m(lambda v: v.get("mass_ratio")); er = m(lambda v: v["lens"]["rlens"]["earliest_layer_tpr50"])
        rows.append([LAB[a], str(len(pr)), f"{au[0]:.3f} $\\to$ {au[1]:.3f}", f"{t3[0]:.2f} $\\to$ {t3[1]:.2f}", f"{sa[0]:.3f} $\\to$ {sa[1]:.3f}", f"{ms[0]:.1f} $\\to$ {ms[1]:.1f}", f"{er[0]:.0f} $\\to$ {er[1]:.0f}"])
    Hjm = AX[joint]["heads"]["lr_nobias"]
    Table("gen_joint", f"Separate fits (one axis per softmax) $\\to$ one joint fit of all 136 rows, per axis. Joint-head row redundancy: mean pairwise cosine {Hjm['row_cos_mean']:.3f}, participation ratio {Hjm['participation_ratio']:.0f} of 136.",
          ["axis", "phrases", "AUROC vs generic", "TPR@$10^{-3}$", "AUROC vs siblings", "mass/prior", "earliest layer (mean)"], "lrrrrrr", rows).emit()

# cross-lingual transfer matrix (within-language control)
if "xling" in AX:
    X = AX["xling"]["heads"]["lr_nobias"]["xfer"]
    langs = ["eng", "spa", "fra", "zho"]; LN = {"eng": "English", "spa": "Spanish", "fra": "French", "zho": "Chinese"}
    M = {}
    for k, v in X.items():
        c, pair = k.split("|"); a, b = pair.split("->"); M.setdefault((a, b), []).append(v)
    rows = []
    for a in langs:
        rows.append([LN[a] + " row"] + [("---" if a == b else f"{np.mean([v['auroc_within_lang'] for v in M[(a, b)]]):.3f} / {np.mean([v.get('tpr@fpr1e-2_within_lang') or 0 for v in M[(a, b)]]):.2f}") for b in langs])
    Table("gen_xling", "Cross-lingual transfer of the bias-free rows, 12 concepts. A row fitted on language $a$ is scored at language $b$'s pre-phrase positions of the same concept against $b$'s pre-phrase positions of the other eleven concepts (language-matched negatives): AUROC / TPR at FPR $10^{-2}$, mean over concepts. The diagonal (own language, other concepts as negatives) is in Table~\\ref{tab:gen_headline}.",
          ["row fitted on"] + [f"scored in {LN[b]}" for b in langs], "l" + "r" * 4, rows).emit()
    # per concept eng->x within-language AUROC
    rows = []
    for c in ["ice cream", "climate change", "credit card", "human rights", "solar system", "black hole", "prime minister", "stock market", "civil war", "middle class", "real estate", "health insurance"]:
        rows.append([c] + [f"{X[f'{c}|eng->{b}']['auroc_within_lang']:.2f}" if f"{c}|eng->{b}" in X else "---" for b in ("spa", "fra", "zho")]
                    + [f"{X[f'{c}|{b}->eng']['auroc_within_lang']:.2f}" if f"{c}|{b}->eng" in X else "---" for b in ("spa", "fra", "zho")])
    Table("gen_xling_concepts", "Cross-lingual transfer per concept, language-matched AUROC: the English row scored in Spanish / French / Chinese, and each language's row scored in English.",
          ["concept", "en$\\to$es", "en$\\to$fr", "en$\\to$zh", "es$\\to$en", "fr$\\to$en", "zh$\\to$en"], "lrrrrrr", rows).emit()

# figure data: per-layer TPR curves and first-token top-10
with open(os.path.join(here, "figures", "gen_layers.dat"), "w") as fh:
    cols = ["layer"] + [f"{a}_R" for a in order] + [f"{a}_first" for a in order]
    fh.write(" ".join(cols) + "\n")
    arr = {a: np.array([v["lens"]["rlens"]["tpr@fpr1e-2_by_layer"] for v in AX[a]["heads"]["lr_nobias"]["per_phrase"].values()]).mean(0) for a in order}
    arrf = {a: np.array([v["lens"]["rlens"]["first_tok_top10_by_layer"] for v in AX[a]["heads"]["lr_nobias"]["per_phrase"].values()]).mean(0) for a in order}
    for i, l in enumerate(lay):
        fh.write(f"{l} " + " ".join(f"{arr[a][i]:.4f}" for a in order) + " " + " ".join(f"{arrf[a][i]:.4f}" for a in order) + "\n")
# position profiles (verbs, abstract): mean over phrases of TPR at generic threshold by offset, for row and probe
with open(os.path.join(here, "figures", "gen_positions.dat"), "w") as fh:
    fh.write("offset " + " ".join(f"{a}_{h}" for a in ("nouns", "verbs", "abstract", "events") if a in AX for h in ("row", "probe")) + "\n")
    for off in range(-1, 5):
        vals = []
        for a in ("nouns", "verbs", "abstract", "events"):
            if a not in AX:
                continue
            for h in ("lr_nobias", "probe"):
                per = AX[a]["heads"][h]["per_phrase"]
                xs = [v["position_profile"].get(str(off)) for v in per.values()]; xs = [x for x in xs if x is not None]
                vals.append(np.mean(xs) if xs else float("nan"))
        fh.write(f"{off} " + " ".join(f"{x:.4f}" for x in vals) + "\n")
json.dump(HTML, open(os.path.join(here, "gen_tables_html.json"), "w"), indent=1)
print("gen tables:", sorted(HTML))
