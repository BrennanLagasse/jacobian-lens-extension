"""Build every table and figure-data file for the report from the four result JSONs.

Outputs (all derived, never hand-typed):
  tables/<name>.tex   booktabs tables for main.tex
  figures/<name>.dat  pgfplots data files
  tables_html.json    the same tables as HTML strings, for report.html
"""
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(here, "..", "modal_exp", "results")
E1 = json.load(open(os.path.join(R, "results.json")))            # experiment 1: calibration vs scale, 10 phrases
E1_ADAM = json.load(open(os.path.join(R, "results_adam.json")))
E1_400 = json.load(open(os.path.join(R, "results_lbfgs_400docs.json")))
B = json.load(open(os.path.join(R, "bench.json")))                # benchmark, 15 phrases
AT = json.load(open(os.path.join(R, "rlens_attribution.json")))  # R-lens pass@10 + ablation
RP = json.load(open(os.path.join(R, "rlens_phrases.json")))      # phrase profiles under matched lenses
os.makedirs(os.path.join(here, "tables"), exist_ok=True)
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
HTML = {}

LABEL = {"proj": "Brennan (proj + unit)", "avgtok": "avg. $W_U$ rows", "lda": "LDA", "lda_wu": "LDA, $W_U$-scaled",
         "lr": "LR (bias, val.\\ pick)", "lr_1e-05": "LR bias, $\\lambda{=}10^{-5}$", "lr_0.0001": "LR bias, $\\lambda{=}10^{-4}$",
         "lr_0.001": "LR bias, $\\lambda{=}10^{-3}$", "lr_nobias": "LR no bias (val.\\ pick)", "lr_nobias_1e-06": "LR no bias, $\\lambda{=}10^{-6}$",
         "lr_nobias_1e-07": "LR no bias, $\\lambda{=}10^{-7}$", "lr_nobias_1e-08": "LR no bias, $\\lambda{=}10^{-8}$",
         "random": "random unit row", "real_wu": "real $W_U$ row"}
HLABEL = {k: v.replace("$W_U$", "W_U").replace("\\lambda{=}", "λ=").replace("$", "").replace("\\ ", " ").replace("^{-", "e-").replace("}", "")
          for k, v in LABEL.items()}


def tex_esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


class Table:
    def __init__(self, name, caption, header, align, rows, label=None, note=None, small=False):
        self.name, self.caption, self.header, self.align, self.rows = name, caption, header, align, rows
        self.label, self.note, self.small = label or f"tab:{name}", note, small

    def tex(self):
        L = ["\\begin{table}[t]", "\\centering", "\\small" if self.small else "\\footnotesize"]
        L.append(f"\\caption{{{self.caption}}}")
        L.append(f"\\label{{{self.label}}}")
        L.append("\\begin{adjustbox}{max width=\\textwidth}")
        L.append(f"\\begin{{tabular}}{{{self.align}}}")
        L.append("\\toprule")
        L.append(" & ".join(self.header) + " \\\\")
        L.append("\\midrule")
        for r in self.rows:
            if r == "hline":
                L.append("\\midrule"); continue
            L.append(" & ".join(r) + " \\\\")
        L.append("\\bottomrule")
        L.append("\\end{tabular}")
        L.append("\\end{adjustbox}")
        if self.note:
            L.append(f"\\par\\smallskip\\begin{{minipage}}{{\\textwidth}}\\footnotesize {self.note}\\end{{minipage}}")
        L.append("\\end{table}")
        return "\n".join(L)

    def html(self):
        def cell(c, tag="td"):
            c = c.replace("\\textbf{", "<b>").replace("$", "").replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
            c = c.replace("\\lambda{=}", "λ=").replace("\\ ", " ").replace("^{-", "e-").replace("}", "</b>" if "<b>" in c and c.count("}") == 1 else "")
            c = c.replace("W_U", "W<sub>U</sub>").replace("\\times", "×").replace("\\pm", "±")
            num = "num" if any(ch.isdigit() for ch in c) and len(c) < 24 else ""
            return f'<{tag} class="{num}">{c}</{tag}>'
        h = ['<div class="tbl"><table><thead><tr>' + "".join(cell(c, "th") for c in self.header) + "</tr></thead><tbody>"]
        for r in self.rows:
            if r == "hline":
                continue
            h.append("<tr>" + "".join(cell(c) for c in r) + "</tr>")
        h.append("</tbody></table></div>")
        cap = self.caption.replace("$W_U$", "W<sub>U</sub>").replace("\\lambda", "λ").replace("$", "").replace("\\%", "%").replace("\\ ", " ").replace("\\emph{", "<em>").replace("\\texttt{", "<code>")
        note = (self.note or "").replace("$W_U$", "W<sub>U</sub>").replace("$", "").replace("\\%", "%").replace("\\ ", " ")
        return f'<p class="cap"><b>{cap}</b> {note}</p>' + "\n".join(h)

    def emit(self):
        open(os.path.join(here, "tables", f"{self.name}.tex"), "w").write(self.tex())
        HTML[self.name] = self.html()


def f(x, d=3, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


def sci(x):
    return f"${x:.1e}$".replace("e-0", "e-").replace("e+0", "e+").replace("e", "\\times10^{") + "}$" if False else f"{x:.1e}".replace("e-0", "e-")


# ------------------------------------------------------------------ data tables
PH15 = B["meta"]["phrases"]
single = set(B["meta"]["single_token_phrases"])
rows = []
for p in PH15:
    s = B["meta"]["summary"][p]
    rows.append([p, "yes" if p in single else "no", str(s["unique"]), str(s["train"]), sci(B["meta"]["pi_true"][p])])
Table("phrases", "The 15 phrases. Unique mined contexts (FineWeb 10BT sample, $14.9$M documents), training contexts available at the 2{,}400 scale after holding out 200, and the estimated per-token prior $\\pi_p$ (occurrences per token in the scanned corpus).",
      ["phrase", "single token", "unique contexts", "train @2400", "$\\pi_p$"], "llrrr", rows,
      note="Phrases 11--15 (New York, New Jersey, John Quincy Adams, Massachusetts, Vermont) were added for the benchmark as confusables; the first ten are Brennan's.").emit()


# ------------------------------------------------------------------ experiment 1 headline
def agg(m, k):
    return float(np.mean([v[k] for v in m["per_phrase"].values()]))


rows = []
for sc in ("150", "600", "2400"):
    S = E1["scales"][sc]
    for h in ("proj", "avgtok", "lda", "lda_wu", "lr"):
        if h == "avgtok" and sc != "150":
            continue
        m = S["methods"][h]
        rows.append([sc if h == "proj" else "", LABEL[h], f(agg(m, "held_top10"), 2), f(np.median([v["held_rank_median"] for v in m["per_phrase"].values()]), 0),
                     f(agg(m, "generic_top10_rate"), 4), sci(m["generic_new_mass"]), f(agg(m, "auroc")), f"{agg(m,'generic_mean'):.1f} $\\pm$ {agg(m,'generic_std'):.1f}"])
    if sc != "2400":
        rows.append("hline")
Table("e1_headline", "Experiment 1 (Brennan's ten phrases): calibration of each head against the number of training contexts per phrase. Means over phrases on held-out data. The avg.\\ $W_U$ row head does not depend on the training set.",
      ["contexts", "head", "own top-10", "own rank", "generic top-10", "new-token mass", "AUROC", "generic logit $\\mu\\pm\\sigma$"], "llrrrrrr", rows,
      note=f"Own = held-out pre-phrase positions (100 per phrase). Generic = {E1['meta']['n_generic_eval_positions']:,} held-out FineWeb positions; true prior mass of the ten phrases together is {sum(E1['meta']['pi_true'].values()):.1e}. A real $W_U$ row has generic logit mean {E1['meta']['wu_row_logit_mean_median']:.1f}, std {E1['meta']['wu_row_logit_std_median']:.2f}.").emit()

# experiment 1: LR optimiser / negatives ablation (adam vs lbfgs-400 vs final)
rows = []
for name, D in (("minibatch Adam, 60 epochs, 400 generic docs", E1_ADAM), ("L-BFGS, 400 generic docs, $\\lambda{=}10^{-6}$", E1_400), ("L-BFGS, 2{,}000 generic docs, $\\lambda$ by held-out NLL", E1)):
    for sc in ("150", "600", "2400"):
        m = D["scales"][sc]["methods"]["lr"]
        rows.append([name if sc == "150" else "", sc, f(agg(m, "auroc")), f(agg(m, "held_top10"), 2), sci(m["generic_new_mass"]), f(agg(m, "generic_top10_rate"), 4)])
    rows.append("hline")
rows.pop()
Table("e1_lr_ablation", "Experiment 1: the logistic head depends on the optimiser and on the number of negative documents. Three passes over the same states.",
      ["fit", "contexts", "AUROC", "own top-10", "new-token mass", "generic top-10"], "llrrrr", rows).emit()

# experiment 1 confusion (2400) for proj / lda / lr, condensed to within-category vs cross-category means
PH10 = list(E1["meta"]["pi_true"])
cat10 = {"crime": PH10[:4], "state": PH10[4:7], "president": PH10[7:]}
rows = []
for h in ("proj", "lda", "lr"):
    C = np.array(E1["scales"]["2400"]["methods"][h]["confusion"])
    diag = np.mean([C[i][i] for i in range(10)])
    within, cross = [], []
    for i, p in enumerate(PH10):
        for j, q in enumerate(PH10):
            if i == j:
                continue
            same = any(p in c and q in c for c in cat10.values())
            (within if same else cross).append(C[i][j])
    rows.append([LABEL[h], f(diag, 2), f(np.mean(within), 2), f(np.mean(cross), 2)])
Table("e1_confusion", "Experiment 1, 2{,}400 contexts: cross-phrase confusion. Entry $(p,q)$ is the fraction of held-out positions before phrase $p$ at which the token for phrase $q$ enters the top-10; shown as the diagonal mean, the mean over same-category pairs (crime / state / president) and the mean over cross-category pairs.",
      ["head", "diagonal", "same category", "cross category"], "lrrr", rows).emit()

# ------------------------------------------------------------------ benchmark
S24 = B["scales"]["2400"]
S150 = B["scales"]["150"]


def pp(m, bucket="all"):
    return {p: v[bucket] for p, v in m["per_phrase"].items() if bucket in v}


def mean_of(m, key, bucket="all", phrases=None):
    d = pp(m, bucket); ps = phrases or list(d)
    vals = [d[p][key] for p in ps if p in d and d[p].get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def top_of(m, key, phrases=None):
    ps = phrases or PH15
    return float(np.mean([m["per_phrase"][p][key] for p in ps]))


def bench_row(h, m, label=None, phrases=None):
    return [label or LABEL[h], f(mean_of(m, "auroc_generic", phrases=phrases)), f(mean_of(m, "auroc_first_sib", phrases=phrases)), f(mean_of(m, "auroc_cat_sib", phrases=phrases)),
            f(mean_of(m, "top10", phrases=phrases), 2), f(np.median([v["rank_median"] for p, v in pp(m).items() if not phrases or p in phrases]), 0),
            f(mean_of(m, "log_rank_ratio_median", phrases=phrases), 2, plus=True), f(top_of(m, "generic_top10", phrases), 4), f(top_of(m, "mass_ratio", phrases), 1), sci(top_of(m, "ece", phrases))]


hdr = ["head", "AUROC vs generic", "vs 1st-tok.\\ sibling", "vs category", "own top-10", "own rank", "$\\log(\\mathrm{rank}/\\mathrm{rank}_{\\mathrm{first}})$", "generic top-10", "mass/prior", "ECE"]
rows = [bench_row("random", S150["methods"]["random"]), bench_row("proj", S24["methods"]["proj"]), bench_row("avgtok", S150["methods"]["avgtok"]),
        bench_row("lda", S24["methods"]["lda"]), bench_row("lda_wu", S24["methods"]["lda_wu"]), "hline",
        bench_row("lr_1e-05", S24["methods"]["lr_1e-05"]), bench_row("lr_0.0001", S24["methods"]["lr_0.0001"]), bench_row("lr_0.001", S24["methods"]["lr_0.001"]), "hline",
        bench_row("lr_nobias_1e-06", S24["methods"]["lr_nobias_1e-06"]), bench_row("lr_nobias_1e-07", S24["methods"]["lr_nobias_1e-07"]), bench_row("lr_nobias_1e-08", S24["methods"]["lr_nobias_1e-08"])]
Table("bench_headline", "Benchmark, 2{,}400 contexts per phrase, means over the 15 phrases. The random row and the avg.\\ $W_U$ row do not depend on the training scale. The validation-loss picks were $\\lambda{=}10^{-4}$ (bias) and $10^{-6}$ (no bias).",
      hdr, "lrrrrrrrrr", rows,
      note="Sibling AUROC: phrase logit at own positions vs positions of phrases sharing the first token (defined for New Hampshire, New York, New Jersey, John Adams, John Quincy Adams). Rank ratio: median over own positions of $\\log(\\mathrm{rank}_{\\mathrm{phrase}}/\\mathrm{rank}_{\\mathrm{first\\ token}})$; $0$ means the row is exactly as confident as the model is about the phrase's first token. mass/prior: mean extended-softmax mass on the phrase token over generic text divided by $\\pi_p$. ECE is prior-weighted.").emit()

rows = []
for h in ("proj", "lda", "lr", "lr_nobias"):
    for sc in ("150", "600", "2400"):
        m = B["scales"][sc]["methods"][h]
        rows.append([LABEL[h] if sc == "150" else "", sc, f(mean_of(m, "auroc_generic")), f(mean_of(m, "auroc_first_sib")), f(mean_of(m, "top10"), 2), f(top_of(m, "generic_top10"), 4), f(top_of(m, "mass_ratio"), 1),
                     f"$10^{{{int(np.log10(B['scales'][sc]['lr_lam']))}}}$" if h == "lr" else (f"$10^{{{int(np.log10(B['scales'][sc]['lr_nobias_lam']))}}}$" if h == "lr_nobias" else "---")])
    rows.append("hline")
rows.pop()
Table("bench_scale", "Benchmark: the four main heads across training scale (means over 15 phrases).",
      ["head", "contexts", "AUROC vs generic", "vs 1st-tok.\\ sibling", "own top-10", "generic top-10", "mass/prior", "$\\lambda$ picked"], "llrrrrrr", rows).emit()

# sibling test
rows = []
for h, lab in (("proj", LABEL["proj"]), ("lda", "LDA"), ("lr", "LR with bias"), ("lr_nobias", "LR, no bias")):
    d = B["orig10"][h]
    for p in ("New Hampshire", "John Adams"):
        v = d[p]
        rows.append([lab if p == "New Hampshire" else "", p, f(v["auroc_first_sib"]), f(v["top10_at_sib"]), f(v["auroc_cat_sib"]), f(v["top10"], 2)])
Table("bench_sibling", "The first-token confound. Heads fitted on Brennan's ten phrases only (never shown New York, New Jersey or John Quincy Adams), scored at those phrases' pre-phrase positions.",
      ["head", "phrase", "AUROC own vs 1st-tok.\\ sibling", "top-10 rate at sibling positions", "AUROC own vs category", "own top-10"], "llrrrr", rows).emit()

# copy sensitivity + stability
rows = []
for h in ("proj", "lda", "lr", "lr_nobias"):
    m = S24["methods"][h]
    rows.append([LABEL[h], f(mean_of(m, "auroc_generic", "single")), f(mean_of(m, "auroc_generic", "copy")), f(mean_of(m, "top10", "single"), 2), f(mean_of(m, "top10", "copy"), 2)])
Table("bench_copy", f"Copy sensitivity at 2{{,}}400 contexts: held-out contexts in which the phrase occurs once ({B['meta']['held_single_frac']*100:.0f}\\% of them) vs contexts in which it already appeared earlier.",
      ["head", "AUROC single", "AUROC copy", "own top-10 single", "own top-10 copy"], "lrrrr", rows).emit()
rows = []
for h, v in B["stability"].items():
    rows.append([LABEL[h], f"{v['auroc_generic']['mean']:.4f} $\\pm$ {v['auroc_generic']['std']:.4f}", f"{v['top10']['mean']:.3f} $\\pm$ {v['top10']['std']:.3f}",
                 f"{v['generic_top10']['mean']:.4f} $\\pm$ {v['generic_top10']['std']:.4f}", f"{v['new_mass']['mean']:.4f} $\\pm$ {v['new_mass']['std']:.4f}"])
Table("bench_stability", "Stability: mean $\\pm$ std over three disjoint 150-context refits.",
      ["head", "AUROC", "own top-10", "generic top-10", "new-token mass"], "lrrrr", rows).emit()

# per-phrase, bias-free head
rows = []
prof_nb = S24["methods"]["lr_nobias"]["lens_profile"]["jlens"]
for p, v in S24["methods"]["lr_nobias"]["per_phrase"].items():
    a = v["all"]
    rows.append([p, f(a["auroc_generic"]), f(a.get("auroc_first_sib")), f(a["rank_median"], 0), f(a["first_rank_median"], 0), f(a["top10"], 2), f(v["mass_ratio"], 1),
                 str(prof_nb["earliest_top10_layer_median"][p]), f(prof_nb["never_top10_frac"][p], 2)])
Table("bench_perphrase", "Benchmark, bias-free logistic head at 2{,}400 contexts, per phrase. Earliest layer: median over the phrase's held-out contexts of the first layer at which the phrase token is in the J-lens top-10 (99 = never in more than half of them).",
      ["phrase", "AUROC vs generic", "vs 1st-tok.\\ sibling", "own rank", "1st-token rank", "own top-10", "mass/prior", "earliest top-10 layer", "never"], "lrrrrrrrr", rows).emit()

# lens-layer profiles (Hub lens, target 31)
lay = B["meta"]["lens_layers"]
sel = [4, 8, 12, 16, 20, 24, 26, 28, 30]
li = {l: i for i, l in enumerate(lay)}


def prof_row(label, arr):
    return [label] + [f(arr[li[l]], 2) for l in sel]


rows = [prof_row("real first token", S24["methods"]["lr_nobias"]["lens_profile"]["jlens"]["first_tok_top10_by_layer"])]
for h in ("proj", "lda", "lr", "lr_nobias"):
    rows.append(prof_row(f"phrase token, {LABEL[h]}", S24["methods"][h]["lens_profile"]["jlens"]["phrase_top10_by_layer"]))
rows.append(prof_row("phrase in top-10, first token not (LR no bias)", S24["methods"]["lr_nobias"]["lens_profile"]["jlens"]["phrase_only_by_layer"]))
rows.append("hline")
rows.append(prof_row("real first token (logit lens)", S24["methods"]["lr_nobias"]["lens_profile"]["logit_lens"]["first_tok_top10_by_layer"]))
rows.append(prof_row("phrase token, LR no bias (logit lens)", S24["methods"]["lr_nobias"]["lens_profile"]["logit_lens"]["phrase_top10_by_layer"]))
Table("bench_profiles", f"Layer profiles over all {B['meta']['n_held']:,} held-out contexts under the Hub J-lens (target layer 31): fraction of contexts in which the token is in the extended top-10 at each layer.",
      ["token"] + [f"L{l}" for l in sel], "l" + "r" * len(sel), rows).emit()

# LR sweeps
rows = []
for sc in ("150", "600", "2400"):
    for r in E1["scales"][sc]["lr_sweep"]:
        rows.append([sc if r is E1["scales"][sc]["lr_sweep"][0] else "", f"$10^{{{int(np.log10(r['lam']))}}}$", f(r["held_nll_generic"], 5), f(r["held_nll_pos_mean"], 2), f(r["row_norm_mean"], 2), "picked" if r["lam"] == E1["scales"][sc]["lr_lam"] else ""])
    rows.append("hline")
rows.pop()
Table("e1_sweep", "Experiment 1: $L_2$ strength for the biased logistic head, chosen by held-out NLL under the true prior.",
      ["contexts", "$\\lambda$", "held NLL generic", "held NLL own phrase", "row norm", ""], "llrrrl", rows).emit()

# ------------------------------------------------------------------ R-lens
p10 = AT["pass10"]
n_layers = len(p10["rlens"]["-2"])
rows = []
for l in range(0, n_layers, 2):
    rows.append([f"L{l}"] + [f(p10[n][pos][l], 2) for pos in ("-2", "-1") for n in ("rlens", "jlens", "logit")])
rows.append("hline")
rows.append(["mean, all layers"] + [f(np.mean(p10[n][pos])) for pos in ("-2", "-1") for n in ("rlens", "jlens", "logit")])
rows.append(["mean, layers 0--15"] + [f(np.mean(p10[n][pos][:16])) for pos in ("-2", "-1") for n in ("rlens", "jlens", "logit")])
Table("rlens_pass10", f"pass@10 on {AT['summary']['n_questions']} two-hop questions: fraction in which the intermediate entity's first token is in the lens top-10, read at the penultimate token ($-2$) and the last token ($-1$).",
      ["layer", "R-lens $-2$", "J-lens $-2$", "logit $-2$", "R-lens $-1$", "J-lens $-1$", "logit $-1$"], "lrrrrrr", rows).emit()

good = [r for r in AT["items"] if r["base_acc"] >= 0.5]
base = np.array([r["base_acc"] for r in good])
conds = AT["conditions"]
acc = {c: np.array([r["abl"][c]["acc"] for r in good]) for c in conds}
rng = np.random.default_rng(0)


def rl(a, b):
    return 1 - a.mean() / b.mean()


NL = {"rlens": "R-lens", "jlens": "J-lens", "logit": "logit lens", "random": "random"}
rows = []
for lname, ltxt in (("first_half", "0--15"), ("all", "all")):
    for pos, ptxt in (("(-2,)", "$-2$"), ("(-2, -1)", "$-2,-1$")):
        for name in ("rlens", "jlens", "logit", "random"):
            c = f"{name}|{lname}|{pos}"
            bs = [rl(acc[c][i], base[i]) for i in (rng.integers(0, len(good), len(good)) for _ in range(4000))]
            rows.append([NL[name], ltxt, ptxt, f(acc[c].mean()), f(rl(acc[c], base), 3, plus=True), f"[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]"])
        rows.append("hline")
rows.pop()
Table("rlens_ablation", f"Direction ablation. For each question and lens the unit direction $d_\\ell = \\mathrm{{normalize}}(J_\\ell^\\top(\\gamma\\odot W_U[t]))$ of the intermediate's first token $t$ is projected out of the residual at the stated positions during the prompt pass, at layers 0--15 or all 32. Eight samples per prompt; accuracy = answer alias in the continuation; {len(good)} of {len(AT['items'])} questions with baseline accuracy $\\ge 0.5$ (mean {base.mean():.3f}). Intervals: bootstrap over questions.",
      ["direction", "layers", "positions", "accuracy after", "relative loss", "95\\% CI"], "lllrrr", rows).emit()
rows = []
for lname, ltxt in (("first_half", "0--15"), ("all", "all")):
    for pos, ptxt in (("(-2,)", "$-2$"), ("(-2, -1)", "$-2,-1$")):
        d = acc[f"jlens|{lname}|{pos}"] - acc[f"rlens|{lname}|{pos}"]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        rows.append([ltxt, ptxt, f(d.mean(), 3, plus=True), f"[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]"])
Table("rlens_paired", "Paired difference in accuracy after ablation, J-lens minus R-lens (positive: the R-lens direction removes more). Bootstrap over questions.",
      ["layers", "positions", "mean", "95\\% CI"], "llrr", rows).emit()

wipe = [r for r in good if r["abl"]["rlens|all|(-2,)"]["acc"] == 0 and r["abl"]["jlens|all|(-2,)"]["acc"] == 0 and r["abl"]["random|all|(-2,)"]["acc"] >= 0.75]
rows = []
for r in sorted(good, key=lambda r: r["abl"]["rlens|all|(-2,)"]["acc"] + r["abl"]["jlens|all|(-2,)"]["acc"])[:12]:
    rows.append([tex_esc(r["prompt"][6:]), r["intermediate"], f(r["base_acc"], 2)] + [f(r["abl"][f"{n}|all|(-2,)"]["acc"], 2) for n in ("rlens", "jlens", "logit", "random")])
Table("rlens_examples", "The twelve questions most affected by removing the lens direction at the penultimate token across all layers (accuracy over 8 samples).",
      ["prompt (after ``Fact:'')", "intermediate", "base", "R-lens", "J-lens", "logit", "random"], "p{7.2cm}lrrrrr", rows, small=False).emit()

# phrase profiles under matched lenses
lay2 = RP["layers"]
sel2 = [4, 8, 12, 16, 20, 24, 28, 30]
li2 = {l: i for i, l in enumerate(lay2)}
rows = []
for n in ("rlens", "jlens", "logit"):
    pr = RP["profiles"][n]
    rows.append([NL[n], "real first token"] + [f(pr["first_top10_by_layer"][li2[l]], 2) for l in sel2])
for n in ("rlens", "jlens", "logit"):
    rows.append([NL[n], "phrase, LR no bias"] + [f(RP["profiles"][n]["phrase_lr_top10_by_layer"][li2[l]], 2) for l in sel2])
for n in ("rlens", "jlens"):
    rows.append([NL[n], "phrase, Brennan proj"] + [f(RP["profiles"][n]["phrase_proj_top10_by_layer"][li2[l]], 2) for l in sel2])
for n in ("rlens", "jlens"):
    rows.append([NL[n], "phrase in top-10, first token not"] + [f(RP["profiles"][n]["phrase_lr_only_by_layer"][li2[l]], 2) for l in sel2])
Table("rlens_phrases", f"Phrase-benchmark layer profiles ({RP['n_held']:,} held-out contexts) under the matched J-lens / R-lens pair (target layer 30) and the logit lens.",
      ["lens", "token"] + [f"L{l}" for l in sel2], "ll" + "r" * len(sel2), rows).emit()

# hand prompts: count of phrase tokens in the top-10 per layer, bench scale 2400
rows = []
for i, pr in enumerate(S24["methods"]["lr_nobias"]["prompts"]):
    cells = []
    for h in ("proj", "lda", "lr_nobias"):
        rr = S24["methods"][h]["prompts"][i]["rows"]
        cells.append(" ".join(f"{r['n_new']}" for r in rr))
    rows.append([tex_esc(pr["prompt"][:58]) + ("\\ldots" if len(pr["prompt"]) > 58 else "")] + cells)
layers_hand = [r["layer"] for r in S24["methods"]["lr_nobias"]["prompts"][0]["rows"]]
Table("hand_prompts", f"Hand prompts (benchmark heads at 2{{,}}400 contexts): number of phrase tokens in the extended top-10 at layers {', '.join(layers_hand)}, read at the last token (penultimate for the first prompt).",
      ["prompt", "Brennan", "LDA", "LR no bias"], "p{7cm}lll", rows).emit()

# multihop questions with baseline accuracy
rows = [[tex_esc(r["prompt"][6:]), r["intermediate"], tex_esc(", ".join(r["answer"])), f(r["base_acc"], 2)] for r in AT["items"]]
Table("multihop", "The two-hop question set with the ablated intermediate alias, accepted answer aliases and baseline sampled accuracy.",
      ["prompt (after ``Fact:'')", "intermediate", "answers", "base"], "p{7.5cm}llr", rows).emit()

# ------------------------------------------------------------------ figure data
def dat(name, cols, arrays):
    with open(os.path.join(here, "figures", f"{name}.dat"), "w") as fh:
        fh.write(" ".join(cols) + "\n")
        for row in zip(*arrays):
            fh.write(" ".join(f"{x}" for x in row) + "\n")


dat("bench_profiles", ["layer", "first", "proj", "lda", "lr", "lrnb", "lrnb_only", "first_logit", "lrnb_logit"],
    [lay, S24["methods"]["lr_nobias"]["lens_profile"]["jlens"]["first_tok_top10_by_layer"], S24["methods"]["proj"]["lens_profile"]["jlens"]["phrase_top10_by_layer"],
     S24["methods"]["lda"]["lens_profile"]["jlens"]["phrase_top10_by_layer"], S24["methods"]["lr"]["lens_profile"]["jlens"]["phrase_top10_by_layer"],
     S24["methods"]["lr_nobias"]["lens_profile"]["jlens"]["phrase_top10_by_layer"], S24["methods"]["lr_nobias"]["lens_profile"]["jlens"]["phrase_only_by_layer"],
     S24["methods"]["lr_nobias"]["lens_profile"]["logit_lens"]["first_tok_top10_by_layer"], S24["methods"]["lr_nobias"]["lens_profile"]["logit_lens"]["phrase_top10_by_layer"]])
dat("pass10", ["layer", "R", "J", "logit"], [list(range(n_layers)), p10["rlens"]["-2"], p10["jlens"]["-2"], p10["logit"]["-2"]])
dat("rlens_phrases", ["layer", "first_R", "first_J", "first_logit", "lr_R", "lr_J", "lr_logit", "proj_R", "proj_J"],
    [lay2] + [RP["profiles"][n][k] for k in ("first_top10_by_layer",) for n in ("rlens", "jlens", "logit")] + [RP["profiles"][n]["phrase_lr_top10_by_layer"] for n in ("rlens", "jlens", "logit")]
    + [RP["profiles"][n]["phrase_proj_top10_by_layer"] for n in ("rlens", "jlens")])
# ablation bars: relative loss with CI for the four directions x (first_half,-2) and (all,-2)
rows_abl = []
for lname in ("first_half", "all"):
    for name in ("rlens", "jlens", "logit", "random"):
        c = f"{name}|{lname}|(-2,)"
        bs = [rl(acc[c][i], base[i]) for i in (rng.integers(0, len(good), len(good)) for _ in range(4000))]
        rows_abl.append((f"{NL[name]}", lname, rl(acc[c], base), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
with open(os.path.join(here, "figures", "ablation.dat"), "w") as fh:
    fh.write("idx label layers loss lo hi\n")
    for i, (n, l, m, lo, hi) in enumerate(rows_abl):
        grp = "0--15" if l == "first_half" else "all"
        fh.write(f"{i} {n.replace(' ', '~')}~({grp}) {l} {m:.4f} {lo:.4f} {hi:.4f}\n")

# scalars for the prose
scalars = {
    "e1_n_generic_eval": E1["meta"]["n_generic_eval_positions"], "e1_prior_sum": sum(E1["meta"]["pi_true"].values()),
    "e1_wu_std": E1["meta"]["wu_row_logit_std_median"], "e1_wu_mean": E1["meta"]["wu_row_logit_mean_median"], "e1_lse": E1["meta"]["mean_lse_generic"],
    "e1_copy_frac": E1["meta"]["n_occ_gt1_frac"], "e1_docs": E1["meta"]["mine_stats"]["docs"],
    "b_n_held": B["meta"]["n_held"], "b_single_frac": B["meta"]["held_single_frac"], "b_n_generic_eval": B["meta"]["n_generic_eval_positions"], "b_prior_sum": B["meta"]["prior_sum"],
    "b_n_generic_store": B["meta"]["n_generic_store"],
    "at_n_q": AT["summary"]["n_questions"], "at_n_good": len(good), "at_base": float(base.mean()), "at_wipe": len(wipe),
    "p10_mean": {n: float(np.mean(p10[n]["-2"])) for n in ("rlens", "jlens", "logit")}, "p10_early": {n: float(np.mean(p10[n]["-2"][:16])) for n in ("rlens", "jlens", "logit")},
    "abl": {c: {"before": float(base.mean()), "after": float(acc[c].mean()), "rel": float(rl(acc[c], base))} for c in conds},
    "rp_n_held": RP["n_held"], "lam_choice": B["lam_choice"],
}
json.dump({"scalars": scalars, "html": HTML}, open(os.path.join(here, "tables_html.json"), "w"), indent=1)
with open(os.path.join(here, "tables", "scalars.tex"), "w") as fh:
    def nc(name, val):
        fh.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    nc("EoneGenericEval", f"{scalars['e1_n_generic_eval']:,}"); nc("EonePriorSum", f"{scalars['e1_prior_sum']:.1e}".replace("e-0", "e-"))
    nc("EoneWuStd", f"{scalars['e1_wu_std']:.2f}"); nc("EoneWuMean", f"{scalars['e1_wu_mean']:.1f}"); nc("EoneLse", f"{scalars['e1_lse']:.1f}")
    nc("EoneCopyFrac", f"{scalars['e1_copy_frac']:.0%}".replace("%", "\\%")); nc("EoneDocs", f"{scalars['e1_docs']:,}")
    nc("BnHeld", f"{scalars['b_n_held']:,}"); nc("BsingleFrac", f"{scalars['b_single_frac']:.0%}".replace("%", "\\%")); nc("BnGenericEval", f"{scalars['b_n_generic_eval']:,}")
    nc("BpriorSum", f"{scalars['b_prior_sum']:.1e}".replace("e-0", "e-")); nc("BnGenericStore", f"{scalars['b_n_generic_store']:,}")
    nc("ATnQ", scalars["at_n_q"]); nc("ATnGood", scalars["at_n_good"]); nc("ATbase", f"{scalars['at_base']:.3f}"); nc("ATwipe", scalars["at_wipe"])
    for n in ("rlens", "jlens", "logit"):
        nc(f"PtenMean{n}", f"{scalars['p10_mean'][n]:.3f}"); nc(f"PtenEarly{n}", f"{scalars['p10_early'][n]:.3f}")
    for c in conds:
        key = c.replace("|", "").replace("(-2,)", "A").replace("(-2, -1)", "B").replace("_", "").replace(",", "").replace(" ", "")
        nc(f"Abl{key}", f"{scalars['abl'][c]['rel']:+.3f}".replace("%", ""))
    nc("RPnHeld", f"{scalars['rp_n_held']:,}")
print("tables:", sorted(HTML), "\nscalars written")
