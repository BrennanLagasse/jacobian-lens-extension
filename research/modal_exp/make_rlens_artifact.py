"""Render results/rlens_attribution.json + rlens_phrases.json into a self-contained HTML report."""
import html
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
rd = os.path.join(here, "results")
A = json.load(open(os.path.join(rd, "rlens_attribution.json")))
Pp = json.load(open(os.path.join(rd, "rlens_phrases.json")))
rng = np.random.default_rng(0)
good = [r for r in A["items"] if r["base_acc"] >= 0.5]
base = np.array([r["base_acc"] for r in good])
conds = A["conditions"]
acc = {c: np.array([r["abl"][c]["acc"] for r in good]) for c in conds}


def esc(s):
    return html.escape(str(s), quote=True)


def rl(a, b):
    return 1 - a.mean() / b.mean()


def boot(c):
    bs = []
    for _ in range(3000):
        i = rng.integers(0, len(good), len(good)); bs.append(rl(acc[c][i], base[i]))
    return rl(acc[c], base), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


LBL = {"rlens": "R-lens", "jlens": "J-lens", "logit": "logit lens", "random": "random"}
abl_rows = []
for lname, ltxt in (("first_half", "layers 0–15"), ("all", "all layers")):
    for pos, ptxt in (("(-2,)", "penultimate"), ("(-2, -1)", "penultimate + last")):
        for name in ("rlens", "jlens", "logit", "random"):
            c = f"{name}|{lname}|{pos}"
            m, lo, hi = boot(c)
            cls = "good" if name in ("rlens", "jlens") and lo > 0 else ""
            abl_rows.append(f'<tr class="{cls}"><td>{LBL[name]}</td><td>{ltxt}</td><td>{ptxt}</td><td class="num">{acc[c].mean():.3f}</td>'
                            f'<td class="num">{m:+.3f}</td><td class="num">[{lo:+.3f}, {hi:+.3f}]</td></tr>')

pair = []
for lname, ltxt in (("first_half", "layers 0–15"), ("all", "all layers")):
    for pos, ptxt in (("(-2,)", "penultimate"), ("(-2, -1)", "penultimate + last")):
        d = acc[f"jlens|{lname}|{pos}"] - acc[f"rlens|{lname}|{pos}"]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        pair.append(f'<tr><td>{ltxt}</td><td>{ptxt}</td><td class="num">{d.mean():+.3f}</td><td class="num">[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]</td></tr>')

p10 = A["pass10"]
n_layers = len(p10["rlens"]["-2"])


def line_chart(series, W=640, H=280, xlabels=None, ylab=""):
    pl, pr, pt, pb = 44, 16, 14, 40
    n = len(series[0][1])
    xs = lambda i: pl + i * (W - pl - pr) / (n - 1)
    ys = lambda v: pt + (1 - v) * (H - pt - pb)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(ylab)}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pl}" y1="{ys(v):.1f}" x2="{W-pr}" y2="{ys(v):.1f}" stroke="var(--rule)"/>'
                   f'<text x="{pl-6}" y="{ys(v)+4:.1f}" text-anchor="end" class="ax">{v:.2f}</text>')
    for i, l in enumerate(xlabels):
        if i % max(1, n // 11) == 0 or i == n - 1:
            out.append(f'<text x="{xs(i):.1f}" y="{H-pb+16}" text-anchor="middle" class="ax">L{l}</text>')
    for name, vals, color, dash in series:
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg"><i style="border-top:2px {"dashed" if d else "solid"} {c}"></i>{esc(nm)}</span>' for nm, _, c, d in series)
    return "\n".join(out), legend


svg1, leg1 = line_chart([("R-lens", p10["rlens"]["-2"], "var(--good)", ""), ("J-lens", p10["jlens"]["-2"], "var(--ink)", ""),
                         ("logit lens", p10["logit"]["-2"], "var(--muted)", "4 3")], xlabels=list(range(n_layers)), ylab="pass@10 by layer")
lay = Pp["layers"]
pr = Pp["profiles"]
svg2, leg2 = line_chart([("first token · R-lens", pr["rlens"]["first_top10_by_layer"], "var(--ink)", ""),
                         ("first token · J-lens", pr["jlens"]["first_top10_by_layer"], "var(--ink)", "4 3"),
                         ("phrase (LR no bias) · R-lens", pr["rlens"]["phrase_lr_top10_by_layer"], "var(--good)", ""),
                         ("phrase (LR no bias) · J-lens", pr["jlens"]["phrase_lr_top10_by_layer"], "var(--good)", "4 3"),
                         ("phrase (LR no bias) · logit lens", pr["logit"]["phrase_lr_top10_by_layer"], "var(--muted)", "4 3")],
                        xlabels=lay, ylab="phrase benchmark top-10 rate by layer")

means = {n: float(np.mean(p10[n]["-2"])) for n in ("rlens", "jlens", "logit")}
early = {n: float(np.mean(p10[n]["-2"][:16])) for n in ("rlens", "jlens", "logit")}
wipe = [r for r in good if r["abl"]["rlens|all|(-2,)"]["acc"] == 0 and r["abl"]["jlens|all|(-2,)"]["acc"] == 0 and r["abl"]["random|all|(-2,)"]["acc"] >= 0.75]
ex_rows = "\n".join(f'<tr><td>{esc(r["prompt"][6:])}</td><td>{esc(r["intermediate"])}</td><td class="num">{r["base_acc"]:.2f}</td>'
                    f'<td class="num">{r["abl"]["rlens|all|(-2,)"]["acc"]:.2f}</td><td class="num">{r["abl"]["jlens|all|(-2,)"]["acc"]:.2f}</td>'
                    f'<td class="num">{r["abl"]["logit|all|(-2,)"]["acc"]:.2f}</td><td class="num">{r["abl"]["random|all|(-2,)"]["acc"]:.2f}</td></tr>'
                    for r in sorted(good, key=lambda r: r["abl"]["rlens|all|(-2,)"]["acc"] + r["abl"]["jlens|all|(-2,)"]["acc"])[:12])

page = f"""<title>R-lens Attribution</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  :root {{ --paper: #f7f6f1; --ink: #1c2028; --muted: #676d78; --rule: #d8d5cc; --soft: #ebe9e2; --good: #0f766e; --good-bg: #e2f1ee; --bad: #b45309;
    --serif: "Fraunces", "Iowan Old Style", Georgia, serif; --sans: "Source Sans 3", "Helvetica Neue", Arial, sans-serif; --mono: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228; --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; }} }}
  :root[data-theme="dark"] {{ --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228; --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 16.5px; line-height: 1.5; padding-inline: 16px; padding-block: 0; }}
  main {{ max-width: 78ch; margin: 0 auto; padding-block: 40px 64px; }}
  h1 {{ font-family: var(--serif); font-weight: 500; font-size: 2.3rem; line-height: 1.1; margin: 0 0 8px; text-wrap: balance; }}
  h2 {{ font-family: var(--serif); font-weight: 600; font-size: 1.45rem; margin: 40px 0 10px; text-wrap: balance; }}
  .kicker {{ font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }}
  .lede {{ font-size: 1.12rem; max-width: 66ch; }} p {{ max-width: 66ch; margin: 0 0 12px; }}
  .verdict {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 22px 0 8px; }}
  .verdict div {{ border-top: 3px solid var(--rule); padding-top: 8px; }} .verdict .good {{ border-color: var(--good); }} .verdict .bad {{ border-color: var(--bad); }}
  .verdict b {{ display: block; font-family: var(--serif); font-size: 1.05rem; font-weight: 600; margin-bottom: 2px; }} .verdict span {{ font-size: 0.93rem; color: var(--muted); }}
  .tbl {{ overflow-x: auto; margin: 12px 0 6px; }} table {{ border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  th {{ font-weight: 600; font-size: 0.74rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
  td.num {{ font-family: var(--mono); font-size: 0.8rem; text-align: right; }} th.num {{ text-align: right; }} tr.good td {{ background: var(--good-bg); }}
  .cap {{ font-size: 0.86rem; color: var(--muted); max-width: 66ch; margin: 4px 0 0; }}
  svg {{ width: 100%; max-width: 100%; height: auto; display: block; }} .ax {{ font-family: var(--mono); font-size: 11px; fill: var(--muted); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px 18px; font-size: 0.82rem; margin: 6px 0 4px; }} .lg i {{ display: inline-block; width: 22px; margin-right: 6px; vertical-align: middle; }}
  code {{ font-family: var(--mono); font-size: 0.84em; background: var(--soft); padding: 0 4px; }} ul {{ max-width: 66ch; padding-left: 20px; }} li {{ margin-bottom: 6px; }}
  .foot {{ margin-top: 40px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 0.86rem; color: var(--muted); }}
</style>
<main>
<div class="kicker">Qwen3.5-9B-Base · matched J-lens / R-lens pair · 50 two-hop questions · 2,942 phrase contexts · 2026-09-12</div>
<h1>R-lens: cleaner early layers, same causal weight as the J-lens</h1>
<p class="lede">A matched Jacobian-lens and RelP-lens pair fitted with the authors' recipe on the base model, then tested three ways: pass@10 on two-hop questions, the direction-ablation attribution experiment with logit-lens and random controls, and the held-out layer profiles of the phrase benchmark.</p>
<div class="verdict">
  <div class="good"><b>pass@10</b><span>R-lens {means["rlens"]:.2f} vs J-lens {means["jlens"]:.2f} vs logit {means["logit"]:.2f} over all layers; {early["rlens"]:.2f} vs {early["jlens"]:.2f} vs {early["logit"]:.2f} over layers 0–15</span></div>
  <div class="good"><b>Ablation</b><span>removing the lens direction at the penultimate token costs 14–16% of accuracy; logit-lens and random directions cost nothing</span></div>
  <div class="bad"><b>R vs J</b><span>indistinguishable on the ablation here (paired difference within ±0.02, intervals span zero); the post's larger R-lens effect does not reproduce on this model</span></div>
  <div class="good"><b>Phrase heads</b><span>R-lens removes the early-layer phrase hits: layer 8 top-10 rate 8% vs 17% under J-lens, late layers unchanged</span></div>
</div>

<h2>What was built</h2>
<p>R-lens is the <code>jlens.fit</code> estimator with three LRP rules in the backward pass: the RMSNorm scale is a constant (LN-rule), SiLU's sigmoid factor is detached (identity-rule), and the SwiGLU product's gradient is split evenly between its branches (half-rule, β = 0.5). The rule set, target layer 30, 25 pile-10k prompts and skip-4 come from the provenance stored in the authors' Qwen3.5-9B lens. The rules are installed as instance-level overrides that graft the RelP gradient onto the standard forward value, so the forward pass is bit-identical (max difference 0.0) and J and R are fitted on the same activations. R-lens per-prompt Jacobian norms come out about half the J-lens's.</p>

<h2>pass@10 on two-hop questions</h2>
<p>Fifty base-model prompts such as “Fact: The capital of the country where the Eiffel Tower stands is”, read at the penultimate token: is the intermediate entity's first token (France) in the lens top-10?</p>
<div class="legend">{leg1}</div>
{svg1}
<p class="cap">Both lenses read the intermediate from layer 6, where the logit lens shows nothing until layer 24. R-lens leads at layers 4–6 and 22–26; J-lens is marginally ahead at 12–16. The gain is in the post's direction and modest on a 9B dense base model.</p>

<h2>Direction ablation</h2>
<p>For each question and lens, the unit direction J<sub>l</sub><sup>T</sup>(γ ⊙ W<sub>U</sub>[t]) of the intermediate's first token is projected out of the residual at the penultimate token during the prompt pass, at the first half of the layers or all of them; a variant also removes it at the last token. Eight samples per prompt at T = 0.7; accuracy is an answer alias in the continuation. {len(good)} of {len(A["items"])} questions with baseline accuracy ≥ 0.5 (mean {base.mean():.3f}) enter the summary; a random unit direction per layer is the control. Intervals are bootstraps over questions.</p>
<div class="tbl"><table>
<thead><tr><th>direction</th><th>layers</th><th>positions</th><th class="num">accuracy after</th><th class="num">relative loss</th><th class="num">95% CI</th></tr></thead>
<tbody>{"".join(abl_rows)}</tbody></table></div>
<p class="cap">Highlighted rows are lens directions whose loss is significantly above zero. With the last position included, the logit-lens direction also bites, because at the answer position the intermediate is already in the unembedding basis at late layers; the penultimate position is what isolates the lens-specific contribution.</p>
<div class="tbl"><table>
<thead><tr><th>paired R-minus-J extra loss</th><th>positions</th><th class="num">mean</th><th class="num">95% CI</th></tr></thead>
<tbody>{"".join(pair)}</tbody></table></div>
<p class="cap">Positive means the R-lens direction hurts more. At the penultimate position the two lenses are within ±0.02 of each other; with the last position included the J-lens hurts slightly more.</p>
<h2>Questions where either lens direction wipes the answer out</h2>
<div class="tbl"><table>
<thead><tr><th>prompt</th><th>intermediate</th><th class="num">base</th><th class="num">R-lens</th><th class="num">J-lens</th><th class="num">logit</th><th class="num">random</th></tr></thead>
<tbody>{ex_rows}</tbody></table></div>
<p class="cap">Accuracy over 8 samples after removing the direction at the penultimate token across all layers. {len(wipe)} questions go from ≥ 0.5 to 0.0 under both lens directions while staying ≥ 0.75 under the random direction.</p>

<h2>The phrase benchmark under the matched lenses</h2>
<p>Held-out profiles of the 15-phrase benchmark ({Pp["n_held"]:,} contexts), recomputed under the matched pair and the logit lens. The bias-free logistic phrase head was refit from the stored states.</p>
<div class="legend">{leg2}</div>
{svg2}
<div class="tbl"><table>
<thead><tr><th>lens</th><th>token</th>{"".join(f'<th class="num">L{l}</th>' for l in lay)}</tr></thead>
<tbody>
{"".join(f'<tr><td>{LBL[n]}</td><td>real first token</td>' + "".join(f'<td class="num">{x:.2f}</td>' for x in pr[n]["first_top10_by_layer"]) + '</tr>' for n in ("rlens", "jlens", "logit"))}
{"".join(f'<tr class="{"good" if n == "rlens" else ""}"><td>{LBL[n]}</td><td>phrase token, LR no bias</td>' + "".join(f'<td class="num">{x:.2f}</td>' for x in pr[n]["phrase_lr_top10_by_layer"]) + '</tr>' for n in ("rlens", "jlens", "logit"))}
{"".join(f'<tr><td>{LBL[n]}</td><td>phrase token, Brennan proj</td>' + "".join(f'<td class="num">{x:.2f}</td>' for x in pr[n]["phrase_proj_top10_by_layer"]) + '</tr>' for n in ("rlens", "jlens"))}
{"".join(f'<tr><td>{LBL[n]}</td><td>phrase in top-10 while first token is not (LR)</td>' + "".join(f'<td class="num">{x:.2f}</td>' for x in pr[n]["phrase_lr_only_by_layer"]) + '</tr>' for n in ("rlens", "jlens"))}
</tbody></table></div>
<p class="cap">Under the J-lens the phrase row was in the top-10 at 6 / 17 / 20% of contexts at layers 4 / 8 / 12, where the real first token is at 0–3%; under the R-lens it is 0 / 8 / 10%. Late layers are unchanged, and the R-lens surfaces the real first token slightly earlier. Brennan's head stays saturated under either lens: the problem is the row, not the lens. Absolute numbers differ from the benchmark page because that used the Hub Base lens (target 31, 458 prompts).</p>

<h2>What to take from this</h2>
<ul>
<li>Use the R-lens as the default readout for phrase heads: same fit cost, same forward, cleaner early layers. Examine the “phrase before its first token” population at layers 8–16, not layer 4.</li>
<li>Keep the ablation protocol (penultimate token, first-half vs all layers, logit-lens and random controls, bootstrap over questions) as the causal check for any new head or lens; it separates lens directions from controls cleanly at n = {len(good)}.</li>
<li>Do not expect the post's R-vs-J ablation gap on a 9B dense base model; that comparison needs Qwen3.6-27B or an MoE, an autorater, and more questions.</li>
</ul>
<div class="foot">Code: <code>~/notes/brennan-jlens-review/modal_exp/jlens_rlens.py</code> (Modal app <code>jlens-rlens</code>). Findings note: <code>RLENS_2026-09-12.md</code>. Source post: camilablank, agam_bhatia, Neel Nanda, “R-lens: making J-lens more faithful on early layers”; artifacts <code>camilablank/workspace-lenses</code>.</div>
</main>
"""
out = os.path.join(rd, "rlens.html")
open(out, "w").write(page)
print("wrote", out, len(page), "wipeouts", len(wipe))
