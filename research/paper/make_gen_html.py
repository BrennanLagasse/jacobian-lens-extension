"""Build gen_report.html (generalisation study page) from gen_tables_html.json and figures/gen_*.dat."""
import html
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(os.path.join(here, "gen_tables_html.json")))
R = os.path.join(here, "..", "modal_exp", "results")
S = json.load(open(os.path.join(R, "gen_summary.json")))["lr_nobias"]
X = json.load(open(os.path.join(R, "gen_xling.json")))["heads"]["lr_nobias"]["xfer"]


def dat(name):
    rows = [l.split() for l in open(os.path.join(here, "figures", f"{name}.dat")).read().strip().splitlines()]
    return {c: [float(r[i]) for r in rows[1:]] for i, c in enumerate(rows[0])}


def line_chart(series, xs, W=680, H_=300, ylab="", xlab="layer", xfmt=lambda x: f"L{int(x)}"):
    pl, pr, pt, pb = 46, 16, 14, 40
    n = len(xs); Xc = lambda i: pl + i * (W - pl - pr) / (n - 1); Yc = lambda v: pt + (1 - v) * (H_ - pt - pb)
    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="{html.escape(ylab)}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pl}" y1="{Yc(v):.1f}" x2="{W-pr}" y2="{Yc(v):.1f}" stroke="var(--rule)"/><text x="{pl-6}" y="{Yc(v)+4:.1f}" text-anchor="end" class="ax">{v:.2f}</text>')
    for i, x in enumerate(xs):
        out.append(f'<text x="{Xc(i):.1f}" y="{H_-pb+16}" text-anchor="middle" class="ax">{xfmt(x)}</text>')
    out.append(f'<text x="{(pl+W-pr)/2:.0f}" y="{H_-6}" text-anchor="middle" class="ax">{xlab}</text>')
    for name, vals, color, dash in series:
        pts = " ".join(f"{Xc(i):.1f},{Yc(v):.1f}" for i, v in enumerate(vals) if v == v)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg"><i style="border-top:2px {"dashed" if d else "solid"} {c}"></i>{html.escape(nm)}</span>' for nm, _, c, d in series)
    return f'<div class="legend">{legend}</div>' + "\n".join(out)


gl = dat("gen_layers")
fig_layers = line_chart([("nouns", gl["nouns_R"], "var(--good)", ""), ("verb phrases", gl["verbs_R"], "var(--blue)", ""), ("abstract concepts", gl["abstract_R"], "var(--warn)", ""),
                         ("events, dates, years", gl["events_R"], "var(--bad)", ""), ("real first token in top-10 (nouns)", gl["nouns_first"], "var(--ink)", "4 3"),
                         ("real first token in top-10 (events)", gl["events_first"], "var(--ink)", "1 3")], gl["layer"], ylab="TPR at FPR 1e-2, R-lens")
gp = dat("gen_positions")
fig_pos = line_chart([("nouns, row", gp["nouns_row"], "var(--good)", ""), ("nouns, presence probe", gp["nouns_probe"], "var(--good)", "4 3"),
                      ("verb phrases, row", gp["verbs_row"], "var(--blue)", ""), ("verb phrases, probe", gp["verbs_probe"], "var(--blue)", "4 3"),
                      ("abstract, row", gp["abstract_row"], "var(--warn)", ""), ("abstract, probe", gp["abstract_probe"], "var(--warn)", "4 3")],
                     gp["offset"], ylab="TPR at the generic FPR 1e-2 threshold", xlab="position relative to the phrase (0 = token before, 1.. = inside)", xfmt=lambda x: f"{int(x):+d}")

langs = ["eng", "spa", "fra", "zho"]; LN = {"eng": "English", "spa": "Spanish", "fra": "French", "zho": "Chinese"}
M = {}
for k, v in X.items():
    c, pair = k.split("|"); a, b = pair.split("->"); M.setdefault((a, b), []).append(v["auroc_within_lang"])
xl = "".join(f"<tr><td>{LN[a]} row</td>" + "".join(f'<td class="num">{"—" if a == b else f"{sum(M[(a,b)])/len(M[(a,b)]):.2f}"}</td>' for b in langs) + "</tr>" for a in langs)

page = f"""<title>Phrase Rows Beyond Proper Nouns</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  :root {{ --paper: #f7f6f1; --ink: #1c2028; --muted: #676d78; --rule: #d8d5cc; --soft: #ebe9e2; --good: #0f766e; --good-bg: #e2f1ee; --bad: #b45309; --warn: #a16207; --blue: #1d4ed8;
    --serif: "Fraunces", "Iowan Old Style", Georgia, serif; --sans: "Source Sans 3", "Helvetica Neue", Arial, sans-serif; --mono: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228; --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; --warn: #e0b04a; --blue: #8ab4ff; }} }}
  :root[data-theme="dark"] {{ --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228; --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; --warn: #e0b04a; --blue: #8ab4ff; }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 16.5px; line-height: 1.5; padding-inline: 16px; padding-block: 0; }}
  main {{ max-width: 80ch; margin: 0 auto; padding-block: 40px 64px; }}
  h1 {{ font-family: var(--serif); font-weight: 500; font-size: 2.2rem; line-height: 1.1; margin: 0 0 8px; text-wrap: balance; }}
  h2 {{ font-family: var(--serif); font-weight: 600; font-size: 1.45rem; margin: 44px 0 10px; text-wrap: balance; }}
  .kicker {{ font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }}
  .lede {{ font-size: 1.1rem; max-width: 68ch; }} p {{ max-width: 68ch; margin: 0 0 12px; }}
  .verdict {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 22px 0 8px; }}
  .verdict div {{ border-top: 3px solid var(--rule); padding-top: 8px; }} .verdict .good {{ border-color: var(--good); }} .verdict .bad {{ border-color: var(--bad); }} .verdict .warn {{ border-color: var(--warn); }}
  .verdict b {{ display: block; font-family: var(--serif); font-size: 1.05rem; font-weight: 600; margin-bottom: 2px; }} .verdict span {{ font-size: 0.93rem; color: var(--muted); }}
  .tbl {{ overflow-x: auto; margin: 8px 0 18px; }} table {{ border-collapse: collapse; font-size: 0.84rem; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 5px 9px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  th {{ font-weight: 600; font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
  td.num, th.num {{ font-family: var(--mono); font-size: 0.78rem; text-align: right; white-space: nowrap; }}
  .cap {{ font-size: 0.88rem; color: var(--muted); max-width: 70ch; margin: 18px 0 4px; }} .cap b {{ color: var(--ink); }}
  svg {{ width: 100%; max-width: 100%; height: auto; display: block; }} .ax {{ font-family: var(--mono); font-size: 11px; fill: var(--muted); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 0.8rem; margin: 6px 0 4px; }} .lg i {{ display: inline-block; width: 22px; margin-right: 6px; vertical-align: middle; }}
  code {{ font-family: var(--mono); font-size: 0.84em; background: var(--soft); padding: 0 4px; }} ul {{ max-width: 68ch; padding-left: 20px; }} li {{ margin-bottom: 6px; }}
  .foot {{ margin-top: 40px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 0.86rem; color: var(--muted); }}
</style>
<main>
<div class="kicker">Generalisation study · 5 axes · 184 phrases · Qwen3.5-9B-Base · 2026-09-14 · companion to main.pdf §{'8'}</div>
<h1>Phrase rows beyond proper nouns: how far the readout reaches</h1>
<p class="lede">Compositional nouns, verb phrases and idioms, abstract concepts, named events and dates, and the same concepts in four languages, each with sibling groups, scored with TPR at fixed false-positive rates and with per-layer detection thresholds, so “when does it pop up” is measured at a controlled FPR rather than by top-10 counts.</p>

<div class="verdict">
  <div class="good"><b>Direction</b><span>AUROC 0.98–0.99 on every axis · at FPR 1e-3 the rows recover {S['verbs']['tpr1e3']:.0%} (verbs) to {S['events']['tpr1e3']:.0%} (events) of upcoming phrases</span></div>
  <div class="good"><b>Siblings</b><span>AUROC {min(S[a]['sib_auroc'] for a in ('nouns','verbs','abstract','events')):.2f}–{max(S[a]['sib_auroc'] for a in ('nouns','verbs','abstract','events')):.2f} · failures are near-synonyms and numerals (hard drive/disk, 1984/1989); successes are different worlds (machine learning/gun)</span></div>
  <div class="warn"><b>When it pops up</b><span>events at layer 8 (median), nouns and verbs at 20, abstract concepts at 24 · the row leads the model's own first-token prediction at every layer</span></div>
  <div><b>Joint fit</b><span>136 rows in one softmax: −0.005 AUROC, −0.01 to −0.03 TPR@1e-3, better calibration (mass/prior 1.0–1.7)</span></div>
  <div class="good"><b>Cross-lingual</b><span>row directions transfer: English rows score {sum(M[('eng','spa')])/len(M[('eng','spa')]):.2f} / {sum(M[('eng','fra')])/len(M[('eng','fra')]):.2f} / {sum(M[('eng','zho')])/len(M[('eng','zho')]):.2f} AUROC in Spanish / French / Chinese against language-matched negatives (0.98 at home); Spanish↔French 0.94</span></div>
</div>

<h2>Headline per axis</h2>
{H['gen_headline']}

<h2>When the phrase pops up</h2>
<p>TPR of the bias-free row at a 1% false-positive rate by R-lens layer, with the threshold at each layer taken from generic text transported through the same lens. The dashed and dotted lines are the rate at which the phrase's real first token is in the lens top-10 at the same positions: the phrase row reads “this phrase is coming” well before the model has settled on its first token.</p>
{fig_layers}
{H['gen_layers']}
{H['gen_earliest']}

<h2>Sibling separation</h2>
<p>Every phrase was fitted with its first-token (or category) siblings present. Separation tracks semantic distance between the siblings, not syntax: the pairs that fail are the ones a reader could not tell apart at that position either.</p>
{H['gen_siblings']}

<h2>Row vs presence probe: the position profile</h2>
<p>The row (solid) fires at the token before the phrase and switches off inside it; the presence probe (dashed), trained on the pre-phrase and inside positions, covers the phrase and lingers after it, longer for abstract concepts than for idioms.</p>
{fig_pos}

<h2>One softmax for everything</h2>
{H['gen_joint']}

<h2>Cross-lingual transfer</h2>
<p>Twelve concepts in English, Spanish, French and Chinese, each language mined from its own corpus. A row fitted on one language is scored at another language's positions of the same concept against that language's positions of the other eleven concepts, so language identity cannot carry the score. Mean AUROC over concepts:</p>
<div class="tbl"><table><thead><tr><th>row fitted on</th>{"".join(f'<th class="num">scored in {LN[b]}</th>' for b in langs)}</tr></thead><tbody>{xl}</tbody></table></div>
{H['gen_xling_concepts']}

<h2>What generalised and what did not</h2>
<ul>
<li><b>Generalised:</b> the pre-phrase readout works for compound nouns, verb phrases and idioms, surface-form-defined abstract concepts, and multi-word events with jagged tokenisation, with the same head, the same prior weighting and no per-axis tuning.</li>
<li><b>Did not:</b> near-synonyms (hard drive / hard disk), numerals sharing a context type (1984 / 1989, World War I / II), abstract nouns with ambiguous surface forms (irony), and the rarest idiom for the presence probe.</li>
<li><b>Partly:</b> cross-lingual transfer of a row's direction (AUROC 0.87–0.94 against language-matched negatives, 0.98 at home); the presence probe transfers better than the row.</li>
<li><b>Measure it at fixed FPR.</b> Top-10 counts inflate under the J-lens at early layers; TPR at FPR 1e-2 with lens-matched thresholds gives the same answer under both lenses.</li>
</ul>
<div class="foot">Code: <code>~/notes/brennan-jlens-review/modal_exp/jlens_gen.py</code> (Modal app <code>jlens-gen</code>). Findings note: <code>GENERALIZATION_2026-09-14.md</code>. Paper section 8 of <code>paper/main.pdf</code>.</div>
</main>
"""
open(os.path.join(here, "gen_report.html"), "w").write(page)
print("wrote gen_report.html", len(page))
