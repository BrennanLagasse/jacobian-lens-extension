"""Render results/bench.json into a self-contained HTML benchmark report."""
import html
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(here, "results", "bench.json")))
meta, scales = R["meta"], R["scales"]
PHR = meta["phrases"]
S = scales["2400"]
LAYERS = meta["lens_layers"]
LABEL = {"random": "random row (floor)", "proj": "Brennan (proj + unit)", "avgtok": "avg W_U rows", "lda": "LDA / whitening",
         "lr_1e-05": "LR with bias, L2 1e-5", "lr_0.0001": "LR with bias, L2 1e-4", "lr_nobias": "LR, no bias, L2 1e-6"}
ORDER = ["random", "proj", "avgtok", "lda", "lr_0.0001", "lr_1e-05", "lr_nobias"]
CLS = {"proj": "bad", "lr_nobias": "good"}


def esc(s):
    return html.escape(str(s), quote=True)


def pp(m, bucket="all"):
    return {p: v[bucket] for p, v in m["per_phrase"].items() if bucket in v}


def mean_of(m, key, bucket="all"):
    vals = [v[key] for v in pp(m, bucket).values() if v.get(key) is not None]
    return float(np.mean(vals))


def top_of(m, key):
    return float(np.mean([m["per_phrase"][p][key] for p in PHR]))


head_rows = []
for h in ORDER:
    m = scales["150"]["methods"][h] if h in ("random", "avgtok") else S["methods"][h]
    head_rows.append(
        f'<tr class="{CLS.get(h, "")}"><td>{LABEL[h]}</td>'
        f'<td class="num">{mean_of(m,"auroc_generic"):.3f}</td><td class="num">{mean_of(m,"auroc_first_sib"):.3f}</td>'
        f'<td class="num">{mean_of(m,"auroc_cat_sib"):.3f}</td><td class="num">{mean_of(m,"top10"):.2f}</td>'
        f'<td class="num">{np.median([v["rank_median"] for v in pp(m).values()]):.0f}</td>'
        f'<td class="num">{mean_of(m,"log_rank_ratio_median"):+.2f}</td><td class="num">{top_of(m,"generic_top10"):.4f}</td>'
        f'<td class="num">{top_of(m,"mass_ratio"):.1f}</td></tr>')

sib_rows = []
for h, lab in (("proj", "Brennan (proj + unit)"), ("lda", "LDA / whitening"), ("lr", "LR with bias"), ("lr_nobias", "LR, no bias")):
    d = R["orig10"][h]
    for p in ("New Hampshire", "John Adams"):
        v = d[p]
        sib_rows.append(f'<tr class="{CLS.get(h, "")}"><td>{lab}</td><td>{p}</td><td class="num">{v["auroc_first_sib"]:.3f}</td>'
                        f'<td class="num">{v["top10_at_sib"]:.3f}</td><td class="num">{v["auroc_cat_sib"]:.3f}</td></tr>')


def profile_svg():
    """Top-10 rate by layer: real first token vs phrase token (bias-free LR) under J-lens and logit lens."""
    W, H, pl, pr, pt, pb = 640, 300, 44, 16, 14, 40
    xs = lambda i: pl + i * (W - pl - pr) / (len(LAYERS) - 1)
    ys = lambda v: pt + (1 - v) * (H - pt - pb)
    series = [
        ("first token · J-lens", S["methods"]["lr_nobias"]["lens_profile"]["jlens"]["first_tok_top10_by_layer"], "var(--ink)", ""),
        ("first token · logit lens", S["methods"]["lr_nobias"]["lens_profile"]["logit_lens"]["first_tok_top10_by_layer"], "var(--ink)", "4 3"),
        ("phrase, LR no bias · J-lens", S["methods"]["lr_nobias"]["lens_profile"]["jlens"]["phrase_top10_by_layer"], "var(--good)", ""),
        ("phrase, LR no bias · logit lens", S["methods"]["lr_nobias"]["lens_profile"]["logit_lens"]["phrase_top10_by_layer"], "var(--good)", "4 3"),
        ("phrase, LR with bias · J-lens", S["methods"]["lr"]["lens_profile"]["jlens"]["phrase_top10_by_layer"], "var(--warn)", ""),
        ("phrase, Brennan · J-lens", S["methods"]["proj"]["lens_profile"]["jlens"]["phrase_top10_by_layer"], "var(--bad)", ""),
    ]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Top-10 rate by layer">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pl}" y1="{ys(v):.1f}" x2="{W-pr}" y2="{ys(v):.1f}" stroke="var(--rule)" stroke-width="1"/>'
                   f'<text x="{pl-6}" y="{ys(v)+4:.1f}" text-anchor="end" class="ax">{v:.2f}</text>')
    for i, l in enumerate(LAYERS):
        out.append(f'<text x="{xs(i):.1f}" y="{H-pb+16}" text-anchor="middle" class="ax">L{l}</text>')
    for name, vals, color, dash in series:
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
        out.append(f'<circle cx="{xs(len(vals)-1):.1f}" cy="{ys(vals[-1]):.1f}" r="3" fill="{color}"/>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg"><i style="border-top:2px {"dashed" if d else "solid"} {c}"></i>{esc(n)}</span>' for n, _, c, d in series)
    return "\n".join(out), legend


svg, legend = profile_svg()


def confusion(mth):
    C = S["methods"][mth]["confusion"]
    short = [("JQA" if p == "John Quincy Adams" else p.split()[-1][:5]) for p in PHR]
    h = ['<table class="conf"><thead><tr><th></th>' + "".join(f"<th>{esc(s)}</th>" for s in short) + "</tr></thead><tbody>"]
    for i, p in enumerate(PHR):
        cells = "".join(f'<td style="--a:{0.06 + 0.94 * C[i][j]:.2f}" class="{"diag" if i == j else ""}">{C[i][j]:.2f}</td>' for j in range(len(PHR)))
        h.append(f"<tr><th>{esc(p)}</th>{cells}</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


def lens_block(idx, keep=("L20", "L24", "L26", "L30", "model")):
    prompt = S["methods"]["lr_nobias"]["prompts"][idx]["prompt"]
    out = [f'<h4>“{esc(prompt)}”</h4><div class="lens">']
    for h in ("proj", "lda", "lr_nobias"):
        out.append(f'<div class="lens-col"><div class="lens-head">{LABEL.get(h, "LDA / whitening")}</div>')
        for r in S["methods"][h]["prompts"][idx]["rows"]:
            if r["layer"] not in keep:
                continue
            chips = "".join(f'<span class="chip new">{esc(t[1:-1])}</span>' if t.startswith("[") else f'<span class="chip">{esc(t.replace(chr(10), "⏎"))}</span>'
                            for t in r["top10"][:7])
            out.append(f'<div class="lens-row"><span class="layer">{r["layer"]}</span>{chips}</div>')
        out.append("</div>")
    out.append("</div>")
    return "\n".join(out)


per_rows = "\n".join(
    f'<tr><td>{esc(p)}</td><td class="num">{v["all"]["auroc_generic"]:.3f}</td>'
    f'<td class="num">{"—" if v["all"].get("auroc_first_sib") is None else f"{v['all']['auroc_first_sib']:.3f}"}</td>'
    f'<td class="num">{v["all"]["rank_median"]:.0f}</td><td class="num">{v["all"]["first_rank_median"]:.0f}</td>'
    f'<td class="num">{v["all"]["top10"]:.2f}</td><td class="num">{v["mass_ratio"]:.1f}</td>'
    f'<td class="num">{S["methods"]["lr_nobias"]["lens_profile"]["jlens"]["earliest_top10_layer_median"][p]}</td></tr>'
    for p, v in S["methods"]["lr_nobias"]["per_phrase"].items())

stab = R["stability"]
stab_rows = "\n".join(
    f'<tr><td>{ {"proj": "Brennan (proj + unit)", "lda": "LDA / whitening", "lr": "LR with bias", "lr_nobias": "LR, no bias"}[h] }</td>'
    f'<td class="num">{v["auroc_generic"]["mean"]:.4f} ± {v["auroc_generic"]["std"]:.4f}</td>'
    f'<td class="num">{v["top10"]["mean"]:.3f} ± {v["top10"]["std"]:.3f}</td>'
    f'<td class="num">{v["generic_top10"]["mean"]:.4f} ± {v["generic_top10"]["std"]:.4f}</td>'
    f'<td class="num">{v["new_mass"]["mean"]:.4f} ± {v["new_mass"]["std"]:.4f}</td></tr>' for h, v in stab.items())

copy_rows = "\n".join(
    f'<tr><td>{LABEL.get(h, "LR with bias")}</td><td class="num">{mean_of(S["methods"][h],"auroc_generic","single"):.3f}</td>'
    f'<td class="num">{mean_of(S["methods"][h],"auroc_generic","copy"):.3f}</td><td class="num">{mean_of(S["methods"][h],"top10","single"):.2f}</td>'
    f'<td class="num">{mean_of(S["methods"][h],"top10","copy"):.2f}</td></tr>' for h in ("proj", "lda", "lr", "lr_nobias"))

early = {h: S["methods"][h]["lens_profile"]["jlens"]["phrase_top10_by_layer"] for h in ("proj", "lda", "lr", "lr_nobias")}
first = S["methods"]["lr_nobias"]["lens_profile"]["jlens"]["first_tok_top10_by_layer"]
only = S["methods"]["lr_nobias"]["lens_profile"]["jlens"]["phrase_only_by_layer"]
li = {l: i for i, l in enumerate(LAYERS)}

page = f"""<title>Phrase-Head Benchmark</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  :root {{
    --paper: #f7f6f1; --ink: #1c2028; --muted: #676d78; --rule: #d8d5cc; --soft: #ebe9e2;
    --good: #0f766e; --good-bg: #e2f1ee; --bad: #b45309; --bad-bg: #f8ead9; --warn: #a16207; --new: #7c3aed; --new-bg: #ede6fb;
    --serif: "Fraunces", "Iowan Old Style", Georgia, serif; --sans: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
    --mono: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228;
    --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; --bad-bg: #3d2a14; --warn: #e0b04a; --new: #c4a7f7; --new-bg: #2d2342; }} }}
  :root[data-theme="dark"] {{
    --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228;
    --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; --bad-bg: #3d2a14; --warn: #e0b04a; --new: #c4a7f7; --new-bg: #2d2342; }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 16.5px; line-height: 1.5; padding-inline: 16px; padding-block: 0; }}
  main {{ max-width: 78ch; margin: 0 auto; padding-block: 40px 64px; }}
  h1 {{ font-family: var(--serif); font-weight: 500; font-size: 2.3rem; line-height: 1.1; margin: 0 0 8px; text-wrap: balance; }}
  h2 {{ font-family: var(--serif); font-weight: 600; font-size: 1.45rem; margin: 40px 0 10px; text-wrap: balance; }}
  h4 {{ font-family: var(--serif); font-weight: 500; font-size: 1.05rem; margin: 20px 0 8px; }}
  .kicker {{ font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }}
  .lede {{ font-size: 1.12rem; max-width: 66ch; }} p {{ max-width: 66ch; margin: 0 0 12px; }}
  .verdict {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 22px 0 8px; }}
  .verdict div {{ border-top: 3px solid var(--rule); padding-top: 8px; }}
  .verdict b {{ display: block; font-family: var(--serif); font-size: 1.05rem; font-weight: 600; margin-bottom: 2px; }}
  .verdict .bad {{ border-color: var(--bad); }} .verdict .good {{ border-color: var(--good); }} .verdict .warn {{ border-color: var(--warn); }}
  .verdict span {{ font-size: 0.93rem; color: var(--muted); }}
  .tbl {{ overflow-x: auto; margin: 12px 0 6px; }}
  table {{ border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  th {{ font-weight: 600; font-size: 0.74rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
  td.num {{ font-family: var(--mono); font-size: 0.8rem; text-align: right; }} th.num {{ text-align: right; }}
  tr.bad td {{ background: var(--bad-bg); }} tr.good td {{ background: var(--good-bg); }}
  .cap {{ font-size: 0.86rem; color: var(--muted); max-width: 66ch; margin: 4px 0 0; }}
  .conf td {{ font-family: var(--mono); font-size: 0.72rem; text-align: right; padding: 3px 6px; background: color-mix(in srgb, var(--new) calc(var(--a) * 100%), var(--paper)); border-bottom: 1px solid var(--paper); }}
  .conf th {{ font-size: 0.66rem; text-transform: none; letter-spacing: 0; padding: 3px 6px; }}
  .conf td.diag {{ outline: 1.5px solid var(--ink); outline-offset: -1.5px; }}
  .confs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }}
  .lens {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
  .lens-col {{ background: var(--soft); padding: 10px 12px; }}
  .lens-head {{ font-weight: 600; font-size: 0.82rem; margin-bottom: 6px; }}
  .lens-row {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin: 4px 0; }}
  .layer {{ font-family: var(--mono); font-size: 0.7rem; color: var(--muted); width: 44px; flex: none; }}
  .chip {{ font-family: var(--mono); font-size: 0.72rem; padding: 1px 6px; border: 1px solid var(--rule); background: var(--paper); white-space: pre; }}
  .chip.new {{ background: var(--new-bg); border-color: var(--new); font-weight: 500; }}
  svg {{ width: 100%; max-width: 100%; height: auto; display: block; }} .ax {{ font-family: var(--mono); font-size: 11px; fill: var(--muted); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px 18px; font-size: 0.82rem; margin: 6px 0 4px; }}
  .lg i {{ display: inline-block; width: 22px; margin-right: 6px; vertical-align: middle; }}
  code {{ font-family: var(--mono); font-size: 0.84em; background: var(--soft); padding: 0 4px; }}
  ul {{ max-width: 66ch; padding-left: 20px; }} li {{ margin-bottom: 6px; }}
  .foot {{ margin-top: 40px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 0.86rem; color: var(--muted); }}
</style>
<main>
<div class="kicker">Qwen3.5-9B-Base · 15 phrases · {meta["n_held"]:,} held-out positions · {meta["n_generic_eval_positions"]:,} generic positions · 2026-09-12</div>
<h1>Phrase-head benchmark: which multi-token rows can a lens trust?</h1>
<p class="lede">A frozen evaluation set with confusable phrases, copy buckets, a random-row floor and the model's own rows as ceiling, scored on direction, calibration, rank behaviour, J-lens versus logit-lens layer profiles, and refit stability. The result: Brennan's rows detect the phrase's first token, not the phrase; a bias term is a lens hazard; a bias-free logistic head behaves like a real unembedding row.</p>

<div class="verdict">
  <div class="bad"><b>Brennan's head</b><span>New Hampshire row fires at 97% of pre-“New York” positions · AUROC vs sibling 0.59 · in the top-10 at 20% of random text</span></div>
  <div><b>LDA / whitening</b><span>false positives gone · sibling AUROC 0.66 · but 74× prior mass, and rank 1 where the model itself ranks “ New” at 8</span></div>
  <div class="warn"><b>LR with bias</b><span>best final-layer calibration · but the +8 bias puts the phrase in the lens top-10 at 49% of contexts at layer 8, where the first token is at 0%</span></div>
  <div class="good"><b>LR, no bias</b><span>AUROC 0.981 · mass 1.0× prior · sibling AUROC 0.73 · rank within 1.25× of the model's own first-token rank · clean early layers</span></div>
</div>

<h2>Evaluation set</h2>
<p>Brennan's ten phrases plus five confusables: New York and New Jersey (share “ New” with New Hampshire), John Quincy Adams (shares “ John” with John Adams), Massachusetts and Vermont (state-category siblings). Every phrase has over 4,200 unique FineWeb contexts except John Quincy Adams (1,317). Per phrase, 200 contexts are held out ({meta["held_single_frac"]:.0%} single-occurrence, the rest contain the phrase earlier); training subsets of 150, 600 and 2,400 are nested. Generic text: 2,000 documents for the baseline and negatives, 500 further documents held out. The state is the post-final-norm residual at the token before the phrase. For every held-out context the residual at 11 layers is also transported through the J-lens and read through the plain logit lens.</p>
<p>Heads: Brennan's <code>PRIOR_REPRESENTATION_EMBED_PROJ</code>; average of the phrase's real rows (for the six single-token phrases this is the real row, so it doubles as the ceiling); LDA with the Gaussian log-odds bias; logistic regression over the extended softmax with a bias and a prior shift; logistic regression <em>without</em> a bias, with positives weighted to the corpus prior; a random unit row as floor. L2 is chosen on a held-out validation loss at the training mixture.</p>

<h2>Headline at 2,400 contexts per phrase, mean over 15 phrases</h2>
<div class="tbl"><table>
<thead><tr><th>head</th><th class="num">AUROC vs generic</th><th class="num">vs first-token sibling</th><th class="num">vs category sibling</th><th class="num">own top-10</th><th class="num">own median rank</th><th class="num">log(rank ÷ first-token rank)</th><th class="num">generic top-10</th><th class="num">mass ÷ prior</th></tr></thead>
<tbody>{"".join(head_rows)}</tbody></table></div>
<p class="cap">Sibling AUROCs score the phrase logit at its own pre-phrase positions against the positions of phrases sharing its first token (New Hampshire vs New York, New Jersey; John Adams vs John Quincy Adams) or its category. The rank ratio compares the phrase token's rank at its own positions to the rank of the phrase's real first token there; zero means the row is as confident as the model is about the first token. Three disjoint 150-context refits move every AUROC by at most 0.0008.</p>

<h2>The first-token confound, measured</h2>
<p>Heads fitted on the original ten phrases only, never shown New York, New Jersey or John Quincy Adams, then scored at those phrases' pre-phrase positions.</p>
<div class="tbl"><table>
<thead><tr><th>head</th><th>phrase</th><th class="num">AUROC own vs first-token sibling</th><th class="num">in top-10 at sibling positions</th><th class="num">AUROC own vs category</th></tr></thead>
<tbody>{"".join(sib_rows)}</tbody></table></div>
<p class="cap">The pre-“New” state does carry which “New …” is coming: a fitted head separates New Hampshire from New York and New Jersey at 0.85 without ever seeing them. John Adams versus John Quincy Adams is at chance for every head.</p>

<h2>Layer profiles over all held-out contexts</h2>
<p>Fraction of the {meta["n_held"]:,} held-out contexts where the token is in the extended top-10 at each layer. The real first token is the standard readout; the J-lens surfaces it earlier than the logit lens, as published. A bias-free phrase row tracks that curve. A biased row, or Brennan's, is already “in the top-10” at layer 8, where nothing real is.</p>
<div class="legend">{legend}</div>
{svg}
<div class="tbl"><table>
<thead><tr><th>J-lens, top-10 rate</th><th class="num">L4</th><th class="num">L8</th><th class="num">L16</th><th class="num">L24</th><th class="num">L30</th></tr></thead>
<tbody>
<tr><td>real first token</td>{"".join(f'<td class="num">{first[li[l]]:.2f}</td>' for l in (4, 8, 16, 24, 30))}</tr>
<tr class="bad"><td>phrase, Brennan</td>{"".join(f'<td class="num">{early["proj"][li[l]]:.2f}</td>' for l in (4, 8, 16, 24, 30))}</tr>
<tr><td>phrase, LDA</td>{"".join(f'<td class="num">{early["lda"][li[l]]:.2f}</td>' for l in (4, 8, 16, 24, 30))}</tr>
<tr><td>phrase, LR with bias</td>{"".join(f'<td class="num">{early["lr"][li[l]]:.2f}</td>' for l in (4, 8, 16, 24, 30))}</tr>
<tr class="good"><td>phrase, LR no bias</td>{"".join(f'<td class="num">{early["lr_nobias"][li[l]]:.2f}</td>' for l in (4, 8, 16, 24, 30))}</tr>
<tr><td>phrase in top-10 while first token is not, LR no bias</td>{"".join(f'<td class="num">{only[li[l]]:.2f}</td>' for l in (4, 8, 16, 24, 30))}</tr>
</tbody></table></div>
<p class="cap">The last row is the population where a multi-token row shows something the single-token readout does not: 13 to 25% of contexts through the middle layers. Its early-layer end should be treated with suspicion until the row's early-layer false-positive rate is characterised.</p>

<h2>Copy sensitivity and stability</h2>
<div class="tbl"><table>
<thead><tr><th>head</th><th class="num">AUROC, single-occurrence</th><th class="num">AUROC, copy contexts</th><th class="num">own top-10, single</th><th class="num">own top-10, copy</th></tr></thead>
<tbody>{copy_rows}</tbody></table></div>
<p class="cap">Copy contexts (phrase already appeared earlier) are easier, but no head depends on them.</p>
<div class="tbl"><table>
<thead><tr><th>three disjoint 150-context refits</th><th class="num">AUROC</th><th class="num">own top-10</th><th class="num">generic top-10</th><th class="num">new-token mass</th></tr></thead>
<tbody>{stab_rows}</tbody></table></div>

<h2>Cross-phrase confusion at 2,400 contexts</h2>
<p>Rows: held-out positions right before the named phrase. Cells: how often the column phrase's token enters the top-10 there.</p>
<div class="confs">
<div><h4>Brennan (proj + unit)</h4>{confusion("proj")}</div>
<div><h4>LDA / whitening</h4>{confusion("lda")}</div>
<div><h4>LR, no bias</h4>{confusion("lr_nobias")}</div>
</div>

<h2>Bias-free logistic head, per phrase</h2>
<div class="tbl"><table>
<thead><tr><th>phrase</th><th class="num">AUROC vs generic</th><th class="num">vs first-token sibling</th><th class="num">own rank</th><th class="num">first-token rank</th><th class="num">own top-10</th><th class="num">mass ÷ prior</th><th class="num">median earliest top-10 layer (J-lens)</th></tr></thead>
<tbody>{per_rows}</tbody></table></div>
<p class="cap">99 in the last column means the phrase token never enters the top-10 in more than half of that phrase's contexts. John Adams and John Quincy Adams split the “ John” evidence between them and both rank poorly.</p>

<h2>J-lens readouts on hand prompts (2,400 contexts)</h2>
<p>Top-7 of the extended vocabulary; highlighted chips are the added phrase tokens.</p>
{lens_block(9)}
{lens_block(5)}
{lens_block(4)}

<h2>What to adopt</h2>
<ul>
<li>Fit phrase rows by logistic regression over the extended softmax <b>without a bias</b>, positives weighted to the corpus prior, L2 about 1e-6, L-BFGS on the stored states. This is the head for <code>extend_model.py</code>.</li>
<li>Report for every phrase row: AUROC vs generic and vs first-token siblings, mass vs prior, and rank relative to the phrase's first token. A row that fails the sibling test is a first-token detector.</li>
<li>When reading the lens with phrase rows, show the phrase's first token alongside, and treat phrase hits below layer 16 as suspect until the row's early-layer false-positive rate is known.</li>
<li>Do not build claims on pairs like John Adams and John Quincy Adams; the pre-phrase state does not separate them.</li>
</ul>
<div class="foot">Code and tables: <code>~/notes/brennan-jlens-review/modal_exp/</code> (Modal app <code>jlens-bench</code>). Findings note: <code>BENCHMARK_2026-09-12.md</code>. Earlier calibration study: the “Multi-token J-lens Heads” page.</div>
</main>
"""
out = os.path.join(here, "results", "bench.html")
open(out, "w").write(page)
print("wrote", out, len(page))
