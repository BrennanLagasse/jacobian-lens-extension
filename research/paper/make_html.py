"""Build report.html, the HTML companion of main.tex, from tables_html.json and figures/*.dat."""
import html
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(here, "tables_html.json")))
H, S = T["html"], T["scalars"]


def dat(name):
    rows = [l.split() for l in open(os.path.join(here, "figures", f"{name}.dat")).read().strip().splitlines()]
    cols = rows[0]
    return {c: [float(r[i]) if c != "label" and c != "layers" else r[i] for r in rows[1:]] for i, c in enumerate(cols)}


def line_chart(series, xs, W=680, H_=300, ylab="", xlab="layer"):
    pl, pr, pt, pb = 46, 16, 14, 40
    n = len(xs)
    X = lambda i: pl + i * (W - pl - pr) / (n - 1)
    Y = lambda v: pt + (1 - v) * (H_ - pt - pb)
    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="{html.escape(ylab)}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W-pr}" y2="{Y(v):.1f}" stroke="var(--rule)"/><text x="{pl-6}" y="{Y(v)+4:.1f}" text-anchor="end" class="ax">{v:.2f}</text>')
    step = max(1, n // 12)
    for i, x in enumerate(xs):
        if i % step == 0 or i == n - 1:
            out.append(f'<text x="{X(i):.1f}" y="{H_-pb+16}" text-anchor="middle" class="ax">L{int(x)}</text>')
    out.append(f'<text x="{(pl+W-pr)/2:.0f}" y="{H_-6}" text-anchor="middle" class="ax">{xlab}</text>')
    for name, vals, color, dash in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg"><i style="border-top:2px {"dashed" if d else "solid"} {c}"></i>{html.escape(nm)}</span>' for nm, _, c, d in series)
    return f'<div class="legend">{legend}</div>' + "\n".join(out)


def bar_chart(d, W=680, H_=280):
    pl, pr, pt, pb = 46, 16, 14, 64
    n = len(d["idx"]); lo, hi = -0.1, 0.35
    X = lambda i: pl + (i + 0.5) * (W - pl - pr) / n
    Y = lambda v: pt + (hi - v) / (hi - lo) * (H_ - pt - pb)
    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="ablation">']
    for v in (-0.1, 0, 0.1, 0.2, 0.3):
        out.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{W-pr}" y2="{Y(v):.1f}" stroke="var(--rule)"/><text x="{pl-6}" y="{Y(v)+4:.1f}" text-anchor="end" class="ax">{v:+.1f}</text>')
    bw = (W - pl - pr) / n * 0.5
    for i in range(n):
        m, l, h = d["loss"][i], d["lo"][i], d["hi"][i]
        color = "var(--good)" if "lens" in d["label"][i] and "logit" not in d["label"][i] else "var(--muted)"
        out.append(f'<rect x="{X(i)-bw/2:.1f}" y="{min(Y(m),Y(0)):.1f}" width="{bw:.1f}" height="{abs(Y(m)-Y(0)):.1f}" fill="{color}" opacity="0.85"/>')
        out.append(f'<line x1="{X(i):.1f}" y1="{Y(l):.1f}" x2="{X(i):.1f}" y2="{Y(h):.1f}" stroke="var(--ink)" stroke-width="1.5"/>')
        out.append(f'<text x="{X(i):.1f}" y="{H_-pb+14}" text-anchor="end" transform="rotate(-30 {X(i):.1f} {H_-pb+14})" class="ax">{html.escape(d["label"][i].replace("~", " "))}</text>')
    out.append("</svg>")
    return "\n".join(out)


bp = dat("bench_profiles")
fig_profiles = line_chart([("real first token, J-lens", bp["first"], "var(--ink)", ""), ("real first token, logit lens", bp["first_logit"], "var(--ink)", "4 3"),
                           ("phrase, Brennan", bp["proj"], "var(--bad)", ""), ("phrase, LR with bias", bp["lr"], "var(--warn)", ""),
                           ("phrase, LDA", bp["lda"], "var(--blue)", ""), ("phrase, LR no bias", bp["lrnb"], "var(--good)", ""),
                           ("phrase in top-10, first token not", bp["lrnb_only"], "var(--good)", "4 3")], bp["layer"], ylab="fraction in top-10")
p10 = dat("pass10")
fig_pass10 = line_chart([("R-lens", p10["R"], "var(--good)", ""), ("J-lens", p10["J"], "var(--ink)", ""), ("logit lens", p10["logit"], "var(--muted)", "4 3")], p10["layer"], ylab="pass@10")
rp = dat("rlens_phrases")
fig_rp = line_chart([("first token, R-lens", rp["first_R"], "var(--ink)", ""), ("first token, J-lens", rp["first_J"], "var(--ink)", "4 3"),
                     ("phrase (LR no bias), R-lens", rp["lr_R"], "var(--good)", ""), ("phrase (LR no bias), J-lens", rp["lr_J"], "var(--good)", "4 3"),
                     ("phrase (LR no bias), logit lens", rp["lr_logit"], "var(--muted)", "2 3"), ("phrase (Brennan), R-lens", rp["proj_R"], "var(--bad)", "")], rp["layer"], ylab="fraction in top-10")
ab = dat("ablation")
fig_abl = bar_chart(ab)
A = S["abl"]

page = f"""<title>Multi-token Readouts for the Jacobian Lens</title>
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
  h3 {{ font-family: var(--serif); font-weight: 600; font-size: 1.12rem; margin: 26px 0 8px; }}
  .kicker {{ font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }}
  .lede {{ font-size: 1.1rem; max-width: 68ch; }} p {{ max-width: 68ch; margin: 0 0 12px; }}
  .abstract {{ background: var(--soft); padding: 14px 18px; max-width: 68ch; font-size: 0.97rem; margin: 18px 0; }}
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
  code {{ font-family: var(--mono); font-size: 0.84em; background: var(--soft); padding: 0 4px; }} ul, ol {{ max-width: 68ch; padding-left: 20px; }} li {{ margin-bottom: 6px; }}
  .eq {{ font-family: var(--mono); font-size: 0.86rem; background: var(--soft); padding: 8px 12px; margin: 8px 0 12px; max-width: 68ch; white-space: pre-wrap; }}
  .foot {{ margin-top: 40px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 0.86rem; color: var(--muted); }}
</style>
<main>
<div class="kicker">Internal report · Ryan &amp; Brennan Lagasse · Qwen3.5-9B-Base · September 12–13, 2026 · companion to main.pdf</div>
<h1>Multi-token readouts for the Jacobian lens: a calibration study, a benchmark, and the R-lens</h1>
<p class="lede">This page tells the story; the LaTeX paper (<code>paper/main.pdf</code>) carries every definition, every table and every caveat. Both are generated from the same result files by the same script, so the numbers agree.</p>

<div class="abstract">Brennan's <code>multi_token</code> extension adds rows to the unembedding matrix so the Jacobian lens can name phrases. In his notebook those rows fill the top-10 of every readout. On the correct model that turns out to be a calibration failure that data cannot fix: the rows are mean-difference directions with ten times a real row's logit spread, identical at 150, 600 and 2,400 contexts. Whitening removes the false positives but is over-confident and bleeds within categories. A logistic regression over the extended softmax is calibrated; without a bias term it behaves like a real unembedding row at every lens layer. A benchmark with confusable phrases shows Brennan's rows detect the phrase's <em>first token</em>, that the state before it nevertheless carries phrase identity which fitted heads read, and that John Adams and John Quincy Adams are not separable there. The R-lens, a matched pair fitted on the base model, reads two-hop intermediates slightly earlier, removes the early-layer phrase hits the benchmark flagged, and is as causally load-bearing as the J-lens but not more so.</div>

<div class="verdict">
  <div class="bad"><b>Brennan's rows</b><span>logit std 18 vs 1.8 · in the top-10 at 20% of random text · unchanged from 150 to 2,400 contexts · fire at 97% of positions before “New York”</span></div>
  <div><b>LDA / whitening</b><span>false positives gone · rank 1 at 70% of own positions · 74× prior mass · category bleed</span></div>
  <div class="warn"><b>LR with bias</b><span>best final-layer calibration · but +8 logit bias fires at 49% of contexts at layer 8 under the lens</span></div>
  <div class="good"><b>LR, no bias</b><span>AUROC 0.981 · mass 1.0× prior · sibling AUROC 0.73 · rank within 1.25× of the model's first-token rank · clean early layers</span></div>
  <div class="good"><b>R-lens</b><span>pass@10 {S['p10_mean']['rlens']:.2f} vs {S['p10_mean']['jlens']:.2f} (J) vs {S['p10_mean']['logit']:.2f} (logit) · ablation {A['rlens|all|(-2,)']['rel']:+.2f} vs {A['jlens|all|(-2,)']['rel']:+.2f} (J), controls ≈ 0 · early phrase hits 8% vs 17%</span></div>
</div>

<h2>1. Setup</h2>
<p>Model: <code>Qwen/Qwen3.5-9B-Base</code> (32 blocks, d = 4096, vocabulary 248k), matched to a Base-model lens (Brennan's notebook had applied a Base lens to the post-trained checkpoint). The Jacobian lens reads layer ℓ as W<sub>U</sub>·norm(J<sub>ℓ</sub>h<sub>ℓ</sub>); a phrase head is a matrix W and bias b applied to the same post-norm state, and the extended softmax runs over the real vocabulary plus the new rows.</p>
<p>Contexts come from Brennan's miner run over the whole FineWeb 10BT sample ({S['e1_docs']:,} documents): matching sentence plus two preceding sentences, case-insensitive; the state is taken at the token <em>before</em> the phrase. Held-out contexts and nested training subsets of 150 / 600 / 2,400 per phrase. Baseline mean and covariance come from 2,000 generic FineWeb documents (712k positions), which also serve as negatives for the fitted heads; further held-out documents give the generic evaluation positions. Each phrase's prior π<sub>p</sub> is its occurrence rate per token in the scanned corpus.</p>
{H['phrases']}

<h2>2. Heads</h2>
<ul>
<li><b>Brennan (proj):</b> unit-normalised phrase mean with the baseline mean projected out; no bias.</li>
<li><b>Average rows:</b> mean of the phrase's real W<sub>U</sub> rows; for the six single-token phrases this is the real row, i.e. the ceiling.</li>
<li><b>LDA:</b> Σ<sup>−1</sup>(μ<sub>p</sub> − h̄) with the Gaussian log-odds bias, log prior and mean log-sum-exp; a W<sub>U</sub>-scaled variant rescales the logit spread.</li>
<li><b>LR with bias:</b> rows and biases fitted by full-batch L-BFGS on the extended softmax with W<sub>U</sub> frozen (positives: pre-phrase positions → phrase token; negatives: in-context and generic positions → real next token), then bias-shifted from the training prior to the corpus prior; L2 chosen on held-out likelihood.</li>
<li><b>LR, no bias:</b> the same without biases, positives weighted to the corpus prior inside the objective, L2 swept down to 10<sup>−8</sup>.</li>
<li><b>Controls:</b> a random unit row (floor).</li>
</ul>

<h2>3. Experiment 1: calibration against training scale</h2>
<p>Brennan's ten phrases, 100 held-out contexts each, {S['e1_n_generic_eval']:,} generic positions. The symptom reproduces on the correct model: each row enters the model's top-10 at 19% of random positions, the ten together carry 32% of the softmax mass on ordinary text against a prior of {S['e1_prior_sum']:.1e}, and their logit spread over generic text is 18 against a real row's {S['e1_wu_std']:.2f}. The rows are numerically identical at 150, 600 and 2,400 contexts, so “undertrained” is ruled out for this construction.</p>
{H['e1_headline']}
<p>Whitening cuts the generic top-10 rate to 0.07% and puts the phrase at rank 1 on 60–70% of its own positions, but its log-odds are ten times too wide, so 60–70× the prior mass remains and category neighbours bleed in. The logistic head lands in the model's own units (mass 2.7e-5 against a prior of 7e-5, confusion below 0.10) at the cost of honest ranks: median rank 7–85 at its own positions, the same range as the model's own first token there.</p>
{H['e1_confusion']}
{H['e1_lr_ablation']}

<h2>4. The benchmark</h2>
<p>Five confusables were added (New York, New Jersey, John Quincy Adams, Massachusetts, Vermont), giving {S['b_n_held']:,} held-out positions ({S['b_single_frac']:.0%} single-occurrence), {S['b_n_generic_eval']:,} generic positions, lens states at 11 layers for every held-out context, three disjoint refits for stability, and the floor and ceiling controls. Every AUROC difference above 0.002 in the tables below is real; the refit spread is 0.0008.</p>
{H['bench_headline']}
{H['bench_scale']}
<h3>The first-token confound</h3>
<p>Heads fitted on Brennan's ten phrases only, never shown the confusables, scored at the confusables' positions. Brennan's New Hampshire row fires at 97% of positions before “New York” or “New Jersey” (AUROC 0.59); his John Adams row at 98% of positions before “John Quincy Adams” (AUROC 0.48). A logistic head fitted on the same ten phrases separates New Hampshire from its siblings at 0.85, so the state before “ New” does carry which “New …” is coming and a mean-difference row discards it. John Adams versus John Quincy Adams is at chance for every head.</p>
{H['bench_sibling']}
<h3>A bias term is a lens hazard</h3>
<p>The biased head is best calibrated at the final layer, but its +7 to +10 logit constant does not shrink when the lens transports an early-layer residual with compressed real logits: under the J-lens it puts the phrase in the top-10 at 49% of contexts at layer 8, where the real first token is at 0%. The bias-free head is at 14% there and tracks the real first token's curve.</p>
{fig_profiles}
{H['bench_profiles']}
<h3>The bias-free head</h3>
<p>Best or joint-best direction, mass at prior, and a rank at its own positions within a factor 1.25 of the model's own first-token rank (LDA is four times more confident than the model about “ New”; Brennan's is fifteen times). Copy contexts are easier for every fitted head, but single-occurrence performance stays high. The dashed line in the figure, the phrase in the top-10 while its first token is not, is the population where a multi-token row shows something the single-token readout does not: 13–25% of contexts through layers 8–24.</p>
{H['bench_perphrase']}
{H['bench_copy']}
{H['bench_stability']}

<h2>5. The R-lens</h2>
<p>The R-lens is the same estimator with three LRP rules in the backward pass: the RMSNorm scale is a constant, SiLU's sigmoid factor is detached, and the SwiGLU product's gradient is split evenly (β = 0.5). The configuration, target layer 30, 25 pile-10k prompts and skip-4 are copied from the provenance inside the authors' Qwen3.5-9B lens. The rules are grafted onto the standard forward so the forward pass is bit-identical, and a matched J-lens / R-lens pair was fitted on the base model (21 minutes each).</p>
<h3>pass@10 on 50 two-hop questions</h3>
<p>Whether the intermediate entity's first token (e.g. France for “the country where the Eiffel Tower stands”) is in the lens top-10 at the penultimate token. Both lenses read it from layer 6, where the logit lens shows nothing until layer 24; the R-lens leads at layers 4–6 and 22–26.</p>
{fig_pass10}
{H['rlens_pass10']}
<h3>Direction ablation</h3>
<p>The unit direction of the intermediate's first token, pulled back through each lens, is projected out of the residual at the penultimate token during the prompt pass, at layers 0–15 or all layers; eight samples per prompt; a random direction is the control. {S['at_n_good']} of {S['at_n_q']} questions with baseline accuracy ≥ 0.5 (mean {S['at_base']:.3f}) enter the summary.</p>
{fig_abl}
{H['rlens_ablation']}
{H['rlens_paired']}
<p>The lens directions are causally load-bearing and the logit-lens direction is not: one rank-one removal per layer costs 14–16% of two-hop accuracy for either lens and nothing for the controls; {S['at_wipe']} questions go from correct to 0/8 under either lens direction and stay correct under the random one. R-lens and J-lens are not distinguishable by this test on this model: the post's larger R-lens effect does not reproduce here. Removing the direction at the last token too doubles the damage and makes even the logit-lens direction bite, which is why the penultimate position isolates the lens-specific contribution.</p>
{H['rlens_examples']}
<h3>The phrase benchmark under the matched lenses</h3>
<p>Under the R-lens the early-layer phrase hits shrink (layers 4 / 8 / 12: 0 / 8 / 10% against 6 / 17 / 20% under the J-lens), the late layers are unchanged, and the real first token surfaces slightly earlier. Brennan's head stays saturated under either lens: its problem is the row.</p>
{fig_rp}
{H['rlens_phrases']}

<h2>6. What to change in the fork</h2>
<ol>
<li>Use the Base model with a Base lens, or fit a lens on the post-trained model.</li>
<li>Replace <code>PRIOR_REPRESENTATION_EMBED*</code> with the bias-free logistic head (no biases, positives weighted to the corpus prior, L-BFGS, L2 ≈ 1e-6). The states from <code>collect_embeddings.py</code> plus a generic-corpus pass are all that is needed; fitting takes under a minute.</li>
<li>Keep the h̄ / Σ pipeline only as a closed-form fallback and label its logits as over-confident.</li>
<li>Before any lens plot, report per row: AUROC vs generic and vs first-token siblings, mass vs prior, rank relative to the phrase's first token. A row that fails the sibling test is a first-token detector.</li>
<li>Read the lens with the R-lens, show the phrase's first token alongside, and treat phrase hits below layer 16 as suspect until the row's early-layer false-positive rate is known.</li>
<li>Use the ablation protocol with logit-lens and random controls as the causal check for any new head.</li>
</ol>
<p><b>Limitations.</b> One model at one size; Brennan's two-sentence contexts and copy bucket; fifty hand-written two-hop questions with alias matching; the post's projection operator is our reading; lenses fitted on 25 prompts as in the authors' artifacts. Full statements in the paper.</p>

<div class="foot">Paper: <code>~/notes/brennan-jlens-review/paper/main.pdf</code> (source <code>main.tex</code>, tables from <code>make_tables.py</code>). Code: <code>~/notes/brennan-jlens-review/modal_exp/</code>. Earlier pages: “Multi-token J-lens Heads”, “Phrase-Head Benchmark”, “R-lens Attribution”.</div>
</main>
"""
out = os.path.join(here, "report.html")
open(out, "w").write(page)
print("wrote", out, len(page))
