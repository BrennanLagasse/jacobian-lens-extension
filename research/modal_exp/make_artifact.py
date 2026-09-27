"""Render results/results.json into a self-contained HTML report page."""
import html
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(here, "results", "results.json")))
meta, scales = R["meta"], R["scales"]
PHR = list(meta["pi_true"])
LABEL = {"proj": "Brennan (proj + unit)", "avgtok": "avg W_U rows", "lda": "LDA / whitening",
         "lda_wu": "LDA, W_U-scaled", "lr": "logistic regression"}
ORDER = ["proj", "avgtok", "lda", "lda_wu", "lr"]
prior_sum = sum(meta["pi_true"].values())


def agg(m, k):
    return float(np.mean([v[k] for v in m["per_phrase"].values()]))


def esc(s):
    return html.escape(s, quote=True)


rows = []
for sc, S in scales.items():
    for mth in ORDER:
        m = S["methods"][mth]
        cls = {"proj": "bad", "lr": "good"}.get(mth, "")
        rows.append(
            f'<tr class="{cls}"><td class="num">{sc}</td><td>{LABEL[mth]}</td>'
            f'<td class="num">{agg(m,"held_top10"):.2f}</td>'
            f'<td class="num">{np.median([v["held_rank_median"] for v in m["per_phrase"].values()]):.0f}</td>'
            f'<td class="num">{agg(m,"generic_top10_rate"):.4f}</td>'
            f'<td class="num">{m["generic_new_mass"]:.1e}</td>'
            f'<td class="num">{agg(m,"auroc"):.3f}</td>'
            f'<td class="num">{agg(m,"generic_mean"):.1f} ± {agg(m,"generic_std"):.1f}</td></tr>'
        )
headline = "\n".join(rows)


def confusion(mth):
    C = scales["2400"]["methods"][mth]["confusion"]
    short = [p.split()[-1][:5] for p in PHR]
    h = ['<table class="conf"><thead><tr><th></th>' + "".join(f"<th>{esc(s)}</th>" for s in short) + "</tr></thead><tbody>"]
    for i, p in enumerate(PHR):
        cells = ""
        for j in range(len(PHR)):
            v = C[i][j]
            a = 0.08 + 0.92 * v
            cells += f'<td style="--a:{a:.2f}" class="{"diag" if i == j else ""}">{v:.2f}</td>'
        h.append(f"<tr><th>{esc(p)}</th>{cells}</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


def lens_block(prompt_idx, layers_keep=("L16", "L20", "L22", "L24", "L26", "L30", "model")):
    out = []
    prompt = scales["2400"]["methods"]["proj"]["lens"][prompt_idx]["prompt"]
    out.append(f'<h4>“{esc(prompt)}”</h4><div class="lens">')
    for mth in ("proj", "lda", "lr"):
        out.append(f'<div class="lens-col"><div class="lens-head">{LABEL[mth]}</div>')
        for r in scales["2400"]["methods"][mth]["lens"][prompt_idx]["rows"]:
            if r["layer"] not in layers_keep:
                continue
            chips = ""
            for t in r["top10"][:7]:
                if t.startswith("["):
                    chips += f'<span class="chip new">{esc(t[1:-1])}</span>'
                else:
                    shown = t.replace("\n", "⏎")
                    chips += f'<span class="chip">{esc(shown)}</span>'
            out.append(f'<div class="lens-row"><span class="layer">{r["layer"]}</span>{chips}</div>')
        out.append("</div>")
    out.append("</div>")
    return "\n".join(out)


lr24 = scales["2400"]["methods"]["lr"]["per_phrase"]
lda24 = scales["2400"]["methods"]["lda"]["per_phrase"]
rank_rows = "\n".join(
    f'<tr><td>{esc(p)}</td><td class="num">{lda24[p]["held_rank_median"]:.0f}</td>'
    f'<td class="num">{lr24[p]["held_rank_median"]:.0f}</td><td class="num">{lr24[p]["held_first_tok_rank_median"]:.0f}</td>'
    f'<td class="num">{lr24[p]["auroc"]:.3f}</td></tr>'
    for p in PHR
)
sweep_rows = "\n".join(
    f'<tr><td class="num">{sc}</td><td class="num">{r["lam"]:g}</td><td class="num">{r["held_nll_generic"]:.5f}</td>'
    f'<td class="num">{r["held_nll_pos_mean"]:.2f}</td><td class="num">{r["row_norm_mean"]:.2f}</td>'
    f'<td>{"chosen" if r["lam"] == S["lr_lam"] else ""}</td></tr>'
    for sc, S in scales.items() for r in S["lr_sweep"]
)

page = f"""<title>Multi-token J-lens Heads</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  :root {{
    --paper: #f7f6f1; --ink: #1c2028; --muted: #676d78; --rule: #d8d5cc; --soft: #ebe9e2;
    --good: #0f766e; --good-bg: #e2f1ee; --bad: #b45309; --bad-bg: #f8ead9; --new: #7c3aed; --new-bg: #ede6fb;
    --serif: "Fraunces", "Iowan Old Style", Georgia, serif;
    --sans: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
    --mono: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228;
    --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; --bad-bg: #3d2a14; --new: #c4a7f7; --new-bg: #2d2342; }} }}
  :root[data-theme="dark"] {{
    --paper: #15171c; --ink: #e8e6df; --muted: #9aa0ab; --rule: #33373f; --soft: #1f2228;
    --good: #5fd1c3; --good-bg: #123a36; --bad: #f2a45a; --bad-bg: #3d2a14; --new: #c4a7f7; --new-bg: #2d2342; }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 16.5px; line-height: 1.5;
         padding-inline: 16px; padding-block: 0; }}
  main {{ max-width: 76ch; margin: 0 auto; padding-block: 40px 64px; }}
  h1 {{ font-family: var(--serif); font-weight: 500; font-size: 2.3rem; line-height: 1.1; margin: 0 0 8px; text-wrap: balance; }}
  h2 {{ font-family: var(--serif); font-weight: 600; font-size: 1.45rem; margin: 40px 0 10px; text-wrap: balance; }}
  h4 {{ font-family: var(--serif); font-weight: 500; font-size: 1.05rem; margin: 20px 0 8px; }}
  .kicker {{ font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }}
  .lede {{ font-size: 1.12rem; max-width: 66ch; }}
  p {{ max-width: 66ch; margin: 0 0 12px; }}
  .verdict {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 22px 0 8px; }}
  .verdict div {{ border-top: 3px solid var(--rule); padding-top: 8px; }}
  .verdict b {{ display: block; font-family: var(--serif); font-size: 1.05rem; font-weight: 600; margin-bottom: 2px; }}
  .verdict .bad {{ border-color: var(--bad); }} .verdict .good {{ border-color: var(--good); }}
  .verdict span {{ font-size: 0.93rem; color: var(--muted); }}
  .tbl {{ overflow-x: auto; margin: 12px 0 6px; }}
  table {{ border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  th {{ font-weight: 600; font-size: 0.74rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
  td.num {{ font-family: var(--mono); font-size: 0.8rem; text-align: right; }} th.num {{ text-align: right; }}
  tr.bad td {{ background: var(--bad-bg); }} tr.good td {{ background: var(--good-bg); }}
  .cap {{ font-size: 0.86rem; color: var(--muted); max-width: 66ch; margin: 4px 0 0; }}
  .conf td {{ font-family: var(--mono); font-size: 0.76rem; text-align: right; background: color-mix(in srgb, var(--new) calc(var(--a) * 100%), var(--paper)); border-bottom: 1px solid var(--paper); }}
  .conf th {{ font-size: 0.7rem; text-transform: none; letter-spacing: 0; }}
  .conf td.diag {{ outline: 1.5px solid var(--ink); outline-offset: -1.5px; }}
  .confs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
  .confs h4 {{ margin-top: 6px; }}
  .lens {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
  .lens-col {{ background: var(--soft); padding: 10px 12px; }}
  .lens-head {{ font-weight: 600; font-size: 0.82rem; margin-bottom: 6px; }}
  .lens-row {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin: 4px 0; }}
  .layer {{ font-family: var(--mono); font-size: 0.7rem; color: var(--muted); width: 44px; flex: none; }}
  .chip {{ font-family: var(--mono); font-size: 0.72rem; padding: 1px 6px; border: 1px solid var(--rule); background: var(--paper); white-space: pre; }}
  .chip.new {{ background: var(--new-bg); border-color: var(--new); color: var(--ink); font-weight: 500; }}
  code {{ font-family: var(--mono); font-size: 0.84em; background: var(--soft); padding: 0 4px; }}
  ul {{ max-width: 66ch; padding-left: 20px; }} li {{ margin-bottom: 6px; }}
  .foot {{ margin-top: 40px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 0.86rem; color: var(--muted); }}
</style>
<main>
<div class="kicker">Qwen3.5-9B-Base · Base lens · 10 phrases · 2026-09-12</div>
<h1>Multi-token J-lens heads: the rows are miscalibrated, not undertrained</h1>
<p class="lede">Brennan's phrase-prototype unembedding rows were rebuilt on the correct base model and compared against a whitened (LDA) head and a logistic-regression head at 150, 600 and 2,400 mined contexts per phrase. The original symptom reproduces at every scale; whitening removes the false positives; only the fitted head is calibrated like a real unembedding row.</p>

<div class="verdict">
  <div class="bad"><b>Brennan's head</b><span>logit std 18 vs 1.8 for a real row · in the top-10 at 19% of random positions · identical at 150 / 600 / 2,400 contexts</span></div>
  <div><b>LDA / whitening</b><span>false positives gone (0.07%) · rank 1 on 70% of own positions · but 70× prior mass and category bleed</span></div>
  <div class="good"><b>Logistic regression</b><span>AUROC 0.985 · mass at prior (2.7e-5 vs 7e-5) · confusion ≤ 0.10 · sensible lens readouts</span></div>
</div>

<h2>Setup</h2>
<p>Model <code>Qwen/Qwen3.5-9B-Base</code> (32 layers, d = 4096) with the matching Base lens from <code>neuronpedia/jacobian-lens</code>. Brennan's ten phrases and his miner recipe, run over the whole FineWeb 10BT sample ({meta["mine_stats"]["docs"]:,} documents); every phrase yielded over 4,200 unique contexts. Per phrase: 100 held-out contexts, then nested training subsets of 150, 600 and 2,400. The state is the post-final-norm residual at the token before the phrase, verified identical to what <code>lens.apply</code> unembeds. Baseline mean and covariance come from 2,000 FineWeb documents (his <code>embed_baseline.py</code> recipe); {meta["n_generic_eval_positions"]:,} positions from a further 200 documents are held out for the false-positive check.</p>
<p>All heads are applied to the same states, so only the rows differ. <em>proj</em> is his <code>PRIOR_REPRESENTATION_EMBED_PROJ</code>. <em>LDA</em> is Σ⁻¹(μ − h̄) with the Gaussian log-odds bias, log prior and mean log-sum-exp so it lives in logit units. <em>Logistic regression</em> fits the ten rows and biases by full-batch L-BFGS on the extended softmax with the real unembedding frozen, then shifts each bias from the training prior to the FineWeb prior. Four of the phrases (blackmail, plagiarism, cheating, Connecticut) are already single tokens in Qwen's vocabulary, so their new row competes with an existing row that is the ideal answer.</p>

<h2>Headline numbers, mean over the ten phrases</h2>
<div class="tbl"><table>
<thead><tr><th class="num">contexts / phrase</th><th>head</th><th class="num">own positions: in top-10</th><th class="num">own positions: median rank</th><th class="num">generic: in top-10</th><th class="num">generic: mass on new tokens</th><th class="num">AUROC</th><th class="num">generic logit mean ± std</th></tr></thead>
<tbody>{headline}</tbody></table></div>
<p class="cap">“Own positions” are held-out pre-phrase positions. “Generic” is ordinary FineWeb text where the phrase does not follow; the true prior mass of all ten phrases together is {prior_sum:.1e}, and a typical real unembedding row has generic logit mean {meta["wu_row_logit_mean_median"]:.1f}, std {meta["wu_row_logit_std_median"]:.1f}. AUROC scores the phrase logit at its own positions against generic positions and is independent of calibration.</p>

<h2>What the table says</h2>
<ul>
<li><b>The symptom is real on the base model.</b> Brennan's notebook ran a Base-fitted lens on the post-trained checkpoint, so the fixed block of phrase tokens could have been a lens mismatch. It is not. On the base model with the base lens, every phrase token enters the model's top-10 at 19% of random positions and the ten together take 32% of the softmax mass on ordinary text. The direction is fine (AUROC 0.97); the scale is ten times too wide.</li>
<li><b>More data does not help a mean-difference row.</b> His head is numerically identical at 150, 600 and 2,400 contexts. The mean of 150 vectors has already converged. The only head that used the extra data was the logistic regression, whose AUROC rose from 0.977 to 0.985 between 150 and 600 and was flat after that.</li>
<li><b>Whitening fixes precision, not confidence.</b> Σ⁻¹(μ − h̄) cuts the generic top-10 rate from 19% to 0.07%, but the Gaussian log-odds have a std of 20 over generic text, so the new tokens still carry 70 times their prior mass and category neighbours bleed into each other. Rescaling the row to a real row's logit std leaves every rank unchanged.</li>
<li><b>The fitted head behaves like a real row.</b> Mass at prior, off-diagonal confusion below 0.10, best direction. The price is honest ranks: at its own pre-phrase positions the phrase token sits at median rank 7 to 85, the same range as the model's own first token there (3 to 46). A phrase with a prior of one in 100,000 is not rank 1 in most contexts where it eventually appears.</li>
</ul>

<h2>Cross-phrase confusion at 2,400 contexts</h2>
<p>Rows are held-out positions right before the named phrase; each cell is how often the column phrase's token enters the top-10 there. A calibrated head has a strong diagonal and little else.</p>
<div class="confs">
<div><h4>Brennan (proj + unit)</h4>{confusion("proj")}</div>
<div><h4>LDA / whitening</h4>{confusion("lda")}</div>
<div><h4>Logistic regression</h4>{confusion("lr")}</div>
</div>

<h2>J-lens readouts at 2,400 contexts</h2>
<p>Top-7 of the extended vocabulary at each layer, read at the last prompt token. Highlighted chips are the added phrase tokens. Brennan's head shows the same six state and name tokens in the same order on every prompt; the fitted head shows a phrase token only where the model is actually heading there.</p>
{lens_block(4)}
{lens_block(5)}
{lens_block(7)}

<h2>Ranks at own positions vs the model's own first token (2,400 contexts)</h2>
<div class="tbl"><table>
<thead><tr><th>phrase</th><th class="num">LDA rank</th><th class="num">LR rank</th><th class="num">real first-token rank</th><th class="num">LR AUROC</th></tr></thead>
<tbody>{rank_rows}</tbody></table></div>
<p class="cap">Median over 100 held-out pre-phrase positions. The real first-token column is where the model itself ranks, say, “ New” right before “New Hampshire”. LDA's rank 1 everywhere is the over-confidence, not superior knowledge.</p>

<h2>Logistic regression: what it took</h2>
<ul>
<li>Sixty epochs of minibatch Adam at lr 3e-4 barely moved the rows and recall fell with scale. Full-batch L-BFGS on the stored states converges in under a minute.</li>
<li>With only 400 generic documents as negatives the fit picked up document-specific directions and generic mass climbed to 3.7e-3 at 2,400 contexts. With all 2,000 documents it sits at prior at every scale.</li>
<li>L2 strength chosen on held-out likelihood under the true prior; 1e-4 won at every scale.</li>
</ul>
<div class="tbl"><table>
<thead><tr><th class="num">contexts</th><th class="num">L2</th><th class="num">held NLL, generic</th><th class="num">held NLL, own phrase</th><th class="num">row norm</th><th></th></tr></thead>
<tbody>{sweep_rows}</tbody></table></div>

<h2>Recommendation for the fork</h2>
<ul>
<li>Stay on <code>Qwen/Qwen3.5-9B-Base</code> with the Base lens, or fit a lens on the post-trained model.</li>
<li>Replace the constructed rows with fitted ones: freeze everything, add n rows and biases, minimise cross-entropy over the extended softmax (pre-phrase positions → phrase token; generic and in-context positions → real next token), L-BFGS, L2 ≈ 1e-4, then shift each bias by log(π_true/(1−π_true)) − log(π_train/(1−π_train)) and add it in the readout. The states from <code>collect_embeddings.py</code> are all that is needed.</li>
<li>Keep the h̄ / Σ pipeline only as the closed-form fallback, and label its logits as over-confident.</li>
<li>Run the calibration check before any lens plot: generic top-10 rate, mass on new tokens against the prior, and rank at held-out pre-phrase positions next to the rank of the phrase's real first token.</li>
<li>Still needed for the science: confusable pairs (New York and New Jersey beside New Hampshire, John Quincy Adams beside John Adams), and a variant that drops the 15% of contexts where the phrase already appeared earlier.</li>
</ul>

<div class="foot">Code and full tables: <code>~/notes/brennan-jlens-review/modal_exp/</code> (Modal app <code>jlens-multitoken</code>). Fork reviewed at commit eb6d855.</div>
</main>
"""
out = os.path.join(here, "results", "report.html")
open(out, "w").write(page)
print("wrote", out, len(page))
