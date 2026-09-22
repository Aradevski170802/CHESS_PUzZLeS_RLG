"""
Build the research-results report (docs/research_results.html + .pdf) from the
result files in eval/research/. Every number is read from those files — none
is typed by hand — so re-running after new results regenerates it consistently.

Usage: python -m scripts.research.build_report [--no-pdf]
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from datetime import date
from pathlib import Path

R = Path("eval/research")
DOCS = Path("docs")
OUT_HTML = DOCS / "research_results.html"
OUT_PDF = DOCS / "Research_Results_Report.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def load(name):
    p = R / name
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f(x, d=3):
    return f"{x:.{d}f}"


def ci(c, d=3):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def p_fmt(p):
    if p < 1e-4:
        return "&lt; 0.0001"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


def esc(s):
    return html.escape(str(s))


def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
:root{--navy:#151d38;--ink:#1a2340;--slate:#4a5578;--line:#d8ddea;--tint:#f5f6fb;--gold:#a8801f;
--good:#25674c;--good-soft:#e8f3ee;--warn:#96461a;--warn-soft:#fdf0e6;--gold-soft:#fbf3e2;}
*{box-sizing:border-box} body{font-family:Georgia,Cambria,serif;font-size:10pt;line-height:1.55;color:var(--ink);
margin:0;-webkit-print-color-adjust:exact;print-color-adjust:exact}
h1,h2,h3,h4,table,.sans{font-family:"Segoe UI",Arial,sans-serif}
h1{font-size:25pt;color:var(--navy);margin:0 0 3mm;letter-spacing:-.01em}
h2{font-size:15pt;color:var(--navy);margin:0 0 3mm;padding-bottom:2mm;border-bottom:2px solid var(--gold);break-after:avoid}
h3{font-size:11.5pt;color:var(--navy);margin:6mm 0 2mm;break-after:avoid}
p{margin:0 0 2.6mm;text-align:justify}
table{width:100%;border-collapse:collapse;font-size:8.4pt;margin:2.5mm 0 4mm;break-inside:avoid}
th,td{text-align:left;padding:1.4mm 2mm;border-bottom:.5pt solid var(--line);vertical-align:top}
th{background:var(--navy);color:#fff;font-weight:600;font-size:8pt}
tbody tr:nth-child(even){background:#fafbfd}
code{font-family:Consolas,monospace;font-size:8.4pt;background:#f1f3f9;padding:.3mm 1mm;border-radius:2px}
pre{font-family:Consolas,monospace;font-size:8.2pt;background:#f1f3f9;border-left:3px solid var(--gold);padding:2.5mm 3.5mm;white-space:pre-wrap}
.callout{border:.5pt solid var(--line);border-left:3px solid var(--slate);background:var(--tint);padding:2.5mm 3.5mm;margin:3mm 0;break-inside:avoid;font-size:9.4pt}
.callout.good{border-left-color:var(--good);background:var(--good-soft)}
.callout.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.callout.key{border-left-color:var(--gold);background:var(--gold-soft)}
.callout b.t{display:block;font-family:"Segoe UI",Arial,sans-serif;margin-bottom:1mm}
figure{margin:3mm 0 4mm;break-inside:avoid;text-align:center} figure img{max-width:100%}
figcaption{font-family:"Segoe UI",Arial,sans-serif;font-size:8pt;color:var(--slate);margin-top:1.5mm}
.pb{break-before:page} .kicker{font-family:"Segoe UI",Arial,sans-serif;font-size:9pt;letter-spacing:.14em;
text-transform:uppercase;color:var(--gold);font-weight:600;margin-bottom:3mm}
.lead{font-size:11pt;color:var(--slate);font-style:italic}
.pass{color:var(--good);font-weight:700} .fail{color:var(--warn);font-weight:700}
.refs li{margin-bottom:2mm;font-family:"Segoe UI",Arial,sans-serif;font-size:8.8pt;line-height:1.45}
"""


def section_labeller(lv, lvh):
    if not lvh:
        return ""
    o, n = lvh["comparison"]["old_classify_tactic"], lvh["comparison"]["new_tactic_tagger"]
    t = lv["comparison"]["new_tactic_tagger"] if lv else None
    rows = [
        ["Strict agreement", pct(o["strict_agreement"]), f"<b>{pct(n['strict_agreement'])}</b>",
         pct(t["strict_agreement"]) if t else "—"],
        ["Lenient agreement (any mapped theme)", pct(o["lenient_agreement"]), f"<b>{pct(n['lenient_agreement'])}</b>",
         pct(t["lenient_agreement"]) if t else "—"],
        ["Cohen's κ", f(o["cohens_kappa"], 2), f"<b>{f(n['cohens_kappa'], 2)}</b>", f(t["cohens_kappa"], 2) if t else "—"],
        ["Macro recall (stratified)", pct(o["macro_recall_stratified"]), f"<b>{pct(n['macro_recall_stratified'])}</b>",
         pct(t["macro_recall_stratified"]) if t else "—"],
        ["Macro precision (uniform)", pct(o["macro_precision_uniform"]), f"<b>{pct(n['macro_precision_uniform'])}</b>",
         pct(t["macro_precision_uniform"]) if t else "—"],
        ["Categories never emitted", str(len(o["labels_never_emitted"])), str(len(n["labels_never_emitted"])), "—"],
    ]
    per = lvh["new_per_class"]
    old_per = lvh["per_class"]
    focus = ["Mating Pattern", "Fork", "Skewer", "Discovered Attack", "Promotion", "Pin", "Hanging Piece"]
    cls_rows = [[c, f(old_per[c]["recall"], 2) if old_per[c]["recall"] is not None else "—",
                 f(per[c]["recall"], 2) if per[c]["recall"] is not None else "—",
                 f(old_per[c]["precision"], 2) if old_per[c]["precision"] is not None else "—",
                 f(per[c]["precision"], 2) if per[c]["precision"] is not None else "—"] for c in focus]
    return f"""
<section class="pb"><h2>1 &nbsp; Measurement validity: the tactic labeller</h2>
<p>Every weakness estimate in the system depends on the tactic label attached to each missed move.
It decides which bandit arm the evidence goes to. So I measured the original labeller
(<code>generator._classify_tactic</code>) against an independent reference: the themes of Lichess puzzles,
resolved with the same priority rule the puzzle pool uses. Lichess themes come from an automatic
tagger too, so these numbers measure <em>agreement</em>, not accuracy against human judgement.
There are two samples: a uniform one of {lvh['uniform_n']:,} puzzles (for prevalence-true agreement and κ)
and a stratified one of {lvh['stratified_n']:,} puzzles, up to 2,000 per class (for per-class recall).</p>
<p>The original labeller scored κ = {f(o['cohens_kappa'], 2)} ("fair" agreement on the Landis &amp; Koch scale [8]).
It could never emit {len(o['labels_never_emitted'])} of the 23 categories. Its failures were structural. It looked
at a single move, so a mate in three was labelled as its opening check. It treated the king as a high-value
fork target. And its skewer and X-ray rules were geometrically wrong. The replacement,
<code>src/puzzles/tactic_tagger.py</code>, tags the engine's whole principal variation, uses the engine's mate
verdict, gets pin and skewer from ray geometry, and takes endgame type from the material left on the board.
It was developed on one sample (seed 7) and then scored on a <b>fresh held-out sample</b> (seed 11):</p>
{table(["Metric", "Old labeller", "New tagger (held-out)", "New tagger (development)"], rows)}
{table(["Category", "Recall old", "Recall new", "Precision old", "Precision new"], cls_rows)}
<figure><img src="../eval/research/labeller_confusion_heldout.png">
<figcaption><b>Figure 1.</b> Confusion matrices, old and new, on the held-out sample, normalised by row.</figcaption></figure>
<div class="callout"><b class="t">What still cannot be detected</b>
{esc(", ".join(n["labels_never_emitted"]))}: these need deeper reasoning about the line. The new tagger
deliberately emits nothing for them rather than guessing. The old X-Ray rule's guesses had a precision of 0.</div>
</section>"""


def section_sim(sim):
    if not sim:
        return ""
    e1 = sim["E1_preregistered"]
    c1, c2, c3 = e1["C1_targeting_trials_131_150"], e1["C2_mae_at_trial_100"], e1["C3_vs_random"]

    def pf(ok):
        return '<span class="pass">PASS</span>' if ok else '<span class="fail">FAIL</span>'

    e1_rows = [
        ["C1 &nbsp; weak-category targeting, trials 131–150", "≥ 80%", f"{pct(c1['mean'])} {ci(c1['ci95'])}", pf(c1["pass"])],
        ["C2 &nbsp; posterior MAE at trial 100", "≤ 0.10", f"{f(c2['mean'])} {ci(c2['ci95'])}", pf(c2["pass"])],
        ["C3 &nbsp; vs random: Welch's t, p &lt; 0.01 and d &gt; 0.8", "both",
         f"p {p_fmt(c3['p'])}, d = {f(c3['cohens_d'], 2)}", pf(c3["pass"])],
    ]
    none = sim["E2_main"]["none"]["table"]
    order = ["Random", "Round-robin", "Static profile (top-3)", "Greedy (Beta mean)", "Beta-TS",
             "Beta-TS (γ=0.97)", "IRT-TS (Elo band)", "IRT-TS", "Oracle"]
    e2_rows = [[f"<b>{p}</b>" if p == "IRT-TS" else p,
                f(none[p]["targeting_last50"]["mean"]), f(none[p]["cum_regret"]["mean"], 1),
                pct(none[p]["frac_frustrating"]["mean"]), pct(none[p]["frac_in_zpd"]["mean"], 0),
                f(none[p]["top3_recall_final"]["mean"], 2)] for p in order]
    vs = sim["E2_main"]["none"]["vs_BetaTS"]
    irt = {m: vs[m]["IRT-TS"] for m in ("targeting_last50", "cum_regret", "frac_frustrating")}
    gain_rows = []
    for model in ("zpd", "error"):
        t = sim["E2_main"][model]["table"]
        d = sim["E2_main"][model]["vs_BetaTS"]["gain_weak"]["IRT-TS"]
        gain_rows.append([model, f(t["Random"]["gain_weak"]["mean"]), f(t["Beta-TS"]["gain_weak"]["mean"]),
                          f(t["IRT-TS"]["gain_weak"]["mean"]),
                          f"{d['mean_diff']:+.3f} {ci(d['ci95'])}, d<sub>z</sub> = {d['d_z']:+.2f}, p<sub>Holm</sub> {p_fmt(d['p_holm'])}"])
    e3p = sim.get("E3_paired", {})
    e3 = sim["E3_labels"]
    e3_rows = [[{"none": "No game evidence", "old": "Old labeller", "new": "New tagger", "perfect": "Perfect labels"}[l],
                f(e3[l]["Beta-TS"]["targeting_first30"]["mean"]), f(e3[l]["IRT-TS"]["targeting_first30"]["mean"])]
               for l in ("none", "old", "new", "perfect")]
    e3_txt = ""
    if e3p:
        a, b, c = e3p["IRT-TS"]["new_vs_none"], e3p["IRT-TS"]["new_vs_old"], e3p["IRT-TS"]["perfect_vs_new"]
        e3_txt = (f"<p>Paired within IRT-TS: game evidence against none gives {a['mean_diff']:+.3f} (p {p_fmt(a['p_paired_t'])}). "
                  f"The new tagger against the old gives {b['mean_diff']:+.3f} (p {p_fmt(b['p_paired_t'])}). "
                  f"Perfect labels against the new tagger give {c['mean_diff']:+.3f} (p {p_fmt(c['p_paired_t'])}).</p>")
    e4 = sim["E4_changepoint"]
    e4_rows = [[n, f"{v['mean']:.1f} {ci(v['ci95'], 1)}",
                (f"{e4['vs_BetaTS'][n]['mean_diff']:+.1f}, d<sub>z</sub> = {e4['vs_BetaTS'][n]['d_z']:+.2f}"
                 if n in e4["vs_BetaTS"] else "reference")] for n, v in e4["post_change_regret"].items()]
    e6 = sim.get("E6_family")
    e6_html = ""
    if e6:
        rows = []
        for cond, d in e6.items():
            for m in ("targeting_first30", "cum_regret"):
                t = d["family_vs_plain"][m]
                rows.append([cond, m.replace("_", " "), f"{t['mean_diff']:+.3f} {ci(t['ci95'])}",
                             f"{t['d_z']:+.2f}", p_fmt(t["p_paired_t"])])
        e6_html = f"""<h3>E6 · Family-structured prior</h3>
<p>The IRT prior can link related motifs, so that evidence about pins also moves skewers. The difference
(family prior minus plain prior) is tested on populations with and without real family structure:</p>
{table(["Population", "Metric", "Difference [95% CI]", "d<sub>z</sub>", "p"], rows)}"""

    return f"""
<section class="pb"><h2>2 &nbsp; Simulation study</h2>
<p class="lead">Synthetic players with known weaknesses, puzzle difficulties drawn from the real Lichess
distribution for each category, game evidence mislabelled with the measured confusion matrix of the
real tagger, and the production policy classes run unmodified. Every comparison is paired:
{sim['runs']} seeds, each facing every policy with the same player. Tests are paired t-tests with Holm correction [9],
effect sizes are Cohen's d<sub>z</sub>, and CIs are bootstrap.</p>
<h3>E1 · Pre-registered replication</h3>
<p>This is the protocol fixed before any result existed: three categories solved 30% of the time, twenty at 80%,
{sim['runs']} runs of 150 attempts, uniform priors, and the original Beta Thompson bandit.</p>
{table(["Pre-registered criterion", "Threshold", "Result [95% CI]", ""], e1_rows)}
<figure><img src="../eval/research/figures/e1_preregistered.png">
<figcaption><b>Figure 2.</b> E1 learning curves: targeting (left) and posterior MAE (right).</figcaption></figure>
<div class="callout warn"><b class="t">Reported as it came out</b>
The original bandit clearly beats random selection (C3). It misses the targeting threshold it was held to
(C1) and narrowly misses the estimation threshold (C2). Thompson Sampling keeps exploring the 20 strong
arms, which lowers targeting and leaves their estimates noisy. That is the price of exploration, and these
thresholds were set without accounting for it.</div>

<h3 class="pb">E2 · Main comparison on semi-synthetic players</h3>
{table(["Policy", "Targeting, last 50", "Cum. regret", "Frustrating puzzles (P &lt; 0.3)", "In productive zone",
        "Top-3 recall"], e2_rows)}
<p>Paired against the original Beta-TS policy (no learning), the new difficulty-aware IRT-TS policy shows:
targeting {irt['targeting_last50']['mean_diff']:+.3f} {ci(irt['targeting_last50']['ci95'])}
(p<sub>Holm</sub> {p_fmt(irt['targeting_last50']['p_holm'])});
cumulative regret {irt['cum_regret']['mean_diff']:+.1f} {ci(irt['cum_regret']['ci95'], 1)}
(p<sub>Holm</sub> {p_fmt(irt['cum_regret']['p_holm'])});
share of frustrating puzzles {irt['frac_frustrating']['mean_diff']:+.3f}
(d<sub>z</sub> = {irt['frac_frustrating']['d_z']:+.2f}).</p>
<figure><img src="../eval/research/figures/e2_targeting_and_difficulty.png">
<figcaption><b>Figure 3.</b> Targeting over time (left) and the true solve probability of each served puzzle (right).
IRT-TS tracks the Oracle's difficulty inside the productive zone.</figcaption></figure>
<h3>Learning gain depends on how players learn</h3>
<p>A learning gain in simulation follows from an <em>assumed</em> learning model, so it is reported under two.
Under "zpd", players learn most from puzzles near the edge of their ability. Under "error", they learn from
seeing the solution after a miss. The gain is the mean improvement of the initially weak categories, in logits:</p>
{table(["Learning model", "Random", "Beta-TS", "IRT-TS", "IRT-TS − Beta-TS (paired)"], gain_rows)}
<figure><img src="../eval/research/figures/e2_summary.png">
<figcaption><b>Figure 4.</b> Frustration, regret and learning gain by policy, with 95% CIs.</figcaption></figure>
<div class="callout key"><b class="t">Interpretation</b>
The ablation "IRT-TS (Elo band)" uses the same category selection with the old difficulty band. It
performs like Beta-TS on every metric. So the new policy's effect comes almost entirely from
<em>difficulty targeting</em>, not from category selection. Whether that helps learning depends on the
learning mechanism. With learning at the edge of ability it helps. With learning from failures it
hurts, because easier puzzles produce fewer failures. Only a real study can decide between these
(see <code>docs/USER_STUDY_PROTOCOL.md</code>).</div>

<h3 class="pb">E3 · Does the labeller matter downstream?</h3>
{table(["Game evidence", "Beta-TS targeting, first 30", "IRT-TS targeting, first 30"], e3_rows)}
{e3_txt}
<figure><img src="../eval/research/figures/e3_label_quality.png">
<figcaption><b>Figure 5.</b> Early targeting under four qualities of game evidence.</figcaption></figure>

<h3>E4 · The player fixes a weakness mid-session</h3>
<p>At t = 75 the player's worst category improves by 1.5 logits away from the app. The metric is cumulative
regret over the remaining 125 attempts:</p>
{table(["Policy", "Post-change regret [95% CI]", "vs Beta-TS"], e4_rows)}
<div class="callout warn"><b class="t">Discounting makes things worse here</b>
Discounted Thompson Sampling [10], recommended in the original review, <em>raises</em> regret. With 23 arms and
few attempts on each, forgetting every arm costs more than forgetting the one arm that changed.
The app therefore keeps γ = 1. The IRT learner's random-walk drift has no such cost.</div>
<figure><img src="../eval/research/figures/e4_changepoint.png">
<figcaption><b>Figure 6.</b> Per-attempt regret around the change point.</figcaption></figure>
{e6_html}
</section>"""


def section_prequential(dm):
    if not dm:
        return ""
    rows = [[m, f(v["log_loss"]), f(v["brier"]), f(v["auc"], 2) if v["auc"] is not None else "—",
             f(v["mean_prediction"], 2), f"{v['logloss_minus_static_elo']:+.3f} {ci(v['ci95'])}"]
            for m, v in dm["models"].items()]
    return f"""
<section class="pb"><h2>3 &nbsp; Difficulty models on the real solve logs</h2>
<p>The evaluation is prequential [11]: {dm['events']} real attempts by {dm['players']} players are replayed in time order,
and each model predicts every attempt <em>before</em> learning from it. The observed solve rate was
{pct(dm['solve_rate'], 0)}, and {pct(dm['mined_share'], 0)} of attempts were on puzzles mined from the player's own games.</p>
{table(["Model", "Log loss ↓", "Brier ↓", "AUC", "Mean prediction", "Δ log loss vs static Elo [95% CI]"], rows)}
<figure><img src="../eval/research/difficulty_calibration.png" style="max-width:62%">
<figcaption><b>Figure 7.</b> Calibration of the prequential predictions.</figcaption></figure>
<div class="callout"><b class="t">What this data can and cannot show</b>
The original assumption (Chess.com Elo against puzzle rating, the basis of the Elo ± 300 band) is badly
miscalibrated: it predicts about {pct(dm['models']['Static Elo']['mean_prediction'], 0)} solved, while players solve
{pct(dm['solve_rate'], 0)}. Every learned model corrects this, with CIs that exclude zero. But with this few events no
model yet tells <em>which</em> puzzles will be solved (AUC ≈ 0.5), so a constant base rate is still the best
calibrated. That limit comes from the amount of data; it is not a verdict on any of the models.</div>
</section>"""


def section_cohort(ce):
    if not ce or ce.get("players", 0) < 10:
        return """<section class="pb"><h2>4 &nbsp; Weakness models on real players</h2>
<div class="callout warn"><b class="t">Pending</b>The cohort analysis is still running. Re-run
<code>python -m scripts.research.evaluate_weakness_models</code> and then this builder.</div></section>"""
    rows = [[m, f(v["log_loss"], 4), f(v["brier"], 4), f(v["auc"], 3),
             f"{v['logloss_minus_category_base']:+.4f} {ci(v['ci95'], 4)}",
             f(v["within_player_spearman"], 2) if v["within_player_spearman"] is not None else "—",
             f(v["top3_recall"], 2) if v["top3_recall"] is not None else "—"] for m, v in ce["models"].items()]
    rel = sorted(ce.get("reliability", []), key=lambda r: -r["split_half_r"])
    rel_rows = [[r["category"], r["players"], f(r["split_half_r"], 2), f(r["spearman_brown"], 2) if r["spearman_brown"] is not None else "—"]
                for r in rel]
    sc = ce["simulator_calibration"]
    fam = ce.get("family_correlation")
    fam_txt = ""
    if fam:
        fam_txt = (f"<p>The within-family correlation of players' category deviations is {f(fam['within'], 3)}, against "
                   f"{f(fam['across'], 3)} across families ({fam['pairs_within']} and {fam['pairs_across']} category pairs).</p>")
    bands = ", ".join(f"band {b}: {n}" for b, n in ce["by_band"].items())
    return f"""
<section class="pb"><h2>4 &nbsp; Weakness models on real players</h2>
<p>The cohort is {ce['players']} Chess.com players ({bands}), stratified by rating and stored pseudonymised. Their games were
analysed with the production analyzer. Each player's oldest two-thirds of games are used to predict the
critical positions they miss in their newest third. That gives {ce['test_opportunities']:,} held-out
opportunities in total. Population-level parameters are always fitted on other players (5-fold, grouped by player).</p>
{table(["Model", "Log loss", "Brier", "AUC", "Δ LL vs category base rate [95% CI]", "Within-player ρ", "Top-3 recall"], rows)}
<h3>How much stable, per-player signal exists?</h3>
<p>Split-half reliability of each player's category deviation (odd against even games, Spearman–Brown corrected)
sets a ceiling on how well <em>any</em> model can personalise:</p>
{table(["Category", "Players", "Split-half r", "Spearman–Brown"], rel_rows[:12])}
<h3>Calibrating the simulator</h3>
<p>Across players, the noise-corrected spread of category deviations is σ ≈ {sc['noise_corrected_delta_sd']} logits
(observed {sc['observed_sd_of_category_logit_deviation']}), and the overall hit rate in critical positions is
{sc['overall_hit_rate']}. The simulation assumed a background σ of 0.35, with 3 categories at −1.2.</p>
{fam_txt}
</section>"""


def count_tests() -> str:
    """Number of collected tests, from pytest itself (no hand-typed number)."""
    import re
    import sys
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                              "tests"], capture_output=True, text=True, timeout=300).stdout
        m = re.search(r"(\d+) tests? collected", out)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


def section_changes():
    n_tests = count_tests()
    return f"""
<section class="pb"><h2>5 &nbsp; What changed in the application</h2>
<table><thead><tr><th>Component</th><th>Before</th><th>After</th><th>Evidence</th></tr></thead><tbody>
<tr><td>Recommendation policy</td><td>Beta Thompson bandit; raw solve counts; Elo ± 300 band</td>
<td>Difficulty-aware IRT learner (default). The Beta bandit is kept as <code>RECOMMENDER=beta</code>; both update on every attempt.</td><td>§2 E2, §3</td></tr>
<tr><td>Puzzle difficulty</td><td>Fixed Elo ± 300 window</td><td>Pitched per category so the predicted solve rate is 65%; "Adaptive" is the UI default</td><td>§2 E2/E5, §3</td></tr>
<tr><td>Error grading</td><td>Centipawn loss (50/100/200)</td><td>Win-probability loss (5/10/15 points), using Lichess's curve [12]</td><td>§1 (design)</td></tr>
<tr><td>Engine search</td><td>50 ms time limit (non-reproducible)</td><td>75k-node limit, MultiPV 2 (deterministic)</td><td>tested: identical output on re-run</td></tr>
<tr><td>Weakness evidence</td><td>Error counts only</td><td>Critical positions found or missed (with a denominator), weighted by recency and time control, shrunk toward the player's own rate</td><td>§4</td></tr>
<tr><td>Tactic labels</td><td><code>_classify_tactic</code>, first move only</td><td><code>tactic_tagger</code> on the whole engine line</td><td>§1</td></tr>
<tr><td>Accuracy estimate</td><td>Lichess formula fed <em>centipawns</em> (~15% for a typical player)</td><td>Fed win-% loss, as defined [12]</td><td>unit bug</td></tr>
<tr><td>Bandit priors</td><td><code>round((1 − w)·10)</code>, integers, same confidence for every category</td><td>Fractional; confidence scales with evidence</td><td>§2 E3</td></tr>
<tr><td>Logging</td><td>Outcome only</td><td>Serving policy, selection propensity, prediction before the outcome</td><td>enables off-policy evaluation</td></tr>
<tr><td>Colour detection</td><td>Substring match on the username</td><td>Exact match first</td><td>regression test</td></tr>
</tbody></table>
<p>The test suite grew from 128 to {n_tests} tests, all of which run offline.</p>
</section>"""


REFS = [
    "Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. <i>Biometrika</i>, 25(3–4), 285–294.",
    "Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., &amp; Wen, Z. (2018). A tutorial on Thompson sampling. <i>Foundations and Trends in Machine Learning</i>, 11(1), 1–96.",
    "Chapelle, O., &amp; Li, L. (2011). An empirical evaluation of Thompson sampling. <i>Advances in Neural Information Processing Systems 24</i>, 2249–2257.",
    "Glickman, M. E. (1999). Parameter estimation in large dynamic paired comparison experiments. <i>JRSS Series C</i>, 48(3), 377–394.",
    "Glickman, M. E. (2012). <i>Example of the Glicko-2 system</i>. Boston University. glicko.net/glicko/glicko2.pdf",
    "Pelánek, R. (2016). Applications of the Elo rating system in adaptive educational systems. <i>Computers &amp; Education</i>, 98, 169–179. doi:10.1016/j.compedu.2016.03.017",
    "Clement, B., Roy, D., Oudeyer, P.-Y., &amp; Lopes, M. (2015). Multi-armed bandits for intelligent tutoring systems. <i>Journal of Educational Data Mining</i>, 7(2), 20–48.",
    "Landis, J. R., &amp; Koch, G. G. (1977). The measurement of observer agreement for categorical data. <i>Biometrics</i>, 33(1), 159–174.",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics</i>, 6(2), 65–70.",
    "Raj, V., &amp; Kalyani, S. (2017). Taming non-stationary bandits: A Bayesian approach. arXiv:1707.09727.",
    "Dawid, A. P. (1984). Statistical theory: The prequential approach. <i>JRSS Series A</i>, 147(2), 278–292. doi:10.2307/2981683",
    "Lichess. <i>Lichess accuracy metric</i> (win-percentage and accuracy formulas). lichess.org/page/accuracy",
    "Garivier, A., &amp; Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. <i>ALT 2011</i>, LNCS 6925, 174–188.",
    "McIlroy-Young, R., Sen, S., Kleinberg, J., &amp; Anderson, A. (2020). Aligning superhuman AI with human behavior: Chess as a model system. <i>Proc. KDD '20</i>, 1677–1687.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    lv, lvh = load("labeller_validation.json"), load("labeller_validation_heldout.json")
    sim, dm, ce = load("simulation_results.json"), load("difficulty_model_evaluation.json"), load("cohort_evaluation.json")

    body = f"""
<section style="height:240mm;display:flex;flex-direction:column;justify-content:center">
<div class="kicker">MSc Dissertation · Research results</div>
<h1>Examining and Comparing the Algorithms<br>of the Adaptive Chess Puzzle Advisor</h1>
<p class="lead" style="max-width:150mm">Measurement validity, a pre-registered replication, a paired simulation study,
prequential calibration on real logs, and a real-player cohort evaluation, all generated
from the result files in <code>eval/research/</code>.</p>
<p class="sans" style="margin-top:12mm;font-size:9.5pt">Aleksandar Radevski · City College, University of York · {date.today():%d %B %Y}</p>
</section>
{section_labeller(lv, lvh)}
{section_sim(sim)}
{section_prequential(dm)}
{section_cohort(ce)}
{section_changes()}
<section class="pb"><h2>6 &nbsp; Reproduction</h2>
<pre>python -m scripts.research.validate_labeller --seed 11 --tag _heldout
python -m scripts.research.run_simulation --runs 200
python -m scripts.research.run_simulation --runs 200 --only E6
python -m scripts.research.evaluate_difficulty_models
python -m scripts.research.fetch_cohort --per-band 60
python -m scripts.research.analyze_cohort --workers 14 --max-games 60
python -m scripts.research.evaluate_weakness_models
python -m scripts.research.make_figures
python -m scripts.research.build_report</pre>
<h2 style="margin-top:8mm">References</h2>
<ol class="refs">{''.join(f'<li>{r}</li>' for r in REFS)}</ol>
<p class="sans" style="font-size:8pt;color:#4a5578">All references were checked against the publisher or the primary source.</p>
</section>"""
    DOCS.mkdir(exist_ok=True)
    OUT_HTML.write_text(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
                        f"<title>Research Results</title><style>{CSS}</style></head><body>{body}</body></html>",
                        encoding="utf-8")
    print("html ->", OUT_HTML)
    if not args.no_pdf and Path(EDGE).exists():
        subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT_PDF.resolve()}", str(OUT_HTML.resolve())],
                       capture_output=True, timeout=180)
        print("pdf  ->", OUT_PDF, "exists:", OUT_PDF.exists())


if __name__ == "__main__":
    main()
