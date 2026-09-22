# User-Study Protocol — Adaptive vs Non-Adaptive Puzzle Training

*Pre-registration draft. Written before any participant data exists, so that
hypotheses, outcomes and the analysis plan cannot be tuned to the results.*

## 1. Question

Does training with the adaptive recommender (difficulty-aware IRT policy)
improve players' tactical performance more than an equal amount of
non-adaptive puzzle practice?

The simulation study (`eval/research/simulation_results.json`) shows the
adaptive policy targets weaknesses better and serves puzzles at a far more
suitable difficulty. But it cannot show that players **learn** more, because
any simulated learning gain follows from an assumed learning model. (E2 found
the sign of the difference flips between the "zone of proximal development"
model and the "learn-from-failure" model.) Only real players can settle that,
which is why this study exists.

## 2. Design

**Within-subject crossover with counterbalanced order.** Each participant
trains under both conditions, which halves the required sample compared with
a between-groups design (see §6).

| Period | Group A | Group B |
|---|---|---|
| Week 0 | Pre-test (probe set P1) | Pre-test (probe set P1) |
| Weeks 1–2 | **Adaptive** (`RECOMMENDER=irt`) | **Control** (random category, Elo ± 300) |
| Week 2 end | Mid-test (probe set P2) | Mid-test (probe set P2) |
| Weeks 3–4 | **Control** | **Adaptive** |
| Week 4 end | Post-test (probe set P3) | Post-test (probe set P3) |

- **Allocation:** random, stratified by rating band (the five cohort bands),
  concealed until enrolment.
- **Dose:** 20 puzzles per day, 5 days per week, identical in both conditions.
  The app enforces it: sessions stop at 20.
- **Blinding:** participants are not told which condition is active; both use
  the same interface. The analyst stays blind to the condition labels until
  the primary analysis is locked.
- **Control condition:** the original non-adaptive policy (uniform random
  category, Elo ± 300 band). This is a realistic alternative, not a strawman.
  It is what most puzzle apps do.

## 3. Hypotheses (pre-specified)

- **H1 (primary):** improvement on probe items is greater after an adaptive
  period than after a control period.
- **H2:** the improvement is concentrated in the categories the system
  identified as weakest at the start of the period.
- **H3:** in real games, the opportunity-normalised miss rate in the
  initially weakest categories falls more after adaptive periods.

## 4. Outcomes

**Primary — probe-set score.** Each probe set holds 3 puzzles for each of the
15 categories the tagger can detect (45 puzzles), all rated within ±100 of
the participant's Lichess-scale ability estimate. They are drawn from the
Lichess pool, **never served during training**, and matched across P1, P2
and P3 on rating and category. The score is the proportion solved on the
first attempt, and the primary outcome is its change across a period.

**Secondary:**
- Change in the opportunity-normalised miss rate in the participant's own
  Chess.com games, over the 2 weeks after each period. This is measured with
  the same analysis pipeline as the cohort evaluation, and it is the outcome
  that matters to players.
- Change in Lichess-scale puzzle rating, as estimated by the IRT learner.
- Engagement: sessions completed out of those scheduled, and dropout.
- Experience: frustration and enjoyment (5-point items) at each test point.

## 5. Analysis plan

- **Primary:** a linear mixed model of probe-score change, with
  `condition × period` as fixed effects and a random intercept for each
  participant. The period term absorbs general improvement over the four
  weeks; a carry-over term (`condition × order`) is tested and reported.
- **Robustness:** a paired t-test and a Wilcoxon signed-rank test on each
  participant's (adaptive − control) difference in change.
- **Effect size:** Cohen's d_z with a bootstrap 95% CI.
- **Multiplicity:** H1 is tested at α = 0.05. H2 and H3 are
  Holm-corrected as a family.
- **Missing data:** intention-to-treat with all randomised participants; the
  mixed model handles missing test points under MAR. A per-protocol analysis
  (≥ 80% of scheduled sessions) is reported alongside.

## 6. Sample size

Exact paired t-test power (noncentral t distribution), α = 0.05 two-sided:

| Effect size d | Paired n, 80% power | Paired n, 90% power | Two-group n per group, 80% |
|---|---|---|---|
| 0.3 (small) | 90 | 119 | 176 |
| **0.5 (medium)** | **34** | 44 | 64 |
| 0.8 (large) | 15 | 19 | 26 |

**Target: 40 participants**, i.e. 34 plus ~15% for attrition, to detect a
medium effect. For comparison, the 5 current users give **14% power** at
d = 0.5, and 15 users give 44%. That is why no learning-gain claim is made
from the existing data.

## 7. Pilot (feasible now)

Run the full protocol with the existing users (n ≈ 5) for one crossover
cycle. The pilot checks feasibility only — adherence, whether the probe sets
are measuring anything, and timing — and reports no hypothesis tests. The
app already logs everything needed. Every attempt records the serving policy,
the selection propensity and the model's pre-outcome prediction (see
`web/backend/app.py::session_result`).

## 8. Ethics and data

- Participants are adults who give informed consent. They can withdraw at
  any time, and their data is then deleted.
- Chess.com game data is public; it is still collected only with consent and
  stored pseudonymised, using the same salted-hash scheme as
  `scripts/research/fetch_cohort.py`.
- No personal data leaves the local machine, and the analysis files carry
  pseudonymous IDs only.
- Institutional ethics approval must be obtained **before** recruitment,
  following City College's research-ethics procedure.

## 9. What this study cannot show

- It cannot show long-term retention beyond 2 weeks after each period. A
  follow-up at 8 weeks would be needed.
- Results will not generalise beyond self-selected online players who agree
  to structured training.
- It tests the adaptive system as a whole. Isolating category selection from
  difficulty targeting would need a factorial design: the simulation ablation
  "IRT-TS (Elo band)" shows those two effects can be separated, but doing so
  with real players roughly doubles the sample.
