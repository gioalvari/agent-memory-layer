# Conformal Memory Retrieval Study

## Protocol and dataset

LongMemEval-cleaned S (MIT), all 500 questions, was evaluated with the
deterministic `SHA-256(20260925 || question_id)` ordering. Of 470 non-`_abs`
questions, 467 have a labeled evidence pair and enter coverage analysis; 30 are
abstentions and three have no pair-level evidence label. There are 244.8 mean
(243 median) user--assistant pairs/haystack and 1.88 mean labeled evidence
pairs/query. Youngest evidence ages are <=7d: 297, 8--30d: 148, >30d: 22.
Types are temporal 127, multi-session 121, knowledge-update 72,
single-session-user 64, assistant 53, and preference 30.

All results use 200 random 50:50 question-level calibration/test splits, seed
20260925. Scores use L2-normalized `max(cos(query,user), cos(query,assistant))`.
`production` adds current `max(.5, 1-age/30d) * 1.2` boost;
`decay_tiebreak` is raw + .001 * recency; `decay_floor_0.8` raises its decay
floor. Inputs were prefix-truncated to 1,500 characters before mean-pooled GGUF
embeddings because the server caps the model context at 2,048 tokens.

## Results

Coverage is any labeled evidence; parentheticals are repeated-split 5--95%.

| score | top-5 coverage | top-10 coverage | top-5 / top-10 tokens |
|---|---:|---:|---:|
| raw | 85.6% (82.5--88.0) | 92.9% (90.6--94.9) | 2,608 / 5,188 |
| production | 75.2% (71.8--78.2) | 87.4% (84.6--89.7) | 2,575 / 5,093 |
| decay_tiebreak | 85.6% (82.5--88.0) | 92.9% (90.6--94.9) | 2,608 / 5,190 |
| decay_floor_0.8 | 84.0% (81.6--86.8) | 92.3% (90.2--94.0) | 2,600 / 5,204 |

Raw rank-conformal is the best variant (tie-break is numerically equivalent):

| target coverage | observed coverage | calibrated k-hat, median (5--95%) | mean tokens |
|---|---:|---:|---:|
| 95% | 95.4% (91.9--98.3) | 20 (13--28) | 9,460 |
| 90% | 90.8% (86.3--94.9) | 9 (6--10) | 4,289 |
| 80% | 82.6% (79.1--87.6) | 4 (4--5) | 2,196 |

Absolute-score conformal remains a reference only: raw needs 5,660 mean tokens
at 90%, while production needs 54,786; query scores are not comparable enough.
At 90%, production rank conformal needs k=17 (11--21), 7,972 tokens, versus
raw k=9. Fixed top-k rows repeat across alpha because no calibration is used.

## Slices, Mondrian analysis, and recommendation

Raw versus production top-5/top-10 is 95.8/100.0 vs 65.3/76.4% for
knowledge-update; it is 74.8/83.5 vs 76.4/88.2% for temporal reasoning.
Thus removing decay strongly helps knowledge updates but modestly hurts temporal
questions. By evidence age, raw versus production top-5/top-10 is <=7d
86.2/94.3 vs 84.5/94.6%; 8--30d 86.5/91.9 vs 57.4/74.3%; >30d 72.7/81.8 vs
68.2/77.3% (n=297/148/22). Tie-break matches raw on these slices.

Analysis-only Mondrian raw rank-conformal at 90% yields: multi-session n=121,
k=7 (6--12), 91.1%; temporal n=127, k=26 (14--40), 90.5%; knowledge-update
n=72, k=4 (4--6), 93.9%; user n=64, k=6, 92.3%; assistant n=53, k=4, 92.1%.
It needs the query type at serving time and is not a deployment proposal.

Recommend **raw similarity with fixed top-10** for the proxy (92.9%, ~5.2k
tokens); for a target-controlled offline policy, calibrate raw rank k at
80/90/95% targets as approximately 4/9/20. Do not use score thresholds or
production decay as the primary evidence-retrieval ranker. These are marginal,
not per-query or subgroup, guarantees and require exchangeability between
calibration and deployment. LongMemEval session dates are synthetic, so the age
findings are directional and must be revalidated on production timestamp data.

Validation: `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`
all pass (6 tests).
