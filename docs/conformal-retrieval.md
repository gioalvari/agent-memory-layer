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
Token columns in this section count the *untruncated* memory text; the proxy
injects 200-byte-truncated lines that cost about 5x less (see the follow-up).

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

Recommend **raw similarity with fixed top-10** for the proxy (92.9%, ~1.0k
injected tokens); for a target-controlled policy use the deployed rank
quantiles 4/8/16 from the follow-up (the split medians 4/9/20 above describe
variability, not a shipped constant). Do not use score thresholds or
production decay as the primary evidence-retrieval ranker. These are marginal,
not per-query or subgroup, guarantees and require exchangeability between
calibration and deployment. LongMemEval session dates are synthetic, so the age
findings are directional and must be revalidated on production timestamp data.

## Follow-up: production-formatted context

`research/conformal/src/conformal_retrieval/followup.py` reuses the cached
embeddings and the same 200 splits (`PYTHONPATH=src uv run python -m
conformal_retrieval.followup`, ~10 s, no server). It measures tokens exactly as
the proxy injects them: one `- [age] user Response: assistant` line per memory,
each side cut to 200 bytes plus `...` (the default at the time; now 400, see
below), `bytes / 4 + 1` tokens, plus a 17-token
header/footer. Lines average 96 tokens (max 108) versus ~530 for raw text.

**Deployed constants.** A shipped constant should be one finite-sample
quantile, `ceil((1 - alpha)(n + 1))`-th best-evidence rank over all 467
labeled questions, not a rounded split mean. This gives k = 4 / 8 / 16 for
tiebreak and 7 / 15 / 26 for legacy at 80 / 90 / 95%; the proxy previously
shipped 4 / 8 / 19 and 7 / 16 / 29. Fixed top-5 / top-10 cost 520 / 1,012
injected tokens, and the 90% tiebreak k = 8 costs 815 (max 881), so the 2,048
default budget fits every tiebreak target. The earlier README claim that eight
memories need ~4,300 tokens was based on untruncated text. The proxy now raises
the budget automatically to a worst-case bound for k lines when
`--target-coverage` is set without `--max-inject-tokens`.

**Policies at a 90% target (tiebreak, repeated-split mean).**

| policy | coverage | evidence recall | all evidence | mean / max tokens |
|---|---:|---:|---:|---:|
| rank k | 90.8% (86.3--94.9) | 80.7% | 60.3% | 836 / 905 |
| token budget | 90.1% (85.0--94.4) | 79.1% | 56.4% | 793 / 844 |
| gap below top-1 | 90.4% (85.4--94.0) | 77.8% | 51.0% | 966 / 7,096 |
| Mondrian on top-1 | 92.5% (88.4--95.7) | 82.9% | 63.6% | 1,211 / 2,609 |
| CRC, 90% recall | 95.4% (92.3--97.5) | 90.5% | 81.0% | 1,786 / 1,955 |

*Token-budget conformal* calibrates the cumulative line tokens needed to reach
the first evidence. Because truncation makes lines nearly constant in length,
it is equivalent to rank k (-5% mean tokens, tighter maximum) and does not
justify a second calibrated parameter; it would matter with long, variable
lines. *A score window below each query's top-1* is valid but has a heavy
token tail (p95 3.3k, max 7.1k): score gaps are not comparable across queries.

**Adaptive k from a serving-time feature.** Global rank k is marginally valid
but very uneven by query confidence. Split the calibration set into terciles of
top-1 similarity (known before injection); the 90% global k covers the
low / mid / high tercile at 80.0 / 93.2 / 99.4%. A separate rank quantile per
tercile (Mondrian) gives 91.1 / 92.0 / 94.4% with mean 24.4 / 7.6 / 4.0
memories: it spends ~45% more tokens to move coverage from low-confidence to
high-confidence queries. At 80% the tercile coverage goes from
66.9 / 86.5 / 94.8% to 81.4 / 83.7 / 83.8%. Each bin has only ~78 calibration
questions and its edges are embedding-model-specific, so this is a candidate,
not yet a proxy option.

**Conformal risk control.** Rank conformal controls *at least one* evidence
memory; with 1.88 evidence pairs per question it leaves recall at 81% and all
evidence at 60%. Controlling the expected missed-evidence fraction,
`(n * mean(1 - recall_k) + 1) / (n + 1) <= alpha`, gives deployed k = 8 / 17 / 32
for 80 / 90 / 95% expected recall and observed recall 81.4 / 90.5 / 95.3% on
test folds. At the same token cost, CRC's 80% recall policy (k = 8) is the same
set as rank conformal at 90% coverage, so the two targets are two views of one
curve.

**Retrieved is not readable.** 82% of evidence user turns and 99.7% of
evidence assistant turns exceed 200 bytes. On the 214 questions whose short
(<=40 characters) answer appears verbatim in an evidence turn, the answer
survives truncation in 52% at 200 bytes, 80% at 400, 92% at 800, and 99.5% at
1,600. Answer visibility by injected depth and line length (tiebreak):

| k \ bytes per side | 200 | 400 | 800 | 1,600 |
|---|---:|---:|---:|---:|
| 2 | 40% / 219 | 56% / 348 | 65% / 534 | 72% / 816 |
| 4 | 49% / 419 | 68% / 670 | 77% / 1,045 | 85% / 1,610 |
| 8 | 57% / 811 | 77% / 1,304 | 87% / 2,045 | 93% / 3,166 |
| 16 | 65% / 1,583 | 85% / 2,542 | 92% / 3,993 | 96% / 6,175 |

Cells are visible answers / mean injected tokens. At ~800 tokens, 2 memories
at 1,600 bytes show the answer in 72% of questions versus 57% for 8 memories
at 200 bytes; 4 x 800 bytes beats 16 x 200 bytes at two-thirds of the tokens.
The conformal guarantees above are about retrieving a labeled memory; they do
not extend to what survives formatting. `--memory-line-chars` exposes the line
length; the end-to-end section below measures its effect on answers. This probe counts only verbatim short answers, so it is a lower bound
on the problem rather than an audit.

## End-to-end answers

`research/conformal/src/conformal_retrieval/endtoend.py` asks Qwen2.5-7B-Instruct
(Q4_K_M, llama-server, greedy, 64 output tokens) the 214 questions above, with
the current date in the system prompt and memories formatted as the proxy's
default `system` mode (tiebreak ranking, relative ages). A reply is correct
when the normalized gold answer occurs in it as whole words; there is no LLM
judge. Differences are paired against 8 x 200 bytes with a 2,000-resample
bootstrap (95% range).

| memories x bytes per side | correct | vs 8 x 200 | prompt tokens |
|---|---:|---:|---:|
| none | 1.9% | -39.3 (-46.3, -32.7) | 84 |
| 5 x 200 | 40.2% | -0.9 (-4.7, +2.3) | 548 |
| 8 x 200 | 41.1% | -- | 808 |
| 16 x 200 | 40.2% | -0.9 (-4.7, +2.3) | 1,498 |
| 4 x 400 | 50.5% | +9.3 (+3.3, +15.4) | 677 |
| 5 x 400 | 53.7% | +12.6 (+5.6, +19.2) | 821 |
| 8 x 400 | 55.1% | +14.0 (+7.9, +19.6) | 1,236 |
| 4 x 800 | 53.7% | +12.6 (+5.6, +19.6) | 1,007 |
| 8 x 800 | 59.8% | +18.7 (+12.1, +25.2) | 1,890 |
| 2 x 1,600 | 50.9% | +9.8 (+3.3, +16.4) | 802 |
| evidence only x 1,600 | 66.4% | +25.2 (+17.8, +32.2) | 796 |

The truncation probe predicted the direction. At 200 bytes, more memories do
not help: 5, 8 and 16 lines are within one point. Doubling the line length
helps at every depth: 5 x 400 beats 5 x 200 by 13.6 points (+7.9, +19.2) and
8 x 400 beats 8 x 200 by 14.0 (+8.4, +19.6). Going on to 800 bytes adds 4.7
more at k = 8 (+0.9, +8.4) for 650 more tokens. Retrieval is still the ceiling:
injecting only the labeled evidence reaches 66.4% at the same token cost as
8 x 200, and multi-session questions stay below 20% in every setting.

The proxy default `--memory-line-chars` is now 400: the longest value at which
the default five memories fit the 2,048-token budget in the worst case
(1,062 estimated tokens), so no line is dropped. With `--target-coverage` the
budget is raised above 2,048 for tiebreak 0.95 and legacy 0.9 / 0.95. The
limits are one 7B model, one prompt, and substring grading on questions whose
answer is a short verbatim string. The coverage guarantees above are unchanged
because they concern ranking, not line length.

Validation: `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`
all pass (20 tests).
