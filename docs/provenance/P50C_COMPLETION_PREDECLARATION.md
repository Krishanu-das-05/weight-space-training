# P50c Completion: Prospective Expectation

This declaration applies to a future edited-model comparison, not a revision to
the completed unedited budget sweep. The unedited results are already known.
It precedes any new matched p50c evaluation on the 30 GSM8K TRAIN questions.
It does not authorize generation, weight edits, downloads, or a new run.

## User-Declared Expectation

- Source: the user's estimate from the article's public p50c logs, not verified
  against locally cached per-response evaluation data.
- At an 8,192-token cap, approximately 30% of edited thinking-on responses are
  expected to close their thinking, with an informal estimated range of 28-40%.
- The corresponding article-based unedited estimate is approximately 9%.
- These are expectations, not measured results, confidence intervals, or
  success/failure thresholds. For context, 30% of 30 questions is 9 responses.

## Outcomes

- Primary: natural completion at or before 8,192 generated tokens, using EOS
  token 248044 only, consistent with the completed baseline's definition.
- Secondary: strict correct delivery at that cap, requiring natural completion
  and a correct answer under the existing frozen boxed-integer grader.
- Diagnostic, thinking-on only: `</think>` closed at or before 8,192 generated
  tokens. At each cap k, decode only generated token IDs `[:k]` with the pinned
  tokenizer, without skipping special tokens or cleaning up spaces, and test for
  the exact literal `</think>`. The complete delimiter must occur within that
  prefix; the prompt is excluded. EOS and correctness are not required.
- This closure diagnostic, not natural EOS, is the like-for-like endpoint for
  the article-based closure estimate. The local model, questions, and sampling
  still need not match the article's evaluation.
- Closing thinking does not necessarily terminate the response; a natural EOS
  also does not by itself establish thinking closure or correctness.
- The article-based closure estimates must not be presented as verified natural
  EOS rates or compared as if the endpoints were identical.
- Token 248046 remains diagnostic only, not an additional EOS.

## Known Unedited Baseline

Source: `results/budget_sweep_20261106T191821_788114Z/summary.json`.
Thinking on, 30 questions, 8,192-token cap:

- Natural completion: 2/30 (6.7%).
- Strict correct delivery: 1/30 (3.3%).
- Thinking closure at or before 8,192 generated tokens: **2/30 (6.7%)**.
- Original target tensor hashes match before and after the completed run.

Closure was computed inference-free from the saved generated token IDs, not from
new model responses. The pinned tokenizer's canonical `</think>` ID is 248069;
local prefix decoding reproduces the saved response text exactly for all 30
thinking-on main records. The closing delimiter completes at generated token
5355 for `gsm8k_train_06229` (natural EOS 5952) and at token 2400 for
`gsm8k_train_01026` (natural EOS 2652). Token indices are 1-based, including the
closing token.

| Generated-Token Cap | Thinking Closed |
|---:|---:|
| 512 | 0/30 |
| 1024 | 0/30 |
| 2048 | 0/30 |
| 4096 | 1/30 |
| 8192 | 2/30 |

Reproducible diagnostic and per-question results:
`p50c_unedited_closure_baseline.json` and `p50c_completion_diagnostic.py`.
This is a new prospective diagnostic added after the baseline was observed;
the completed sweep's report, raw records, and original scoring are unchanged.

The baseline is descriptive evidence from GSM8K TRAIN, which is likely present
in Qwen's training data; per-item inclusion is unverified. Absolute accuracy is
not a capability or benchmark claim.

## Execution Status

Declaration only. A complete comparison protocol and explicit run approval are
still required. Existing plans, scoring rules, reports, notebooks, and saved
responses are not changed by this declaration.
