# Weight-Space Lab: First Controlled Pilot

> Curated copy of a completed historical report. Original flat-layout paths refer to the local workbench reconstructed by `scripts/lab.py`; full raw runs are not distributed with the code. The original report is preserved locally.

Date: 2026-11-06. Status: completed local pilot; no validated weight-edit gain.

## Scope And Protocol

- Qwen3.5-0.8B, original pinned BF16 checkpoint.
- Transformers 5.20.2 / PyTorch 2.14.1+cu130 on the local NVIDIA GPU.
- Two handmade development questions: the existing powers and discount probes.
- One seed group, with per-question seeds 0 and 1, paired across conditions.
- Temperature 0.6, top-p 0.95, top-k 20, generated-token presence penalty 1.5.
- Maximum 1,024 generated tokens per response; 900-second wall-time budget
  and 12,288-token total allocation.
- No training, optimizer, model-file rewrite, network access, or new packages.

The harness now distinguishes a boxed answer inside reasoning, a final
answer after thinking closes, a natural EOS stop, and token/time limits.
"Delivered correct" requires both a correct final answer and natural EOS.
Exact-string instruction controls remove known trailing message/EOS markers,
not arbitrary generated words.

The installed Transformers processor order is tested: presence penalty,
temperature, top-k, then top-p. No penalty-order bug was found.
This does not establish numerical equivalence with vLLM.

## Behavioral Result

| Condition | Delivered Correct | Natural EOS | Hit Token Cap | Mean Tokens |
| --- | --- | --- | --- | --- |
| Original, thinking on | 0/2 | 0/2 | 2/2 | 1024 |
| Original, thinking off | 1/2 | 1/2 | 1/2 | 632 |
| Published edit, both layers | 0/2 | 0/2 | 2/2 | 1024 |
| Layer 15 only | 0/2 | 0/2 | 2/2 | 1024 |
| Layer 19 only | 0/2 | 0/2 | 2/2 | 1024 |
| Half-strength edit | 0/2 | 0/2 | 2/2 | 1024 |

Thinking-off returned the correct final answer, 1048, on the powers
question in 240 tokens. It still hit the cap on the discount question.
It is therefore a useful baseline, not a universal cure.

The edited conditions modestly reduced the repeated-four-token-gram
heuristic relative to thinking-on, but did not improve delivered answers.
That heuristic is not a validated loop detector.

All 12 planned generations completed. Original target tensor hashes were
verified after the run. Raw text, per-response records, source hashes,
settings, and the gate decision are retained locally in:

`results/pilot_20261106T105240_469212Z/`

## What The Edit Looks Like

| Property | Layer 15 | Layer 19 |
| --- | --- | --- |
| BF16 entries actually changed | 65,260 | 64,928 |
| Changed columns | 64 | 64 |
| Delta Frobenius norm / original matrix norm | 10.636% | 8.106% |
| Delta energy in first singular direction | 99.99945% | 99.99915% |
| Components needed for 99% energy | 1 | 1 |

The absolute cosine between the leading directions is 0.8490:
related, but not identical. The correction is approximately rank one in
each matrix after BF16 rounding. Sparse global coverage does not imply
a negligible perturbation to each affected matrix.

The anatomy notebook saves column-norm plots, cumulative singular-energy
curves, and binned signed heatmaps. These describe weight geometry;
they do not establish that a direction semantically represents looping.

## Activation Check

Four short teacher-forced forward passes compare the same fixed token
inputs under base and published weights. They capture only projections
and norms from the two MLP outputs, not full activation dumps.

For example, mean absolute cosine on the synthetic rechecking text at
layer 15 changes from about 0.1345 to 0.3719. Other projections also change.
This is not evidence that the edit simply deletes a "looping direction":
these are synthetic texts, absolute projection is not a semantic label,
and intermediate activations change under intervention.

The original hashes are checked afterward, hooks are removed even on
errors, and the notebook releases its own model memory.

Figures and traces are in:
`results/anatomy_20261106T125019_166402Z/`

## Stopping Decision

The predeclared gate requires a complete paired pilot and an absolute
25-percentage-point delivered-correct improvement over the better of
thinking-on and thinking-off, without reducing natural completion
relative to thinking-on.

No weight variant passes. **Stop this candidate search here.**
The 20 handmade held-out math questions and 10 instruction/QA controls
remain unused for model inference. The runner refuses to promote this
failed pilot into a held-out run.

This is a bounded negative result, not a refutation of the article:
two easy questions, one seed group, a much shorter generation limit,
different inference machinery, and different grading do not reproduce
its MATH/GSM8K benchmark protocol. Side effects on general behavior have
not been established because held-out evaluation was deliberately gated.

The next investigation needs a new, explicit hypothesis or a carefully
budgeted reference-protocol comparison, not a search expanded until a
favorable example appears.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Both new notebooks execute offline, including GPU teacher forcing.
Tests cover edit/identity/half-strength hashes and rollback, sparse-index
boundaries, scoring and seed behavior, sampler order, budget handling,
partial reports, held-out gating, SVG escaping, and hook cleanup.
