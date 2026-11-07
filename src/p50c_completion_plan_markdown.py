"""Human-readable rendering of the prospective edited-model plan."""

import json


def render_plan(plan, digest):
    rows = "\n".join(f"| {item['id']} | {item['seed']} |" for item in plan["questions_and_seeds"])
    expected = plan["edit"]["target_edited_sha256"]
    original = plan["edit"]["target_original_sha256"]
    instruction = plan["prompt"]["instruction"]
    return f"""# P50c Vs. Original: Thinking-On Completion

Status: **AWAITING EXPLICIT APPROVAL. No edited generation has run.**

Plan SHA-256: `{digest}`. Approval must bind to this exact `plan.json`.
The frozen plan is not rewritten to authorize execution; a separate approval
record and a tested, source-fingerprinted runtime adapter are required.

## Question And Predeclaration

Does the exact full p50c edit improve **natural completion** versus the saved
original thinking-on model on this fixed development sample?

Primary endpoint: natural completion at 8,192 tokens. Secondary outcome: strict
correct delivery. Diagnostic: `</think>` closed within the generated-token cap.
All three outcomes are also tabulated at 512, 1024, 2048, 4096 and 8192 tokens.
Natural EOS and thinking closure are distinct events.

The user's article-based expectation is about 30% edited thinking closure
(informal range 28-40%), versus about 9% unedited. These estimates are
**unverified**, not natural-EOS measurements, confidence intervals or a
success threshold. Only the endpoint is like-for-like; the article's evaluation
population/runtime may differ. The baseline and earlier local toy pilots were
already observed before this new prospective declaration.

Known original thinking-on baseline, n=30:

| Outcome | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| Natural completion | 0 | 0 | 0 | 1 | 2 |
| Strict correct delivery | 0 | 0 | 0 | 1 | 1 |
| Thinking closed | 0 | 0 | 0 | 1 | 2 |

Closure at 8192 is **2/30 = 6.7%**, 95% Wilson 1.8-21.3%.
The two closing delimiters finish at generated-token indices 2400 and 5355;
the corresponding natural EOS indices are 2652 and 5952. Calculation used
only saved token IDs and the verified local tokenizer, no model inference.
Details are in `baseline_closure.json` and the prospective predeclaration.
The completed sweep's report, scorer, records and notebooks remain unchanged.

## Fixed Design

- Model: pinned Qwen3.5-0.8B revision `{plan['model']['revision']}`,
  original BF16 checkpoint, native conditional-generation class, CUDA/eager,
  no quantization/compilation, batch size 1.
- Data: the **same 30 GSM8K main TRAIN items**, sampling seed 20261106,
  unchanged integer answers, paired seeds 0-29 in the same question order.
  Dataset revision `{plan['dataset']['revision']}`; pool SHA-256
  `{plan['dataset']['dev_sha256']}`. No new sampling/downloads.
- Edited condition only: **enable_thinking=True**. Reuse the 30 saved original
  thinking-on main responses; never regenerate baseline or include off/short
  records in the paired analysis.
- Exact instruction: `{instruction}`. Each actual user message and rendered
  prompt must match the saved baseline for its question. Gold is not supplied.
- Sampling: do_sample=True, temperature 0.6, top-p 0.95, top-k 20,
  generated-token presence penalty 1.5 before warpers, repetition penalty 1.0,
  one beam, dynamic cache/use_cache=True, min_new_tokens=0.
- Generate each completed response once with **max_new_tokens=8192**;
  no separate generation per cap and no other probes/warmups.
- EOS and padding: **248044 only**, forced EOS=None, no stop on box or
  `</think>`, no extra stopping criteria. Token 248046 is counted/reported
  per response and original/edited condition but never becomes an EOS.

## Exact In-Memory Intervention

Cached p50c source: `RikiyaT/lm_understand-logs`,
commit `{plan['edit']['revision']}`, `exp8_math/runs/p50c/edit`,
CC BY 4.0. Both cached artifacts are size/SHA-256 verified:
manifest `83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225`,
NPZ `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7`.
Use the existing `weight_edit.edited_weights` context, layers (15,19),
strength=1.0, exactly 130,188 changed BF16 entries. No new edit implementation.

1. Load the pinned original locally and verify both original hashes.
2. Enter the verified context; apply both tensors and verify both edited hashes
   before generating. Verify edited weights are active before each call and log
   that provenance with the raw response.
3. In `finally`, restore and independently verify both original hashes after
   success, budget stop, exception and handled interrupt.
4. Any apply/hash/restore failure is fatal; never silently continue with
   originals or claim restoration without verification.
5. A hard kill/power loss cannot execute cleanup. No edited weights exist on
   disk; any new process reloads/verifies the original checkpoint, then reapplies
   the edit before resuming pending work.

Original target hashes:
```json
{json.dumps(original, indent=2)}
```
Required active edited hashes:
```json
{json.dumps(expected, indent=2)}
```
**Never save model weights or target tensors.** The original checkpoint's full
SHA-256 must also remain unchanged.

## Determinism And Reused Prefix Evidence

- Same Python {plan['execution']['python']}, Torch {plan['execution']['torch']},
  Transformers {plan['execution']['transformers']}, CUDA {plan['execution']['cuda_runtime']},
  RTX 2000 Ada laptop GPU and driver {plan['hardware']['nvidia_driver']};
  full OS/hardware metadata is frozen in JSON and rechecked at launch.
- Set **CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch/CUDA initialization**.
  Strict deterministic algorithms, warn_only=False; threads **1 intra/1 inter**;
  TF32 off, highest float32 matmul precision, cuDNN deterministic=True and benchmark=False,
  eval/inference mode. Failure to enable determinism => STOP, no relaxation.
- Same per-question CPU/CUDA RNG reset as the baseline. Reuse the unchanged,
  source-hashed `budget_sweep_generate.generate_record` implementation.
- **No new truncation check is scheduled.** All four original short/long
  checks passed. Architecture, tokenizer, prompts, batching, seeded sampler,
  forward/cache implementation and determinism configuration are identical;
  only values in two weight tensors change. `max_new_tokens` remains a stop
  bound. This reuses prior evidence, not a claim that edited prefixes have
  newly been empirically verified on every input.
- Any required configuration/core change => STOP and present a revised plan,
  including a new prefix check if needed, for fresh approval. No silent change.
- No wall-clock generation limit, `max_time`, subprocess kill timer or
  time-based stopping criterion. Record all elapsed times, including pauses.

## Frozen Outcomes And Analysis

- Primary natural completion at cap k: first EOS 248044 index <=k,
  regardless of correctness or thinking closure.
- Secondary strict correct delivery: primary condition AND the existing
  unchanged boxed-integer/final-answer grader marks it correct.
- Thinking-on closure diagnostic: locally decode generated token IDs `[:k]`
  with the pinned tokenizer, skip_special_tokens=False and no cleanup, then
  test for the exact complete literal `</think>`. Do not count prompt tokens.
  Count a response once; EOS/correctness are not required. Split delimiter
  pieces count only after the whole literal is inside the prefix.
- For original and edited, each outcome/cap has count/n, rate and
  **95% pointwise Wilson interval**. Also report mean(min(actual tokens,k)),
  a point estimate only, not a Wilson interval or extra compute.
- Pair all 30 questions by ID/seed for each outcome/cap: edited wins,
  original wins, ties, signed percentage-point difference and **two-sided exact
  McNemar p**; no discordance gives p=1.
- Raw and Holm-adjusted p-values across the five caps **separately per outcome**:
  primary, secondary and diagnostic are three distinct families, not global
  familywise control across all 15 tests. Secondary/diagnostic evidence cannot
  replace the primary endpoint. No adaptive endpoint/subset/cap selection.
- Partial run: disclose missing IDs and actual denominators, no planned
  complete-n=30 inference. Investigate all naturally finished short (<=1024)
  wrong edited replies without changing grading.
- n=30 generally detects only large effects (~25+ points); exact paired power
  depends on discordances and multiplicity. Non-significance is not equivalence.

## Budget And Resumption

**30 x 8192 = 245,760 tokens maximum without interruptions**, including zero
new short checks. Limit **250,000**, headroom **4,240**. Only these edited
generations use this allocation; no unused allocation carries over.

Use one exclusive run lock. Durably reserve the full next cap before starting;
reconcile to actual tokens only after its canonical raw record is flushed/fsynced
and the journal is durable. Preserve token IDs/text, prompts, seed, outcomes,
edit provenance, EOS index, timings and memory per response before the next call.
Records/ledger bind to this plan SHA and the published_p50c variant.

After a crash, validate the plan, approval, sources, baseline, assets and runtime;
skip all completed records. Only an unfinished item can restart with its same
seed under a new reservation. An unresolved 8192-token call remains fully
charged, so a retry may need savings from earlier natural EOS: headroom alone
does not guarantee recovery of a worst-case full run. If a call cannot be funded,
stop cleanly as partial, restore/verify originals and report accounting. Never
exceed the limit, erase reservations, or repeat completed work.

## Reporting And Stop

After two main edited responses, report measured tokens/s, peak GPU allocation
and a provisional finish projection, then continue automatically. No extra
generation for profiling. Host delays can invalidate an ETA; it is no deadline.

After the approved run, save records/summary and verified weight/error history;
produce a five-sentence-summary report, all three outcome tables and paired
comparisons, one labelled three-panel PNG+SVG with Wilson bands, and read-only
notebook 07 (RUN_EXPERIMENT=False). Recompute from saved records, validate
offline tests/budget/rollback/prior-artifact integrity, then **STOP**.

GSM8K TRAIN is likely in Qwen's training data; per-item inclusion is unverified.
This small fixed one-model intervention is not training, a capability/benchmark
claim, an article-rate reproduction or generalization to other questions/models.
No held-out use, new questions, downloads, installs, extra seeds/conditions,
strength/layer search, weight saves, commits or pushes.

## Frozen Question Pairing

| GSM8K TRAIN ID | Seed |
|---|---:|
{rows}

**Paused for explicit approval. No edited inference is authorized or started.**
"""
