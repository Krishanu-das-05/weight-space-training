# Prompt Controls: Completion Is Not Correctness

> Curated copy of a completed historical report. Original flat-layout paths refer to the local workbench reconstructed by `scripts/lab.py`; full raw runs are not distributed with the code. The original report is preserved locally.

Date: 2026-11-06. Status: all 16 planned responses completed.

## Question And Fixed Design

Does requesting step-by-step reasoning contribute to poor completion,
independently of a weight edit?

The original pinned Qwen3.5-0.8B checkpoint is used in all four conditions.
No weight-edit context is entered. The powers and discount development
questions are paired across seed groups 0 and 100; actual per-question
seeds are 0, 1, 100, and 101.

- Thinking on/off is controlled by the chat template's `enable_thinking`.
- Step instruction: `Please reason step by step, and put your final answer within \boxed{}.`
- Final-only instruction: `Please give only your final answer within \boxed{}, with no explanation.`
- Sampling: temperature 0.6, top-p 0.95, top-k 20, generated-token
  presence penalty 1.5, repetition penalty 1.0.
- Maximum 1,024 new tokens per response; 16,384 total token allocation;
  900-second soft wall-time limit.
- Same strict boxed-integer final-answer grader as the earlier pilot.
  Delivered correct requires a correct final answer and natural EOS.

The run uses 10,771 generated tokens and approximately 410 seconds of
experiment wall time. No generation hits its time limit. Timings are
local observations, not speed benchmarks.

## Results

Each row contains four responses from two questions and two seed groups.

| Thinking | Instruction | Strict Delivered Correct | Natural EOS | Hit Token Cap | Mean Tokens |
| --- | --- | --- | --- | --- | --- |
| On | Step by step | 0/4 | 0/4 | 4/4 | 1024.00 |
| On | Final only | 1/4 | 1/4 | 3/4 | 899.75 |
| Off | Step by step | 2/4 | 2/4 | 2/4 | 659.50 |
| Off | Final only | 2/4 | 4/4 | 0/4 | 109.50 |

With thinking off, final-only instructions improve observed completion
from 2/4 to 4/4 and reduce mean generated tokens by about 83.4%.
They do **not** improve aggregate strict correct delivery: both modes
score 2/4, with different successful cases.

The thinking-on final-only condition completes the powers question in
527 tokens for seed group 0, but hits the cap on all three other responses.
It is not a reliable fix in this tiny sample.

## Paired Effects

Wins/losses refer to delivered correctness for the same question and seed.
Token deltas are treatment minus reference; negative means fewer tokens.

| Reference -> Treatment | Wins | Losses | Ties | Mean Token Delta |
| --- | --- | --- | --- | --- |
| Thinking on: steps -> final | 1 | 0 | 3 | -124.25 |
| Thinking off: steps -> final | 1 | 1 | 2 | -550.00 |
| Steps: thinking on -> off | 2 | 0 | 2 | -364.50 |
| Final: thinking on -> off | 2 | 1 | 1 | -790.25 |

The final-only versus step comparison with thinking off has one win and
one loss, not an across-the-board improvement.

Seed group 0 delivers 0/2, 1/2, 1/2, and 1/2 for the four table rows.
Seed group 100 delivers 0/2, 0/2, 1/2, and 1/2 respectively.

## Inspect The Short Answers

The two shortest thinking-off/final-only responses are wrong:

- Powers, seed group 0: `\boxed{264}` in 10 tokens; expected 1048.
- Discount, seed group 100: `\boxed{70.4}` in 11 tokens; expected 66.

The integer-only grader rejects the second decimal box, but this is not
merely a formatting disagreement: 70.4 is also numerically wrong.
The original scoring rule is unchanged after inspecting the results.

The two correct thinking-off/final-only responses are much longer:
194 tokens for discount/group 0 and 223 for powers/group 100. Both include
explanation despite the final-only instruction. The requested style is
an experimental input, not a guarantee of compliance.

## Interpretation And Stop

Instructions and thinking mode materially affect completion and cost on
these development examples. They do not give us a universally correct,
short-answer configuration. Stopping earlier and producing less text
cannot serve as proxies for correctness.

This is not evidence that the model learns new math, that a weight edit
works, or that the article's benchmark is reproduced. Two handmade
questions with two seed groups are not four independent tasks, and the
generation cap and grading differ from the published evaluation.

Stop this diagnostic at the predeclared 16 responses. The held-out set
remains unused for inference, and the earlier weight-edit gate stays
closed. A new prompt variant, larger budget, or reference-protocol
comparison needs a separate hypothesis and declared scope.

## Artifacts And Reproduction

Local artifacts:
`results/prompt_control_20261106T173626_294729Z/`

- `plan.json`: fixed conditions, instructions, sampling, source hashes.
- `records.jsonl`: each completed response flushed before the next starts.
- `summary.json`: raw responses, metrics, seed summaries, paired effects.

Notebook `05_prompt_controls.ipynb` reads this run by default without
launching another GPU experiment. Its reader validates the design and
recomputes summary/effect tables from response records.

To deliberately repeat the bounded experiment:

```powershell
.\.venv\Scripts\python.exe experiment_runner.py --stage prompt-control --max-tokens 1024 --seconds 900 --token-budget 16384
```

The original target tensor hashes are verified before and after the run.
No weights are edited or saved. No packages, downloads, commits, or pushes
are required.
