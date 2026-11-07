"""Read-only reporting for the approved, provenance-checked edited comparison."""

import json
from pathlib import Path

import nbformat

from budget_sweep_io import ROOT, json_bytes, read_json
from budget_sweep_outputs import fenced
from p50c_completion_analysis import OUTCOMES, original_records, recompute
from p50c_completion_io import RUN, EditedRecordStore, load_approved, make_decoder
from p50c_restore_audit import restoration_failure_report
from p50c_amendment import DECLARATION, load_amended
from p50c_amended_analysis import recompute_amended, verify_legacy_outcomes
from p50c_amended_store import AmendedRecordStore

LABELS = {
    "natural_completion": "Natural Completion (Primary)",
    "strict_correct_delivery": "Strict Correct Delivery (Secondary)",
    "thinking_closed": "Thinking Closed (Diagnostic)",
}


def load_saved_analysis(directory=RUN, *, require_complete=True):
    directory = Path(directory).resolve()
    if directory != RUN:
        raise ValueError("Not the approved run directory.")
    if DECLARATION.exists():
        plan, pool, declaration, digest = load_amended(sealed=True)
        store = AmendedRecordStore(directory, plan, pool, declaration, digest, make_decoder(plan))
    else:
        plan, pool = load_approved(directory)
        store = EditedRecordStore(directory, plan, pool)
    records = store.records()
    raw_journal = directory / "records.jsonl"
    if not raw_journal.exists():
        raise ValueError("Missing completed-record journal.")
    with raw_journal.open("rb") as handle:
        journal_bytes = handle.read(50_000_001)
    if len(journal_bytes) > 50_000_000 or journal_bytes != b"".join(json_bytes(row) for row in records):
        raise ValueError("Journal/canonical snapshot mismatch; resume the guarded worker for recovery.")
    references = {item["id"]: item for item in plan["baseline"]["selection"]}
    for row in records:
        reference = references[row["problem_id"]]
        if row.get("user_message") != reference["user_message"] or row.get("rendered_prompt") != reference["rendered_prompt"]:
            raise ValueError("The edited prompt differs from its fixed original counterpart.")
    if DECLARATION.exists():
        analysis = recompute_amended(plan, pool, records, original_records(plan), store.decode, declaration, digest)
        verify_legacy_outcomes(analysis, read_json(DECLARATION.parent / "legacy_analysis.json"))
        if any(row["attempt_id"] == declaration["retained_unreconciled_attempt_id"] for row in records):
            raise ValueError("The original unpersisted attempt must remain conservatively charged.")
    else:
        analysis = recompute(plan, pool, records, original_records(plan), make_decoder(plan))
    summary = read_json(directory / "summary.json")
    budget = store.budget(records)
    if require_complete:
        if (
            summary["status"] != "complete" or not analysis["complete_data"]
            or summary["target_hash_checks"]["verified_after_restore"] is not True
            or summary["target_hash_checks"]["original_after_restore"] != plan["model"]["target_base_sha256"]
            or summary["target_hash_checks"]["edited_after_apply"] != plan["edit"]["target_edited_sha256"]
            or budget != summary["budget"]
            or DECLARATION.exists() and (
                summary.get("amendment_sha256") != digest or summary.get("legacy_11_outcomes_unchanged") is not True
                or budget["unreconciled_reserved_tokens"] < declaration["retained_reservation_tokens"]
            )
        ):
            raise ValueError("Reporting requires all 30 verified responses and verified weight restoration.")
        for key, value in analysis.items():
            if json_bytes({"value": value}) != json_bytes({"value": summary[key]}):
                raise ValueError("Saved summary does not match raw paired recomputation.")
    return plan, pool, records, summary, analysis, budget


def _rate(row):
    if not row["n"]:
        return "not available"
    lo, hi = row["wilson95"]
    return f"{row['successes']}/{row['n']} ({row['rate']:.1%}; {lo:.1%}-{hi:.1%})"


def tables_markdown(analysis):
    parts = []
    for outcome in OUTCOMES:
        data = analysis["tables"][outcome]
        parts.extend([
            f"### {LABELS[outcome]}", "",
            "Counts, rates and 95% pointwise Wilson intervals.", "",
            "| Token cap | Original | Published p50c |",
            "|---:|---:|---:|",
        ])
        for original, edited in zip(data["original"], data["edited"]):
            parts.append(f"| {original['cap']:,} | {_rate(original)} | {_rate(edited)} |")
        parts.append("")
    parts.extend(["### Mean Tokens", "", "Mean(min(actual generated length, cap)); no Wilson interval applies to this continuous mean.",
                  "These are counterfactual capped lengths, not separate generations or additional compute.", "",
                  "| Token cap | Original | Published p50c |", "|---:|---:|---:|"])
    for original, edited in zip(analysis["tables"]["natural_completion"]["original"],
                                analysis["tables"]["natural_completion"]["edited"]):
        value = f"{edited['mean_tokens']:.1f}" if edited["n"] else "not available"
        parts.append(f"| {original['cap']:,} | {original['mean_tokens']:.1f} | {value} |")
    return "\n".join(parts)


def paired_markdown(analysis):
    if not analysis["complete_data"]:
        return "Not performed: the planned complete 30-question edited comparison is not yet available."
    parts = [
        "Each row pairs the same 30 question IDs and seeds. Difference = edited minus original, in percentage points.",
        "Exact two-sided McNemar tests use discordant pairs; zero discordance gives p=1.",
        "Holm adjustment covers five caps separately for each outcome, not all 15 tests jointly.",
        "Secondary/diagnostic results cannot replace the primary natural-completion endpoint.", "",
    ]
    for outcome in OUTCOMES:
        parts.extend([f"### {LABELS[outcome]}", "",
                      "| Cap | Edited wins | Original wins | Ties | Difference (pp) | Exact p | Holm p |",
                      "|---:|---:|---:|---:|---:|---:|---:|"])
        for row in analysis["paired_comparisons"][outcome]:
            parts.append(
                f"| {row['cap']:,} | {row['edited_wins']} | {row['original_wins']} | {row['ties']} | "
                f"{row['difference_percentage_points']:+.1f} | {row['exact_mcnemar_p']:.6g} | {row['holm_adjusted_p']:.6g} |"
            )
        parts.append("")
    return "\n".join(parts)


def public_summary(analysis, summary):
    if not analysis["complete_data"] or summary["status"] != "complete":
        sentences = [
            f"The edited run has {analysis['edited_main_responses']} of 30 planned responses and is not a completed comparison.",
            "Natural completion is the primary outcome, with strict correct delivery secondary and thinking closure diagnostic.",
            "No planned complete-sample paired inference is reported before all 30 edited responses are available.",
            "Any partial counts use their actual denominator and do not justify a general model-performance claim.",
            "GSM8K TRAIN is likely in Qwen's training data, so absolute accuracy is not a capability or benchmark claim",
        ]
    else:
        endpoint = {outcome: {variant: rows[-1]["successes"] for variant, rows in data.items()}
                    for outcome, data in analysis["tables"].items()}
        natural, correct, closed = (endpoint[outcome] for outcome in OUTCOMES)
        sentences = [
            f"On 30 fixed GSM8K TRAIN questions, the published p50c edit yielded {natural['edited']}/30 naturally finished thinking-on responses by 8,192 generated tokens, versus {natural['original']}/30 for the saved original model.",
            (f"Under a text-bound handling amendment declared after 11 saved responses and before the remaining 19, "
             f"strict correct delivery was {correct['edited']}/30 edited versus {correct['original']}/30 original.")
            if summary.get("amendment_sha256") else
            f"Strict correct delivery was {correct['edited']}/30 edited versus {correct['original']}/30 original.",
            f"The separate thinking-closure diagnostic was {closed['edited']}/30 edited versus {closed['original']}/30 original; it is the endpoint relevant to the unverified article-based expectation, not the same event as natural EOS.",
            f"The edited run accounted for {summary['budget']['accounted_tokens']:,} of 250,000 tokens, every accepted edited response carries verified published tensor hashes, and original target weights were restored.",
            "GSM8K TRAIN is likely in Qwen's training data (per-item inclusion is unverified), so these one-model development results are not a capability or benchmark claim",
        ]
    return " ".join(sentence.rstrip(".") + "." for sentence in sentences)


def amendment_markdown(analysis, summary):
    if not summary.get("amendment_sha256"):
        return ""
    _, _, declaration, digest = load_amended(sealed=True)
    if digest != summary["amendment_sha256"]:
        raise ValueError("Report amendment identity differs from the sealed run.")
    parts = [
        "## Pre-Declared Amendment", "",
        "The user approved this amendment **after 11 saved responses and before any of the remaining 19 generations**.",
        f"Declaration time: `{declaration['declared_utc']}`; amendment SHA-256: `{digest}`.",
        "The original plan, original 23 source files, 11 canonical records and their outcomes remain unchanged.",
        "Prior partial report/summary/analysis and the original journal are archived under `amendments/text_bound_01/before/`.",
        "",
        "- Persist generated token IDs, then raw text when decodable, before response validation or grading.",
        "- Flag decoded lengths outside 1-200,000 characters as `text_bound_violation`, recording the exact character length.",
        "- Such responses have strict-correct=false even if boxed text appears; natural EOS and literal thinking closure come from token IDs only.",
        "- Full oversized/empty raw text is retained separately; the accepted JSON record need not embed unbounded text.",
        "- Structural token/EOS/provenance checks and strict determinism remain mandatory; decoding failures retain IDs and stop.",
        "",
        "An inference-free regression proved byte-for-byte preservation and identical outcomes/tables for the saved 11.",
        "The sampling call, forked RNG setup and seed calls are AST-identical to the original core; no new prefix generation was added.",
        "Item `gsm8k_train_02851` was regenerated with the same seed and configuration, followed by the 18 never-started items.",
        "The original failed attempt had no persisted raw output, so exact equality to that old output cannot be empirically verified.",
        "Its full 8,192-token reservation remains charged and is not reconciled against the regenerated response.",
        "",
        "This is an approved prospective handling change after partial results were known, **not an untouched initial pre-registration**.",
        "It changes no saved outcome among the original 11 and must not be hidden when sharing results.", "",
        "### Text-Bound Violations", "",
        "| Question | Decoded characters | Natural EOS index | Thinking closed by 8192 |",
        "|---|---:|---:|---|",
    ]
    for row in analysis.get("text_bound_violations", []):
        parts.append(f"| {row['id']} | {row['character_length']:,} | {row['natural_eos_index']} | {row['thinking_closed_at_8192']} |")
    if not analysis.get("text_bound_violations"):
        parts.append("| None | - | - | - |")
    return "\n".join(parts)


def diagnostic_markdown(analysis):
    parts = [
        "Token 248046 remains diagnostic only, never an extra EOS.", "",
        "| Variant | Responses containing 248046 | Total occurrences |",
        "|---|---:|---:|",
    ]
    for variant, data in analysis["token_248046_diagnostics"].items():
        parts.append(f"| {variant} | {data['responses_with_248046']} | {data['occurrences']} |")
    for variant, data in analysis["token_248046_diagnostics"].items():
        if data["affected_responses"]:
            entries = ", ".join(f"{row['id']} ({row['occurrences']})" for row in data["affected_responses"])
            parts.extend(["", f"Containing {variant} responses (occurrence counts): {entries}."])
    return "\n".join(parts)


def short_wrong_markdown(analysis):
    rows = analysis["short_wrong_answers"]
    if not rows:
        return "No edited response both ends naturally within 1,024 tokens and fails the frozen strict grader."
    parts = [
        f"All {len(rows)} edited responses with natural EOS <=1,024 and strict correct delivery false are listed.",
        "Excerpts are bounded to 1,500 characters; complete unmodified responses remain in the durable records.",
    ]
    for row in rows:
        parts.extend([
            "", f"### {row['id']}",
            f"Seed {row['seed']}; {row['tokens']} tokens; parsed final integer "
            f"`{row['parsed_final_answer']}`; reference `{row['expected']}`.", "",
            fenced(row["text"][:1500]),
        ])
    return "\n".join(parts)


def render_report(plan, records, summary, analysis, *, image_relative, restore_audit=None):
    checks = summary.get("target_hash_checks", {})
    if (
        not isinstance(restore_audit, dict) or restore_audit.get("passed") is not True
        or checks.get("verified_after_restore") is not True
        or checks.get("original_after_restore") != plan["model"]["target_base_sha256"]
    ):
        return restoration_failure_report(plan, restore_audit or {"passed": False, "issue": "missing independent evidence"})
    if not analysis["complete_data"] or summary["status"] != "complete":
        raise ValueError("A final report cannot silently promote an unfinished run.")
    primary = analysis["paired_comparisons"]["natural_completion"][-1]
    closure = analysis["tables"]["thinking_closed"]["edited"][-1]
    budget = summary["budget"]
    seconds = sum(row["seconds"] for row in records)
    rate = sum(len(row["token_ids"]) for row in records) / seconds
    peaks = max(row["peak_gpu_allocated_bytes"] for row in records) / 1024**3
    durations = "\n".join(
        f"| {row['problem_id']} | {len(row['token_ids'])} | {row['seconds']:.2f} | "
        f"{len(row['token_ids']) / row['seconds']:.2f} |" for row in records
    )
    return f"""# P50c Thinking-On Completion Report

{public_summary(analysis, summary)}

## Question

Does the full published p50c edit improve natural completion versus the saved
original model on these same 30 development questions? The primary endpoint is
natural EOS by 8,192 generated tokens. Strict correct delivery is secondary.
Thinking closure is a separate diagnostic, not a substitute primary outcome.

## Fixed Design

- Model: Qwen3.5-0.8B, revision `{plan['model']['revision']}`,
  original BF16 checkpoint, eager attention/dynamic cache, batch size 1.
- Questions: exactly the same 30 GSM8K `main` TRAIN items and integer answers,
  sampling seed 20261106, per-question generation seeds 0-29.
  Dataset revision `{plan['dataset']['revision']}`.
- Thinking on only (`enable_thinking=True`); fixed instruction:
  `{plan['prompt']['instruction']}`. Every actual prompt matches its original.
- Same sampling: temperature 0.6, top-p 0.95, top-k 20, generated-token presence
  penalty 1.5, repetition penalty 1, one beam. One 8,192-token call per completed
  item; caps 512/1024/2048/4096/8192 are post-hoc views, not extra generations.
- Full published edit: layers 15 and 19, strength 1.0, 130,188 BF16 values.
  Rikiya Takehi et al., `RikiyaT/lm_understand-logs`,
  commit `{plan['edit']['revision']}`, CC BY 4.0, used without modification.
- EOS remains 248044 only. A closed `</think>` and token 248046 do not stop
  generation. Correct delivery requires natural EOS within the cap. Bounded
  decoded text uses the unchanged strict final boxed-integer grader; an approved
  text-bound exception, when present, is disclosed separately and never correct.
- Closure means exact literal `</think>` wholly inside locally decoded generated
  token IDs `[:k]`, skip_special_tokens=False, no cleanup, excluding the prompt.
  It does not require EOS or correctness; each response counts once per cap.
- Original control: reuse only its 30 thinking-on main records, never regenerate,
  and exclude thinking-off/short-validation responses from the comparison.
- Strict determinism: CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA, threads 1/1,
  deterministic algorithms with warn_only=False, TF32 off, eval/inference mode.
  No wall-clock cutoff. No new prefix checks: identical frozen generation core
  and determinism reuse the four passing original checks, not a newly measured
  guarantee for all edited prefixes.

The baseline and earlier local toy pilots were known before this predeclaration;
new matched edited GSM8K responses were not. The article-based ~30% edited
closure expectation (informal 28-40% range, unedited ~9%) was estimated and
unverified, not a confidence interval, natural-EOS expectation or pass/fail gate.

## Results

![Primary, secondary and diagnostic outcome curves]({image_relative})

{tables_markdown(analysis)}

{amendment_markdown(analysis, summary)}

## Paired Effects

{paired_markdown(analysis)}

At the primary 8,192-token endpoint, the paired effect is
**{primary['difference_percentage_points']:+.1f} percentage points**:
{primary['edited_wins']} edited wins, {primary['original_wins']} original wins
and {primary['ties']} ties; exact p={primary['exact_mcnemar_p']:.6g},
Holm p={primary['holm_adjusted_p']:.6g}.

The observed edited thinking-closure rate is {closure['successes']}/30
({closure['rate']:.1%}). Only this diagnostic is like-for-like in endpoint with
the article-based estimate; it is not a reproduction/verification of the
article's evaluation population or rate. Closure and EOS remain distinct even
when their counts coincide. No secondary/diagnostic result replaces the primary.

## Short Wrong Answers

{short_wrong_markdown(analysis)}

## Token 248046 Diagnostic

{diagnostic_markdown(analysis)}

## Reliability And Cost

- Each start/resume loads and verifies originals, reapplies p50c and verifies both
  edited tensors before work. Every response carries its actual
  `verified_edited_tensor_sha256`; storage and analysis reject missing/mismatched
  hashes. A fresh-process gate cannot inherit verification from prior records.
- All {len(records)} accepted records match the planned edited hashes and fixed
  original prompts. Verification is repeated immediately before every generation.
- The existing context restores originals in `finally`, including handled
  exceptions/interrupts; independent post-run original hashes are verified.
  A hard kill cannot execute cleanup, but no edited weights are saved and a new
  process must reload/reverify pristine disk weights before reapplying the edit.
- Edited allocation: {budget['accounted_tokens']:,}/250,000 accounted tokens:
  {budget['completed_actual_tokens']:,} completed, plus
  {budget['unreconciled_reserved_tokens']:,} unreconciled full-cap reservations.
  Reservations are durable before generation; completed records are fsynced
  before the next call, and completed work is never repeated.
- Recorded generation wall time totals {seconds / 3600:.2f} hours, aggregate
  {rate:.2f} tokens/s, peak allocated GPU memory {peaks:.2f} GiB. Times may include
  host pauses; the original baseline had an unusually long timing observation.
  These are not controlled GPU performance comparisons.

All individual edited timings are retained, without outlier removal:

| Question ID | Generated tokens | Generation wall seconds | Tokens/s |
|---|---:|---:|---:|
{durations}

## Limitations

- Only 30 selected TRAIN development questions, one seed per item and one 0.8B
  checkpoint. The original and edited runs occurred at different times.
- GSM8K TRAIN is likely in Qwen's training data; per-item inclusion is unverified.
  Absolute accuracy is not a capability or benchmark claim.
- n=30 generally detects only large effects (~25+ percentage points); paired
  discordance and multiplicity matter. Non-significance is not equivalence.
- Wilson bands are pointwise binomial intervals, not simultaneous bands or
  confidence intervals on paired differences/mean tokens. Holm control applies
  within each five-cap outcome family, not globally across the 15 comparisons.
- The original prefix validation is inherited under an identical implementation;
  it is not a fresh empirical edited-weight check. Integer annotation parsing
  does not independently prove all reference solutions or rule out near duplicates.
- The closure estimate from public article logs remains unverified locally and
  may come from different questions, sampling and stop semantics.
- If present, the text-bound handling amendment was declared after 11 outcomes
  were known, before the remaining 19; its scope and all violations are disclosed.

## Stop

Independent restoration check: **PASSED in both artifacts**. In `summary.json`
and `{Path(restore_audit['latest_history_file']).name}`, `verified_after_restore`
is true and `original_after_restore` exactly matches the planned base hashes.
The full observed/planned hash maps are retained in `restoration_audit.json`.
No status field is used as proof of restoration; missing or mismatched evidence
would replace this Stop section with a restoration-failure warning.

The one fixed edited comparison is complete. No held-out use, model/tensor saves,
new downloads/installations, extra conditions/seeds/strength search, commits or
pushes. Original target weights are restored and verified; prior reports,
scorers, records and notebooks remain unchanged. No further generation is
scheduled, regardless of these results.
"""


def make_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(
        kernelspec={"display_name": "Python (weight-space)", "language": "python", "name": "weight-space"},
        language_info={"name": "python", "version": "3.12.3"},
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# 07 - Published p50c Completion Comparison\n\n"
            "Read-only analysis of the fixed edited run and its saved unedited thinking-on control. "
            "No model loading, generation, weight edits or downloads occur here. "
            "Natural EOS is primary; strict correct delivery is secondary; thinking closure is diagnostic. "
            "GSM8K TRAIN is likely in Qwen's training data, so absolute accuracy is not a capability claim."
        ),
        nbformat.v4.new_code_cell(
            "import json\nimport sys\nfrom pathlib import Path\n\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "cwd = Path.cwd().resolve()\n"
            "ROOT = cwd.parent.parent if cwd.name.startswith('p50c_completion_') and cwd.parent.name == 'results' else cwd\n"
            "if not (ROOT / 'p50c_completion_outputs.py').is_file():\n"
            "    raise RuntimeError('Open this notebook from the project or its approved run directory.')\n"
            "if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n"
            "from p50c_completion_io import RUN\n"
            "from p50c_completion_outputs import (\n"
            "    diagnostic_markdown, load_saved_analysis, paired_markdown,\n"
            "    public_summary, short_wrong_markdown, tables_markdown,\n"
            ")\n"
            "from budget_sweep_io import read_json, sha256\n\n"
            "RUN_EXPERIMENT = False\n"
            "if RUN_EXPERIMENT:\n"
            "    raise RuntimeError('This is a read-only notebook; use the explicitly approved guarded worker for execution.')\n"
            "plan, pool, records, saved, analysis, budget = load_saved_analysis(RUN)\n"
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
            "print('Loaded verified saved records:', len(records), 'edited + 30 original')\n"
            "print('No inference performed. Accounted edited tokens:', budget['accounted_tokens'])"
        ),
        nbformat.v4.new_markdown_cell(
            "## Public Summary And Outcomes\n\nAll tables below are recomputed from raw saved token IDs/text, "
            "not copied from the report. Wilson intervals are pointwise; they do not apply to mean tokens."
        ),
        nbformat.v4.new_code_cell(
            "display(Markdown(public_summary(analysis, saved)))\n"
            "display(Markdown(tables_markdown(analysis)))"
        ),
        nbformat.v4.new_code_cell(
            "figure_path = RUN / 'p50c_vs_original_token_budget.png'\n"
            "finalization = read_json(RUN / 'finalization.json')\n"
            "assert sha256(figure_path) == finalization['artifact_sha256'][figure_path.relative_to(ROOT).as_posix()]\n"
            "display(Image(filename=str(figure_path)))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Paired Effects\n\nSame question ID and seed, edited versus original at each fixed cap. "
            "Exact McNemar and Holm adjustment across five caps separately per outcome; "
            "secondary/diagnostic evidence cannot replace the primary endpoint."
        ),
        nbformat.v4.new_code_cell("display(Markdown(paired_markdown(analysis)))"),
        nbformat.v4.new_markdown_cell("## Provenance, Closure And Short Wrong Answers"),
        nbformat.v4.new_code_cell(
            "expected = plan['edit']['target_edited_sha256']\n"
            "assert all(row['verified_edited_tensor_sha256'] == expected and row['edit_verified_before_generation'] is True for row in records)\n"
            "assert saved['target_hash_checks']['verified_after_restore'] is True\n"
            "print('All edited record hashes match:', json.dumps(expected, indent=2))\n"
            "print('Original target restoration:', saved['target_hash_checks']['verified_after_restore'])\n"
            "display(Markdown(diagnostic_markdown(analysis)))\n"
            "display(Markdown(short_wrong_markdown(analysis)))\n"
            "display(Markdown('### Per-Question Events\\n\\n' + '\\n'.join(\n"
            "    f\"- {row['id']} (seed {row['seed']}): edited EOS={row['edited']['graded']['natural_eos_index']}, \"\n"
            "    f\"closure at 8192={row['edited']['caps']['8192']['thinking_closed']}, \"\n"
            "    f\"correct delivery={row['edited']['caps']['8192']['strict_correct_delivery']}\"\n"
            "    for row in analysis['per_question']\n"
            ")))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Stop And Limitations\n\nOnly 30 TRAIN development items, one seed per item, one pinned 0.8B model. "
            "The baseline was already observed, the article-based closure estimate remains unverified, "
            "and the original and edited runs occurred at different times. "
            "The inherited prefix checks are not fresh edited-prefix validation. "
            "No held-out use, model saves, downloads, installations, extra conditions or further generation are scheduled."
        ),
    ]
    notebook.cells.insert(-1, nbformat.v4.new_code_cell(
        "from p50c_completion_outputs import amendment_markdown\n"
        "display(Markdown(amendment_markdown(analysis, saved)))"
    ))
    nbformat.validate(notebook)
    return notebook
