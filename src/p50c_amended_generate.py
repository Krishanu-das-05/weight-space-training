"""Identical sampling call, followed by durable raw capture before response checks."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import time

from budget_sweep_generate import GeneratedPresencePenalty, LogitsProcessorList, build_probe_content, torch
from p50c_completion_generate import VerifiedEditedGenerator, verify_edited_targets
from weight_edit import tensor_sha256


def generate_raw_record(model, tokenizer, plan, problem, job, attempt, plan_hash, store, actual):
    thinking = job["condition"] == "thinking_on"
    content = build_probe_content(problem, "step_by_step")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )
    reference = next(item for item in plan["baseline"]["selection"] if item["id"] == job["problem_id"])
    if content != reference["user_message"] or prompt != reference["rendered_prompt"]:
        raise ValueError("The amended prompt differs from its fixed original counterpart.")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    sampling = plan["sampling"]
    capture = store.capture(attempt)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
        torch.manual_seed(job["seed"])
        torch.cuda.manual_seed_all(job["seed"])
        output = model.generate(
            **inputs, max_new_tokens=job["max_new_tokens"], do_sample=True,
            temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
            repetition_penalty=1.0, num_beams=1, use_cache=True, cache_implementation="dynamic",
            min_new_tokens=0, eos_token_id=248044, pad_token_id=248044, forced_eos_token_id=None,
            logits_processor=LogitsProcessorList([GeneratedPresencePenalty(prompt_length, 1.5)]),
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = output[0, prompt_length:].tolist()
    capture.tokens({
        **job, "attempt_id": attempt, "plan_sha256": plan_hash,
        "amendment_sha256": store.amendment_hash, "expected": problem["answer"], "thinking": thinking,
        "verified_edited_tensor_sha256": actual, "edit_verified_before_generation": True,
        "raw_captured_before_response_validation": True, "seconds": seconds,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "user_message": content, "rendered_prompt": prompt, "prompt_tokens": prompt_length,
    }, tokens)
    try:
        text = tokenizer.decode(tokens, skip_special_tokens=False)
    except Exception as exception:
        capture.decode_error(exception)
        raise
    capture.text(text)
    return store.build_record(attempt)


class RawFirstEditedGenerator(VerifiedEditedGenerator):
    def __init__(self, model, tokenizer, plan, plan_hash, store, *, generator=generate_raw_record, hasher=tensor_sha256):
        super().__init__(model, tokenizer, plan, plan_hash, hasher=hasher)
        self.store, self.raw_generator = store, generator

    def __call__(self, problem, job, attempt):
        self.assert_ready()
        try:
            actual = verify_edited_targets(self.model, self.plan, self.hasher)
        except BaseException:
            self._verified = False
            raise
        return self.raw_generator(
            self.model, self.tokenizer, self.plan, problem, job, attempt, self.plan_hash, self.store, actual,
        )
