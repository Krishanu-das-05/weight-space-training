"""Strict deterministic GPU generation; no clock-based stopping or downloads."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import hashlib
import importlib.metadata
import platform
import subprocess
import time

import torch
from transformers import AutoTokenizer, LogitsProcessorList, Qwen3_5ForConditionalGeneration

from budget_sweep_io import ROOT, project_path, sha256
from budget_sweep_scoring import grade_generation
from math_probe import GeneratedPresencePenalty, build_probe_content


def configure_determinism():
    before_cuda = not torch.cuda.is_initialized()
    if not before_cuda or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic setup requires the approved environment before CUDA initialization.")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError("Strict determinism is not enabled.")
    except Exception:
        raise RuntimeError("Could not enable strict deterministic execution; stopping without a fallback.") from None
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "set_before_cuda_initialization": before_cuda,
        "deterministic_algorithms": True, "deterministic_warn_only": False,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "wall_time_limit": None,
    }


def check_hardware(plan):
    execution = plan["execution"]
    for actual, expected in (
        (platform.python_version(), execution["python"]), (str(torch.__version__), execution["torch"]),
        (importlib.metadata.version("transformers"), execution["transformers"]),
        (torch.version.cuda, execution["cuda_runtime"]), (platform.platform(), plan["hardware"]["os"]),
    ):
        if actual != expected:
            raise RuntimeError("Approved software/OS configuration changed.")
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.strip()
    if (
        properties.name != plan["hardware"]["gpu"]
        or properties.total_memory != plan["hardware"]["gpu_memory_bytes"]
        or driver != plan["hardware"]["nvidia_driver"]
    ):
        raise RuntimeError("Approved GPU configuration changed.")


def load_model(plan):
    check_hardware(plan)
    weights = project_path(plan["model"]["weights_file"])
    if sha256(weights) != plan["model"]["checkpoint_sha256"]:
        raise RuntimeError("The original checkpoint hash changed.")
    for relative, digest in plan["model"]["cached_config_sha256"].items():
        path = project_path(relative)
        if path.parent != weights.parent or sha256(path) != digest:
            raise RuntimeError("A cached model configuration changed.")
    tokenizer = AutoTokenizer.from_pretrained(weights.parent, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        weights.parent, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).to("cuda:0").eval()
    if model.generation_config.eos_token_id != 248044 or model.generation_config.max_time is not None:
        raise RuntimeError("Unexpected EOS or hidden generation time limit.")
    return model, tokenizer


def verify_targets(model, plan):
    actual = {}
    for name, expected in plan["model"]["target_base_sha256"].items():
        weight = model.get_parameter(name)
        if weight.dtype != torch.bfloat16:
            raise RuntimeError("Original BF16 target dtype changed.")
        raw = weight.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
        actual[name] = hashlib.sha256(raw).hexdigest()
        if actual[name] != expected:
            raise RuntimeError("Original target tensor hash changed.")
    return actual


def generate_record(model, tokenizer, plan, problem, job, attempt, plan_hash):
    thinking = job["condition"] == "thinking_on"
    content = build_probe_content(problem, "step_by_step")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    sampling = plan["sampling"]
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
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    graded = grade_generation(
        text, tokens, problem["answer"], thinking=thinking, eos_ids={248044},
        terminal_tokens=plan["stopping"]["terminal_tokens_for_grading"],
    )
    return {
        **job, "attempt_id": attempt, "plan_sha256": plan_hash,
        "expected": problem["answer"], "thinking": thinking, "token_ids": tokens, "text": text,
        "graded": graded, "seconds": seconds,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "token_248046_count": tokens.count(248046), "contains_token_248046": 248046 in tokens,
        "user_message": content, "rendered_prompt": prompt, "prompt_tokens": prompt_length,
    }
