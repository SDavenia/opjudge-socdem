"""
LLM judges, as in opinion_judge_stability_bias: local models (REGISTRY), run with vLLM (default)
or transformers, and models served by OpenRouter (any other id, an OpenRouter slug such as
openai/gpt-5.4-nano, with OPENROUTER_API_KEY in the environment or a .env file). All decode
greedily (temperature 0).

run_judge(judge_model_id, prompts, ...) returns one answer per prompt, in order, and calls
on_progress(answers, n_done) as answers come in (pending ones are None).
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class ModelSpec:
    name: str
    model_class: str = "AutoModelForCausalLM"
    proc_class: str = "AutoTokenizer"
    is_vlm: bool = False
    slow_tokenizer: bool = False
    batch_size_divide: int = 1  # transformers
    num_params: float | None = None


REGISTRY = {
    "gemma3": ModelSpec("google/gemma-3-27b-it", "Gemma3ForConditionalGeneration", "AutoProcessor",
                        is_vlm=True, num_params=27.0),
    "llama-3.3-70b": ModelSpec("meta-llama/Llama-3.3-70B-Instruct", batch_size_divide=2, num_params=70.0),
    "qwen2.5-72b": ModelSpec("Qwen/Qwen2.5-72B-Instruct", batch_size_divide=2, num_params=72.0),
    # The fast Mistral tokenizer mis-splits some batched inputs, so the slow one is used.
    "mistral-24b": ModelSpec("mistralai/Mistral-Small-24B-Instruct-2501", slow_tokenizer=True, num_params=24.0),
    "qwen2.5-05b": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", num_params=0.5),
    "qwen2.5-1b": ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", num_params=1.5),
    "qwen2.5-3b": ModelSpec("Qwen/Qwen2.5-3B-Instruct", num_params=3.0),
    "qwen2.5-7b": ModelSpec("Qwen/Qwen2.5-7B-Instruct", num_params=7.0),
    "qwen2.5-14b": ModelSpec("Qwen/Qwen2.5-14B-Instruct", num_params=14.0),
    "qwen2.5-32b": ModelSpec("Qwen/Qwen2.5-32B-Instruct", batch_size_divide=2, num_params=32.0),
    # Small enough for a CPU smoke test of the pipeline, not a judge of the study.
    "llama-3.2-1b": ModelSpec("meta-llama/Llama-3.2-1B-Instruct", num_params=1.2),
}


def is_local(judge_model_id):
    return judge_model_id in REGISTRY


def run_judge(judge_model_id, prompts, *, max_new_tokens, group_size, backend="vllm", batch_size=32,
              tensor_parallel_size=None, pipeline_parallel_size=None, max_model_len=4096, gpu_memory_utilization=0.9,
              chunk_size=4096, max_concurrency=8, max_retries=5, on_progress=None):
    if is_local(judge_model_id):
        spec = REGISTRY[judge_model_id]
        if backend == "vllm":
            return run_vllm(spec, prompts, max_new_tokens=max_new_tokens, group_size=group_size,
                            tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
                            max_model_len=max_model_len,
                            gpu_memory_utilization=gpu_memory_utilization, chunk_size=chunk_size,
                            on_progress=on_progress)
        return run_local(spec, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
                         group_size=group_size, on_progress=on_progress)
    return run_openrouter(judge_model_id, prompts, max_new_tokens=max_new_tokens, max_concurrency=max_concurrency,
                          max_retries=max_retries, on_progress=on_progress)


# --- Local judges (vLLM) -----------------------------------------------------------------------

def run_vllm(spec, prompts, *, max_new_tokens, group_size=1, tensor_parallel_size=None, pipeline_parallel_size=None,
             max_model_len=4096, gpu_memory_utilization=0.9, chunk_size=4096, on_progress=None):
    """Greedy answers to prompts, with the model's chat template. vLLM batches continuously, so
    the prompts are sent in chunks (a multiple of group_size, so that the versions of a pair go
    in the same chunk) only to report progress and checkpoint between them.

    The model is split over the GPUs by pipeline rather than tensor parallelism by default: the
    cluster's GPUs have no P2P link, so the per-layer all-reduces of tensor parallelism go through
    host memory and are slow."""
    import torch

    # FlashInfer's sampler is JIT-compiled with the system nvcc, which on the cluster (12.5) is too old
    # for it; greedy decoding does not need it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    tp = tensor_parallel_size or 1
    pp = pipeline_parallel_size or max(1, torch.cuda.device_count() // tp)
    kwargs = {}
    if spec.slow_tokenizer:
        # vLLM no longer has a "slow" tokenizer mode (and refuses "hf" for Mistral repos with
        # transformers < 5.15); Mistral's own mistral_common tokenizer avoids the fast HF one.
        kwargs["tokenizer_mode"] = "mistral"
    if spec.is_vlm:
        kwargs["limit_mm_per_prompt"] = {"image": 0}  # text only: no memory for the vision encoder
    llm = LLM(model=spec.name, dtype="bfloat16", tensor_parallel_size=tp, pipeline_parallel_size=pp,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, **kwargs)
    # transformers' generate applies the repetition_penalty of the model's generation_config even with
    # do_sample=False (Qwen2.5: 1.05 to 1.1, over the prompt tokens too); it is kept, so that the two
    # backends give the same answers.
    repetition_penalty = llm.get_default_sampling_params().repetition_penalty
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, repetition_penalty=repetition_penalty)

    size = max(1, chunk_size)
    if size >= group_size:
        size -= size % group_size
    print(f"Judge {spec.name} (vLLM, tensor parallel {tp}, pipeline parallel {pp}) | {len(prompts)} prompts | "
          f"chunks of {size} | repetition penalty {repetition_penalty}")

    answers = [None] * len(prompts)
    with tqdm(total=len(prompts), desc=f"Scoring ({spec.name})") as bar:
        for start in range(0, len(prompts), size):
            chunk = prompts[start:start + size]
            outputs = llm.chat([[{"role": "user", "content": p}] for p in chunk], params, use_tqdm=False)
            answers[start:start + len(chunk)] = [o.outputs[0].text.strip() for o in outputs]
            bar.update(len(chunk))
            if start == 0:
                for i in range(min(3, len(chunk))):
                    print(f"[sample {i}] {answers[i]!r}")
            if on_progress is not None:
                on_progress(answers, start + len(chunk))
    return answers


# --- Local judges (transformers) ---------------------------------------------------------------

def tokenizer_of(proc, spec):
    return proc.tokenizer if spec.is_vlm else proc


def load_model(spec):
    import torch
    import transformers

    token = os.getenv("HF_TOKEN")
    proc = None
    if spec.slow_tokenizer:
        try:
            proc = transformers.AutoTokenizer.from_pretrained(spec.name, token=token, use_fast=False)
        except Exception as e:
            print(f"Could not load the slow tokenizer of {spec.name}, using the fast one: {e}")
    if proc is None:
        proc = getattr(transformers, spec.proc_class).from_pretrained(spec.name, token=token)

    model = getattr(transformers, spec.model_class).from_pretrained(
        spec.name, token=token, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    tok = tokenizer_of(proc, spec)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, proc


def chat_text(tok, spec, prompt):
    content = [{"type": "text", "text": prompt}] if spec.is_vlm else prompt
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def generate_batch(model, proc, spec, texts, max_new_tokens):
    import torch

    tok = tokenizer_of(proc, spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if spec.is_vlm:
        inputs = proc(text=texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    else:
        inputs = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                    do_sample=False)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return [text.strip() for text in tok.batch_decode(new_tokens, skip_special_tokens=True)]


def run_local(spec, prompts, *, max_new_tokens, batch_size, group_size=1, on_progress=None):
    """Greedy answers to prompts. The batch size (divided by the model's batch_size_divide) is
    rounded down to a multiple of group_size, so that each run of group_size consecutive prompts
    (the versions of one pair that differ only in the sociodemographic) shares a batch and its
    padding, and their scores cannot differ because of the batching."""
    import torch

    model, proc = load_model(spec)
    tok = tokenizer_of(proc, spec)
    texts = [chat_text(tok, spec, p) for p in prompts]
    size = max(1, batch_size // spec.batch_size_divide)
    if size >= group_size:
        size -= size % group_size
    print(f"Judge {spec.name} | {len(texts)} prompts | batch size {size}")

    answers = [None] * len(texts)
    for start in tqdm(range(0, len(texts), size), desc=f"Scoring ({spec.name})"):
        answers[start:start + size] = generate_batch(model, proc, spec, texts[start:start + size], max_new_tokens)
        if on_progress is not None:
            on_progress(answers, min(start + size, len(texts)))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: max allocated {torch.cuda.max_memory_allocated(i) / 1024 ** 3:.2f} GB")
    return answers


# --- OpenRouter judges -------------------------------------------------------------------------

def run_openrouter(judge_model_id, prompts, *, max_new_tokens, max_concurrency, max_retries, on_progress=None):
    from dotenv import load_dotenv
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to a .env file).")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    transient_errors = (RateLimitError, APITimeoutError, APIError)

    def call(prompt):
        """One chat completion, with exponential backoff on transient errors ("" if it keeps failing)."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(model=judge_model_id, max_tokens=max_new_tokens, temperature=0.0,
                                                      messages=[{"role": "user", "content": prompt}])
                content = (resp.choices[0].message.content or "").strip()
                if not content and attempt == 0:
                    print(f"[warn] empty answer (finish reason: {resp.choices[0].finish_reason})")
                return content
            except transient_errors as e:
                if attempt == max_retries - 1:
                    print(f"[error] giving up after {max_retries} attempts: {e}")
                    return ""
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 30)
        return ""

    print(f"Judge {judge_model_id} (OpenRouter) | {len(prompts)} requests | {max_concurrency} in flight")
    answers = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {pool.submit(call, prompt): i for i, prompt in enumerate(prompts)}
        for n_done, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                             desc=f"Scoring ({judge_model_id})"), start=1):
            i = futures[future]
            try:
                answers[i] = future.result()
            except Exception as e:
                print(f"[error] prompt {i} failed: {e}")
                answers[i] = ""
            if n_done <= 3:
                print(f"[sample {i}] {answers[i]!r}")
            if on_progress is not None:
                on_progress(answers, n_done)
    return answers
