#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np


START_CONTEXT = -1
DEFAULT_PROMPTS = [
    "Explain knowledge distillation in one concise sentence.",
    "Why can a small student model learn from a larger teacher?",
    "Define soft labels in machine learning.",
    "Give one practical limitation of model distillation.",
    "Describe KL divergence in plain language.",
    "What does a teacher model provide during distillation?",
]


@dataclass(frozen=True)
class TeacherProb:
    id: int
    token: str
    logprob: float
    prob: float


@dataclass(frozen=True)
class TrainingSample:
    prompt: str
    position: int
    context_id: int
    token_id: int
    token_text: str
    token_bytes: list[int]
    teacher_topk: list[TeacherProb]


@dataclass
class TinyStudent:
    logits: np.ndarray
    context_ids: list[int]
    vocab_ids: list[int]

    def __post_init__(self) -> None:
        self._context_to_row = {value: idx for idx, value in enumerate(self.context_ids)}
        self._vocab_to_col = {value: idx for idx, value in enumerate(self.vocab_ids)}

    def probs_for_context(self, context_id: int) -> dict[int, float]:
        row = self._context_to_row[context_id]
        probs = softmax(self.logits[row])
        return {token_id: float(probs[col]) for col, token_id in enumerate(self.vocab_ids)}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def normalize_top_logprobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("top_logprobs is empty")

    logprobs = np.array([float(row["logprob"]) for row in rows], dtype=np.float64)
    probs = softmax(logprobs)
    normalized: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs):
        normalized.append(
            {
                "id": int(row["id"]),
                "token": str(row.get("token", "")),
                "logprob": float(row["logprob"]),
                "prob": float(prob),
            }
        )
    return normalized


def parse_completion_samples(
    payload: dict[str, Any],
    prompt: str,
    start_context: int = START_CONTEXT,
) -> list[TrainingSample]:
    content = _completion_content(payload)
    samples: list[TrainingSample] = []
    context_id = start_context

    for position, item in enumerate(content):
        top_rows = item.get("top_logprobs") or []
        if not top_rows and "logprob" in item:
            top_rows = [item]
        teacher_topk = [
            TeacherProb(
                id=int(row["id"]),
                token=str(row.get("token", "")),
                logprob=float(row["logprob"]),
                prob=float(row["prob"]),
            )
            for row in normalize_top_logprobs(top_rows)
        ]
        token_id = int(item["id"])
        samples.append(
            TrainingSample(
                prompt=prompt,
                position=position,
                context_id=context_id,
                token_id=token_id,
                token_text=str(item.get("token", "")),
                token_bytes=[int(value) for value in item.get("bytes", [])],
                teacher_topk=teacher_topk,
            )
        )
        context_id = token_id

    return samples


def _completion_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "choices" in payload:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("completion response has no choices")
        logprobs = choices[0].get("logprobs") or {}
        content = logprobs.get("content")
        if content is None:
            raise ValueError("completion response has no logprobs.content")
        return list(content)

    content = payload.get("completion_probabilities")
    if content is None:
        raise ValueError("completion response has no completion probabilities")
    return list(content)


def train_student(
    samples: list[TrainingSample],
    epochs: int = 200,
    lr: float = 0.4,
) -> tuple[TinyStudent, dict[str, float]]:
    if not samples:
        raise ValueError("cannot train without samples")

    context_ids = sorted({sample.context_id for sample in samples})
    vocab_ids = sorted({prob.id for sample in samples for prob in sample.teacher_topk})
    context_to_row = {value: idx for idx, value in enumerate(context_ids)}
    vocab_to_col = {value: idx for idx, value in enumerate(vocab_ids)}

    logits = np.zeros((len(context_ids), len(vocab_ids)), dtype=np.float64)
    initial = evaluate(logits, samples, context_to_row, vocab_to_col)

    for _ in range(epochs):
        grad = np.zeros_like(logits)
        for sample in samples:
            row = context_to_row[sample.context_id]
            target = target_vector(sample, vocab_to_col, len(vocab_ids))
            grad[row] += softmax(logits[row]) - target
        grad /= len(samples)
        logits -= lr * grad

    final = evaluate(logits, samples, context_to_row, vocab_to_col)
    model = TinyStudent(logits=logits, context_ids=context_ids, vocab_ids=vocab_ids)
    metrics = {
        "sample_count": float(len(samples)),
        "context_count": float(len(context_ids)),
        "vocab_size": float(len(vocab_ids)),
        "initial_kl": initial["kl"],
        "final_kl": final["kl"],
        "initial_cross_entropy": initial["cross_entropy"],
        "final_cross_entropy": final["cross_entropy"],
        "top1_accuracy": final["top1_accuracy"],
    }
    return model, metrics


def target_vector(
    sample: TrainingSample,
    vocab_to_col: dict[int, int],
    vocab_size: int,
) -> np.ndarray:
    target = np.zeros(vocab_size, dtype=np.float64)
    for item in sample.teacher_topk:
        target[vocab_to_col[item.id]] = item.prob
    total = np.sum(target)
    if total <= 0:
        raise ValueError("target distribution has no mass")
    return target / total


def evaluate(
    logits: np.ndarray,
    samples: list[TrainingSample],
    context_to_row: dict[int, int],
    vocab_to_col: dict[int, int],
) -> dict[str, float]:
    eps = 1e-12
    total_cross_entropy = 0.0
    total_entropy = 0.0
    top1_hits = 0

    for sample in samples:
        row = context_to_row[sample.context_id]
        target = target_vector(sample, vocab_to_col, logits.shape[1])
        pred = softmax(logits[row])
        total_cross_entropy += float(-np.sum(target * np.log(np.maximum(pred, eps))))
        total_entropy += float(-np.sum(target * np.log(np.maximum(target, eps))))

        teacher_top1 = max(sample.teacher_topk, key=lambda item: item.prob).id
        student_top1 = max(
            ((token_id, pred[col]) for token_id, col in vocab_to_col.items()),
            key=lambda item: item[1],
        )[0]
        top1_hits += int(teacher_top1 == student_top1)

    n = len(samples)
    cross_entropy = total_cross_entropy / n
    entropy = total_entropy / n
    return {
        "cross_entropy": cross_entropy,
        "entropy": entropy,
        "kl": cross_entropy - entropy,
        "top1_accuracy": top1_hits / n,
    }


def collect_teacher_samples(
    prompts: list[str],
    server_url: str,
    n_predict: int,
    top_k: int,
    temperature: float,
    seed: int,
) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for idx, prompt in enumerate(prompts):
        payload = request_completion(
            server_url=server_url,
            prompt=prompt,
            n_predict=n_predict,
            top_k=top_k,
            temperature=temperature,
            seed=seed + idx,
        )
        samples.extend(parse_completion_samples(payload, prompt=prompt))
    return samples


def request_completion(
    server_url: str,
    prompt: str,
    n_predict: int,
    top_k: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": "local-gguf-teacher",
        "prompt": prompt,
        "max_tokens": n_predict,
        "temperature": temperature,
        "top_k": max(top_k, 1),
        "logprobs": max(top_k, 1),
        "seed": seed,
    }
    return post_json(f"{server_url.rstrip('/')}/v1/completions", payload)


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(server_url: str, timeout_seconds: float = 180.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = get_json(f"{server_url.rstrip('/')}/health", timeout=2.0)
            if health.get("status") == "ok":
                return
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"llama-server did not become healthy: {last_error}")


def start_llama_server(
    model_path: Path,
    host: str,
    port: int,
    ctx_size: int,
    gpu_layers: str,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    cmd = [
        "llama-server",
        "-m",
        str(model_path),
        "-c",
        str(ctx_size),
        "-ngl",
        gpu_layers,
        "--host",
        host,
        "--port",
        str(port),
        "--no-ui",
    ]
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in prompts if line and not line.startswith("#")]


def write_samples(path: Path, samples: list[TrainingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample_to_dict(sample), ensure_ascii=False) + "\n")


def sample_to_dict(sample: TrainingSample) -> dict[str, Any]:
    return {
        "prompt": sample.prompt,
        "position": sample.position,
        "context_id": sample.context_id,
        "token_id": sample.token_id,
        "token_text": sample.token_text,
        "token_bytes": sample.token_bytes,
        "teacher_topk": [prob.__dict__ for prob in sample.teacher_topk],
    }


def write_student(path: Path, model: TinyStudent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        logits=model.logits,
        context_ids=np.array(model.context_ids, dtype=np.int64),
        vocab_ids=np.array(model.vocab_ids, dtype=np.int64),
    )


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/distill"))
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--server-url", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--n-predict", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.n_predict < 1:
        parser.error("--n-predict must be >= 1")

    model_path = args.model
    if not model_path.exists():
        parser.error(f"model not found: {model_path}")

    out_dir = args.out_dir
    server_url = args.server_url or f"http://{args.host}:{args.port}"
    proc: subprocess.Popen[bytes] | None = None

    try:
        if not args.server_url:
            proc = start_llama_server(
                model_path=model_path,
                host=args.host,
                port=args.port,
                ctx_size=args.ctx_size,
                gpu_layers=args.gpu_layers,
                log_path=out_dir / "llama-server.log",
            )
        wait_for_server(server_url)

        prompts = load_prompts(args.prompts_file)
        samples = collect_teacher_samples(
            prompts=prompts,
            server_url=server_url,
            n_predict=args.n_predict,
            top_k=args.top_k,
            temperature=args.temperature,
            seed=args.seed,
        )
        student, metrics = train_student(samples, epochs=args.epochs, lr=args.lr)

        write_samples(out_dir / "teacher_samples.jsonl", samples)
        write_student(out_dir / "student.npz", student)
        write_metrics(out_dir / "metrics.json", metrics)

        summary = {
            "samples": len(samples),
            "contexts": int(metrics["context_count"]),
            "vocab_size": int(metrics["vocab_size"]),
            "initial_kl": metrics["initial_kl"],
            "final_kl": metrics["final_kl"],
            "top1_accuracy": metrics["top1_accuracy"],
            "out_dir": str(out_dir),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
