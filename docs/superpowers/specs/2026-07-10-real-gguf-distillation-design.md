# Real GGUF Distillation Minimal Demo Design

Date: 2026-07-10
Status: approved by user confirmation

## Goal

Build a small local distillation demo that uses the existing `Qwen2.5-7B-Instruct-Q4_K_M.gguf` model in `week-01-llama-cpp/models/` as the real teacher, then publish a Chinese learning post to Lei's blog explaining the principle, workflow, and key implementation code.

## Reader

The reader knows basic LLM terms, but may not yet understand what "teacher logits", "soft labels", KL loss, or student training mean in a distillation pipeline.

## Chosen Approach

Use `llama-server` as the teacher runtime. The script requests short completions with `logprobs`, extracts generated-token top-k log probabilities, and trains a tiny NumPy conditional softmax student from those truncated teacher distributions.

This is a real teacher distillation path because the teacher is the local GGUF Qwen model and the supervision is a soft probability distribution, not only generated text. It is intentionally minimal: the student is a teaching model, not a production chat model.

Rejected alternatives:

- `llama-cpp-python` plus PyTorch student: closer to a neural LM workflow, but requires extra heavy dependencies or compilation.
- Hard-label response imitation: fastest, but loses the central distillation idea of learning from teacher probability mass.
- Full-vocabulary logits distillation: conceptually ideal, but not exposed by the installed `llama-server` path and unnecessary for a minimal local demo.

## Components

1. `tools/distill_tiny_from_gguf.py`
   - Starts or connects to local `llama-server`.
   - Requests teacher completions with top-k `logprobs`.
   - Converts top-k log probabilities into normalized soft labels.
   - Trains a tiny NumPy student table `P(next_token | previous_token)`.
   - Writes `reports/distill/teacher_samples.jsonl`, `student.npz`, and `metrics.json`.

2. `tests/test_distill_tiny_from_gguf.py`
   - Tests top-k logprob normalization.
   - Tests teacher response parsing.
   - Tests that training reduces KL on synthetic soft-label data.

3. `docs/blog/real-gguf-distillation-minimal.md`
   - Source note owned by the learning project.
   - Includes commands, observed metrics, key code excerpts, and boundaries.

4. Lei blog post
   - Published under `src/content/blog/`.
   - Uses public-safe project-relative paths only.
   - Includes key implementation code and explains the flow in plain Chinese.

## Data Flow

```text
prompts
  -> llama-server + GGUF teacher
  -> generated token + top-k logprobs
  -> normalized teacher soft labels
  -> tiny NumPy student training
  -> KL/top-1 metrics
  -> source note
  -> Lei blog post
```

## Testing

Follow TDD for the implementation:

1. Add failing unit tests for distribution normalization and sample parsing.
2. Add a failing unit test showing KL decreases on synthetic data.
3. Implement the minimum production code needed for those tests.
4. Run the real teacher pipeline once with a small prompt set.
5. Run Lei blog checks: `pnpm validate:blog` and `pnpm build`.

## Boundaries

- The demo distills a top-k truncated distribution, not the full vocabulary logits.
- The student is a compact teaching artifact, not a useful general chat model.
- No local absolute paths should appear in the source note or public blog post.
- The existing GGUF model file is read only.
