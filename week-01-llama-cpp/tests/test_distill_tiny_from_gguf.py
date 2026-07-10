from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "distill_tiny_from_gguf.py"
spec = importlib.util.spec_from_file_location("distill_tiny_from_gguf", MODULE_PATH)
distill = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = distill
spec.loader.exec_module(distill)


class DistillTinyFromGgufTests(unittest.TestCase):
    def test_normalizes_top_logprobs_into_truncated_distribution(self) -> None:
        rows = [
            {"id": 10, "token": "A", "logprob": math.log(0.70)},
            {"id": 20, "token": "B", "logprob": math.log(0.20)},
            {"id": 30, "token": "C", "logprob": math.log(0.05)},
        ]

        probs = distill.normalize_top_logprobs(rows)

        self.assertEqual([item["id"] for item in probs], [10, 20, 30])
        self.assertAlmostEqual(sum(item["prob"] for item in probs), 1.0)
        self.assertGreater(probs[0]["prob"], probs[1]["prob"])
        self.assertGreater(probs[1]["prob"], probs[2]["prob"])

    def test_parses_v1_completion_logprobs_into_training_samples(self) -> None:
        response = {
            "choices": [
                {
                    "text": " AB",
                    "logprobs": {
                        "content": [
                            {
                                "id": 101,
                                "token": " A",
                                "bytes": [32, 65],
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"id": 101, "token": " A", "logprob": -0.2},
                                    {"id": 202, "token": " B", "logprob": -2.2},
                                ],
                            },
                            {
                                "id": 303,
                                "token": "B",
                                "bytes": [66],
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"id": 303, "token": "B", "logprob": -0.1},
                                    {"id": 404, "token": "C", "logprob": -1.7},
                                ],
                            },
                        ]
                    },
                }
            ]
        }

        samples = distill.parse_completion_samples(response, prompt="Explain:", start_context=-1)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].context_id, -1)
        self.assertEqual(samples[0].token_id, 101)
        self.assertEqual(samples[1].context_id, 101)
        self.assertEqual(samples[1].token_id, 303)
        self.assertAlmostEqual(sum(item.prob for item in samples[0].teacher_topk), 1.0)

    def test_training_reduces_kl_on_synthetic_soft_labels(self) -> None:
        samples = [
            distill.TrainingSample(
                prompt="p0",
                position=0,
                context_id=-1,
                token_id=10,
                token_text="A",
                token_bytes=[65],
                teacher_topk=[
                    distill.TeacherProb(id=10, token="A", logprob=math.log(0.8), prob=0.8),
                    distill.TeacherProb(id=20, token="B", logprob=math.log(0.2), prob=0.2),
                ],
            ),
            distill.TrainingSample(
                prompt="p1",
                position=0,
                context_id=-1,
                token_id=10,
                token_text="A",
                token_bytes=[65],
                teacher_topk=[
                    distill.TeacherProb(id=10, token="A", logprob=math.log(0.8), prob=0.8),
                    distill.TeacherProb(id=20, token="B", logprob=math.log(0.2), prob=0.2),
                ],
            ),
        ]

        model, metrics = distill.train_student(samples, epochs=120, lr=0.5)

        self.assertLess(metrics["final_kl"], metrics["initial_kl"])
        self.assertGreater(model.probs_for_context(-1)[10], model.probs_for_context(-1)[20])


if __name__ == "__main__":
    unittest.main()
