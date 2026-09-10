"""Regression tests for V71+ hypothesis-result projection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import autonomous_v71plus as loop


class HypothesisRegistrySyncTests(unittest.TestCase):
    def test_status_falls_back_to_model_comparison_evaluation(self) -> None:
        """Legacy DEV reports without a top-level status must not project null."""
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            research = root / "research"
            output = root / "output_V211"
            research.mkdir()
            output.mkdir()

            hypothesis_path = research / "HYPOTHESIS_REGISTRY.json"
            experiment_path = research / "EXPERIMENT_REGISTRY.json"
            hypothesis_path.write_text(
                json.dumps({"hypotheses": [{"id": "H", "status": "READY"}]}),
                encoding="utf-8",
            )
            experiment_path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id": "H",
                                "status": "COMMITTED",
                                "assigned_version": 211,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (output / "DEV_ROBUSTNESS_REPORT.json").write_text(
                json.dumps(
                    {
                        "selected": {
                            "metrics": {},
                            "robustness": {},
                            "research_gate": {},
                            "material_gate": {"material_pass": False},
                        },
                        "fallback_is_exact_entire_v69_frame": True,
                    }
                ),
                encoding="utf-8",
            )
            expected = "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
            (output / "MODEL_COMPARISON.json").write_text(
                json.dumps({"evaluation": {"status": expected}}),
                encoding="utf-8",
            )

            with (
                patch.object(loop, "ROOT", root),
                patch.object(loop, "HYPOTHESIS_REGISTRY_PATH", hypothesis_path),
                patch.object(loop, "REGISTRY_PATH", experiment_path),
            ):
                loop.sync_hypothesis_registry()

            payload = json.loads(hypothesis_path.read_text(encoding="utf-8"))
            result = payload["hypotheses"][0]["last_result"]
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["material_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
