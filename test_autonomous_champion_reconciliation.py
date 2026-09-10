"""Regression tests for crash-safe champion reconciliation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import autonomous_v37plus as loop


class ChampionReconciliationTests(unittest.TestCase):
    def test_new_noop_commit_cannot_displace_older_tied_champion(self) -> None:
        """Reconciliation must rediscover V69 when called with only new V72."""
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for version, rank, hypothesis in (
                (58, 1, "BOOTSTRAP"),
                (69, 9, "V69_CHAMPION"),
                (72, 9, "V72_EXACT_NOOP"),
            ):
                directory = root / f"output_V{version}"
                directory.mkdir()
                (directory / "DEV_ROBUSTNESS_REPORT.json").write_text(
                    json.dumps({"rank": rank, "hypothesis": hypothesis}),
                    encoding="utf-8",
                )

            controller = loop.Controller.__new__(loop.Controller)
            controller.state = {}

            def valid_commit(directory: Path):
                return {"valid": True} if directory.name in {"output_V69", "output_V72"} else None

            def score(report: dict):
                return (int(report["rank"]),)

            champion_path = root / "research" / "CURRENT_CHAMPION.json"
            with (
                patch.object(loop, "ROOT", root),
                patch.object(loop, "CHAMPION_PATH", champion_path),
                patch.object(controller, "valid_commit", side_effect=valid_commit),
                patch.object(loop.Controller, "champion_score", side_effect=score),
            ):
                controller.reconcile_champion([(72, {"valid": True})])

            payload = json.loads(champion_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 69)
            self.assertEqual(payload["hypothesis"], "V69_CHAMPION")
            self.assertEqual(controller.state["current_champion"]["version"], 69)


if __name__ == "__main__":
    unittest.main(verbosity=2)
