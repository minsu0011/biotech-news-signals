from __future__ import annotations

import json
from pathlib import Path

import historical_news_10h_orchestrator as module


def test_allowlist_rejects_model_and_v224_commands() -> None:
    module.assert_allowed(["historical_news_article_backfill.py", "--resume"])
    for arguments in (
        ["train_model.py"],
        ["historical_news_build_freeze.py", "--model"],
        ["historical_news_build_freeze.py", "--v224"],
    ):
        try:
            module.assert_allowed(arguments)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsafe command accepted: {arguments}")


def test_work_signature_ignores_state_json_but_tracks_data(tmp_path: Path) -> None:
    old_root = module.ROOT
    try:
        module.ROOT = tmp_path
        data = tmp_path / "data"
        data.mkdir()
        before = module.work_signature((data,))
        (data / "status.json").write_text(json.dumps({"updated": 1}), encoding="utf-8")
        assert module.work_signature((data,)) == before
        (data / "articles.jsonl").write_text("{}\n", encoding="utf-8")
        assert module.work_signature((data,)) != before
    finally:
        module.ROOT = old_root


def test_default_state_is_explicitly_data_only() -> None:
    state = module.default_state(36000)
    assert state["target_productive_seconds"] == 36000
    assert state["model_building_forbidden"] is True
    assert state["model_building_performed"] is False
    assert state["research_seal_touched"] is False
    assert state["final_meta_touched"] is False
    assert state["v224_touched"] is False
