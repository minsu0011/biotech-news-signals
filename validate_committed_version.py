from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = "2"
if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
    if not _kernel32.SetProcessAffinityMask(_kernel32.GetCurrentProcess(), 0xC0000000):
        raise OSError("failed to pin validator to CPU 30-31")

import pandas as pd


ROOT = Path(__file__).resolve().parent
V69_OOF = ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate(version: int, baseline: pd.DataFrame) -> dict:
    output = ROOT / f"output_V{version}"
    commit_path = output / "COMMIT.json"
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    commit = load(commit_path)
    manifest = load(manifest_path)
    manifest_sha = digest(manifest_path)
    assert commit["manifest_sha256"] == manifest_sha
    files = manifest["files"]
    assert len(files) == 18
    for relative, expected in files.items():
        artifact = output / relative
        assert artifact.is_file()
        assert artifact.stat().st_size == expected["bytes"]
        assert digest(artifact) == expected["sha256"]
    required = {"MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"}
    assert required.issubset(files)

    matches = list(output.glob(f"V{version}*SELECTED_OOF.csv.gz"))
    assert len(matches) == 1
    selected_frame = pd.read_csv(matches[0])
    assert len(selected_frame) == 7568
    assert selected_frame.equals(baseline)

    dev = load(output / "DEV_ROBUSTNESS_REPORT.json")
    source = load(output / "SOURCE_TRANSFER_REPORT.json")
    model = load(output / "MODEL_COMPARISON.json")
    evaluation = model["evaluation"]
    selected = dev["selected"]
    candidate = dev["candidate"]
    material = selected["material_gate"]
    assert material == dev["material_gate"] == source["material_gate"]
    assert material["nested_bootstrap"] == evaluation["nested_bootstrap"]
    if "nested_bootstrap" in dev:
        assert dev["nested_bootstrap"] == material["nested_bootstrap"]
    if "nested_bootstrap" in source:
        assert source["nested_bootstrap"] == material["nested_bootstrap"]
    assert dev.get("fallback_is_exact_entire_v69_frame") is True
    fallback = evaluation["fallback"]
    assert fallback.get("activated") is True
    assert fallback.get("exact_entire_frame_verified") is True
    canonical = evaluation["canonical_gate_audit"]
    assert canonical.get("reported_equals_controller_recomputed") is True
    assert canonical["selected"] == selected["research_gate"]
    assert canonical["candidate"] == candidate["research_gate"]
    assert "candidate_native_robustness_bootstrap" in evaluation

    registry = load(ROOT / "research" / "EXPERIMENT_REGISTRY.json")
    matches = [job for job in registry["jobs"] if job.get("assigned_version") == version]
    assert len(matches) == 1
    job = matches[0]
    assert job["status"] == "COMMITTED"
    runner = ROOT / job["runner"]
    assert digest(runner) == job["runner_sha256"]

    return {
        "version": version,
        "status": evaluation["status"],
        "manifest_sha256": manifest_sha,
        "manifest_files_exact": len(files),
        "selected_oof_rows": len(selected_frame),
        "selected_oof_exact_entire_v69": True,
        "material": f'{material["passed"]}/{material["total"]}',
        "selected_canonical": f'{selected["research_gate"]["passed"]}/{selected["research_gate"]["total"]}',
        "candidate_canonical": f'{candidate["research_gate"]["passed"]}/{candidate["research_gate"]["total"]}',
        "robust_survivor": selected["research_gate"]["robust_survivor"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "runner_sha256": job["runner_sha256"],
        "verification": "PASS_INDEPENDENT_EXACT",
    }


parser = argparse.ArgumentParser()
parser.add_argument("versions", nargs="+", type=int)
args = parser.parse_args()
baseline_frame = pd.read_csv(V69_OOF)
print(json.dumps([validate(version, baseline_frame) for version in args.versions], indent=2))
