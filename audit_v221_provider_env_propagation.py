"""Fail-closed, value-free provider environment propagation audit."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from provider_env_runtime import (
    PROVIDER_ENV_NAMES,
    all_present,
    import_user_environment_if_missing,
    presence,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "V221_PROVIDER_ENV_PROPAGATION_AUDIT.json"


def pin_bounded() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 0xC0000000):
            raise OSError("failed to pin propagation audit to CPU 30-31")


def child(layer: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--layer", layer],
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {"layer": layer, "status": "FAIL", "exit_code": completed.returncode}
    return json.loads(completed.stdout)


def orchestrator_layer() -> dict:
    import_user_environment_if_missing()
    current = presence()
    acquisition = child("acquisition_worker")
    passed = all(value == "PRESENT" for value in current.values()) and acquisition.get("status") == "PASS"
    return {
        "layer": "python_orchestrator",
        "variables": current,
        "status": "PASS" if passed else "FAIL",
        "child": acquisition,
    }


def acquisition_layer() -> dict:
    import_user_environment_if_missing()
    current = presence()
    return {
        "layer": "acquisition_worker",
        "variables": current,
        "status": "PASS" if all_present() else "FAIL",
    }


def watchdog_layer() -> dict:
    # Run a real PowerShell child, import missing values from User scope there,
    # then launch the Python-orchestrator audit.  No value is written to stdout.
    python_path = str(Path(sys.executable).resolve()).replace("'", "''")
    script_path = str(Path(__file__).resolve()).replace("'", "''")
    names = ",".join(f"'{name}'" for name in PROVIDER_ENV_NAMES)
    command = f"""
$ErrorActionPreference = 'Stop'
$names = @({names})
foreach ($name in $names) {{
  if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {{
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if (-not [string]::IsNullOrWhiteSpace($value)) {{
      [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }}
  }}
}}
$missing = @($names | Where-Object {{ [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_, 'Process')) }})
if ($missing.Count -ne 0) {{ exit 86 }}
& '{python_path}' '{script_path}' --layer python_orchestrator
exit $LASTEXITCODE
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {
            "layer": "powershell_watchdog",
            "variables": {name: "MISSING_OR_UNPROVEN" for name in PROVIDER_ENV_NAMES},
            "status": "FAIL",
            "child_exit_code": completed.returncode,
        }
    nested = json.loads(completed.stdout)
    return {
        "layer": "powershell_watchdog",
        "variables": {name: "PRESENT" for name in PROVIDER_ENV_NAMES},
        "status": "PASS" if nested.get("status") == "PASS" else "FAIL",
        "child": nested,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=("python_orchestrator", "acquisition_worker"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    pin_bounded()
    if args.layer == "python_orchestrator":
        print(json.dumps(orchestrator_layer(), sort_keys=True))
        return 0
    if args.layer == "acquisition_worker":
        print(json.dumps(acquisition_layer(), sort_keys=True))
        return 0

    before = presence()
    after = import_user_environment_if_missing()
    watchdog = watchdog_layer()
    passed = all(value == "PRESENT" for value in after.values()) and watchdog.get("status") == "PASS"
    report = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "values_serialized": False,
        "current_shell_inherited": before,
        "current_process_after_user_fallback": after,
        "powershell_watchdog": watchdog,
        "status": "PASS" if passed else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"], "values_serialized": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
