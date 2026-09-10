"""Small value-free Massive/KIS network preflight for the V221 resume."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "V221_API_PREFLIGHT_STATUS.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
            raise OSError("failed to pin API preflight to CPU 30-31")


def main() -> int:
    pin_bounded()
    import pandas as pd

    import v59_provider_acquisition as provider
    from provider_env_runtime import PROVIDER_ENV_NAMES, import_user_environment_if_missing

    # provider imports the full-run resource module; restore bounded preflight
    # policy before any network or QA work.
    pin_bounded()
    credentials = import_user_environment_if_missing()
    semantics_paths = (
        ROOT / "data" / "BAR_SEMANTICS_MASSIVE.json",
        ROOT / "data" / "BAR_SEMANTICS_KIS.json",
    )
    semantics_before = {str(path.relative_to(ROOT)): sha256(path) for path in semantics_paths}

    massive_client = provider.MassiveClient(min_interval=0)
    massive_start = pd.Timestamp("2026-08-25T13:30:00Z")
    massive_bars, massive_status = massive_client.get(
        "AAPL", massive_start, massive_start + pd.Timedelta(minutes=10)
    )
    massive_qa = provider.bar_qa(massive_bars, "US")

    kis_client = provider.KISClient()
    kis_auth = kis_client.authenticate()
    if kis_auth.get("ok"):
        kis_bars, kis_status = kis_client.get_day("005930", "20260825", max_pages=1)
    else:
        kis_bars, kis_status = pd.DataFrame(), kis_auth
    kis_qa = provider.bar_qa(kis_bars, "KR")

    semantics_after = {str(path.relative_to(ROOT)): sha256(path) for path in semantics_paths}
    semantics_unchanged = semantics_before == semantics_after
    massive_ok = bool(massive_status.get("ok") and massive_qa.get("pass"))
    kis_ok = bool(kis_auth.get("ok") and kis_status.get("ok") and kis_qa.get("pass"))
    report = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_provider_calls": True,
        "values_serialized": False,
        "credentials": credentials,
        "massive": {
            "request_count": 1,
            "ok": massive_ok,
            "rows": int(len(massive_bars)),
            "qa": massive_qa,
            "sample_ticker": "AAPL",
            "sample_interval_utc": [str(massive_start), str(massive_start + pd.Timedelta(minutes=10))],
        },
        "kis": {
            "token_acquisition": bool(kis_auth.get("ok")),
            "request_count": 1 if kis_auth.get("ok") else 0,
            "ok": kis_ok,
            "rows": int(len(kis_bars)),
            "qa": kis_qa,
            "sample_ticker": "005930",
            "sample_date": "20260825",
        },
        "semantics_authority_sha256_before": semantics_before,
        "semantics_authority_sha256_after": semantics_after,
        "semantics_authority_unchanged": semantics_unchanged,
        "status": "PASS" if massive_ok and kis_ok and semantics_unchanged else "FAIL",
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for name in PROVIDER_ENV_NAMES:
        value = os.environ.get(name)
        if value and value in serialized:
            raise RuntimeError("provider value appeared in preflight serialization")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + f".tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "status": report["status"],
                "massive_ok": massive_ok,
                "kis_ok": kis_ok,
                "semantics_authority_unchanged": semantics_unchanged,
                "values_serialized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
