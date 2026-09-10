"""Full-machine runtime policy for BIO_NEWS_30M V24 and later.

Use every logical processor and expose the primary NVIDIA GPU.  Archived
certificates retain their own sealed source snapshots and are unaffected.
"""
from __future__ import annotations

import os
import sys


CPU_IDS = tuple(range(os.cpu_count() or 1))
THREAD_COUNT = len(CPU_IDS)
GPU_ENABLED = True
CATBOOST_TASK_TYPE = "GPU"
CATBOOST_DEVICES = "0"


def _set_environment() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(THREAD_COUNT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ.pop("HIP_VISIBLE_DEVICES",None)
    os.environ.pop("ROCR_VISIBLE_DEVICES",None)


def _set_affinity() -> None:
    if sys.platform == "win32":
        import ctypes

        mask = sum(1 << cpu_id for cpu_id in CPU_IDS)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(CPU_IDS))
        return
    raise RuntimeError("strict CPU affinity is unsupported on this platform")


def configure() -> None:
    _set_environment()
    _set_affinity()


def status() -> dict[str, object]:
    if sys.platform == "win32":
        import ctypes

        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        ok = kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        active = [i for i in range(64) if process_mask.value & (1 << i)]
    else:
        active = sorted(os.sched_getaffinity(0))
    return {
        "cpu_ids": active,
        "thread_count": THREAD_COUNT,
        "gpu_enabled": GPU_ENABLED,
        "catboost_task_type": CATBOOST_TASK_TYPE,
        "catboost_devices": CATBOOST_DEVICES,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }


configure()
