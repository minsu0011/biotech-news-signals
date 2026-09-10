"""Transfer the previously declared V64 causal adapter to the frozen V221 epoch.

This is a dry-run diagnostic only.  It updates only the immutable input hash
and row-count bindings, preserves the V64 architecture and predeclared grids,
and writes no model/output/cache artifacts.
"""
from __future__ import annotations

import sys

import experiment_v64_new_data_retrain as experiment


experiment.EXPECTED_SHA256[experiment.EXTENSION_PATH.name] = experiment.sha256(
    experiment.EXTENSION_PATH
)
experiment.EXPECTED_EXTENSION_ROWS = 430
experiment.EXTENSION_CLUSTER_POLICY = "LABEL_BLIND_FIRST_EVENT_PER_CLUSTER"
experiment.ALLOW_UNSCORABLE_PREFIX_QUARANTINE = True


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--dry-run"]
    experiment.main()
