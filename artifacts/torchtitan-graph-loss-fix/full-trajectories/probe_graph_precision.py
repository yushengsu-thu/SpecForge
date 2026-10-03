"""Correctness-only ablation wrapper; elapsed times are not benchmark results."""

import json
import os
import runpy
import sys
from pathlib import Path

import torch
import torch._inductor.config as ic

preserve = os.environ.get("PROBE_PRESERVE_CASTS", "0") == "1"
ic.emulate_precision_casts = preserve
ic.eager_numerics.division_rounding = os.environ.get("PROBE_DIVISION_ROUNDING", "1") == "1"
driver = Path(os.environ["PROBE_DRIVER"])
sys.path.insert(0, str(driver.parent))
runpy.run_path(str(driver), run_name="__main__")
if int(os.environ.get("RANK", "0")) == 0:
    path = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(path.read_text())
    result["correctness_probe"] = {
        "emulate_precision_casts": preserve,
        "division_rounding": os.environ.get("PROBE_DIVISION_ROUNDING", "1") == "1",
        "timings_are_not_validated_performance": True,
        "wrapper": str(Path(__file__).resolve()),
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
