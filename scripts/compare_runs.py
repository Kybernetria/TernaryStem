#!/usr/bin/env python3
"""Compare development diagnostics across experiment JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ternarystem.evaluation.comparison import summarize_record

parser = argparse.ArgumentParser(
    description="Compare model/equal-share, direct/mask, and matched FP32/QAT diagnostics"
)
parser.add_argument("records", nargs="+", type=Path, help="experiment.json files")
parser.add_argument("--output", type=Path, help="optional JSON output path")
args = parser.parse_args()

summaries = [
    summarize_record(json.loads(path.read_text(encoding="utf-8")), str(path))
    for path in args.records
]
payload = {
    "metric_warning": "All values are development global_sdr diagnostics, never BSSEval.",
    "runs": summaries,
}
text = json.dumps(payload, indent=2) + "\n"
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
print(text, end="")
