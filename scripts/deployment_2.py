#!/usr/bin/env python3
"""Offline readiness and authorized local-CUDA evidence for deployment #2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ternarystem.deployment.readiness import cuda_probe, freeze_source, readiness_matrix
from ternarystem.training import atomic_json_save


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--allowlist", type=Path, default=Path("configs/deployment_2/allowlist.yaml")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--cuda-evidence", type=Path)
    freeze = subparsers.add_parser("freeze-source")
    freeze.add_argument("--output", type=Path)
    probe = subparsers.add_parser("cuda-probe")
    probe.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "freeze-source":
        payload = freeze_source(args.repository, args.allowlist)
        output = args.output or args.repository / payload["source_lock_path"]
        atomic_json_save(payload, output)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "cuda-probe":
        payload = cuda_probe(args.repository, args.allowlist)
        atomic_json_save(payload, args.output)
        print(json.dumps(payload, indent=2))
        return 0
    payload = readiness_matrix(args.repository, args.allowlist, args.cuda_evidence)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
