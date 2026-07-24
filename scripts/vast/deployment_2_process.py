#!/usr/bin/env python3
"""Record, verify, or terminate a deployment #2 process group."""

import argparse
import json
from pathlib import Path

from ternarystem.deployment.process import record_process, terminate_verified_group, verify_process

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
record = subparsers.add_parser("record")
record.add_argument("pid", type=int)
record.add_argument("identity", type=Path)
record.add_argument("--token", required=True)
verify = subparsers.add_parser("verify")
verify.add_argument("identity", type=Path)
terminate = subparsers.add_parser("terminate")
terminate.add_argument("identity", type=Path)
terminate.add_argument("--timeout", type=float, default=10.0)
args = parser.parse_args()

if args.command == "record":
    payload = record_process(args.pid, args.identity, args.token)
elif args.command == "verify":
    payload = verify_process(args.identity)
else:
    terminate_verified_group(args.identity, args.timeout)
    payload = {"terminated": True}
print(json.dumps(payload, indent=2))
