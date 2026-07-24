#!/usr/bin/env python3
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Evaluate generated MUSDB stems with museval")
parser.add_argument("--musdb-root", required=True)
parser.add_argument("--estimates", required=True)
args = parser.parse_args()
try:
    import musdb
    import museval
except ImportError as error:
    raise SystemExit("install evaluation dependencies: pip install -e '.[eval]'") from error
mus = musdb.DB(root=args.musdb_root, subsets="test", is_wav=True)
museval.eval_mus_dir(mus, args.estimates)
