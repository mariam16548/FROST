"""Command line: ``frost --config config/default.yaml``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frost",
        description="Find coldspots (Find Regions Of Substitution Tolerance).",
    )
    p.add_argument("--config", type=Path, required=True, metavar="YAML",
                   help="settings file, e.g. config/default.yaml")
    p.add_argument("--variant-csv", dest="data.variant_csv", metavar="CSV",
                   help="input table (overrides the config)")
    p.add_argument("--out-dir", dest="output.dir", metavar="DIR",
                   help="output folder (default: results)")
    p.add_argument("--prefix", dest="output.prefix", metavar="NAME",
                   help="output file prefix (default: coldspots)")
    p.add_argument("--evaluate", dest="evaluation.enabled",
                   action="store_const", const=True,
                   help="also benchmark against ClinVar and MAVE data")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k != "config" and v is not None}
    try:
        cfg = load_config(args.config, overrides)
        run(cfg)
    except (OSError, KeyError, ValueError) as err:
        print(f"frost: error: {err}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
