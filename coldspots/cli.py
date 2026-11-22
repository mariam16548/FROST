"""Command-line entry point: ``python -m coldspots.cli --config config/...``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coldspots",
        description="Detect and evaluate mutational cold spots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML configuration file")
    p.add_argument("--variant-csv", dest="data.variant_csv",
                   help="override data.variant_csv (the input table)")
    p.add_argument("--out-dir", dest="output.dir",
                   help="override output.dir")
    p.add_argument("--prefix", dest="output.prefix",
                   help="override output.prefix")
    p.add_argument("--evaluate", dest="evaluation.enabled",
                   action="store_const", const=True,
                   help="also run the ClinVar / MAVE enrichment analysis")
    p.add_argument("--unit", dest="evaluation.unit",
                   choices=["variant", "position"],
                   help="counting unit for the enrichment tests")
    p.add_argument("--mave-filter-level", dest="evaluation.mave_filter_level",
                   choices=["gene", "interval"],
                   help="apply the MAVE support threshold per gene or per interval")
    p.add_argument("--min-mave-positions", dest="evaluation.min_mave_positions",
                   type=int, help="minimum distinct PLP/BLB MAVE positions")
    p.add_argument("--min-mave-coverage", dest="evaluation.min_mave_coverage",
                   type=float, help="minimum assayed fraction of a cold spot")
    p.add_argument("--min-cold-run", dest="detection.min_cold_run", type=int,
                   help="minimum cold-run span in amino acids")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k != "config" and v is not None}
    cfg = load_config(args.config, overrides)
    print(f"Config: {args.config}  (fingerprint {cfg.fingerprint()})")
    run(cfg)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
