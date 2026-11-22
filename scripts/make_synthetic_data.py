"""Write synthetic input tables to data/synthetic/ for a quick smoke test.

    python scripts/make_synthetic_data.py
    python -m coldspots.cli --config config/synthetic.yaml

See coldspots/synthetic.py for what the data contains. Run from the
repository root: config/synthetic.yaml reads data/synthetic/ relative to the
current directory.
"""
from __future__ import annotations

import argparse

from coldspots.synthetic import make_synthetic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--n-genes", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    paths = make_synthetic(args.out_dir, n_genes=args.n_genes, seed=args.seed)
    for name, path in paths.items():
        print(f"  {name:<9} {path}")


if __name__ == "__main__":
    main()
