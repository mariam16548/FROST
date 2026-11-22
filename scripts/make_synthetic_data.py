"""Write a small synthetic input table to data/synthetic/ for a quick test.

    python scripts/make_synthetic_data.py
    frost --config config/default.yaml \
        --variant-csv data/synthetic/variants.csv.gz --out-dir results/synthetic
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
