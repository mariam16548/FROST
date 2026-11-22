"""Configuration objects for the cold-spot pipeline.

All tunable parameters and every filesystem path live here (or in the YAML
file that populates these objects). No path is hard-coded anywhere else in
the package, so a third party can reproduce a run by editing one file.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config"]


@dataclass
class DataPaths:
    # The single variant-level input table (CSV, optionally .gz): one row per
    # missense substitution, carrying the per-residue predictor scores plus
    # ClinVar and MAVE calls. The per-residue table used for training and
    # detection is derived from it; both evaluations run on it directly.
    # See docs/DATA.md for the schema.
    variant_csv: str


@dataclass
class Columns:
    clinvar: str = "CLNSIG"
    mave: str = "functional_assignment"
    # Tried in order if `mave` is absent; the resolved name is recorded in the
    # run manifest.
    mave_candidates: list[str] = field(
        default_factory=lambda: [
            "functional_assignment", "ExC_assignment", "assignment",
            "mave_assignment", "MAVE_assignment", "mave_class", "MAVE_class",
        ]
    )
    gene: str = "gene_symbol"
    # Amino-acid position. If the table has no such column, it is parsed from
    # `protein_variant` (e.g. "C2R" -> 2).
    aa_pos: str = "aa_pos"
    protein_variant: str = "protein_variant"
    # Precomputed membership columns that may ship with the table. They are
    # never read, so cold-spot membership is always recomputed from the
    # intervals produced by the current run.
    drop_precomputed: list[str] = field(
        default_factory=lambda: [
            "variant_in_coldspot", "gene_has_coldspot",
            "in_coldspot", "in_coldspot_bw",
        ]
    )


@dataclass
class Features:
    # Feature order matters: the first feature defines which HMM state is
    # labelled "cold" (the state with the lower mean).
    primary: list[str] = field(
        default_factory=lambda: [
            "max_AlphaMissense", "min_ESM1b", "max_EVE", "max_phyloP",
        ]
    )
    # Features dropped (by marginalising the trained Gaussian) to build the
    # fallback model used for genes missing any value of those scores.
    fallback_drop: list[str] = field(default_factory=lambda: ["max_EVE"])
    # Features whose sign is flipped so that "higher == more damaging" holds
    # for every feature before normalisation.
    invert: list[str] = field(default_factory=lambda: ["min_ESM1b"])
    # Winsorisation percentiles applied before min-max scaling.
    winsorize: tuple[float, float] = (1.0, 99.0)


@dataclass
class Training:
    # gene -> list of [start, end] amino-acid intervals treated as cold
    # (state 0) when estimating emission and transition parameters.
    intervals: dict[str, list[list[int]]] = field(
        default_factory=lambda: {"BRCA1": [[224, 1366]]}
    )
    # Genes never decoded, and always removed from both evaluation sets.
    exclude_genes: list[str] = field(default_factory=lambda: ["BRCA1"])


@dataclass
class Detection:
    # Minimum span, in amino-acid coordinates, of a retained cold run.
    min_cold_run: int = 40


@dataclass
class Evaluation:
    # ClinVar gene filter: keep genes with >= this many PLP and BLB rows.
    # 0 disables the filter.
    min_class_count: int = 0
    # MAVE cold-spot filter: distinct aa positions carrying a PLP/BLB call.
    min_mave_positions: int = 10
    # Minimum fraction of a region with any MAVE call. 0 disables.
    min_mave_coverage: float = 0.0
    # "gene": threshold the sum over a gene's cold spots and keep them all if
    #         the gene clears it.
    # "interval": each cold spot must clear the threshold on its own.
    mave_filter_level: str = "gene"
    # "variant":  count one observation per row of the input table.
    # "position": collapse to one observation per (gene, aa_pos, label).
    # See docs/METHODS.md - "variant" is the primary analysis, "position" is
    # the pseudo-replication sensitivity analysis.
    unit: str = "variant"
    # Canonical name -> label as it appears in the data. PLP and BLB are
    # required; any other entries are reported but not tested.
    clinvar_labels: dict[str, str] = field(
        default_factory=lambda: {"PLP": "PLP", "BLB": "BLB", "VUS": "VUS"}
    )
    mave_labels: dict[str, str] = field(
        default_factory=lambda: {"PLP": "Abnormal", "BLB": "Normal",
                                 "IR": "Indeterminate"}
    )


@dataclass
class Output:
    dir: str = "results"
    prefix: str = "coldspots"
    write_pickle: bool = True


@dataclass
class Config:
    data: DataPaths
    columns: Columns = field(default_factory=Columns)
    features: Features = field(default_factory=Features)
    training: Training = field(default_factory=Training)
    detection: Detection = field(default_factory=Detection)
    evaluation: Evaluation = field(default_factory=Evaluation)
    output: Output = field(default_factory=Output)
    seed: int = 42

    # -- derived -----------------------------------------------------------
    @property
    def fallback_features(self) -> list[str]:
        """Primary feature list minus the marginalised features."""
        return [f for f in self.features.primary
                if f not in self.features.fallback_drop]

    @property
    def fallback_keep_indices(self) -> list[int]:
        return [i for i, f in enumerate(self.features.primary)
                if f not in self.features.fallback_drop]

    @property
    def training_intervals(self) -> dict[str, list[tuple[int, int]]]:
        return {g: [tuple(iv) for iv in ivs]
                for g, ivs in self.training.intervals.items()}

    @property
    def exclude_genes(self) -> set[str]:
        return set(self.training.exclude_genes)

    def validate(self) -> None:
        e = self.evaluation
        if e.mave_filter_level not in ("gene", "interval"):
            raise ValueError(
                f"evaluation.mave_filter_level must be 'gene' or 'interval', "
                f"got {e.mave_filter_level!r}"
            )
        if e.unit not in ("variant", "position"):
            raise ValueError(
                f"evaluation.unit must be 'variant' or 'position', "
                f"got {e.unit!r}"
            )
        if not self.features.primary:
            raise ValueError("features.primary must not be empty")
        missing = [f for f in self.features.fallback_drop
                   if f not in self.features.primary]
        if missing:
            raise ValueError(
                f"features.fallback_drop entries not in features.primary: "
                f"{missing}"
            )
        if not self.fallback_features:
            raise ValueError("fallback_drop removes every feature")
        for name, labels in (("clinvar_labels", e.clinvar_labels),
                             ("mave_labels", e.mave_labels)):
            if not {"PLP", "BLB"} <= set(labels):
                raise ValueError(
                    f"evaluation.{name} must map both PLP and BLB to the "
                    f"labels used in the data, got {labels}"
                )
        if self.detection.min_cold_run < 1:
            raise ValueError("detection.min_cold_run must be >= 1")
        for gene, ivs in self.training.intervals.items():
            for iv in ivs:
                if len(iv) != 2 or iv[0] > iv[1]:
                    raise ValueError(
                        f"bad training interval for {gene}: {iv}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of the resolved config, recorded in the run manifest."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


_SECTIONS = {
    "columns": Columns,
    "features": Features,
    "training": Training,
    "detection": Detection,
    "evaluation": Evaluation,
    "output": Output,
}


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply dotted-key overrides, and validate it.

    ``overrides`` keys use dotted paths, e.g. ``{"evaluation.unit": "position"}``.
    """
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    raw = copy.deepcopy(raw)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = raw
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    if "data" not in raw:
        raise KeyError(f"{path}: missing required 'data' section")
    if "variant_csv" not in raw["data"]:
        raise KeyError(
            f"{path}: 'data' must set variant_csv, the single variant-level "
            f"table (see docs/DATA.md). The older two-table layout "
            f"(vep_residue_csv + vep_expanded_csv) is no longer supported."
        )
    cfg = Config(
        data=DataPaths(**raw["data"]),
        **{name: cls(**raw.get(name, {})) for name, cls in _SECTIONS.items()},
        seed=raw.get("seed", 42),
    )
    if isinstance(cfg.features.winsorize, list):
        cfg.features.winsorize = tuple(cfg.features.winsorize)
    cfg.validate()
    return cfg
