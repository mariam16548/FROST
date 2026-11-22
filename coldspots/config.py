"""Settings, loaded from a YAML file (see config/default.yaml)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config"]


@dataclass
class DataPaths:
    # The input table: one row per missense substitution (docs/DATA.md).
    variant_csv: str


@dataclass
class Columns:
    gene: str = "gene_symbol"
    transcript: str = "transcript_id"      # copied into the outputs
    # The position is read from protein_variant (e.g. "C2R" -> 2) unless the
    # table has an aa_pos column.
    aa_pos: str = "aa_pos"
    protein_variant: str = "protein_variant"
    clinvar: str = "CLNSIG"                 # --evaluate only
    mave: str = "functional_assignment"     # --evaluate only


@dataclass
class Features:
    # Per-residue scores. The labelled training region must have the lower
    # mean on the first feature (AlphaMissense), which defines the Cold state.
    primary: list[str] = field(
        default_factory=lambda: [
            "max_AlphaMissense", "min_ESM1b", "max_EVE", "max_phyloP",
        ]
    )
    # Genes with any missing value of these features use the others.
    fallback_drop: list[str] = field(default_factory=lambda: ["max_EVE"])
    # Sign-inverted so that higher = more deleterious for every feature.
    invert: list[str] = field(default_factory=lambda: ["min_ESM1b"])
    # Percentiles each feature is clipped to before min-max scaling.
    winsorize: tuple[float, float] = (1.0, 99.0)


@dataclass
class Training:
    # gene -> [start, end] amino-acid intervals labelled Cold.
    intervals: dict[str, list[list[int]]] = field(
        default_factory=lambda: {"BRCA1": [[224, 1366]]}
    )
    # Never decoded or evaluated.
    exclude_genes: list[str] = field(default_factory=lambda: ["BRCA1"])


@dataclass
class Detection:
    # Cold runs spanning fewer positions are reassigned to Not-Cold.
    min_cold_run: int = 40


@dataclass
class Evaluation:
    # Benchmarking against ClinVar and MAVE data (--evaluate).
    enabled: bool = False
    # A gene enters the MAVE evaluation if its abnormal/normal calls cover at
    # least this many distinct positions inside its coldspots, summed across
    # all of its coldspots.
    min_mave_positions: int = 10
    # Canonical name -> label as written in the table.
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


@dataclass
class Config:
    data: DataPaths
    columns: Columns = field(default_factory=Columns)
    features: Features = field(default_factory=Features)
    training: Training = field(default_factory=Training)
    detection: Detection = field(default_factory=Detection)
    evaluation: Evaluation = field(default_factory=Evaluation)
    output: Output = field(default_factory=Output)

    @property
    def fallback_features(self) -> list[str]:
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
        f, e = self.features, self.evaluation
        if not f.primary:
            raise ValueError("features.primary must not be empty")
        if len(set(f.primary)) != len(f.primary):
            raise ValueError("features.primary contains duplicates")
        for name in ("fallback_drop", "invert"):
            bad = [x for x in getattr(f, name) if x not in f.primary]
            if bad:
                raise ValueError(
                    f"features.{name} entries not in features.primary: {bad}")
        if not self.fallback_features:
            raise ValueError("features.fallback_drop removes every feature")
        if (len(f.winsorize) != 2
                or not all(_is_number(x) for x in f.winsorize)
                or not 0 <= f.winsorize[0] < f.winsorize[1] <= 100):
            raise ValueError(
                f"features.winsorize must be [low, high] with "
                f"0 <= low < high <= 100, got {list(f.winsorize)}")

        if not self.training.intervals:
            raise ValueError("training.intervals must not be empty")
        for gene, ivs in self.training.intervals.items():
            if not isinstance(ivs, list) or not all(
                    isinstance(iv, (list, tuple)) and len(iv) == 2
                    and all(_is_int(x) for x in iv) and iv[0] <= iv[1]
                    for iv in ivs):
                raise ValueError(
                    f"training.intervals.{gene} must be a list of "
                    f"[start, end] pairs, e.g. [[224, 1366]]; got {ivs}")
        not_excluded = [g for g in self.training.intervals
                        if g not in self.training.exclude_genes]
        if not_excluded:
            raise ValueError(
                f"training gene(s) {not_excluded} must also be listed in "
                f"training.exclude_genes")

        if not _is_int(self.detection.min_cold_run) \
                or self.detection.min_cold_run < 1:
            raise ValueError("detection.min_cold_run must be a whole "
                             "number >= 1")
        if not isinstance(e.enabled, bool):
            raise ValueError("evaluation.enabled must be true or false")
        if not _is_int(e.min_mave_positions) or e.min_mave_positions < 1:
            raise ValueError("evaluation.min_mave_positions must be a whole "
                             "number >= 1")
        for name in ("clinvar_labels", "mave_labels"):
            labels = getattr(e, name)
            if not {"PLP", "BLB"} <= set(labels):
                raise ValueError(f"evaluation.{name} must map both PLP and "
                                 f"BLB, got {labels}")
            if labels["PLP"] == labels["BLB"]:
                raise ValueError(f"evaluation.{name}: PLP and BLB map to the "
                                 f"same label {labels['PLP']!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short hash of the settings, recorded in the run manifest."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


_SECTIONS = {
    "data": DataPaths,
    "columns": Columns,
    "features": Features,
    "training": Training,
    "detection": Detection,
    "evaluation": Evaluation,
    "output": Output,
}


def _build(section: str, cls, values) -> Any:
    values = values or {}
    if not isinstance(values, dict):
        raise ValueError(f"config section '{section}' must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown setting(s) in '{section}': {unknown}; "
                         f"allowed: {sorted(known)}")
    return cls(**values)


def load_config(path: str | Path,
                overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply dotted-key overrides, and validate it.

    ``overrides`` keys use dotted paths, e.g. ``{"output.dir": "results/run2"}``.
    """
    path = Path(path)
    with open(path) as fh:
        raw = copy.deepcopy(yaml.safe_load(fh) or {})

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[leaf] = value

    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}; "
                         f"allowed: {sorted(_SECTIONS)}")
    if not (raw.get("data") or {}).get("variant_csv"):
        raise ValueError(f"{path}: data.variant_csv (the input table) "
                         f"is required")
    built = {name: _build(name, cls, raw.get(name))
             for name, cls in _SECTIONS.items()}
    cfg = Config(**built)
    if isinstance(cfg.features.winsorize, list):
        cfg.features.winsorize = tuple(cfg.features.winsorize)
    cfg.validate()
    return cfg
