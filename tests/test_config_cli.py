"""Config validation and the command-line entry point."""
from pathlib import Path

import pytest
import yaml

from coldspots.cli import main
from coldspots.config import load_config
from coldspots.pipeline import _format_p
from coldspots.synthetic import make_synthetic

REPO = Path(__file__).resolve().parents[1]


def _write_cfg(tmp_path, **sections):
    raw = yaml.safe_load((REPO / "config" / "default.yaml").read_text())
    for dotted, value in sections.items():
        node = raw
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_shipped_configs_load():
    for name in ("default",):
        load_config(REPO / "config" / f"{name}.yaml")


@pytest.mark.parametrize("override, message", [
    ({"features__winsorize": [99, 1]}, "winsorize"),
    ({"features__invert": ["not_a_feature"]}, "invert"),
    ({"features__primary": ["max_AlphaMissense", "min_ESM1b", "max_EVE",
                            "max_phyloP", "max_phyloP"]}, "duplicates"),
    ({"evaluation__min_mave_positions": 0}, "min_mave_positions"),
    ({"evaluation__enabled": "false"}, "true or false"),
    ({"detection__min_cold_run": 40.5}, "whole number"),
    ({"training__intervals": {"BRCA1": [224, 1366]}}, "pairs"),
    ({"training__exclude_genes": []}, "exclude_genes"),
    ({"evaluation__mave_labels": {"PLP": "X", "BLB": "X"}}, "same label"),
    ({"evaluation__mave_labels": {"PLP": "Abnormal"}}, "PLP and BLB"),
    ({"evaluation__unit": "position"}, "unknown setting"),
    ({"evaluation__mave_filter_level": "interval"}, "unknown setting"),
])
def test_bad_settings_are_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write_cfg(tmp_path, **override))


def test_unknown_or_empty_sections(tmp_path):
    raw = yaml.safe_load((REPO / "config" / "default.yaml").read_text())
    raw["detections"] = {"min_cold_run": 40}           # misspelled
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown section"):
        load_config(path)
    path.write_text("data:\n  variant_csv: x.csv\nevaluation:\n")
    assert load_config(path).evaluation.enabled is False   # empty = defaults


def test_cli_requires_config():
    with pytest.raises(SystemExit):
        main([])


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    rc = main(["--config", str(REPO / "config" / "default.yaml"),
               "--variant-csv", str(tmp_path / "missing.csv")])
    assert rc == 1
    assert "frost: error" in capsys.readouterr().err


def test_cli_end_to_end(tmp_path, capsys):
    paths = make_synthetic(tmp_path / "synth", n_genes=20, seed=3)
    rc = main(["--config", str(REPO / "config" / "default.yaml"),
               "--variant-csv", str(paths["variants"]),
               "--out-dir", str(tmp_path / "out"), "--prefix", "cli"])
    assert rc == 0
    assert (tmp_path / "out" / "cli_regions.tsv").exists()
    assert (tmp_path / "out" / "cli_variants.csv.gz").exists()
    assert not (tmp_path / "out" / "cli_metrics.json").exists()
    assert "Done." in capsys.readouterr().out


def test_cli_evaluate_flag(tmp_path):
    paths = make_synthetic(tmp_path / "synth", n_genes=20, seed=3)
    main(["--config", str(REPO / "config" / "default.yaml"),
          "--variant-csv", str(paths["variants"]), "--evaluate",
          "--out-dir", str(tmp_path / "out"), "--prefix", "cli"])
    assert (tmp_path / "out" / "cli_metrics.json").exists()


def test_p_value_display():
    assert _format_p(0.0) == "p < 1e-300"
    assert _format_p(0.01234) == "p = 0.0123"
