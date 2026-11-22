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
    for name in ("default", "sensitivity_position_level", "synthetic"):
        load_config(REPO / "config" / f"{name}.yaml")


@pytest.mark.parametrize("override, message", [
    ({"features__winsorize": [99, 1]}, "winsorize"),
    ({"features__invert": ["not_a_feature"]}, "invert"),
    ({"features__primary": ["max_AlphaMissense", "min_ESM1b", "max_EVE",
                            "max_phyloP", "max_phyloP"]},
     "duplicates"),
    ({"evaluation__min_mave_coverage": 1.5}, "min_mave_coverage"),
    ({"evaluation__min_mave_positions": -1}, ">= 0"),
    ({"training__exclude_genes": []}, "exclude_genes"),
    ({"evaluation__mave_labels": {"PLP": "X", "BLB": "X"}}, "same label"),
    ({"evaluation__mave_labels": {"PLP": "Abnormal"}}, "PLP and BLB"),
    ({"evaluation__unit": "residue"}, "unit"),
])
def test_bad_settings_are_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write_cfg(tmp_path, **override))


def test_cli_end_to_end(tmp_path, capsys):
    paths = make_synthetic(tmp_path / "synth", n_genes=20, seed=3)
    rc = main(["--config", str(REPO / "config" / "synthetic.yaml"),
               "--variant-csv", str(paths["variants"]),
               "--out-dir", str(tmp_path / "out"), "--prefix", "cli"])
    assert rc == 0
    assert (tmp_path / "out" / "cli_intervals.csv").exists()
    assert "Done." in capsys.readouterr().out


def test_p_value_display():
    assert _format_p(0.0) == "<1e-300"
    assert _format_p(0.01234) == "0.01234"
