"""End-to-end runs on synthetic data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coldspots.config import load_config
from coldspots.evaluation import in_intervals
from coldspots.pipeline import run
from coldspots.synthetic import make_synthetic

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth"), n_genes=40, seed=1)


def _cfg(synth, out_dir, **extra):
    overrides = {"data.variant_csv": str(synth["variants"]),
                 "output.dir": str(out_dir), **extra}
    return load_config(REPO / "config" / "synthetic.yaml", overrides)


def test_end_to_end_recovers_planted_cold_spots(synth, tmp_path):
    result = run(_cfg(synth, tmp_path))
    for key in ("intervals", "metrics", "manifest", "pickle"):
        assert result["written"][key].exists()

    calls = result["coldspots"]
    truth = pd.read_csv(synth["truth"])
    res = pd.read_csv(synth["variants"], low_memory=False)
    res["aa_pos"] = res["protein_variant"].str[1:-1].astype(int)
    res = res.drop_duplicates(["gene_symbol", "aa_pos"])
    res = res[res["gene_symbol"] != "BRCA1"]
    predicted = in_intervals(res, calls, gene_col="gene_symbol", aa_col="aa_pos")
    actual = in_intervals(res, truth, gene_col="gene_symbol", aa_col="aa_pos")
    precision = (predicted & actual).sum() / predicted.sum()
    recall = (predicted & actual).sum() / actual.sum()
    assert precision > 0.95 and recall > 0.95

    assert "BRCA1" not in set(calls["gene"])       # training gene excluded
    assert "G00000" not in set(calls["gene"])      # has a missing AM value
    assert set(calls["model"]) == {"4D", "3D"}     # fallback model was used

    metrics = json.loads(result["written"]["metrics"].read_text())
    assert metrics["or_clinvar"]["OR"] > 1         # synthetic cold = benign

    manifest = json.loads(result["written"]["manifest"].read_text())
    assert manifest["trained_models"][0]["cold_state"] == 0
    assert "scikit-learn" not in manifest["package_versions"]


def test_two_runs_give_identical_files(synth, tmp_path):
    a = run(_cfg(synth, tmp_path / "a"))["written"]
    b = run(_cfg(synth, tmp_path / "b", seed=7))["written"]
    for key in ("intervals", "metrics"):
        assert a[key].read_bytes() == b[key].read_bytes()


def test_sensitivity_settings_run(synth, tmp_path):
    result = run(_cfg(synth, tmp_path, **{"evaluation.unit": "position",
                                          "evaluation.mave_filter_level":
                                              "interval"}))
    assert result["eval"]["unit"] == "position"


def test_table_without_pathogenic_labels(synth, tmp_path):
    """No PLP rows at all must not crash the ClinVar gene filter."""
    df = pd.read_csv(synth["variants"], low_memory=False)
    df.loc[df["CLNSIG"] == "PLP", "CLNSIG"] = np.nan
    path = tmp_path / "no_plp.csv"
    df.to_csv(path, index=False)
    cfg = _cfg(synth, tmp_path, **{"data.variant_csv": str(path),
                                   "evaluation.min_class_count": 1})
    run(cfg)


def test_misplaced_training_interval_is_an_error(synth, tmp_path):
    """Labelling a hot stretch of BRCA1 as cold must stop, not silently flip."""
    cfg = _cfg(synth, tmp_path)
    cfg.training.intervals = {"BRCA1": [[1400, 1800]]}
    with pytest.raises(ValueError, match="cannot be the cold state"):
        run(cfg)


def test_mave_labels_are_read_from_config(synth, tmp_path):
    """The released table words MAVE calls Normal/Abnormal, not BLB/PLP."""
    ev = run(_cfg(synth, tmp_path))["eval"]
    assert ev["n_mave_intervals"] > 0
    assert ev["mave_summary"]["PLP"]["total"] > 0
    assert ev["mave_summary"]["BLB"]["total"] > 0
    assert not np.isnan(ev["or_mave"]["OR"])


def test_precomputed_membership_columns_are_ignored(synth, tmp_path):
    """Flipping the shipped variant_in_coldspot column changes nothing."""
    df = pd.read_csv(synth["variants"], low_memory=False)
    df["variant_in_coldspot"] = ~df["variant_in_coldspot"]
    path = tmp_path / "flipped.csv"
    df.to_csv(path, index=False)
    a = run(_cfg(synth, tmp_path / "a"))["written"]["metrics"].read_bytes()
    b = run(_cfg(synth, tmp_path / "b",
                 **{"data.variant_csv": str(path)}))["written"]["metrics"]
    assert a == b.read_bytes()


def test_aa_pos_column_is_used_when_present(synth, tmp_path):
    df = pd.read_csv(synth["variants"], low_memory=False)
    df["aa_pos"] = df["protein_variant"].str[1:-1].astype(int)
    df = df.drop(columns="protein_variant")
    path = tmp_path / "with_pos.csv"
    df.to_csv(path, index=False)
    a = run(_cfg(synth, tmp_path / "a"))["coldspots"]
    b = run(_cfg(synth, tmp_path / "b",
                 **{"data.variant_csv": str(path)}))["coldspots"]
    pd.testing.assert_frame_equal(a, b)


def test_scores_that_vary_within_a_residue_are_rejected(synth, tmp_path):
    df = pd.read_csv(synth["variants"], low_memory=False)
    df.loc[df.index[::2], "max_AlphaMissense"] += 0.01
    path = tmp_path / "varying.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must be per-residue"):
        run(_cfg(synth, tmp_path, **{"data.variant_csv": str(path)}))


def test_old_two_table_config_gives_a_clear_error(tmp_path):
    cfg = tmp_path / "old.yaml"
    cfg.write_text("data:\n  vep_residue_csv: a.csv\n  vep_expanded_csv: b.csv\n")
    with pytest.raises(KeyError, match="variant_csv"):
        load_config(cfg)
