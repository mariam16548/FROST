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


def _cfg(synth, out_dir, evaluate=True, **extra):
    overrides = {"data.variant_csv": str(synth["variants"]),
                 "output.dir": str(out_dir),
                 "evaluation.enabled": evaluate, **extra}
    return load_config(REPO / "config" / "default.yaml", overrides)


def test_end_to_end_recovers_planted_cold_spots(synth, tmp_path):
    result = run(_cfg(synth, tmp_path))
    for key in ("regions", "variants", "metrics", "manifest"):
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
    no_eve = res.groupby("gene_symbol")["max_EVE"].apply(lambda x: x.isna().any())
    assert set(calls["gene"]) & set(no_eve[no_eve].index)  # 3-feature genes too

    metrics = json.loads(result["written"]["metrics"].read_text())
    assert metrics["or_clinvar"]["OR"] > 1         # synthetic cold = benign

    manifest = json.loads(result["written"]["manifest"].read_text())
    assert manifest["fitted_models"][0]["cold_state"] == 0
    assert "scikit-learn" not in manifest["package_versions"]


def test_two_runs_give_identical_files(synth, tmp_path):
    a = run(_cfg(synth, tmp_path / "a"))["written"]
    b = run(_cfg(synth, tmp_path / "b"))["written"]
    for key in ("regions", "metrics"):
        assert a[key].read_bytes() == b[key].read_bytes()


def test_table_without_pathogenic_labels(synth, tmp_path):
    """No PLP rows at all must not crash the evaluation."""
    df = pd.read_csv(synth["variants"], low_memory=False)
    df.loc[df["CLNSIG"] == "PLP", "CLNSIG"] = np.nan
    path = tmp_path / "no_plp.csv"
    df.to_csv(path, index=False)
    ev = run(_cfg(synth, tmp_path, **{"data.variant_csv": str(path)}))["eval"]
    assert ev["clinvar_summary"]["PLP"]["total"] == 0


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


def test_missing_aa_pos_is_an_error(synth, tmp_path):
    """A blank position must stop the run, not silently drop coldspots."""
    df = pd.read_csv(synth["variants"], low_memory=False)
    df["aa_pos"] = df["protein_variant"].str[1:-1].astype(float)
    df.loc[df.index[500], "aa_pos"] = np.nan
    path = tmp_path / "blank_pos.csv"
    df.drop(columns="protein_variant").to_csv(path, index=False)
    with pytest.raises(ValueError, match="amino-acid position"):
        run(_cfg(synth, tmp_path, **{"data.variant_csv": str(path)}))


def test_default_run_writes_regions_and_membership(synth, tmp_path):
    """Evaluation off: regions TSV and per-variant membership, nothing else."""
    result = run(_cfg(synth, tmp_path, evaluate=False))
    assert result["eval"] is None
    assert set(result["written"]) == {"regions", "variants", "manifest"}

    regions = pd.read_csv(result["written"]["regions"], sep="\t")
    assert list(regions.columns) == ["gene_symbol", "transcript_id",
                                     "aa_start", "aa_end"]
    assert len(regions) == len(result["coldspots"])
    assert (regions["transcript_id"] == "ENST_" + regions["gene_symbol"]).all()

    src = pd.read_csv(synth["variants"], low_memory=False)
    out = pd.read_csv(result["written"]["variants"], low_memory=False,
                      dtype={"gene_has_coldspot": "boolean",
                             "variant_in_coldspot": "boolean"})
    assert list(out.columns) == ["transcript_id", "gene_symbol",
                                 "protein_variant", "gene_has_coldspot",
                                 "variant_in_coldspot"]
    assert len(out) == len(src)
    assert (out["protein_variant"].values == src["protein_variant"].values).all()

    brca1 = out["gene_symbol"] == "BRCA1"
    assert out.loc[brca1, ["gene_has_coldspot", "variant_in_coldspot"]].isna().all().all()
    rest = out[~brca1].copy()
    rest["aa_pos"] = rest["protein_variant"].str[1:-1].astype(int)
    expected = in_intervals(rest, regions.rename(columns={"gene_symbol": "gene"}),
                            gene_col="gene_symbol", aa_col="aa_pos")
    assert (rest["variant_in_coldspot"].astype(bool).values == expected).all()
    has = rest["gene_symbol"].isin(set(regions["gene_symbol"]))
    assert (rest["gene_has_coldspot"].astype(bool) == has).all()


def test_label_columns_not_needed_when_evaluation_is_off(synth, tmp_path):
    df = pd.read_csv(synth["variants"], low_memory=False)
    path = tmp_path / "no_labels.csv"
    df.drop(columns=["CLNSIG", "functional_assignment"]).to_csv(path, index=False)
    a = run(_cfg(synth, tmp_path / "a", evaluate=False))["coldspots"]
    b = run(_cfg(synth, tmp_path / "b", evaluate=False,
                 **{"data.variant_csv": str(path)}))["coldspots"]
    pd.testing.assert_frame_equal(a, b)
