"""Opt-in checks against the LUNA16 challenge evaluator."""
import argparse
import json
import os
from pathlib import Path

import pytest

from lung_attention.data import write_rows
from lung_attention.evaluation import evaluate


@pytest.mark.parametrize("bootstraps", [0, 20])
def test_external_froc_and_scan_denominator(tmp_path, bootstraps):
    evaluator = os.environ.get("LUNA16_EVALUATOR")
    if not evaluator:
        pytest.skip("Set LUNA16_EVALUATOR to a trusted evaluator to run integration checks")
    assert Path(evaluator).is_file()
    write_rows(tmp_path / "scans.csv",
               [dict(seriesuid=uid, fold=0) for uid in ("scan-a", "scan-b")],
               ["seriesuid", "fold"])
    fields = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
    annotations = [dict(zip(fields, [uid, 10, 10, 10, 6])) for uid in ("scan-a", "scan-b")]
    prediction_fields = ["seriesuid", "coordX", "coordY", "coordZ", "probability"]
    predictions = [dict(zip(prediction_fields, ["scan-a", 10, 10, 10, 0.9])),
                   dict(zip(prediction_fields, ["scan-a", 30, 30, 30, 0.1]))]
    if bootstraps:
        # Every sampled scan has both classes, avoiding undefined ROC samples.
        predictions += [dict(row, seriesuid="scan-b") for row in predictions]
        annotations += [dict(zip(fields, [uid, 50, 50, 50, 6]))
                        for uid in ("scan-a", "scan-b")]
    # Without bootstrapping, scan-b has a missed nodule and no predictions.
    write_rows(tmp_path / "annotations.csv", annotations, fields)
    write_rows(tmp_path / "excluded.csv", [], fields)
    write_rows(tmp_path / "predictions.csv", predictions, prediction_fields)
    evaluate(argparse.Namespace(
        evaluator=evaluator, data=str(tmp_path), fold=0, seed=2021,
        predictions=str(tmp_path / "predictions.csv"),
        annotations=str(tmp_path / "annotations.csv"),
        excluded=str(tmp_path / "excluded.csv"),
        output=str(tmp_path / "result"), bootstraps=bootstraps))
    result = json.loads((tmp_path / "result/metrics.json").read_text())
    assert result["scans"] == 2
    assert result["cpm"] == pytest.approx(0.5)
    assert result["sensitivity"] == pytest.approx([0.5] * 7)
    assert (tmp_path / "result/froc_mp_afnet.png").stat().st_size > 1000
    if bootstraps:
        assert (tmp_path / "result/froc_mp_afnet_bootstrapping.csv").is_file()
