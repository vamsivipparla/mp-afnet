"""Adapter to an explicitly supplied LUNA16 evaluator (not vendored)."""
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

from .data import file_hash, read_rows, write_rows
from .fusion import key, hybrid_score


# noduleCADEvaluationLUNA16.py distributed in LUNA16's evaluationScript.zip.
_LUNA16_PYTHON2_SHA256 = "5c22276353294a6e9de40e7ec8f63f8efa83252c34110b0c0f3c00cbe6b1dcd8"


def _csv_records(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def _load_evaluator(path):
    spec = importlib.util.spec_from_file_location("luna_external", path)
    evaluator = importlib.util.module_from_spec(spec)
    source = path.read_bytes()
    python2_release = hashlib.sha256(source).hexdigest() == _LUNA16_PYTHON2_SHA256
    if python2_release:
        # Adapt the challenge distribution in memory. Candidate matching,
        # bootstrap sampling, and FROC calculations come from the source file.
        source = re.sub(rb"(?m)^([ \t]*)print ([^\r\n]+)", rb"\1print(\2)", source)
        source = source.replace(b".iteritems()", b".items()")
    exec(compile(source, str(path), "exec"), evaluator.__dict__)
    if python2_release:
        evaluator.csvTools.readCSV = _csv_records
    return evaluator


class _EvaluatorPlots:
    """Adapt the LUNA16 evaluator's plotting calls to current Matplotlib."""

    def __init__(self, pyplot):
        self.pyplot = pyplot

    def __getattr__(self, name):
        return getattr(self.pyplot, name)

    def grid(self, *args, **kwargs):
        if "b" in kwargs:
            kwargs["visible"] = kwargs.pop("b")
        return self.pyplot.grid(*args, **kwargs)

    def xscale(self, value, **kwargs):
        if "basex" in kwargs:
            kwargs["base"] = kwargs.pop("basex")
        return self.pyplot.xscale(value, **kwargs)

    def legend(self, *args, **kwargs):
        if self.pyplot.gca().get_legend_handles_labels()[0]:
            return self.pyplot.legend(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        # The caller plots the returned FROC and all seven CPM operating points.
        return None


def merge_predictions(paths, data, output):
    rows = [r for p in paths for r in read_rows(p)]
    keys = [key(r) for r in rows]
    expected = {key(r) for r in read_rows(Path(data) / "manifest.csv")}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Merged predictions must cover every manifest candidate exactly once")
    write_rows(output, sorted(rows, key=key),
               ["seriesuid", "coordX", "coordY", "coordZ", "probability"])


def evaluate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if args.bootstraps < 0:
        raise ValueError("Bootstrap sample count cannot be negative")
    path, output = Path(args.evaluator).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scans = [r["seriesuid"] for r in read_rows(Path(args.data) / "scans.csv")
             if args.fold is None or int(r["fold"]) == args.fold]
    if not scans or len(scans) != len(set(scans)):
        raise ValueError("Empty or duplicate evaluation scan IDs")
    predictions = read_rows(args.predictions)
    keys = [key(r) for r in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prediction coordinates")
    for r in predictions:
        if r["seriesuid"] not in scans or not np.isfinite(key(r)[1:]).all():
            raise ValueError("Prediction outside declared scan set or invalid coordinates")
        hybrid_score([float(r["probability"])] * 3)
    # Evaluation code is executable: supply a reviewed, trusted evaluator file.
    existing_figures = set(plt.get_fignums())
    sys.path.insert(0, str(path.parent))
    try:
        evaluator = _load_evaluator(path)
        evaluator.plt = _EvaluatorPlots(plt)
        nodules = evaluator.collectNoduleAnnotations(
            _csv_records(args.annotations), _csv_records(args.excluded), scans)
        np.random.seed(args.seed)
        fps, sensitivity, _, bootstrap_fps, bootstrap_mean, lower, upper = evaluator.evaluateCAD(
            scans, str(args.predictions), str(output), nodules, "mp_afnet",
            maxNumberOfCADMarks=100, performBootstrapping=args.bootstraps > 0,
            numberOfBootstrapSamples=args.bootstraps)
    finally:
        sys.path.pop(0)
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)
    rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    sampled = np.interp(rates, fps, sensitivity)
    figure, axes = plt.subplots()
    grid = np.geomspace(rates[0], rates[-1], 200)
    axes.plot(grid, np.interp(grid, fps, sensitivity), label="Observed FROC")
    axes.scatter(rates, sampled, s=15)
    if bootstrap_fps is not None:
        axes.plot(bootstrap_fps, bootstrap_mean, linestyle="--", label="Bootstrap mean")
        axes.fill_between(bootstrap_fps, lower, upper, alpha=0.15, label="Pointwise 95% band")
    axes.set(xlabel="False positives per scan", ylabel="Sensitivity",
             xlim=(rates[0], rates[-1]), ylim=(0, 1), title="Candidate false-positive reduction")
    axes.set_xscale("log", base=2)
    axes.set_xticks(rates, labels=[str(rate) for rate in rates])
    axes.grid(visible=True, alpha=0.3)
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "froc_mp_afnet.png", dpi=150)
    plt.close(figure)
    result = dict(cpm=float(sampled.mean()), fp_per_scan=rates, sensitivity=sampled.tolist(),
                  scans=len(scans), scan_ids=scans, evaluator_sha256=file_hash(path),
                  predictions_sha256=file_hash(args.predictions),
                  annotations_sha256=file_hash(args.annotations), excluded_sha256=file_hash(args.excluded),
                  max_marks=100, bootstraps=args.bootstraps, seed=args.seed)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"cpm": result["cpm"], "scans": len(scans)}))
