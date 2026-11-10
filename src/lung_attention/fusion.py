"""Paper hybrid voting with explicit candidate identity validation."""
import math
from .data import read_rows, write_rows


def hybrid_score(scores):
    if len(scores) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Expected three finite nodule probabilities")
    votes = [p > 0.5 for p in scores]
    majority = sum(votes) >= 2
    agreeing = [p for p, vote in zip(scores, votes) if vote == majority]
    return sum(agreeing) / len(agreeing)


def key(row):
    return (row["seriesuid"], *(float(row[k]) for k in ("coordX", "coordY", "coordZ")))


def fuse(paths, output, method="hybrid"):
    if len(paths) != 3 or method not in {"hybrid", "mean"}:
        raise ValueError("Select three inputs and hybrid or mean fusion")
    tables = []
    for path in paths:
        rows = read_rows(path)
        table = {key(row): row for row in rows}
        if not rows or len(rows) != len(table):
            raise ValueError("Empty predictions or duplicate candidate coordinates")
        tables.append(table)
    if any(set(t) != set(tables[0]) for t in tables[1:]):
        raise ValueError("Candidate keys differ; refusing to silently align unequal files")
    result = []
    for candidate in sorted(tables[0]):
        scores = [float(t[candidate]["probability"]) for t in tables]
        hybrid = hybrid_score(scores)  # validates scores for either method
        row = dict(tables[0][candidate])
        row["probability"] = hybrid if method == "hybrid" else sum(scores) / 3
        result.append(row)
    write_rows(output, result, ["seriesuid", "coordX", "coordY", "coordZ", "probability"])
