"""Candidate-centered CT crops with scan-disjoint partitions."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .preprocessing import segment_lung_field


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(raw, candidates, output, *, preprocessing="lung"):
    """Write memory-mappable 1 mm CT volumes and candidate/scan manifests."""
    if preprocessing not in {"lung", "resample"}:
        raise ValueError("Preprocessing must be 'lung' or 'resample'")
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.csv").exists():
        raise ValueError("Output already prepared; use a fresh directory")
    scans = {}
    for fold in range(10):
        for path in sorted((raw / f"subset{fold}").glob("*.mhd")):
            if path.stem in scans:
                raise ValueError("Duplicate scan ID across subsets")
            scans[path.stem] = (fold, path)
    if not scans:
        raise ValueError("No subset0..subset9/*.mhd files found")
    grouped = {uid: [] for uid in scans}
    seen = set()
    for row in read_rows(candidates):
        uid = row["seriesuid"]
        if uid not in scans:
            raise ValueError(f"Candidate scan absent from raw data: {uid}")
        point = tuple(float(row[k]) for k in ("coordX", "coordY", "coordZ"))
        if not np.isfinite(point).all() or row["class"] not in {"0", "1"}:
            raise ValueError("Invalid candidate coordinates or class")
        key = (uid, *point)
        if key in seen:
            raise ValueError("Duplicate candidate coordinates")
        seen.add(key)
        grouped[uid].append((row, point))
    manifest, scan_rows = [], []
    for uid, (fold, path) in scans.items():
        original = sitk.ReadImage(str(path), sitk.sitkFloat32)
        size = [max(1, round(n * s)) for n, s in zip(original.GetSize(), original.GetSpacing())]
        image = sitk.Resample(original, size, sitk.Transform(), sitk.sitkNearestNeighbor,
                              original.GetOrigin(), [1.0] * 3, original.GetDirection(),
                              -1000.0, sitk.sitkFloat32)
        if preprocessing == "lung":
            try:
                image = segment_lung_field(image)
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"Lung field preprocessing failed for {uid}: {error}") from error
        volume_name = f"{uid}.npy"
        np.save(output / volume_name, sitk.GetArrayFromImage(image))
        for row, point in grouped[uid]:
            index = image.TransformPhysicalPointToContinuousIndex(point)
            if any(i < -0.5 or i > n - 0.5 for i, n in zip(index, size)):
                raise ValueError(f"Candidate outside image: {uid}, {point}")
            candidate_id = hashlib.sha256(json.dumps([uid, *point]).encode()).hexdigest()[:24]
            manifest.append(dict(candidate_id=candidate_id, seriesuid=uid, fold=fold,
                                 volume=volume_name, label=int(row["class"]),
                                 coordX=point[0], coordY=point[1], coordZ=point[2],
                                 x=index[0], y=index[1], z=index[2]))
        scan_rows.append(dict(seriesuid=uid, fold=fold))
    fields = ["candidate_id", "seriesuid", "fold", "volume", "label",
              "coordX", "coordY", "coordZ", "x", "y", "z"]
    write_rows(output / "manifest.csv", manifest, fields)
    write_rows(output / "scans.csv", scan_rows, ["seriesuid", "fold"])
    (output / "preprocessing.json").write_text(json.dumps({
        "spacing_mm": [1, 1, 1], "interpolation": "nearest",
        "preprocessing": preprocessing,
        "gaussian_sigma_mm": 1.0 if preprocessing == "lung" else None,
        "hu_range": [-1000, 400], "outside_hu": -1000,
        "candidates_sha256": file_hash(candidates), "scan_count": len(scans),
        "candidate_count": len(manifest), "simpleitk": sitk.Version_VersionString()
    }, indent=2))


def crop(volume, center_zyx, size):
    """Fixed-size, centered, air-padded crop; never stretch truncated anatomy."""
    start = np.rint(center_zyx).astype(int) - size // 2
    end = start + size
    lo, hi = np.maximum(start, 0), np.minimum(end, volume.shape)
    output = np.full((size,) * 3, -1000, dtype=np.float32)
    if np.all(hi > lo):
        dst = tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(lo, hi, start))
        src = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        output[dst] = volume[src]
    return np.clip((output + 1000) / 1400, 0, 1)


class CandidateDataset(Dataset):
    def __init__(self, rows, root, sizes=(16, 32, 48), augment=False):
        self.rows, self.root, self.sizes, self.augment = rows, Path(root), sizes, augment

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=2)
    def volume(self, name):
        return np.load(self.root / name, mmap_mode="r", allow_pickle=False)

    def __getitem__(self, index):
        row = self.rows[index]
        center = np.array([float(row[k]) for k in ("z", "y", "x")])
        positive_augmentation = self.augment and int(row["label"]) == 1
        # Apply the same spatial transform to all scales of the candidate.
        if positive_augmentation:
            center += np.random.randint(-1, 2, size=3)
        rotation = np.random.randint(4) if positive_augmentation else 0
        flips = np.random.randint(2, size=3) if positive_augmentation else [0, 0, 0]
        patches = []
        for size in self.sizes:
            patch = np.rot90(crop(self.volume(row["volume"]), center, size), rotation, axes=(1, 2))
            for axis, flip in enumerate(flips):
                if flip:
                    patch = np.flip(patch, axis)
            patches.append(torch.from_numpy(patch.copy()).unsqueeze(0))
        return patches, torch.tensor(int(row["label"]), dtype=torch.long)


def partition(rows, test_fold, validation_fold=None, *, seed=2021):
    if test_fold not in range(10) or (validation_fold is not None and (
            test_fold == validation_fold or validation_fold not in range(10))):
        raise ValueError("Test and validation folds must be different values in 0..9")
    uid_fold = {}
    for row in rows:
        uid, fold = row["seriesuid"], int(row["fold"])
        if uid in uid_fold and uid_fold[uid] != fold:
            raise ValueError("Scan occurs in multiple folds")
        uid_fold[uid] = fold
    development = sorted(uid for uid, fold in uid_fold.items() if fold != test_fold)
    if validation_fold is None:
        if len(development) < 2:
            raise ValueError("At least two development scans are required")
        count = max(1, len(development) // 5)
        validation_ids = set(np.random.default_rng(seed).permutation(development)[:count])
    else:
        validation_ids = {uid for uid, fold in uid_fold.items() if fold == validation_fold}
    train = [r for r in rows if int(r["fold"]) != test_fold and r["seriesuid"] not in validation_ids]
    val = [r for r in rows if r["seriesuid"] in validation_ids]
    test = [r for r in rows if int(r["fold"]) == test_fold]
    if not all((train, val, test)):
        raise ValueError("Train, validation and test partitions must all be nonempty")
    return train, val, test


def balanced_pools(rows, iterations, positive_repeats, seed):
    """Cycle through shuffled negatives; reuse expanded positive training pool."""
    if iterations < 1 or positive_repeats < 1:
        raise ValueError("Iterations and positive repeats must be positive")
    positives = [r for r in rows if int(r["label"]) == 1] * positive_repeats
    negatives = [r for r in rows if int(r["label"]) == 0]
    if not positives or not negatives:
        raise ValueError("Training requires both classes")
    rng = np.random.default_rng(seed)
    order, cursor = rng.permutation(len(negatives)), 0
    for _ in range(iterations):
        selected = []
        while len(selected) < len(positives):
            if cursor == len(order):
                order, cursor = rng.permutation(len(negatives)), 0
            take = min(len(positives) - len(selected), len(order) - cursor)
            selected.extend(negatives[i] for i in order[cursor:cursor + take])
            cursor += take
        pool = positives + selected
        rng.shuffle(pool)
        yield pool
