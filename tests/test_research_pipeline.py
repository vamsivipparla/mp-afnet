import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from lung_attention.data import crop, partition, balanced_pools, prepare, read_rows, write_rows
from lung_attention.models import AttentionGate3D, MultiPatchCNN
from lung_attention.fusion import fuse, hybrid_score
from lung_attention.engine import train, predict
from lung_attention.evaluation import merge_predictions

torch.set_num_threads(1)


@pytest.mark.parametrize("family", ["custom", "residual", "dense"])
def test_multiscale_forward_backward_and_reload(family):
    model = MultiPatchCNN(family=family, width=0.0625)
    patches = [torch.randn(2, 1, s, s, s) for s in (16, 32, 48)]
    logits = model(patches)
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    gates = [m for m in model.modules() if isinstance(m, AttentionGate3D)]
    assert gates and all(m.score.weight.grad is not None for m in gates)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    copy = MultiPatchCNN(**model.spec)
    copy.load_state_dict(model.state_dict())
    model.eval(); copy.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(patches), copy(patches))


def test_attention_multiresolution_and_ablation():
    gate = AttentionGate3D(4)
    x = torch.ones(2, 4, 8, 8, 8)
    output = gate(x, torch.randn(2, 4, 2, 2, 2))
    assert output.shape == x.shape
    assert (output >= 0).all() and (output <= 1).all()
    model = MultiPatchCNN("dense", attention=False, width=0.0625, sizes=(16,))
    assert not any(isinstance(m, AttentionGate3D) for m in model.modules())
    assert model([torch.randn(2, 1, 16, 16, 16)]).shape == (2, 2)
    with pytest.raises(ValueError):
        model([])


def test_padding_preserves_candidate_and_intensity():
    volume = np.full((8, 8, 8), -1000.0)
    volume[0, 0, 0] = 400
    patch = crop(volume, [0, 0, 0], 16)
    assert patch.shape == (16, 16, 16)
    assert patch[8, 8, 8] == 1 and patch.sum() == 1
    assert crop(volume, [-100, 0, 0], 16).sum() == 0


def test_fixed_partitions_and_balanced_coverage():
    rows = [dict(seriesuid=f"s{f}", fold=str(f), label=str(y)) for f in range(3) for y in (0, 1)]
    training, val, test = partition(rows, 0, 1)
    assert {r["seriesuid"] for r in training} == {"s2"}
    assert {r["seriesuid"] for r in val}.isdisjoint(r["seriesuid"] for r in test)
    with pytest.raises(ValueError):
        partition(rows + [dict(seriesuid="s0", fold="2", label="1")], 0, 1)
    pool_rows = [dict(label="1", id="p")] + [dict(label="0", id=str(i)) for i in range(6)]
    pools = list(balanced_pools(pool_rows, 3, 2, 7))
    assert all(sum(int(r["label"]) for r in p) == 2 for p in pools)
    assert {r["id"] for p in pools for r in p if r["label"] == "0"} == set(map(str, range(6)))


def test_validation_uses_twenty_percent_of_development_scans():
    rows = [dict(seriesuid=f"scan{i}", fold=str(i % 10), label=str(label))
            for i in range(100) for label in (0, 1)]
    training, validation, test = partition(rows, 0)
    groups = [{row["seriesuid"] for row in split} for split in (training, validation, test)]
    assert [len(group) for group in groups] == [72, 18, 10]
    assert len(set.union(*groups)) == 100
    _, reversed_validation, _ = partition(list(reversed(rows)), 0)
    assert {row["seriesuid"] for row in reversed_validation} == groups[1]


def test_fusion_aligns_keys_and_rejects_mismatch(tmp_path):
    assert hybrid_score([0.482, 0.997, 0.993]) == pytest.approx(0.995)
    assert hybrid_score([0.5, 0.4, 0.9]) == pytest.approx(0.45)
    assert hybrid_score([0.001, 0.0006, 0.984]) == pytest.approx(0.0008)
    with pytest.raises(ValueError):
        hybrid_score([float("nan"), 0.1, 0.2])
    fields = ["seriesuid", "coordX", "coordY", "coordZ", "probability"]
    a = dict(zip(fields, ["a", 0, 0, 0, 0.9]))
    b = dict(zip(fields, ["b", 0, 0, 0, 0.1]))
    paths = [tmp_path / f"{i}.csv" for i in range(3)]
    for i, path in enumerate(paths):
        write_rows(path, [a, b] if i == 0 else [b, a], fields)
    fuse(paths, tmp_path / "fused.csv")
    assert float(read_rows(tmp_path / "fused.csv")[0]["probability"]) == pytest.approx(0.9)
    write_rows(paths[1], [a], fields)
    with pytest.raises(ValueError):
        fuse(paths, tmp_path / "bad.csv")


def test_real_cli_stages_on_synthetic_ct(tmp_path):
    """Exercise lung masking, physical transforms, training, and held-out inference."""
    candidates = []
    z, y, x = np.indices((32, 64, 72))
    volume = np.full(z.shape, -1000, dtype=np.float32)
    volume[((x - 36) / 32)**2 + ((y - 32) / 28)**2 < 1] = 40
    for side in (23, 49):
        lung = ((x - side) / 10)**2 + ((y - 32) / 15)**2 + ((z - 16) / 10)**2 < 1
        volume[lung] = -800
    volume[15:18, 31:34, 22:25] = 25
    for fold in range(3):
        directory = tmp_path / "raw" / f"subset{fold}"
        directory.mkdir(parents=True)
        image = sitk.GetImageFromArray(volume)
        image.SetOrigin((50, -20, 100))
        image.SetDirection((-1., 0., 0., 0., 1., 0., 0., 0., -1.))
        image.SetSpacing((1., 1., 2.))
        sitk.WriteImage(image, str(directory / f"scan{fold}.mhd"))
        for i, label in enumerate((0, 1)):
            point = image.TransformIndexToPhysicalPoint((22 + i, 32, 16))
            candidates.append(dict(seriesuid=f"scan{fold}", coordX=point[0], coordY=point[1],
                                   coordZ=point[2], **{"class": label}))
    write_rows(tmp_path / "candidates.csv", candidates, ["seriesuid", "coordX", "coordY", "coordZ", "class"])
    data = tmp_path / "data"
    prepare(tmp_path / "raw", tmp_path / "candidates.csv", data)
    rows = read_rows(data / "manifest.csv")
    assert float(rows[0]["x"]) == 22 and float(rows[0]["z"]) == 32
    prepared = np.load(data / "scan0.npy")
    assert prepared[32, 32, 23] > -500  # Solid tissue inside the lung mask.
    assert -900 < prepared[32, 32, 49] < -700
    assert prepared[32, 8, 36] == -1000  # Tissue outside the lungs.
    args = argparse.Namespace(data=str(data), output=str(tmp_path / "run"), family="dense",
                              test_fold=0, validation_fold=1, iterations=1, epochs_per_iteration=1,
                              positive_repeats=1, batch_size=2, learning_rate=0.001,
                              seed=7, device="cpu", width=0.0625, sizes=[16, 32, 48], no_attention=False,
                              allow_nondeterministic=False)
    train(args)
    checkpoint = tmp_path / "run" / "best.pt"
    assert checkpoint.exists()
    predict(argparse.Namespace(data=str(data), checkpoint=str(checkpoint),
                               output=str(tmp_path / "pred.csv"), device="cpu", batch_size=1))
    predictions = read_rows(tmp_path / "pred.csv")
    assert len(predictions) == 2 and {r["seriesuid"] for r in predictions} == {"scan0"}
    assert all(0 <= float(r["probability"]) <= 1 for r in predictions)
    with pytest.raises(ValueError):
        merge_predictions([tmp_path / "pred.csv"], data, tmp_path / "all.csv")
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["train_scans"] == ["scan2"] and config["validation_scans"] == ["scan1"]
