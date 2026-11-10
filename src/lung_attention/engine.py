"""Training and held-out prediction with explicit fold/checkpoint provenance."""
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import CandidateDataset, balanced_pools, file_hash, partition, read_rows, write_rows
from .models import MultiPatchCNN


def seed_all(seed, strict=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(strict)


def batch_to(batch, device):
    patches, labels = batch
    return [p.to(device) for p in patches], labels.to(device)


def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            patches, labels = batch_to(batch, device)
            total += F.cross_entropy(model(patches), labels, reduction="sum").item()
            count += len(labels)
    return total / count


def train(args):
    if args.batch_size < 2 or args.epochs_per_iteration < 1:
        raise ValueError("Batch size >=2 and epochs per iteration >=1 required")
    seed_all(args.seed, strict=not args.allow_nondeterministic)
    data, output = Path(args.data), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh run directory to preserve existing experiment results")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(data / "manifest.csv")
    train_rows, val_rows, test_rows = partition(rows, args.test_fold, args.validation_fold, seed=args.seed)
    spec = dict(family=args.family, attention=not args.no_attention,
                width=args.width, sizes=args.sizes)
    model = MultiPatchCNN(**spec).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    validation = DataLoader(CandidateDataset(val_rows, data, args.sizes), batch_size=args.batch_size)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(model=spec, manifest_sha256=file_hash(data / "manifest.csv"),
                  scans_sha256=file_hash(data / "scans.csv"), torch=str(torch.__version__),
                  numpy=np.__version__, python=platform.python_version(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_scans=sorted({r["seriesuid"] for r in train_rows}),
                  validation_scans=sorted({r["seriesuid"] for r in val_rows}),
                  test_scans=sorted({r["seriesuid"] for r in test_rows}))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    history, best = [], float("inf")
    for iteration, pool in enumerate(balanced_pools(train_rows, args.iterations,
                                                   args.positive_repeats, args.seed)):
        if len(pool) < args.batch_size:
            raise ValueError("Balanced pool smaller than batch; reduce --batch-size")
        loader = DataLoader(CandidateDataset(pool, data, args.sizes, augment=True),
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
        for epoch in range(args.epochs_per_iteration):
            model.train()
            total, count = 0.0, 0
            for batch in loader:
                patches, labels = batch_to(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(patches), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(labels)
                count += len(labels)
            val = validation_loss(model, validation, args.device)
            if not np.isfinite(val):
                raise ValueError("Non-finite validation loss")
            history.append(dict(iteration=iteration, epoch=epoch, training_loss=total / count,
                                validation_loss=val))
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if val < best:
                best = val
                torch.save(dict(state=model.state_dict(), spec=spec, config=config,
                                validation_loss=val), output / "best.pt")
            print(json.dumps(history[-1]), flush=True)


def predict(args):
    # Only load checkpoints you trust; weights_only avoids arbitrary pickle objects.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data = Path(args.data)
    if checkpoint["config"]["manifest_sha256"] != file_hash(data / "manifest.csv"):
        raise ValueError("Prediction manifest differs from training manifest")
    if checkpoint["config"]["scans_sha256"] != file_hash(data / "scans.csv"):
        raise ValueError("Prediction scan manifest differs from training scan manifest")
    fold = checkpoint["config"]["test_fold"]
    rows = [r for r in read_rows(data / "manifest.csv") if int(r["fold"]) == fold]
    model = MultiPatchCNN(**checkpoint["spec"])
    model.load_state_dict(checkpoint["state"])
    model.to(args.device).eval()
    dataset = CandidateDataset(rows, data, checkpoint["spec"]["sizes"])
    scores = []
    with torch.no_grad():
        for patches, _ in DataLoader(dataset, batch_size=args.batch_size):
            scores.extend(model([p.to(args.device) for p in patches]).softmax(1)[:, 1].cpu().tolist())
    output = [dict(row, probability=score) for row, score in zip(rows, scores)]
    write_rows(args.output, output, ["seriesuid", "coordX", "coordY", "coordZ", "probability"])
    Path(str(args.output) + ".json").write_text(json.dumps({
        "test_fold": fold, "manifest_sha256": file_hash(data / "manifest.csv"),
        "checkpoint_sha256": file_hash(args.checkpoint), "model": checkpoint["spec"]
    }, indent=2))
