"""MP-AFNet training and evaluation for LUNA16 candidate classification."""
import argparse
from .data import prepare
from .engine import train, predict
from .fusion import fuse
from .evaluation import evaluate, merge_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for field in ("raw", "candidates", "output"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--preprocessing", choices=["lung", "resample"], default="lung")
    p.set_defaults(func=lambda a: prepare(a.raw, a.candidates, a.output, preprocessing=a.preprocessing))

    p = commands.add_parser("train")
    for field in ("data", "output"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--family", choices=["custom", "residual", "dense"], required=True)
    p.add_argument("--test-fold", type=int, required=True)
    p.add_argument("--validation-fold", type=int, choices=range(10),
                   help="Use one specific validation subset instead of a 20%% scan split")
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--epochs-per-iteration", type=int, default=3)
    p.add_argument("--positive-repeats", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default="cpu")
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 48])
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--allow-nondeterministic", action="store_true",
                   help="Allow operations without deterministic implementations; recorded in run config")
    p.set_defaults(func=train)

    p = commands.add_parser("predict")
    for field in ("data", "checkpoint", "output"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.set_defaults(func=predict)

    p = commands.add_parser("fuse")
    p.add_argument("predictions", nargs=3)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["hybrid", "mean"], default="hybrid")
    p.set_defaults(func=lambda a: fuse(a.predictions, a.output, a.method))

    p = commands.add_parser("merge")
    p.add_argument("predictions", nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=lambda a: merge_predictions(a.predictions, a.data, a.output))

    p = commands.add_parser("evaluate")
    for field in ("data", "predictions", "evaluator", "annotations", "excluded", "output"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--fold", type=int, choices=range(10))
    p.add_argument("--bootstraps", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
