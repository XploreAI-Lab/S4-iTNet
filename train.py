"""Train S4-iTNet from preprocessed training and validation arrays."""

import argparse
import json
from pathlib import Path

from config import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke", action="store_true", help="Run two CPU-sized optimization updates"
    )
    parser.add_argument("--stage1-only", action="store_true", help="Train only the auxiliary detection task")
    args = parser.parse_args()
    model_args, config = configure(args.config, args.device)
    model_args.stage1_only = args.stage1_only
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    model_args.data_root = str(args.data_root.resolve())
    if args.smoke:
        model_args.batch_size = 1
        model_args.train_epochs = 1
        model_args.max_updates = 2
        model_args.epoch_class_targets = [1, 1, 1, 1]
    args.output.mkdir(parents=True, exist_ok=True)
    model_args.checkpoints = str(args.output.resolve())
    config["runtime"] = {
        "arguments": vars(model_args),
        "smoke": args.smoke,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    from experiments.trainer import S4iTNetExperiment

    experiment = S4iTNetExperiment(model_args)
    experiment.train("model")


if __name__ == "__main__":
    main()
