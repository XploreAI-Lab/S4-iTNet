"""Load an experiment configuration and initialize its runtime settings."""

import json
import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def configure(path=None, device="cpu"):
    config = json.loads(Path(path or ROOT / "configs/s4_itnet_10s.json").read_text())
    os.environ["MPLBACKEND"] = "Agg"
    if device == "cpu":
        os.environ["S4ITNET_DISABLE_KEOPS"] = "1"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    args = Namespace(**config["model"])
    args.use_gpu = device.startswith("cuda")
    args.gpu = int(device.split(":")[1]) if ":" in device else 0
    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use --device cpu for smoke tests"
        )
    return args, config
