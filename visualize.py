"""Render EEG traces and detector-derived channel weights for one selected row."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import ROOT, configure
from evaluate import CLASSES
from preprocessing.edf_io import CHANNEL_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/s4_itnet_10s_wf1.pth"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    cfg, _ = configure(device=a.device)
    from model.s4_itnet import S4iTNet

    torch.set_num_threads(4)
    model = S4iTNet(cfg).to(a.device).eval()
    model.load_state_dict(
        torch.load(a.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    data = np.load(a.data_root / "test.npy", mmap_mode="r")
    if not 0 <= a.index < len(data):
        p.error("index is out of range")
    waveform = np.array(data[a.index], dtype=np.float32)
    labels = np.load(a.data_root / "test_multi_label.npy")
    ictal = np.load(a.data_root / "test_bi_label.npy")[a.index].astype(bool)
    with torch.inference_mode():
        _, logits, weights = model.forward_with_alpha(
            torch.from_numpy(waveform[None]).to(a.device)
        )
    prob = logits.softmax(-1)[0].cpu().numpy()
    weights = weights[0].cpu().numpy()
    display = (waveform - waveform.mean(-1, keepdims=True)) / (
        waveform.std(-1, keepdims=True) + 1e-6
    )
    fig, (ax, bar) = plt.subplots(
        1, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [4, 1]}, sharey=True
    )
    positions = np.arange(22)[::-1]
    colors = ["#b1252d" if active else "#242424" for active in ictal]
    for c, pos in enumerate(positions):
        ax.plot(
            np.arange(2000) / 200,
            np.clip(display[c], -3, 3) * 0.13 + pos,
            color=colors[c],
            lw=0.55,
        )
    ax.set_yticks(positions, CHANNEL_NAMES, fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.6, 21.6)
    bar.barh(positions, weights, height=0.65, color=colors)
    bar.axvline(1 / 22, color="gray", ls="--", lw=1)
    bar.set_xlabel("Channel weight")
    bar.tick_params(axis="y", left=False)
    synthetic = (a.data_root / "SYNTHETIC_DATA.json").exists()
    prefix = "Synthetic smoke sample | " if synthetic else ""
    fig.suptitle(
        f"{prefix}True: {CLASSES[int(labels[a.index])]} | Predicted: {CLASSES[int(prob.argmax())]} ({prob.max():.3f})",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "Red: annotated ictal channel. Traces standardized for display only; dashed line: 1/22.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
