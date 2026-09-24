"""Check full-size checkpoint loading, forward outputs, and stage gradient routing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import ROOT, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/smoke_report.json"
    )
    args = parser.parse_args()
    config, _ = configure(device=args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    from model.s4_itnet import S4iTNet

    model = S4iTNet(config).to(args.device)
    state = torch.load(
        ROOT / "checkpoints/s4_itnet_10s_wf1.pth", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    x = torch.from_numpy(np.load(ROOT / "dataset/smoke_demo/test.npy")[:1]).to(
        args.device
    )
    model.eval()
    with torch.no_grad():
        detection, classification, alpha = model.forward_with_alpha(x)
    assert detection.shape == (1, 22) and classification.shape == (1, 4)
    assert torch.isfinite(classification).all() and torch.isfinite(alpha).all()
    assert torch.allclose(alpha.sum(-1), torch.ones(1, device=args.device), atol=1e-6)
    model.train()
    model.freeze_classifier()
    det, cls = model.forward_with_logits(x)
    torch.nn.functional.binary_cross_entropy_with_logits(
        det, torch.zeros_like(det)
    ).backward()
    assert model.projector1.weight.grad is not None
    assert all(p.grad is None for p in model.cls_head.parameters())
    model.zero_grad(set_to_none=True)
    model.enable_finetuning()
    det, cls = model.forward_with_logits(x)
    torch.nn.functional.cross_entropy(
        cls, torch.zeros(1, dtype=torch.long, device=args.device)
    ).backward()
    assert model.projector1.weight.grad is None
    assert model.cls_head[-1].weight.grad is not None
    assert model.enc_embedding.value_embedding.weight.grad is not None
    report = {
        "passed": True,
        "device": args.device,
        "strict_checkpoint_load": True,
        "input_shape": list(x.shape),
        "stage1_classifier_frozen": True,
        "stage2_backbone_trainable": True,
        "detection_gate_detached": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
