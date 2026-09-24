import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import build_loaders
from model.s4_itnet import S4iTNet

SELECTION_RULES = {
    "macro_f1": ("max", "macro_f1"),
    "weighted_f1": ("max", "weighted_f1"),
    "accuracy": ("max", "accuracy"),
    "loss": ("min", "classification_loss"),
}

STAGE1_SELECTION_RULES = {
    "channel_f1": ("max", "detection_f1"),
    "channel_accuracy": ("max", "detection_accuracy"),
    "detection_loss": ("min", "detection_loss"),
}


def classification_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=precision + recall > 0,
    )
    total = support.sum()
    weights = support / total if total else np.zeros_like(support)
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": f1.tolist(),
    }


def sanitize_s4_gradients(model, max_bad_fraction):
    """Replace sparse non-finite S4 gradients and reject larger failures."""
    replaced = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not parameter.requires_grad:
            continue
        finite = torch.isfinite(parameter.grad)
        if bool(finite.all()):
            continue

        bad = int((~finite).sum().item())
        fraction = bad / parameter.grad.numel()
        is_s4_parameter = ".s4." in f".{name}." or name.startswith("s4.")
        if not is_s4_parameter or fraction > max_bad_fraction:
            return replaced, name, bad, parameter.grad.numel()
        parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        replaced += bad
    return replaced, None, 0, 0


class CheckpointSelector:
    def __init__(self, output_dir, rules=None, summary_name="selection.json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rules = rules or SELECTION_RULES
        self.summary_name = summary_name
        self.best = {
            name: float("-inf") if mode == "max" else float("inf")
            for name, (mode, _) in self.rules.items()
        }
        self.epochs = {name: 0 for name in self.rules}

    def update(self, epoch, metrics, confusion, model):
        for name, (mode, field) in self.rules.items():
            value = float(metrics[field])
            improved = (
                value > self.best[name] if mode == "max" else value < self.best[name]
            )
            if not improved:
                continue

            self.best[name] = value
            self.epochs[name] = epoch
            torch.save(model.state_dict(), self.output_dir / f"best_by_{name}.pth")
            np.save(
                self.output_dir / f"best_by_{name}_val_confusion_matrix.npy",
                confusion,
            )
            print(f"  saved best {name}: {value:.6f} (epoch {epoch})")

    def write_summary(self):
        summary = {
            name: {"epoch": self.epochs[name], "value": self.best[name]}
            for name in self.rules
            if self.epochs[name] > 0
        }
        (self.output_dir / self.summary_name).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


class S4iTNetExperiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
        self.model = S4iTNet(args).float().to(self.device)
        self.train_loader, self.val_loader = build_loaders(
            args.data_root,
            args.batch_size,
            args.epoch_class_targets,
            args.sampler_seed,
            args.num_workers,
        )
        self.classification_loss = nn.CrossEntropyLoss()
        self.detection_loss = nn.BCEWithLogitsLoss()
        self.validation_detection_loss = nn.BCELoss()

    def _train_epoch(self, stage, optimizer, max_updates, updates):
        self.model.train()
        detection_losses = []
        classification_losses = []
        true_binary, pred_binary = [], []
        true_class, pred_class = [], []

        for batch_index, (signals, binary_targets, class_targets) in enumerate(
            self.train_loader, start=1
        ):
            if max_updates and updates >= max_updates:
                break

            signals = signals.float().to(self.device)
            binary_targets = binary_targets.float().to(self.device)
            class_targets = class_targets.long().to(self.device)
            if not torch.isfinite(signals).all():
                raise ValueError(f"Non-finite input at training batch {batch_index}")

            optimizer.zero_grad(set_to_none=True)
            binary_logits, class_logits = self.model.forward_with_logits(signals)
            if (
                not torch.isfinite(binary_logits).all()
                or not torch.isfinite(class_logits).all()
            ):
                print(f"  skipped non-finite output at batch {batch_index}")
                continue

            detection_loss = self.detection_loss(binary_logits, binary_targets)
            class_loss = self.classification_loss(class_logits, class_targets)
            loss = detection_loss if stage == 1 else class_loss
            if not torch.isfinite(loss):
                print(f"  skipped non-finite loss at batch {batch_index}")
                continue

            loss.backward()
            replaced, bad_name, bad_count, total = sanitize_s4_gradients(
                self.model, self.args.s4_grad_max_bad_frac
            )
            if bad_name is not None:
                print(
                    f"  skipped invalid gradient in {bad_name} "
                    f"({bad_count}/{total} values)"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            if replaced and batch_index % 50 == 0:
                print(f"  replaced {replaced} non-finite S4 gradient values")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            updates += 1

            detection_losses.append(detection_loss.item())
            classification_losses.append(class_loss.item())
            true_binary.extend(binary_targets.detach().cpu().view(-1).int().tolist())
            pred_binary.extend(
                (binary_logits.detach().sigmoid() > 0.5).cpu().view(-1).int().tolist()
            )
            true_class.extend(class_targets.detach().cpu().tolist())
            pred_class.extend(class_logits.detach().argmax(dim=1).cpu().tolist())

        return self._summarize(
            detection_losses,
            classification_losses,
            true_binary,
            pred_binary,
            true_class,
            pred_class,
        ), updates

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        detection_losses = []
        classification_losses = []
        true_binary, pred_binary = [], []
        true_class, pred_class = [], []

        for batch_index, (signals, binary_targets, class_targets) in enumerate(
            self.val_loader, start=1
        ):
            signals = signals.float().to(self.device)
            binary_targets = binary_targets.float().to(self.device)
            class_targets = class_targets.long().to(self.device)
            if not torch.isfinite(signals).all():
                raise ValueError(f"Non-finite input at validation batch {batch_index}")
            binary_logits, class_logits = self.model.forward_with_logits(signals)
            if (
                not torch.isfinite(binary_logits).all()
                or not torch.isfinite(class_logits).all()
            ):
                raise RuntimeError("Non-finite output encountered during validation")

            detection_losses.append(
                self.validation_detection_loss(
                    binary_logits.sigmoid(), binary_targets
                ).item()
            )
            classification_losses.append(
                self.classification_loss(class_logits, class_targets).item()
            )
            true_binary.extend(binary_targets.cpu().view(-1).int().tolist())
            pred_binary.extend(
                (binary_logits.sigmoid() > 0.5).cpu().view(-1).int().tolist()
            )
            true_class.extend(class_targets.cpu().tolist())
            pred_class.extend(class_logits.argmax(dim=1).cpu().tolist())

        return self._summarize(
            detection_losses,
            classification_losses,
            true_binary,
            pred_binary,
            true_class,
            pred_class,
        )

    @staticmethod
    def _summarize(
        detection_losses,
        classification_losses,
        true_binary,
        pred_binary,
        true_class,
        pred_class,
    ):
        class_confusion = confusion_matrix(true_class, pred_class, labels=range(4))
        summary = classification_metrics(class_confusion)
        summary.update(
            {
                "detection_loss": float(np.mean(detection_losses)),
                "detection_accuracy": accuracy_score(true_binary, pred_binary),
                "detection_precision": precision_score(
                    true_binary, pred_binary, zero_division=0
                ),
                "detection_recall": recall_score(
                    true_binary, pred_binary, zero_division=0
                ),
                "detection_f1": f1_score(true_binary, pred_binary, zero_division=0),
                "classification_loss": float(np.mean(classification_losses)),
                "class_confusion": class_confusion,
                "channel_confusion": confusion_matrix(true_binary, pred_binary, labels=[0, 1]),
            }
        )
        return summary

    def train(self, run_name="model"):
        output_dir = Path(self.args.checkpoints) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        selector = CheckpointSelector(output_dir / "selection_checkpoints")
        stage1_selector = CheckpointSelector(
            output_dir / "stage1_selection_checkpoints", STAGE1_SELECTION_RULES
        )
        metrics_path = output_dir / "metrics.csv"

        fields = [
            "epoch",
            "stage",
            "train_detection_loss",
            "train_classification_loss",
            "val_detection_loss",
            "val_detection_accuracy",
            "val_detection_f1",
            "val_classification_loss",
            "val_accuracy",
            "val_weighted_f1",
            "val_macro_f1",
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

        stage = 1
        self.model.freeze_classifier()
        optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.train_epochs)
        best_detection_loss = float("inf")
        best_classification_loss = float("inf")
        stalled_epochs = 0
        updates = 0

        for epoch in range(1, self.args.train_epochs + 1):
            train_metrics, updates = self._train_epoch(
                stage, optimizer, self.args.max_updates, updates
            )
            scheduler.step()
            val_metrics = self.validate()

            print(
                f"Epoch {epoch:03d} stage={stage} "
                f"val_loss={val_metrics['classification_loss']:.6f} "
                f"val_wF1={val_metrics['weighted_f1']:.4f} "
                f"val_macroF1={val_metrics['macro_f1']:.4f}"
            )
            row = {
                "epoch": epoch,
                "stage": stage,
                "train_detection_loss": train_metrics["detection_loss"],
                "train_classification_loss": train_metrics["classification_loss"],
                "val_detection_loss": val_metrics["detection_loss"],
                "val_detection_accuracy": val_metrics["detection_accuracy"],
                "val_detection_f1": val_metrics["detection_f1"],
                "val_classification_loss": val_metrics["classification_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)

            if stage == 1:
                stage1_selector.update(
                    epoch, val_metrics, val_metrics["channel_confusion"], self.model
                )
                current = val_metrics["detection_loss"]
                if current < best_detection_loss - 1e-8:
                    best_detection_loss = current
                    stalled_epochs = 0
                else:
                    stalled_epochs += 1

                if stalled_epochs >= self.args.freeze_patience:
                    if getattr(self.args, "stage1_only", False):
                        print("Stopping Stage 1-only training after detection-loss early stopping")
                        break
                    stage = 2
                    self.model.unfreeze_classifier()
                    self.model.enable_finetuning()
                    optimizer = AdamW(
                        self.model.parameters(),
                        lr=self.args.learning_rate * 0.5,
                        weight_decay=self.args.weight_decay,
                    )
                    remaining = max(1, self.args.train_epochs - epoch)
                    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
                    best_classification_loss = float("inf")
                    stalled_epochs = 0
                    print("Starting Stage 2")
            else:
                selector.update(
                    epoch,
                    val_metrics,
                    val_metrics["class_confusion"],
                    self.model,
                )
                current = val_metrics["classification_loss"]
                if current < best_classification_loss - 1e-8:
                    best_classification_loss = current
                    stalled_epochs = 0
                else:
                    stalled_epochs += 1
                if stalled_epochs >= self.args.stage2_patience:
                    print("Stopping after Stage 2 validation loss stopped improving")
                    break

            if self.args.max_updates and updates >= self.args.max_updates:
                print(f"Stopping after {updates} optimizer updates")
                break

        selector.write_summary()
        stage1_selector.write_summary()
        torch.save(self.model.state_dict(), output_dir / "final_model.pth")
        return self.model
