from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

CLASS_NAMES = ("CFSZ", "GNSZ", "ABSZ", "CTSZ")


class NpySeizureDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        self.data = np.load(self.root / f"{split}.npy", mmap_mode="r")
        self.binary_labels = np.load(self.root / f"{split}_bi_label.npy", mmap_mode="r")
        self.class_labels = np.load(
            self.root / f"{split}_multi_label.npy", mmap_mode="r"
        )
        expected = (len(self.class_labels), 22, 2000)
        if self.data.shape != expected:
            raise ValueError(
                f"{split}.npy has shape {self.data.shape}; expected {expected}"
            )
        if self.binary_labels.shape != (len(self.class_labels), 22):
            raise ValueError(f"Invalid channel labels for {split}")
        if not np.isin(self.class_labels, range(4)).all():
            raise ValueError(f"Invalid class labels for {split}")

        counts = np.bincount(self.class_labels, minlength=4)
        print(
            f"{split}: shape={self.data.shape}, class counts={counts.tolist()}",
            flush=True,
        )

    def __getitem__(self, index):
        return (
            np.array(self.data[index], copy=True),
            np.array(self.binary_labels[index], copy=True),
            int(self.class_labels[index]),
        )

    def __len__(self):
        return len(self.data)


class ClassEpochSampler(Sampler):
    """Sample a fixed number of windows from each class per epoch."""

    def __init__(self, labels, targets, seed):
        self.seed = int(seed)
        self.epoch = 0
        labels = np.asarray(labels, dtype=np.int64)
        if isinstance(targets, dict):
            targets = [targets[class_id] for class_id in range(4)]
        self.indices = [np.flatnonzero(labels == class_id) for class_id in range(4)]
        self.counts = [
            min(len(indices), int(target))
            for indices, target in zip(self.indices, targets)
        ]

        summary = ", ".join(
            f"{name}={count}/{len(indices)}"
            for name, count, indices in zip(CLASS_NAMES, self.counts, self.indices)
        )
        print(f"Epoch sampling: {summary}", flush=True)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = []
        for indices, count in zip(self.indices, self.counts):
            if count == len(indices):
                sample = indices.copy()
                rng.shuffle(sample)
            else:
                sample = rng.choice(indices, size=count, replace=False)
            selected.append(sample)

        selected = np.concatenate(selected)
        rng.shuffle(selected)
        self.epoch += 1
        return iter(selected.tolist())

    def __len__(self):
        return sum(self.counts)


def build_loaders(
    data_root,
    batch_size,
    class_targets,
    sampler_seed,
    num_workers=0,
):
    train_set = NpySeizureDataset(data_root, "train")
    val_set = NpySeizureDataset(data_root, "val")
    sampler = ClassEpochSampler(train_set.class_labels, class_targets, sampler_seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    print(
        f"Batches per epoch: train={len(train_loader)}, val={len(val_loader)}",
        flush=True,
    )
    return train_loader, val_loader
