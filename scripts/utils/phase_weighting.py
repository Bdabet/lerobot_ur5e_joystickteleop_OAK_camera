"""Oversample frames by their recorded phase label before training.

Scripted auto_pick_place datasets carry a per-frame `phase` feature (see
`scripts/core/auto_pickplace.Phase`). Grasping and releasing are the moments
the policy most needs to get right but the fewest frames of an episode, and
PERTURB frames are deliberately wrong motion recorded only so the CORRECT
frames that follow have something to recover from. This module repeats each
frame's index `weight` times (0 drops it) so the stock shuffled DataLoader
sees the reweighted distribution without touching the training loop.
"""

from pathlib import Path

import numpy as np
import yaml

from scripts.core.auto_pickplace import Phase

PHASE_KEY = "phase"


def load_phase_weights(path: str | Path) -> dict[int, int]:
    """Read a {phase name: integer repeat count} yaml; unlisted phases get 1."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    weights = {int(phase): 1 for phase in Phase}
    for name, weight in data.items():
        if int(weight) != weight or weight < 0:
            raise ValueError(f"Phase weight for {name} must be a non-negative integer, got {weight}.")
        weights[int(Phase[name])] = int(weight)
    return weights


class PhaseWeightedDataset:
    """Wraps a dataset so index i maps to the i-th entry of an expanded index
    list in which every frame appears as many times as its phase's weight.

    Composition (not subclassing), matching `ParaphrasedDataset` in
    `task_paraphrase.py` — delegates all other attribute access to the
    wrapped dataset via `__getattr__`.
    """

    def __init__(self, dataset, indices: np.ndarray):
        self._dataset = dataset
        self._indices = indices

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        item = self._dataset[int(self._indices[idx])]
        # Bookkeeping only; keep it out of the policy's batch.
        item.pop(PHASE_KEY, None)
        return item


def weight_dataset_by_phase(dataset, weights: dict[int, int]):
    """Return dataset wrapped with phase-based oversampling, or unchanged if it
    has no phase labels (e.g. teleop-recorded or pre-phase datasets)."""
    if PHASE_KEY not in dataset.meta.features:
        return dataset

    hf_dataset = dataset.hf_dataset
    phases = np.asarray(hf_dataset.select_columns([PHASE_KEY]).with_format("numpy")[PHASE_KEY]).reshape(-1)
    repeats = np.array([weights.get(int(phase), 1) for phase in phases], dtype=np.int64)
    indices = np.repeat(np.arange(len(phases)), repeats)
    if len(indices) == 0:
        raise ValueError("Phase weights drop every frame of the dataset.")

    counts = {Phase(p).name: int(np.sum(phases == p)) for p in np.unique(phases)}
    print(
        f"[phase_weighting] frames per phase {counts}; "
        f"{len(phases)} frames -> {len(indices)} samples per epoch"
    )
    return PhaseWeightedDataset(dataset, indices)
