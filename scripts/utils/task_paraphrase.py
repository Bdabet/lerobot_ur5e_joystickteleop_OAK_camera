"""Randomized language-instruction paraphrasing for policy training.

Each recorded episode has a single fixed task string baked in at record time
(stored 1:1 in the dataset's meta/tasks.parquet). Training on a small dataset
with only one phrasing per task risks the policy overfitting to that exact
wording. This module swaps in a randomly sampled, meaning-preserving
paraphrase for each `__getitem__` call, analogous to image color-jitter but
for the language instruction.
"""

import random
from pathlib import Path

import yaml


def load_paraphrase_map(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {task: list(alternatives) for task, alternatives in data.items()}


class ParaphrasedDataset:
    """Wraps a dataset so `__getitem__` resamples `item["task"]` on every call.

    Composition rather than subclassing `type(dataset)`: `LeRobotDataset`'s
    object layout isn't compatible with a `__class__` reassignment trick, so
    this instead delegates every other attribute access to the wrapped
    dataset via `__getattr__` (needed by the training loop for `.meta`,
    `.num_frames`, `.num_episodes`, etc.).
    """

    def __init__(self, dataset, paraphrase_map: dict[str, list[str]], seed: int | None = None):
        self._dataset = dataset
        self._paraphrase_map = paraphrase_map
        self._paraphrase_rng = random.Random(seed)

    def __getattr__(self, name):
        # Guard against recursion when `_dataset` itself isn't set yet (e.g.
        # during unpickling, before `__init__`/`__setstate__` has run) and
        # avoid intercepting dunder lookups (pickle probes __getstate__ etc.).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        alternatives = self._paraphrase_map.get(item["task"])
        if alternatives:
            item["task"] = self._paraphrase_rng.choice(alternatives)
        return item


def wrap_dataset_with_task_paraphrases(dataset, paraphrase_map: dict[str, list[str]], seed: int | None = None):
    return ParaphrasedDataset(dataset, paraphrase_map, seed=seed)
