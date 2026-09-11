"""Restrict `observation.state` to a subset of named dims before training.

SmolVLA's pretrained checkpoint (`lerobot/smolvla_base`) was trained on
small-arm (so100/so101) datasets with compact, Cartesian/TCP-style
proprioceptive state, and its state-input projection only pads vectors up to
`max_state_dim=32` — it never truncates. The UR5e dataset's `observation.state`
is 48 dims (full sensor telemetry: TCP pose/speed/force/accel, gripper, and
joint pos/vel/acc/force), which is both too wide for the pretrained checkpoint
and far more than what a Cartesian-delta-action policy needs.

`DEFAULT_STATE_SUBSET` keeps only the TCP pose + gripper position, mirroring
the dataset's 7-dim TCP-delta action space (`delta_x/y/z/rx/ry/rz,
gripper_position`) and staying close to what SmolVLA actually saw in
pretraining, rather than raw force/velocity/acceleration telemetry it never
saw.
"""

DEFAULT_STATE_SUBSET = [
    "tcp_pose.x",
    "tcp_pose.y",
    "tcp_pose.z",
    "tcp_pose.rx",
    "tcp_pose.ry",
    "tcp_pose.rz",
    "gripper_raw_position",
]


class StateSubsetDataset:
    """Wraps a dataset so `item[state_key]` is sliced to `indices` on every call.

    Composition (not subclassing), matching `ParaphrasedDataset` in
    `task_paraphrase.py` — delegates all other attribute access to the
    wrapped dataset via `__getattr__`.
    """

    def __init__(self, dataset, indices: list[int], state_key: str = "observation.state"):
        self._dataset = dataset
        self._indices = indices
        self._state_key = state_key

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item[self._state_key] = item[self._state_key][..., self._indices]
        return item


def restrict_state_dims(
    dataset, selected_names: list[str] = DEFAULT_STATE_SUBSET, state_key: str = "observation.state"
):
    """Mutate `dataset.meta.features`/`.stats` to the selected dims and return a wrapped dataset."""
    feature = dataset.meta.features[state_key]
    original_names = feature["names"]
    indices = [original_names.index(name) for name in selected_names]

    stats = dataset.meta.stats.get(state_key)
    if stats is not None:
        for stat_key, arr in stats.items():
            if hasattr(arr, "shape") and arr.shape and arr.shape[0] == len(original_names):
                stats[stat_key] = arr[indices]

    feature["shape"] = (len(indices),)
    feature["names"] = list(selected_names)

    return StateSubsetDataset(dataset, indices, state_key=state_key)
