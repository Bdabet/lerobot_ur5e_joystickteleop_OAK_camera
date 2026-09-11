"""Thin wrapper around lerobot's stock training entrypoint that injects
random task-instruction paraphrasing (see scripts/utils/task_paraphrase.py)
without forking the training loop.

Usage matches `lerobot-train` exactly, e.g.:

    ur5e-train-smolvla \\
        --dataset.repo_id=<your_dataset_id> \\
        --policy.type=smolvla \\
        --policy.pretrained_path=lerobot/smolvla_base \\
        --output_dir=outputs/train/smolvla_ur5e \\
        --job_name=smolvla_ur5e
"""

from pathlib import Path

from lerobot.datasets.factory import make_dataset as _make_dataset
from lerobot.scripts import lerobot_train

from scripts.utils.state_subset import DEFAULT_STATE_SUBSET, restrict_state_dims
from scripts.utils.task_paraphrase import load_paraphrase_map, wrap_dataset_with_task_paraphrases

PARAPHRASE_MAP_PATH = Path(__file__).resolve().parents[2] / "configs" / "task_paraphrases.yaml"
PARAPHRASE_MAP = load_paraphrase_map(PARAPHRASE_MAP_PATH)


def make_dataset_with_paraphrases(cfg):
    dataset = _make_dataset(cfg)
    # lerobot/smolvla_base expects state vectors <=32 dims; the UR5e dataset's
    # observation.state is 48 dims (full sensor telemetry). Restrict to the
    # TCP pose + gripper subset that matches the dataset's TCP-delta action
    # space, before wrapping with task paraphrasing.
    dataset = restrict_state_dims(dataset, DEFAULT_STATE_SUBSET)
    return wrap_dataset_with_task_paraphrases(dataset, PARAPHRASE_MAP)


# `train()` calls the module-local name `make_dataset`, so redirecting it here
# is enough to inject these without editing the lerobot submodule.
lerobot_train.make_dataset = make_dataset_with_paraphrases


def main():
    lerobot_train.train()


if __name__ == "__main__":
    main()
