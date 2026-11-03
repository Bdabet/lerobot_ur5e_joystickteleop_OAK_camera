"""Point lerobot's data root at record.storage.data_root from cfg.yaml.

lerobot reads HF_LEROBOT_HOME once, when lerobot.utils.constants is first
imported, so this must run before any lerobot import. Importing the `scripts`
package (e.g. `python -m scripts.core.run_record`) triggers it. An
HF_LEROBOT_HOME already set in the shell takes precedence over the config.
"""
import os
from pathlib import Path

import yaml


def _apply_data_root() -> None:
    if "HF_LEROBOT_HOME" in os.environ:
        return
    try:
        with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        data_root = cfg["record"]["storage"].get("data_root")
    except (OSError, KeyError, AttributeError, yaml.YAMLError):
        return
    if data_root:
        os.environ["HF_LEROBOT_HOME"] = str(Path(data_root).expanduser())


_apply_data_root()
