"""Config loading for the Tool Delta prediction benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml and resolve relative paths against its own directory.

    Args:
        path: Path to config.yaml.

    Returns:
        Config dict with ``data.v3_results``, ``output.tables_dir``, and
        ``output.logs_dir`` resolved to absolute paths.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base = path.parent
    cfg["data"]["v3_results"] = (base / cfg["data"]["v3_results"]).resolve()
    cfg["output"]["tables_dir"] = (base / cfg["output"]["tables_dir"]).resolve()
    cfg["output"]["logs_dir"] = (base / cfg["output"]["logs_dir"]).resolve()
    return cfg
