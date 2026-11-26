"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the experiment configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
