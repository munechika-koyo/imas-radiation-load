"""Read values supplied by Pixi without defining a second command-line interface."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def task_inputs() -> Mapping[str, str]:
    """Return task environment values; Pixi owns argument definitions and defaults."""
    if len(sys.argv) != 1:
        raise SystemExit("Use the positional arguments defined in pixi.toml; extra flags are not accepted.")
    return os.environ


def optional_path(value: str) -> Path | None:
    """Interpret an empty task value as no override."""
    return Path(value).expanduser().resolve() if value else None
