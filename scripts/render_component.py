"""Run one component of a frozen multi-component calculation, without final IDS writes."""

import os
from pathlib import Path

from _configuration import load_config
from _runtime import render_component
from _task_inputs import task_inputs


def main() -> int:
    values = task_inputs()
    directory = Path(values["RADLOAD_RUN_DIR"]).expanduser().resolve()
    config = load_config(directory / "config.toml")
    workers = int(values["RADLOAD_WORKERS"] or os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    return 0 if render_component(config, values["RADLOAD_COMPONENT"], directory, workers) else 2


if __name__ == "__main__":
    raise SystemExit(main())
