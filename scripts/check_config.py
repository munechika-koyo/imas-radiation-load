"""Inspect TOML settings without loading external input data."""

from pathlib import Path

from _configuration import load_config, save_config
from _task_inputs import optional_path, task_inputs


def main() -> int:
    """Print the resolved configuration identity and optionally save a snapshot."""
    values = task_inputs()
    config = load_config(Path(values["RADLOAD_CONFIG"]))
    output = optional_path(values["RADLOAD_OUTPUT"])
    if output is not None:
        save_config(config, output, overwrite=False)
    print(f"Valid configuration: {config.source}; emitter.kind={config.emitter.kind}")
    if config.emitter.kind == "plasma_total":
        print("Configuration support only; rendering, submission and merge are not implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
