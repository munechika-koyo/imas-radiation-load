"""Combine component results into one IMAS wall entry."""

from dataclasses import replace
from pathlib import Path

from _configuration import load_config, resolve_output_uri, validate_config
from _task_inputs import optional_path, task_inputs
from _wall_io import merge_results
from _workspace import remove_temporary_workspace


def main() -> int:
    """Read the run's frozen configuration and save a new result entry."""
    values = task_inputs()
    run_directory = optional_path(values["RADLOAD_RUN_DIR"])
    if run_directory is None:
        raise ValueError("run_dir must not be empty")
    config_path = optional_path(values["RADLOAD_CONFIG"]) or run_directory / "config.toml"
    config = load_config(config_path)
    if values["RADLOAD_OUTPUT_URI"]:
        config = replace(
            config,
            output=replace(
                config.output,
                uri=resolve_output_uri(values["RADLOAD_OUTPUT_URI"], Path.cwd()),
            ),
        )
    validate_config(config)
    print(f"Wall IDS: {merge_results(config, run_directory)}", flush=True)
    if remove_temporary_workspace(run_directory, config.output.uri):
        print(f"Removed temporary workspace: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
