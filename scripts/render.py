"""Render one component in the current environment and save its IMAS wall result."""

import os
from dataclasses import replace
from pathlib import Path

from _configuration import load_config, require_runtime, resolve_output_uri, validate_config
from _runtime import render_component
from _task_inputs import task_inputs
from _wall_io import merge_results, wall_geometry
from _workspace import create_run, remove_temporary_workspace


def main() -> int:
    values = task_inputs()
    config = load_config(Path(values["RADLOAD_CONFIG"]))
    require_runtime(config)
    if values["RADLOAD_OUTPUT_URI"]:
        config = replace(
            config,
            output=replace(
                config.output, uri=resolve_output_uri(values["RADLOAD_OUTPUT_URI"], Path.cwd())
            ),
        )
    validate_config(config)
    component = values["RADLOAD_COMPONENT"]
    workers = int(values["RADLOAD_WORKERS"] or os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if workers < 1:
        raise ValueError("workers must be positive")
    geometry = wall_geometry(config)
    if component not in geometry["parts"]:
        raise ValueError(
            f"Unknown component {component!r}; available: {', '.join(geometry['parts'])}"
        )
    config, directory = create_run(config, geometry, [component])
    print(f"Work directory: {directory}; final output: {config.output.uri}", flush=True)
    if not render_component(config, component, directory, workers):
        print(f"Not converged; intermediate results retained in {directory}", flush=True)
        return 2
    print(f"Wall IDS: {merge_results(config, directory)}", flush=True)
    remove_temporary_workspace(directory, config.output.uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
