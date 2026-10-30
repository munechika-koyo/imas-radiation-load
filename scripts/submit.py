"""Generate concrete Slurm scripts and optionally submit component and merge jobs."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from _configuration import (
    hdf5_entry_path,
    load_config,
    require_runtime,
    save_config,
    validate_config,
)
from _slurm import load_template, write_job
from _task_inputs import optional_path, task_inputs
from _wall_io import wall_geometry
from _workspace import create_run, write_json


def main() -> int:
    values = task_inputs()
    config = load_config(Path(values["RADLOAD_CONFIG"]))
    require_runtime(config)
    submit = values["RADLOAD_SUBMIT_MODE"] == "submit"
    config = replace(
        config,
        slurm=replace(
            config.slurm,
            cpus_per_task=int(values["RADLOAD_CPUS_PER_TASK"] or config.slurm.cpus_per_task),
            partition=values["RADLOAD_PARTITION"] or config.slurm.partition,
            time=values["RADLOAD_TIME"] or config.slurm.time,
        ),
    )
    validate_config(config)
    if not config.slurm.job_template.is_file() or not config.slurm.working_directory.is_dir():
        raise ValueError("Slurm job_template and working_directory must exist")
    geometry = wall_geometry(config)
    components = list(geometry["parts"])
    config, directory = create_run(
        config,
        geometry,
        components,
        directory=optional_path(values["RADLOAD_WORK_DIR"]),
    )
    (directory / "jobs").mkdir()
    frozen_template = directory / "job-template.sh.j2"
    shutil.copyfile(config.slurm.job_template, frozen_template)
    config = replace(config, slurm=replace(config.slurm, job_template=frozen_template))
    save_config(config, directory / "config.toml")
    template = load_template(frozen_template)
    # Generated #SBATCH directives define resources; inherited SBATCH_* defaults must not override them.
    env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    record: dict = dict(status="generating", components=[], merge=None)
    print(f"Work directory: {directory}; final output: {config.output.uri}", flush=True)

    def launch(script: Path) -> str:
        command = ["sbatch", "--parsable", str(script)]
        print(shlex.join(command), flush=True)
        response = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        ).stdout.strip()
        if not re.fullmatch(r"[0-9]+(?:;[^;\s]+)?", response):
            raise RuntimeError(f"Unexpected sbatch response: {response!r}")
        return response.split(";", 1)[0]

    try:
        for component in components:
            script = write_job(
                template,
                config,
                directory,
                f"render_{component}",
                "render-component",
                [component, str(directory), str(config.slurm.cpus_per_task)],
            )
            record["components"].append(dict(component=component, script=str(script), job_id=None))
        merge_script = write_job(
            template,
            config,
            directory,
            "merge",
            "merge",
            [str(directory)],
            merge=True,
            dependency="afterok:PENDING_COMPONENT_JOB_IDS",
        )
        record.update(
            status="ready" if submit else "preview",
            merge=dict(script=str(merge_script), job_id=None),
        )
        write_json(directory / "jobs.json", record)
        print(f"Scripts saved to: {directory / 'jobs'}", flush=True)
        if not submit:
            print(
                "Preview only: merge.sh contains a pending dependency placeholder; no jobs submitted.",
                flush=True,
            )
            return 0
        record["status"] = "submitting"
        for job in record["components"]:
            job["job_id"] = launch(Path(job["script"]))
            write_json(directory / "jobs.json", record)
        dependency = ":".join(job["job_id"] for job in record["components"])
        merge_script = write_job(
            template,
            config,
            directory,
            "merge",
            "merge",
            [str(directory)],
            merge=True,
            dependency=f"afterok:{dependency}",
            overwrite=True,
        )
        record["status"] = "components_submitted"
        write_json(directory / "jobs.json", record)
        record["merge"]["job_id"] = launch(merge_script)
        record["status"] = "submitted"
        try:
            write_json(directory / "jobs.json", record)
        except FileNotFoundError:
            if not hdf5_entry_path(config.output.uri).is_dir():
                raise
        print(f"Submitted: {len(components)} component jobs + one afterok merge job", flush=True)
        return 0
    except Exception as error:
        record.update(
            status="submission_failed" if submit else "generation_failed", error=str(error)
        )
        if directory.is_dir():
            write_json(directory / "jobs.json", record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
