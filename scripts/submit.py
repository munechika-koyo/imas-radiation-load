"""Generate concrete Slurm scripts and optionally submit component and merge jobs."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _configuration import (
    hdf5_entry_path,
    load_config,
    require_runtime,
    save_config,
    validate_config,
)
from _slurm import job_log_paths, load_template, write_job
from _submission import claim_submission, submission_path
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
    record: dict[str, Any] = dict(
        schema_version=1,
        status="generating",
        created_at=datetime.now(UTC).isoformat(),
        config=str(config.source),
        output_uri=config.output.uri,
        output_occurrence=config.output.occurrence,
        run_directory=str(directory),
        max_passes=config.render.max_passes,
        cluster=os.environ.get("SLURM_CLUSTER_NAME") or None,
        components=[],
        merge=None,
    )
    persistent = submission_path(config.output.uri) if submit else None
    if persistent is not None:
        if persistent.is_relative_to(directory):
            raise ValueError("The submission record must be outside the run directory")
        claim_submission(persistent, record)
        print(f"Submission record: {persistent}", flush=True)
    print(f"Work directory: {directory}; final output: {config.output.uri}", flush=True)

    def save_record() -> None:
        record["updated_at"] = datetime.now(UTC).isoformat()
        # The merge may already have removed the temporary run when sbatch returns.
        # Store job IDs durably first, independently of that directory's lifetime.
        if persistent is not None:
            write_json(persistent, record)
        try:
            write_json(directory / "jobs.json", record)
        except FileNotFoundError:
            if not hdf5_entry_path(config.output.uri).is_dir():
                raise

    def job_record(name: str, script: Path, component: str | None = None) -> dict[str, Any]:
        stdout, stderr = job_log_paths(config, directory, name)
        return dict(
            component=component,
            script=str(script),
            job_id=None,
            cluster=record["cluster"],
            stdout=str(stdout),
            stderr=str(stderr),
        )

    def launch(job: dict[str, Any]) -> None:
        script = Path(job["script"])
        command = ["sbatch", "--parsable", str(script)]
        print(shlex.join(command), flush=True)
        response = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        ).stdout.strip()
        if not re.fullmatch(r"[0-9]+(?:;[^;\s]+)?", response):
            raise RuntimeError(f"Unexpected sbatch response: {response!r}")
        job_id, _, cluster = response.partition(";")
        job.update(job_id=job_id, submitted_at=datetime.now(UTC).isoformat())
        if cluster:
            job["cluster"] = cluster
            record["cluster"] = record["cluster"] or cluster
        for key in ("stdout", "stderr"):
            job[key] = job[key].replace("%j", job_id)

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
            record["components"].append(job_record(f"render_{component}", script, component))
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
            merge=job_record("merge", merge_script),
        )
        save_record()
        print(f"Scripts saved to: {directory / 'jobs'}", flush=True)
        if not submit:
            print(
                "Preview only: merge.sh contains a pending dependency placeholder; no jobs submitted.",
                flush=True,
            )
            return 0
        record["status"] = "submitting"
        for job in record["components"]:
            launch(job)
            save_record()
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
        save_record()
        launch(record["merge"])
        record["status"] = "submitted"
        save_record()
        print(f"Submitted: {len(components)} component jobs + one afterok merge job", flush=True)
        return 0
    except Exception as error:
        record.update(
            status="submission_failed" if submit else "generation_failed", error=str(error)
        )
        save_record()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
