"""Render self-contained, shell-quoted Slurm job scripts from a frozen Jinja template."""

from __future__ import annotations

import getpass
import json
import math
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from _configuration import Config


def collect_slurm_timings(
    directory: Path, components: list[str], *, timeout: float | None
) -> dict[str, dict[str, Any]]:
    """Fetch finished allocation times once per cluster; missing accounting is optional."""
    if timeout is None or not (directory / "jobs.json").is_file():
        return {}
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Slurm accounting timeout must be finite and positive")
    executable = shutil.which("sacct")
    if executable is None:
        print(
            "Slurm timing unavailable: sacct is not installed; keeping worker timings.", flush=True
        )
        return {}
    try:
        record = json.loads((directory / "jobs.json").read_text())
        since = (
            datetime.fromisoformat(record["created_at"])
            .astimezone(UTC)
            .strftime("%Y-%m-%dT%H:%M:%S")
        )
        groups: dict[str | None, dict[str, str]] = {}
        for job in record["components"]:
            if job["component"] in components and job["job_id"] is not None:
                cluster = job.get("cluster") or record.get("cluster")
                groups.setdefault(cluster, {})[str(job["job_id"])] = job["component"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Slurm timing unavailable: cannot read submission record: {error}", flush=True)
        return {}
    result: dict[str, dict[str, Any]] = {}
    env = dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="standard")
    for cluster, jobs in groups.items():
        command = [
            executable,
            *(["--clusters", cluster] if cluster else ["--local"]),
            "--allocations",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(jobs),
            "--starttime",
            since,
            "--format=JobIDRaw,State%80,Start,End,ElapsedRaw",
        ]
        try:
            response = subprocess.run(
                command, env=env, capture_output=True, text=True, check=True, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Slurm timing unavailable: {error}; keeping worker timings.", flush=True)
            continue
        for line in response.stdout.splitlines():
            fields = [value.strip() for value in line.split("|")]
            if len(fields) != 5:
                continue
            job_id, state, started, finished, elapsed = fields
            if job_id not in jobs or not elapsed.isdecimal():
                continue
            try:
                started_at = datetime.fromisoformat(started).replace(tzinfo=UTC)
                finished_at = datetime.fromisoformat(finished).replace(tzinfo=UTC)
            except ValueError:
                # Accounting may not have received the job's final update yet.
                continue
            result[jobs[job_id]] = dict(
                job_id=job_id,
                cluster=cluster,
                state=state,
                started_at=started_at.isoformat(),
                finished_at=finished_at.isoformat(),
                elapsed_seconds=int(elapsed),
            )
    missing = sorted(set(components) - result.keys())
    if missing:
        print(
            f"Slurm timing unavailable for: {', '.join(missing)}; keeping worker timings.",
            flush=True,
        )
    return result


def load_template(path: Path) -> Template:
    environment = Environment(
        loader=FileSystemLoader(path.parent),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["quote"] = lambda value: shlex.quote(str(value))
    return environment.get_template(path.name)


def job_log_paths(config: Config, directory: Path, name: str) -> tuple[Path, Path]:
    """Resolve known log substitutions, leaving the job ID for sbatch's response."""
    job_name = f"radload-{name}"
    patterns = (
        config.slurm.stdout or str(directory / "logs" / f"{name}-%j.out"),
        config.slurm.stderr or str(directory / "logs" / f"{name}-%j.err"),
    )
    stdout, stderr = (
        Path(pattern.replace("%u", getpass.getuser()).replace("%x", job_name))
        for pattern in patterns
    )
    return stdout, stderr


def write_job(
    template: Template,
    config: Config,
    directory: Path,
    name: str,
    task: str,
    arguments: list[str],
    *,
    merge: bool = False,
    dependency: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one concrete script and check its Bash syntax before it can be submitted."""
    slurm = config.slurm
    cpus = slurm.merge.cpus_per_task if merge else slurm.cpus_per_task
    memory = (slurm.merge.memory or slurm.memory) if merge else slurm.memory
    time = (slurm.merge.time or slurm.time) if merge else slurm.time
    job_name = f"radload-{name}"
    stdout, stderr = job_log_paths(config, directory, name)
    for pattern in (stdout, stderr):
        parent = pattern.parent
        if "%" in str(parent):
            raise ValueError(
                "Slurm log parent directories may use %u/%x, but not runtime-only substitutions"
            )
        parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, str | int] = {
        "job-name": job_name,
        "nodes": 1,
        "ntasks-per-node": 1,
        "cpus-per-task": cpus,
        "partition": slurm.partition,
        "time": time,
        "output": str(stdout),
        "error": str(stderr),
        "export": "ALL",
        "chdir": str(slurm.working_directory),
    }
    for option, value in (("mem", memory), ("account", slurm.account), ("qos", slurm.qos)):
        if value is not None:
            options[option] = value
    if dependency is not None:
        options.update(dependency=dependency, **{"kill-on-invalid-dep": "yes"})
    if any(any(character in str(value) for character in "\r\n\0") for value in options.values()):
        raise ValueError("SBATCH directive values must be single-line strings")
    command = shlex.join(
        [
            "pixi",
            "run",
            "--manifest-path",
            str(slurm.working_directory / "pixi.toml"),
            task,
            *arguments,
        ]
    )
    content = template.render(
        options=options,
        working_directory=str(slurm.working_directory),
        command=command,
    )
    destination = directory / "jobs" / f"{name}.sh"
    with destination.open("w" if overwrite else "x") as stream:
        stream.write(content)
    os.chmod(destination, 0o700)
    subprocess.run(["bash", "-n", str(destination)], check=True, capture_output=True, text=True)
    return destination
