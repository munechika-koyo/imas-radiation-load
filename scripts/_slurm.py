"""Render self-contained, shell-quoted Slurm job scripts from a frozen Jinja template."""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from _configuration import Config


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
    stdout = slurm.stdout or str(directory / "logs" / f"{name}-%j.out")
    stderr = slurm.stderr or str(directory / "logs" / f"{name}-%j.err")
    for pattern in (stdout, stderr):
        # User/job-name substitutions are known before sbatch; %j remains Slurm's job ID.
        parent = Path(pattern.replace("%u", getpass.getuser()).replace("%x", job_name)).parent
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
        "output": stdout,
        "error": stderr,
        "export": "ALL",
        "chdir": str(slurm.working_directory),
    }
    for option, value in (("mem", memory), ("account", slurm.account), ("qos", slurm.qos)):
        if value is not None:
            options[option] = value
    if dependency is not None:
        options.update(dependency=f"afterok:{dependency}", **{"kill-on-invalid-dep": "yes"})
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
