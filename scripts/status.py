"""Read Slurm states and saved calculation progress without modifying a run."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from _configuration import hdf5_entry_path, load_config
from _submission import submission_path
from _task_inputs import task_inputs

TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)
SUBMITTING_STATES = frozenset({"generating", "ready", "submitting", "components_submitted"})
JobKey = tuple[str | None, str]


@dataclass(frozen=True)
class JobState:
    state: str = "UNKNOWN"
    elapsed: str = "-"
    exit_code: str = "-"
    reason: str = "-"
    nodes: str = "-"


def read_record(path: Path, output_uri: str) -> dict[str, Any]:
    """Validate identities before querying job IDs or reading component checkpoints."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No submission record: {path}. Use the configuration's submitted output.uri. "
            "Records are created by new submit-mode runs, not by preview or older submissions."
        )
    record = json.loads(path.read_text())
    if record.get("schema_version") != 1:
        raise ValueError(f"Unsupported submission record: {path}")
    if hdf5_entry_path(record["output_uri"]) != hdf5_entry_path(output_uri):
        raise ValueError("Submission record does not match the configured output URI")
    datetime.fromisoformat(record["created_at"])
    if not Path(record["run_directory"]).is_absolute():
        raise ValueError("Submission run_directory must be absolute")
    if not isinstance(record.get("components"), list):
        raise ValueError("Submission record must contain component jobs")
    names: set[str] = set()
    for job in record["components"]:
        name = job["component"]
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or any(char in name for char in ("/", "\\", "\0"))
            or name in names
        ):
            raise ValueError(f"Invalid or duplicate component: {name!r}")
        names.add(name)
    for job in jobs(record):
        job_id = job["job_id"]
        if job_id is not None and not re.fullmatch(r"[0-9]+", str(job_id)):
            raise ValueError(f"Invalid Slurm job ID: {job_id!r}")
        cluster = job.get("cluster") or record.get("cluster")
        if cluster is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", cluster):
            raise ValueError(f"Invalid Slurm cluster: {cluster!r}")
    return record


def jobs(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the component jobs and final merge stage in their submission order."""
    return [*record["components"], *([record["merge"]] if record.get("merge") else [])]


def job_key(job: dict[str, Any], record: dict[str, Any]) -> JobKey:
    return job.get("cluster") or record.get("cluster"), str(job["job_id"])


def query(command: list[str], timeout: float, warnings: list[str]) -> str | None:
    """Treat inaccessible Slurm services as missing information, never completion."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("SQUEUE_", "SACCT_"))
    }
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        warnings.append(f"{command[0]}: {error}")
        return None
    if result.returncode:
        warnings.append(f"{command[0]}: {result.stderr.strip() or 'query failed'}")
        return None
    return result.stdout


def query_states(
    record: dict[str, Any], timeout: float, terminal: dict[JobKey, JobState]
) -> tuple[dict[JobKey, JobState], list[str]]:
    """Batch queue/accounting queries per cluster; cache finished jobs during watch."""
    result = dict(terminal)
    warnings: list[str] = []
    groups: dict[str | None, list[str]] = defaultdict(list)
    for job in jobs(record):
        if job["job_id"] is not None:
            key = job_key(job, record)
            groups[key[0]].append(key[1])
    since = datetime.fromisoformat(record["created_at"]).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    for cluster, ids in groups.items():
        scope = ["--clusters", cluster] if cluster else ["--local"]
        queued = query(
            [
                "squeue",
                *scope,
                "--noheader",
                "--states=all",
                "--jobs",
                ",".join(ids),
                "--format=%i|%T|%M|%r|%N",
            ],
            timeout,
            warnings,
        )
        if queued is not None:
            for line in queued.splitlines():
                fields = [value.strip() for value in line.split("|")]
                if len(fields) != 5 or fields[0] not in ids:
                    if line.strip():
                        warnings.append(f"Unrecognized squeue row: {line}")
                    continue
                job_id, state, elapsed, reason, nodes = fields
                result[cluster, job_id] = JobState(
                    state=state, elapsed=elapsed, reason=reason, nodes=nodes
                )
                if state not in TERMINAL_STATES:
                    terminal.pop((cluster, job_id), None)
        # Queue entries are authoritative for active/requeued jobs. Accounting
        # supplies disappeared jobs and exit codes for terminal queue entries.
        missing = [
            job_id
            for job_id in ids
            if (cluster, job_id) not in result
            or (
                result[cluster, job_id].state in TERMINAL_STATES
                and (cluster, job_id) not in terminal
            )
        ]
        if missing:
            accounted = query(
                [
                    "sacct",
                    *scope,
                    "--allocations",
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    ",".join(missing),
                    "--starttime",
                    since,
                    "--format=JobIDRaw,State%80,Elapsed,ExitCode,NodeList,Reason",
                ],
                timeout,
                warnings,
            )
            if accounted is not None:
                for line in accounted.splitlines():
                    fields = [value.strip() for value in line.split("|")]
                    if len(fields) != 6 or fields[0] not in missing:
                        # Ignore job-step rows even if a site emits them with -X.
                        if line.strip() and not (fields and "." in fields[0]):
                            warnings.append(f"Unrecognized sacct row: {line}")
                        continue
                    job_id, state, elapsed, exit_code, nodes, reason = fields
                    state = state.split()[0].rstrip("+") if state else "UNKNOWN"
                    result[cluster, job_id] = JobState(state, elapsed, exit_code, reason, nodes)
    for key, state in result.items():
        if state.state in TERMINAL_STATES:
            terminal[key] = state
    return result, warnings


def read_progress(record: dict[str, Any], warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Prefer merged sampling records; otherwise read the last saved checkpoints."""
    final = hdf5_entry_path(record["output_uri"]) / "rendering.json"
    try:
        if final.is_file():
            return json.loads(final.read_text())["components"]
    except (OSError, ValueError, KeyError) as error:
        warnings.append(f"Cannot read merged progress: {error}")
    progress: dict[str, dict[str, Any]] = {}
    for job in record["components"]:
        name = job["component"]
        path = Path(record["run_directory"]) / name / "status.json"
        try:
            progress[name] = json.loads(path.read_text())
        except FileNotFoundError:
            pass  # The job may not have started, or merge may be cleaning up.
        except (OSError, ValueError) as error:
            warnings.append(f"Cannot read {name} checkpoint: {error}")
    return progress


def state_label(state: str, *, count: int | None = None) -> Text:
    """Keep state names readable alongside colors and terminal-width-aware emoji."""
    if state == "COMPLETED":
        icon, style = "✅", "bold green"
    elif state in {"RUNNING", "COMPLETING"}:
        icon, style = "🚀", "bold cyan"
    elif state == "PENDING":
        icon, style = "⏳", "yellow"
    elif state == "CANCELLED":
        icon, style = "🛑", "yellow"
    elif state == "TIMEOUT":
        icon, style = "⏱", "bold red"
    elif state in TERMINAL_STATES:
        icon, style = "❌", "bold red"
    elif state == "UNKNOWN":
        icon, style = "❔", "magenta"
    elif state == "NOT_SUBMITTED":
        icon, style = "➖", "dim"
    else:
        icon, style = "🔄", "blue"
    label = state if count is None else f"{state}={count}"
    return Text(f"{icon} {label}", style=style)


def render_status(
    record: dict[str, Any],
    path: Path,
    states: dict[JobKey, JobState],
    warnings: list[str],
    *,
    width: int,
    footer: Text | None = None,
) -> Group:
    """Build a Rich snapshot; never interpret data, node names or paths as markup."""
    progress = read_progress(record, warnings)
    metadata = Table.grid(padding=(0, 1), expand=True)
    metadata.add_column(style="cyan", no_wrap=True)
    metadata.add_column(ratio=1, overflow="fold")
    for label, value in (
        ("Submission", record["status"]),
        ("Created", record["created_at"]),
        ("Updated", datetime.now().astimezone().isoformat(timespec="seconds")),
    ):
        metadata.add_row(label, Text(str(value)))
    table = Table(
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="bright_black",
        expand=True,
        padding=(0, 1),
    )
    compact = width < 140
    if compact:
        table.add_column("Component / Job", style="bold", ratio=2, overflow="fold")
        table.add_column("Slurm state", ratio=2, overflow="fold")
        table.add_column("Saved pass", justify="right", ratio=1)
        table.add_column("Converged", justify="center", width=9, no_wrap=True)
        table.add_column("Details", ratio=3, overflow="fold")
    else:
        for label in (
            "Component / stage",
            "Job ID",
            "Slurm state",
            "Elapsed",
            "Saved pass",
            "Converged",
            "Exit",
            "Nodes",
            "Reason / calculation stop",
        ):
            table.add_column(label, overflow="fold")
    counts: Counter[str] = Counter()
    failed: list[dict[str, Any]] = []
    for index, job in enumerate(jobs(record)):
        state = (
            states.get(job_key(job, record), JobState())
            if job["job_id"] is not None
            else JobState(state="NOT_SUBMITTED")
        )
        counts[state.state] += 1
        name = job.get("component")
        saved = progress.get(name, {}) if name is not None else {}
        converged = saved.get("converged")
        convergence = (
            Text("✅ yes", style="green")
            if converged is True
            else Text("⚠ no", style="yellow")
            if converged is False
            else Text("—", style="dim")
        )
        render_pass = saved.get("render_pass")
        pass_text = f"{render_pass}/{record['max_passes']}" if render_pass is not None else "-"
        reason = state.reason if state.reason not in {"", "None", "(null)", "-"} else ""
        if stop := saved.get("stop_reason"):
            reason = "; ".join(filter(None, (reason, stop)))
        component = (
            Text(name, style="bold")
            if name is not None
            else Text("🔗 merge (stage)", style="bold blue")
        )
        job_id = Text(str(job["job_id"] or "-"), style="dim cyan")
        pass_label = Text(pass_text, style="cyan" if render_pass is not None else "dim")
        if name is None and index:
            table.add_section()
        if compact:
            component.append("\n")
            component.append_text(job_id)
            details = Text(f"Elapsed {state.elapsed}")
            if state.exit_code != "-":
                details.append(f" · Exit {state.exit_code}")
            if state.nodes not in {"", "None", "(null)", "-"}:
                details.append(f"\n{state.nodes}", style="dim")
            if reason:
                details.append(f"\n{reason}")
            table.add_row(component, state_label(state.state), pass_label, convergence, details)
        else:
            table.add_row(
                component,
                job_id,
                state_label(state.state),
                Text(state.elapsed),
                pass_label,
                convergence,
                Text(state.exit_code),
                Text(state.nodes or "-"),
                Text(reason or "-"),
            )
        if state.state in TERMINAL_STATES - {"COMPLETED"}:
            failed.append(job)
    summary = Text("Jobs: ", style="bold")
    for index, (state, count) in enumerate(sorted(counts.items())):
        if index:
            summary.append("  ")
        summary.append_text(state_label(state, count=count))
    if not counts:
        summary.append("none recorded yet", style="dim")
    locations = Table.grid(padding=(0, 1), expand=True)
    locations.add_column(style="dim cyan", no_wrap=True)
    locations.add_column(ratio=1, overflow="fold")
    for label, value in (
        ("Record", path),
        ("Output", record["output_uri"]),
        ("Work directory", record["run_directory"]),
    ):
        locations.add_row(label, Text(str(value), style="dim"))
    renderables = [
        Panel(metadata, title="📡 IMAS radiation load · Slurm status", border_style="cyan"),
        summary,
        table,
        Text(
            "Saved pass is the last checkpoint, not live ray progress. COMPLETED does not imply convergence.",
            style="dim",
        ),
        locations,
    ]
    if error := record.get("error"):
        renderables.append(Text(f"❌ Submission error: {error}", style="bold red"))
    for job in failed:
        renderables.append(
            Text(
                f"📄 Logs for {job.get('component') or 'merge (stage)'}: {job.get('stdout', '-')} | {job.get('stderr', '-')}",
                style="yellow",
            )
        )
    for warning in dict.fromkeys(warnings):
        renderables.append(Text(f"⚠ Warning: {warning}", style="yellow"))
    if footer is not None:
        renderables.append(footer)
    return Group(*renderables)


def main() -> int:
    values = task_inputs()
    config = load_config(Path(values["RADLOAD_CONFIG"]))
    path = submission_path(config.output.uri)
    mode = values["RADLOAD_STATUS_MODE"]
    interval = float(values["RADLOAD_STATUS_INTERVAL"])
    timeout = float(values["RADLOAD_STATUS_TIMEOUT"])
    if mode not in {"once", "watch"}:
        raise ValueError("Use the once/watch modes defined in the status Pixi task")
    if not all(math.isfinite(value) and value > 0 for value in (interval, timeout)):
        raise ValueError("Status interval and query timeout must be finite and positive")
    terminal: dict[JobKey, JobState] = {}
    console = Console(highlight=False)
    display = (
        Live(console=console, auto_refresh=False, vertical_overflow="ellipsis")
        if mode == "watch" and console.is_terminal
        else nullcontext()
    )
    with display as live:
        while True:
            record = read_record(path, config.output.uri)
            if record.get("output_occurrence") != config.output.occurrence:
                raise ValueError(
                    "Submission record does not match the configured output occurrence"
                )
            states, warnings = query_states(record, timeout, terminal)
            query_failed = bool(warnings)
            submitted = [job_key(job, record) for job in jobs(record) if job["job_id"] is not None]
            unknown = any(states.get(key, JobState()).state == "UNKNOWN" for key in submitted)
            finished = record["status"] not in SUBMITTING_STATES and all(
                states.get(key, JobState()).state in TERMINAL_STATES for key in submitted
            )
            unavailable = unknown and query_failed
            footer = None
            if mode == "watch":
                if finished:
                    success = all(states[key].state == "COMPLETED" for key in submitted)
                    footer = Text(
                        "🏁 All submitted jobs have finished.",
                        style="bold green" if success else "bold yellow",
                    )
                elif unavailable:
                    footer = Text(
                        "⚠ Cannot continue watching with incomplete Slurm queries; retry on the submitting cluster.",
                        style="bold red",
                    )
                else:
                    footer = Text(
                        "🔄 Watching; Ctrl+C stops the display. Missing job records remain UNKNOWN.",
                        style="cyan",
                    )
            snapshot = render_status(
                record, path, states, warnings, width=console.width, footer=footer
            )
            if live is not None:
                live.update(snapshot, refresh=True)
            else:
                console.print(snapshot)
            if mode == "once":
                return 1 if unavailable else 0
            if finished:
                return 0
            if unavailable:
                return 1
            time.sleep(interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        Console().print(Text("🛑 Stopped watching; jobs are unchanged.", style="yellow"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        Console(stderr=True).print(Text(f"❌ Status unavailable: {error}", style="bold red"))
        raise SystemExit(1) from None
